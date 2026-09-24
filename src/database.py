import os
import json
import time
import uuid
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()


class Database:
    _instance: Optional["Database"] = None

    @classmethod
    def get_instance(cls) -> "Database":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, database_url: Optional[str] = None):
        raw_url = database_url or os.getenv("DATABASE_URL")
        
        if raw_url:
            raw_url = raw_url.strip()
            # Standardize postgresql prefix for SQLAlchemy
            if raw_url.startswith("postgres://"):
                raw_url = "postgresql://" + raw_url[len("postgres://"):]

            # Safely encode special characters in password (such as '@', '#', '%', etc.)
            if "://" in raw_url and "@" in raw_url:
                scheme, rest = raw_url.split("://", 1)
                auth, endpoint = rest.rsplit("@", 1)
                if ":" in auth:
                    user, password = auth.split(":", 1)
                    encoded_password = urllib.parse.quote_plus(urllib.parse.unquote_plus(password))
                    raw_url = f"{scheme}://{user}:{encoded_password}@{endpoint}"

            self.database_url = raw_url
            self.is_postgres = "postgresql" in raw_url
            endpoint_display = raw_url.rsplit("@", 1)[-1] if "@" in raw_url else "configured host"
            print(f"[INFO] Connecting to database: {endpoint_display}")
            self.engine: Engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_size=10 if self.is_postgres else 5,
                max_overflow=20 if self.is_postgres else 10,
            )
        else:
            # Local SQLite fallback if DATABASE_URL is not yet provided in .env
            local_db_path = Path("data/ragkno.sqlite3")
            local_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{local_db_path.resolve()}"
            self.is_postgres = False
            print(f"[INFO] DATABASE_URL not found in environment. Using local SQLite at {local_db_path}")
            self.engine: Engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
            )

        self._init_schema()

    def _init_schema(self) -> None:
        turns_id_def = "BIGSERIAL PRIMARY KEY" if self.is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        ts_def = "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP" if self.is_postgres else "DATETIME DEFAULT CURRENT_TIMESTAMP"

        schema_statements = [
            f"""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                picture TEXT,
                created_at {ts_def},
                last_login_at {ts_def}
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at {ts_def},
                updated_at {ts_def}
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                action_json TEXT,
                created_at {ts_def}
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS user_sources (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'file',
                chunk_count INTEGER DEFAULT 0,
                created_at {ts_def}
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS chat_memory_turns (
                id {turns_id_def},
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at {ts_def}
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS chat_memory_summaries (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                last_turn_id BIGINT NOT NULL DEFAULT 0,
                updated_at {ts_def},
                PRIMARY KEY (user_id, session_id)
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS user_drive_tokens (
                user_id TEXT PRIMARY KEY,
                token TEXT NOT NULL,
                refresh_token TEXT,
                token_uri TEXT NOT NULL,
                client_id TEXT NOT NULL,
                client_secret TEXT NOT NULL,
                scopes_json TEXT NOT NULL DEFAULT '[]',
                expiry_ts REAL,
                updated_at {ts_def}
            );
            """,
            f"""
            CREATE TABLE IF NOT EXISTS feedbacks (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                rating TEXT NOT NULL,
                feedback TEXT NOT NULL,
                created_at {ts_def}
            );
            """,
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);",
            "CREATE INDEX IF NOT EXISTS idx_sources_user ON user_sources(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_sources_key ON user_sources(user_id, source_key);",
            "CREATE INDEX IF NOT EXISTS idx_mem_user_session ON chat_memory_turns(user_id, session_id);",
            "CREATE INDEX IF NOT EXISTS idx_feedbacks_user ON feedbacks(user_id);",
        ]

        with self.engine.begin() as conn:
            for stmt in schema_statements:
                conn.execute(text(stmt.strip()))
            for idx in indexes:
                conn.execute(text(idx.strip()))

        print(f"[INFO] Database tables initialized successfully (Postgres={self.is_postgres}).")

    # -----------------------------------------------------------------------
    # User Operations
    # -----------------------------------------------------------------------
    def upsert_user(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(user_info.get("sub") or user_info.get("id") or "").strip()
        email = str(user_info.get("email") or "").strip()
        if not user_id or not email:
            raise ValueError("user_id and email are required for upsert_user.")

        name = str(user_info.get("name") or email.split("@")[0]).strip()
        picture = str(user_info.get("picture") or "").strip()

        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT id, email, name, picture FROM users WHERE id = :id OR email = :email"),
                {"id": user_id, "email": email},
            ).mappings().first()

            if row:
                conn.execute(
                    text(
                        """
                        UPDATE users
                        SET name = :name, picture = :picture, last_login_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                        """
                    ),
                    {"id": user_id, "name": name, "picture": picture},
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO users (id, email, name, picture)
                        VALUES (:id, :email, :name, :picture)
                        """
                    ),
                    {"id": user_id, "email": email, "name": name, "picture": picture},
                )

        return {"id": user_id, "email": email, "name": name, "picture": picture}

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, email, name, picture FROM users WHERE id = :id"),
                {"id": user_id},
            ).mappings().first()
            return dict(row) if row else None

    # -----------------------------------------------------------------------
    # Thread Operations
    # -----------------------------------------------------------------------
    def get_user_threads(self, user_id: str) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, user_id, title, created_at, updated_at
                    FROM threads
                    WHERE user_id = :user_id
                    ORDER BY updated_at DESC
                    """
                ),
                {"user_id": user_id},
            ).mappings().fetchall()

            result = []
            for r in rows:
                item = dict(r)
                item["createdAt"] = item.get("created_at")
                item["updatedAt"] = item.get("updated_at")
                item["sessionId"] = item["id"]
                result.append(item)
            return result

    def create_thread(self, user_id: str, title: str = "New Chat", thread_id: Optional[str] = None) -> Dict[str, Any]:
        tid = thread_id or f"thread_{uuid.uuid4().hex}"
        t_title = (title or "New Chat").strip()[:100]

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO threads (id, user_id, title, created_at, updated_at)
                    VALUES (:id, :user_id, :title, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """
                ),
                {"id": tid, "user_id": user_id, "title": t_title},
            )

        return {
            "id": tid,
            "sessionId": tid,
            "user_id": user_id,
            "title": t_title,
            "messages": [],
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000),
        }

    def get_thread(self, thread_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            if user_id:
                row = conn.execute(
                    text("SELECT id, user_id, title, created_at, updated_at FROM threads WHERE id = :id AND user_id = :user_id"),
                    {"id": thread_id, "user_id": user_id},
                ).mappings().first()
            else:
                row = conn.execute(
                    text("SELECT id, user_id, title, created_at, updated_at FROM threads WHERE id = :id"),
                    {"id": thread_id},
                ).mappings().first()

            if not row:
                return None
            item = dict(row)
            item["sessionId"] = item["id"]
            return item

    def rename_thread(self, thread_id: str, user_id: str, title: str) -> bool:
        clean_title = (title or "New Chat").strip()[:100]
        with self.engine.begin() as conn:
            res = conn.execute(
                text(
                    """
                    UPDATE threads
                    SET title = :title, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id AND user_id = :user_id
                    """
                ),
                {"id": thread_id, "user_id": user_id, "title": clean_title},
            )
            return res.rowcount > 0

    def touch_thread(self, thread_id: str, user_id: Optional[str] = None, title: Optional[str] = None) -> None:
        params: Dict[str, Any] = {"id": thread_id}
        clauses = ["updated_at = CURRENT_TIMESTAMP"]
        if title:
            clauses.append("title = :title")
            params["title"] = title.strip()[:100]

        sql = f"UPDATE threads SET {', '.join(clauses)} WHERE id = :id"
        if user_id:
            sql += " AND user_id = :user_id"
            params["user_id"] = user_id

        with self.engine.begin() as conn:
            conn.execute(text(sql), params)

    def delete_thread(self, thread_id: str, user_id: str) -> bool:
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM messages WHERE thread_id = :id AND user_id = :user_id"),
                {"id": thread_id, "user_id": user_id},
            )
            conn.execute(
                text("DELETE FROM chat_memory_turns WHERE session_id = :id AND user_id = :user_id"),
                {"id": thread_id, "user_id": user_id},
            )
            conn.execute(
                text("DELETE FROM chat_memory_summaries WHERE session_id = :id AND user_id = :user_id"),
                {"id": thread_id, "user_id": user_id},
            )
            res = conn.execute(
                text("DELETE FROM threads WHERE id = :id AND user_id = :user_id"),
                {"id": thread_id, "user_id": user_id},
            )
            return res.rowcount > 0

    # -----------------------------------------------------------------------
    # Message Operations
    # -----------------------------------------------------------------------
    def get_thread_messages(self, thread_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT id, thread_id, user_id, role, text, sources_json, action_json, created_at FROM messages WHERE thread_id = :thread_id"
        params: Dict[str, Any] = {"thread_id": thread_id}
        if user_id:
            sql += " AND user_id = :user_id"
            params["user_id"] = user_id
        sql += " ORDER BY created_at ASC"

        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().fetchall()

            messages = []
            for r in rows:
                item = dict(r)
                sources = []
                if item.get("sources_json"):
                    try:
                        sources = json.loads(item["sources_json"])
                    except Exception:
                        sources = []
                action = None
                if item.get("action_json"):
                    try:
                        action = json.loads(item["action_json"])
                    except Exception:
                        action = None

                messages.append({
                    "id": item["id"],
                    "role": item["role"],
                    "text": item["text"],
                    "sources": sources,
                    "action": action,
                    "createdAt": item.get("created_at"),
                })
            return messages

    def append_message(
        self,
        thread_id: str,
        user_id: str,
        role: str,
        text_content: str,
        sources: Optional[List[Any]] = None,
        message_id: Optional[str] = None,
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mid = message_id or f"msg_{uuid.uuid4().hex}"
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        action_json = json.dumps(action, ensure_ascii=False) if action else None

        with self.engine.begin() as conn:
            # Ensure thread exists
            thread_row = conn.execute(
                text("SELECT id, title FROM threads WHERE id = :id"),
                {"id": thread_id},
            ).mappings().first()

            if not thread_row:
                initial_title = text_content.strip()[:48] if role == "user" else "New Chat"
                conn.execute(
                    text(
                        """
                        INSERT INTO threads (id, user_id, title, created_at, updated_at)
                        VALUES (:id, :user_id, :title, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """
                    ),
                    {"id": thread_id, "user_id": user_id, "title": initial_title or "New Chat"},
                )
            else:
                # If first user message and title is "New Chat", auto-update title
                if role == "user" and thread_row["title"] == "New Chat":
                    auto_title = text_content.strip()[:48]
                    if auto_title:
                        conn.execute(
                            text("UPDATE threads SET title = :title, updated_at = CURRENT_TIMESTAMP WHERE id = :id"),
                            {"title": auto_title, "id": thread_id},
                        )
                else:
                    conn.execute(
                        text("UPDATE threads SET updated_at = CURRENT_TIMESTAMP WHERE id = :id"),
                        {"id": thread_id},
                    )

            conn.execute(
                text(
                    """
                    INSERT INTO messages (id, thread_id, user_id, role, text, sources_json, action_json, created_at)
                    VALUES (:id, :thread_id, :user_id, :role, :text, :sources_json, :action_json, CURRENT_TIMESTAMP)
                    """
                ),
                {
                    "id": mid,
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "role": role,
                    "text": text_content,
                    "sources_json": sources_json,
                    "action_json": action_json,
                },
            )

        return {
            "id": mid,
            "threadId": thread_id,
            "userId": user_id,
            "role": role,
            "text": text_content,
            "sources": sources or [],
            "action": action,
        }

    # -----------------------------------------------------------------------
    # User Sources Operations (File, URL, Drive)
    # -----------------------------------------------------------------------
    def record_user_source(
        self,
        user_id: str,
        source_key: str,
        source_name: str,
        source_type: str = "file",
        chunk_count: int = 0,
    ) -> Dict[str, Any]:
        normalized_key = str(source_key or "").strip().lower().rstrip("/")
        name = str(source_name or source_key).strip()
        uid = str(user_id or "system").strip()
        sid = f"src_{uuid.uuid4().hex[:12]}"

        with self.engine.begin() as conn:
            existing = conn.execute(
                text("SELECT id, chunk_count FROM user_sources WHERE user_id = :user_id AND source_key = :key"),
                {"user_id": uid, "key": normalized_key},
            ).mappings().first()

            if existing:
                conn.execute(
                    text(
                        """
                        UPDATE user_sources
                        SET chunk_count = chunk_count + :chunks, source_name = :name
                        WHERE id = :id
                        """
                    ),
                    {"id": existing["id"], "chunks": chunk_count, "name": name},
                )
                return {"id": existing["id"], "source_key": normalized_key, "source_name": name, "chunk_count": existing["chunk_count"] + chunk_count}
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO user_sources (id, user_id, source_key, source_name, source_type, chunk_count)
                        VALUES (:id, :user_id, :key, :name, :type, :chunks)
                        """
                    ),
                    {
                        "id": sid,
                        "user_id": uid,
                        "key": normalized_key,
                        "name": name,
                        "type": source_type,
                        "chunks": chunk_count,
                    },
                )
                return {"id": sid, "source_key": normalized_key, "source_name": name, "chunk_count": chunk_count}

    def get_user_sources(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT id, user_id, source_key, source_name, source_type, chunk_count, created_at FROM user_sources"
        params: Dict[str, Any] = {}
        if user_id and user_id.strip() not in ("*", "all"):
            sql += " WHERE user_id = :user_id OR user_id = 'system'"
            params["user_id"] = user_id.strip()
        sql += " ORDER BY created_at DESC"

        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().fetchall()
            return [dict(r) for r in rows]

    def delete_user_source(self, user_id: str, source_key: str) -> bool:
        normalized_key = str(source_key or "").strip().lower().rstrip("/")
        with self.engine.begin() as conn:
            res = conn.execute(
                text("DELETE FROM user_sources WHERE user_id = :user_id AND source_key = :key"),
                {"user_id": user_id, "key": normalized_key},
            )
            return res.rowcount > 0

    # -----------------------------------------------------------------------
    # Chat Memory Operations (Scoped per user_id + session_id)
    # -----------------------------------------------------------------------
    def append_memory_turn(self, user_id: str, session_id: str, role: str, text_content: str) -> None:
        safe_text = (text_content or "").strip()[:4000]
        if not safe_text:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO chat_memory_turns (user_id, session_id, role, text)
                    VALUES (:user_id, :session_id, :role, :text)
                    """
                ),
                {"user_id": user_id, "session_id": session_id, "role": role, "text": safe_text},
            )

    def get_recent_memory_turns(self, user_id: str, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, role, text, created_at
                    FROM chat_memory_turns
                    WHERE user_id = :user_id AND session_id = :session_id
                    ORDER BY id DESC
                    LIMIT :limit
                    """
                ),
                {"user_id": user_id, "session_id": session_id, "limit": limit},
            ).mappings().fetchall()

            items = [dict(r) for r in rows]
            items.reverse()
            return items

    def get_memory_turn_count(self, user_id: str, session_id: str) -> int:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) AS total FROM chat_memory_turns WHERE user_id = :user_id AND session_id = :session_id"),
                {"user_id": user_id, "session_id": session_id},
            ).mappings().first()
            return int(row["total"]) if row else 0

    def get_memory_summary_record(self, user_id: str, session_id: str) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT summary_text, last_turn_id FROM chat_memory_summaries WHERE user_id = :user_id AND session_id = :session_id"),
                {"user_id": user_id, "session_id": session_id},
            ).mappings().first()
            if not row:
                return {"summary_text": "", "last_turn_id": 0}
            return {"summary_text": row["summary_text"], "last_turn_id": int(row["last_turn_id"])}

    def get_memory_turns_since(self, user_id: str, session_id: str, last_turn_id: int) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, role, text, created_at
                    FROM chat_memory_turns
                    WHERE user_id = :user_id AND session_id = :session_id AND id > :last_turn_id
                    ORDER BY id ASC
                    """
                ),
                {"user_id": user_id, "session_id": session_id, "last_turn_id": last_turn_id},
            ).mappings().fetchall()
            return [dict(r) for r in rows]

    def upsert_memory_summary(self, user_id: str, session_id: str, summary_text: str, last_turn_id: int) -> None:
        clean_text = (summary_text or "").strip()
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT user_id FROM chat_memory_summaries WHERE user_id = :user_id AND session_id = :session_id"),
                {"user_id": user_id, "session_id": session_id},
            ).mappings().first()

            if row:
                conn.execute(
                    text(
                        """
                        UPDATE chat_memory_summaries
                        SET summary_text = :summary_text, last_turn_id = :last_turn_id, updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = :user_id AND session_id = :session_id
                        """
                    ),
                    {
                        "summary_text": clean_text,
                        "last_turn_id": last_turn_id,
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO chat_memory_summaries (user_id, session_id, summary_text, last_turn_id)
                        VALUES (:user_id, :session_id, :summary_text, :last_turn_id)
                        """
                    ),
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "summary_text": clean_text,
                        "last_turn_id": last_turn_id,
                    },
                )

    def clear_memory_session(self, user_id: str, session_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM chat_memory_turns WHERE user_id = :user_id AND session_id = :session_id"),
                {"user_id": user_id, "session_id": session_id},
            )
            conn.execute(
                text("DELETE FROM chat_memory_summaries WHERE user_id = :user_id AND session_id = :session_id"),
                {"user_id": user_id, "session_id": session_id},
            )

    # -----------------------------------------------------------------------
    # User Google Drive Token Operations
    # -----------------------------------------------------------------------

    def save_user_drive_token(self, user_id: str, token_data: Dict[str, Any]) -> None:
        uid = str(user_id or "").strip()
        if not uid:
            raise ValueError("user_id is required to save drive token")

        token = token_data.get("token") or ""
        refresh_token = token_data.get("refresh_token")
        token_uri = token_data.get("token_uri") or "https://oauth2.googleapis.com/token"
        client_id = token_data.get("client_id") or ""
        client_secret = token_data.get("client_secret") or ""
        scopes = token_data.get("scopes") or []
        scopes_json = json.dumps(scopes)
        expiry_ts = token_data.get("expiry_ts")

        with self.engine.begin() as conn:
            existing = conn.execute(
                text("SELECT user_id, refresh_token FROM user_drive_tokens WHERE user_id = :uid"),
                {"uid": uid},
            ).mappings().first()

            if existing:
                # Keep existing refresh_token if new one is omitted by Google
                final_refresh = refresh_token or existing.get("refresh_token")
                conn.execute(
                    text(
                        """
                        UPDATE user_drive_tokens
                        SET token = :token,
                            refresh_token = :refresh_token,
                            token_uri = :token_uri,
                            client_id = :client_id,
                            client_secret = :client_secret,
                            scopes_json = :scopes_json,
                            expiry_ts = :expiry_ts,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = :uid
                        """
                    ),
                    {
                        "uid": uid,
                        "token": token,
                        "refresh_token": final_refresh,
                        "token_uri": token_uri,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scopes_json": scopes_json,
                        "expiry_ts": expiry_ts,
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO user_drive_tokens (
                            user_id, token, refresh_token, token_uri, client_id, client_secret, scopes_json, expiry_ts
                        ) VALUES (
                            :uid, :token, :refresh_token, :token_uri, :client_id, :client_secret, :scopes_json, :expiry_ts
                        )
                        """
                    ),
                    {
                        "uid": uid,
                        "token": token,
                        "refresh_token": refresh_token,
                        "token_uri": token_uri,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scopes_json": scopes_json,
                        "expiry_ts": expiry_ts,
                    },
                )

    def get_user_drive_token(self, user_id: str) -> Optional[Dict[str, Any]]:
        uid = str(user_id or "").strip()
        if not uid:
            return None
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT * FROM user_drive_tokens WHERE user_id = :uid"),
                {"uid": uid},
            ).mappings().first()
            if not row:
                return None
            res = dict(row)
            try:
                res["scopes"] = json.loads(res.get("scopes_json") or "[]")
            except Exception:
                res["scopes"] = []
            return res

    def delete_user_drive_token(self, user_id: str) -> bool:
        uid = str(user_id or "").strip()
        if not uid:
            return False
        with self.engine.begin() as conn:
            res = conn.execute(
                text("DELETE FROM user_drive_tokens WHERE user_id = :uid"),
                {"uid": uid},
            )
            return bool(res.rowcount and res.rowcount > 0)

    # -----------------------------------------------------------------------
    # Feedback Operations
    # -----------------------------------------------------------------------
    def save_feedback(self, user_id: Optional[str], rating: str, feedback: str) -> Dict[str, Any]:
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO feedbacks (id, user_id, rating, feedback, created_at)
                VALUES (:id, :user_id, :rating, :feedback, CURRENT_TIMESTAMP)
                """),
                {
                    "id": feedback_id,
                    "user_id": str(user_id or "anonymous").strip(),
                    "rating": str(rating or "").strip(),
                    "feedback": str(feedback or "").strip(),
                },
            )
        return {
            "id": feedback_id,
            "user_id": user_id,
            "rating": rating,
            "feedback": feedback,
            "created_at": now,
        }

    def list_feedbacks(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            if user_id:
                stmt = text("""
                    SELECT id, user_id, rating, feedback, created_at
                    FROM feedbacks
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                rows = conn.execute(stmt, {"user_id": str(user_id).strip(), "limit": limit}).fetchall()
            else:
                stmt = text("""
                    SELECT id, user_id, rating, feedback, created_at
                    FROM feedbacks
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                rows = conn.execute(stmt, {"limit": limit}).fetchall()

            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "user_id": r[1],
                    "rating": r[2],
                    "feedback": r[3],
                    "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4]),
                })
            return results



