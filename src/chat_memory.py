import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class ChatMemoryStore:
    def __init__(self, db_path: str = "data/chat_memory.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    last_turn_id INTEGER NOT NULL DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session_id ON turns(session_id)")

    @staticmethod
    def _clip_text(text: str, max_len: int = 4000) -> str:
        value = (text or "").strip()
        if len(value) <= max_len:
            return value
        return value[:max_len]

    def append_turn(self, session_id: str, role: str, text: str) -> None:
        safe_text = self._clip_text(text)
        if not safe_text:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO turns(session_id, role, text) VALUES (?, ?, ?)",
                (session_id, role, safe_text),
            )

    def get_recent_turns(self, session_id: str, limit: int = 8) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, text, created_at
                FROM turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        items = [dict(row) for row in rows]
        items.reverse()
        return items

    def get_turn_count(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["total"]) if row else 0

    def get_summary_record(self, session_id: str) -> Dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_text, last_turn_id FROM summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if not row:
            return {"summary_text": "", "last_turn_id": 0}
        return {"summary_text": row["summary_text"], "last_turn_id": int(row["last_turn_id"])}

    def get_turns_since(self, session_id: str, last_turn_id: int) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, text, created_at
                FROM turns
                WHERE session_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (session_id, last_turn_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_summary(self, session_id: str, summary_text: str, last_turn_id: int) -> None:
        text = (summary_text or "").strip()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries(session_id, summary_text, last_turn_id)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET
                  summary_text = excluded.summary_text,
                  last_turn_id = excluded.last_turn_id,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, text, last_turn_id),
            )

    def clear_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))

    def build_memory_context(self, session_id: str, recent_limit: int = 8) -> Optional[str]:
        summary = self.get_summary_record(session_id).get("summary_text", "").strip()
        recent = self.get_recent_turns(session_id, limit=recent_limit)

        parts = []
        if summary:
            parts.append("Conversation summary:\n" + summary)

        if recent:
            recent_lines = [f"{item['role'].title()}: {item['text']}" for item in recent]
            parts.append("Recent turns:\n" + "\n".join(recent_lines))

        if not parts:
            return None
        return "\n\n".join(parts)
