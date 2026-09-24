from typing import Dict, List, Optional
from src.database import Database


class ChatMemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db = Database.get_instance()

    def append_turn(self, session_id: str, role: str, text: str, user_id: str = "default") -> None:
        self.db.append_memory_turn(
            user_id=user_id or "default",
            session_id=session_id,
            role=role,
            text_content=text,
        )

    def get_recent_turns(self, session_id: str, limit: int = 8, user_id: str = "default") -> List[Dict]:
        return self.db.get_recent_memory_turns(
            user_id=user_id or "default",
            session_id=session_id,
            limit=limit,
        )

    def get_turn_count(self, session_id: str, user_id: str = "default") -> int:
        return self.db.get_memory_turn_count(
            user_id=user_id or "default",
            session_id=session_id,
        )

    def get_summary_record(self, session_id: str, user_id: str = "default") -> Dict:
        return self.db.get_memory_summary_record(
            user_id=user_id or "default",
            session_id=session_id,
        )

    def get_turns_since(self, session_id: str, last_turn_id: int, user_id: str = "default") -> List[Dict]:
        return self.db.get_memory_turns_since(
            user_id=user_id or "default",
            session_id=session_id,
            last_turn_id=last_turn_id,
        )

    def upsert_summary(self, session_id: str, summary_text: str, last_turn_id: int, user_id: str = "default") -> None:
        self.db.upsert_memory_summary(
            user_id=user_id or "default",
            session_id=session_id,
            summary_text=summary_text,
            last_turn_id=last_turn_id,
        )

    def clear_session(self, session_id: str, user_id: str = "default") -> None:
        self.db.clear_memory_session(
            user_id=user_id or "default",
            session_id=session_id,
        )

    def build_memory_context(self, session_id: str, recent_limit: int = 8, user_id: str = "default") -> Optional[str]:
        uid = user_id or "default"
        summary = self.get_summary_record(session_id, user_id=uid).get("summary_text", "").strip()
        recent = self.get_recent_turns(session_id, limit=recent_limit, user_id=uid)

        parts = []
        if summary:
            parts.append("Conversation summary:\n" + summary)

        if recent:
            recent_lines = [f"{item['role'].title()}: {item['text']}" for item in recent]
            parts.append("Recent turns:\n" + "\n".join(recent_lines))

        if not parts:
            return None
        return "\n\n".join(parts)
