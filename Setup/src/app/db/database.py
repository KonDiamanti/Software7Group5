"""SQLite access. Swapping to another database later only means changing this module and schema.sql."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- conversations -------------------------------------------------
    def create_conversation(self) -> int:
        cur = self.conn.execute("INSERT INTO conversations DEFAULT VALUES")
        self.conn.commit()
        return cur.lastrowid

    # --- messages ------------------------------------------------------
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        model_name: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, model_name) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, model_name),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_messages(self, conversation_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()

    # --- guard logs ----------------------------------------------------
    def add_guard_log(
        self,
        conversation_id: int,
        direction: str,
        guard_name: str,
        allowed: bool,
        reason: str = "",
        score: float | None = None,
        message_id: int | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO guard_logs "
            "(conversation_id, message_id, direction, guard_name, allowed, reason, score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                message_id,
                direction,
                guard_name,
                int(allowed),
                reason,
                score,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_guard_logs(self, conversation_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM guard_logs WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
