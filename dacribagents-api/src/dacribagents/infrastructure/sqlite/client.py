"""SQLite client for SMS conversation storage (short-term memory)."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Literal

from loguru import logger


@dataclass
class Message:
    """A single SMS message."""

    id: str
    conversation_id: str
    direction: Literal["inbound", "outbound"]
    provider: str
    provider_message_id: str | None
    ts: str
    from_number: str
    to_number: str
    text: str
    status: str = "stored"


@dataclass
class Conversation:
    """An SMS conversation."""

    conversation_id: str
    created_at: str
    updated_at: str
    state_json: str | None = None


class SQLiteClient:
    """SQLite client for SMS conversation and message storage."""

    def __init__(self, db_path: str | Path = "/app/data/conversations.db"):
        self.db_path = Path(db_path)
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connection() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('inbound','outbound')),
                    provider TEXT NOT NULL,
                    provider_message_id TEXT,
                    ts TEXT NOT NULL,
                    from_number TEXT NOT NULL,
                    to_number TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'stored',

                    UNIQUE(provider, provider_message_id),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv_ts
                    ON messages(conversation_id, ts);
            """)
            logger.info(f"SQLite database initialized at {self.db_path}")

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_or_create_conversation(self, conversation_id: str) -> Conversation:
        """Get existing conversation or create a new one."""
        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = cursor.fetchone()

            if row:
                return Conversation(
                    conversation_id=row["conversation_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    state_json=row["state_json"],
                )

            # Create new conversation
            conn.execute(
                """INSERT INTO conversations (conversation_id, created_at, updated_at)
                   VALUES (?, ?, ?)""",
                (conversation_id, now, now),
            )
            logger.info(f"Created new conversation: {conversation_id}")

            return Conversation(
                conversation_id=conversation_id,
                created_at=now,
                updated_at=now,
            )

    def add_message(
        self,
        conversation_id: str,
        direction: Literal["inbound", "outbound"],
        provider: str,
        from_number: str,
        to_number: str,
        text: str,
        provider_message_id: str | None = None,
        status: str = "stored",
    ) -> Message:
        """Add a message to a conversation."""
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            # Ensure conversation exists
            self.get_or_create_conversation(conversation_id)

            # Insert message
            conn.execute(
                """INSERT INTO messages 
                   (id, conversation_id, direction, provider, provider_message_id, ts, from_number, to_number, text, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, conversation_id, direction, provider, provider_message_id, now, from_number, to_number, text, status),
            )

            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )

        logger.debug(f"Added {direction} message to {conversation_id}: {text[:50]}...")

        return Message(
            id=msg_id,
            conversation_id=conversation_id,
            direction=direction,
            provider=provider,
            provider_message_id=provider_message_id,
            ts=now,
            from_number=from_number,
            to_number=to_number,
            text=text,
            status=status,
        )

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[Message]:
        """Get recent messages for a conversation (oldest first)."""
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM messages 
                   WHERE conversation_id = ?
                   ORDER BY ts DESC
                   LIMIT ?""",
                (conversation_id, limit),
            )
            rows = cursor.fetchall()

        # Return in chronological order (oldest first)
        messages = [
            Message(
                id=row["id"],
                conversation_id=row["conversation_id"],
                direction=row["direction"],
                provider=row["provider"],
                provider_message_id=row["provider_message_id"],
                ts=row["ts"],
                from_number=row["from_number"],
                to_number=row["to_number"],
                text=row["text"],
                status=row["status"],
            )
            for row in reversed(rows)
        ]

        return messages

    def update_message_status(self, message_id: str, status: str) -> None:
        """Update the status of a message."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE messages SET status = ? WHERE id = ?",
                (status, message_id),
            )

    def get_messages_for_archival(self, limit: int = 100) -> list[Message]:
        """Get old messages that should be archived to Milvus."""
        # Get messages from conversations with more than 20 messages
        # (keeping only the 20 most recent in SQLite)
        with self._connection() as conn:
            cursor = conn.execute(
                """
                WITH ranked AS (
                    SELECT *, 
                           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY ts DESC) as rn
                    FROM messages
                )
                SELECT * FROM ranked WHERE rn > 20
                ORDER BY ts ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            Message(
                id=row["id"],
                conversation_id=row["conversation_id"],
                direction=row["direction"],
                provider=row["provider"],
                provider_message_id=row["provider_message_id"],
                ts=row["ts"],
                from_number=row["from_number"],
                to_number=row["to_number"],
                text=row["text"],
                status=row["status"],
            )
            for row in rows
        ]

    def delete_archived_messages(self, message_ids: list[str]) -> int:
        """Delete messages that have been archived to Milvus."""
        if not message_ids:
            return 0

        with self._connection() as conn:
            placeholders = ",".join("?" * len(message_ids))
            cursor = conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                message_ids,
            )
            return cursor.rowcount


# Singleton instance
_client: SQLiteClient | None = None


def get_sqlite_client(db_path: str | None = None) -> SQLiteClient:
    """Get or create SQLite client singleton."""
    global _client
    if _client is None:
        from dacribagents.infrastructure import get_settings
        settings = get_settings()
        path = db_path or getattr(settings, "sqlite_db_path", "/app/data/conversations.db")
        _client = SQLiteClient(db_path=path)
    return _client