"""Milvus-based checkpoint store for email ingestion cursors."""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger
from pymilvus import DataType

from dacribagents.application.ports.checkpoint_store import CheckpointStore
from dacribagents.application.ports.email_source import EmailCursor
from dacribagents.infrastructure.milvus_client import MilvusClientWrapper


COLLECTION_NAME = "email_checkpoints"


class MilvusCheckpointStore(CheckpointStore):
    """Store email ingestion checkpoints in Milvus."""

    def __init__(self, client: MilvusClientWrapper):
        self.client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create checkpoints collection if it doesn't exist."""
        if self.client.client.has_collection(COLLECTION_NAME):
            return

        logger.info(f"Creating collection {COLLECTION_NAME}")
        # Simple key-value store, no vector search needed
        self.client.client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=8,  # Dummy dimension, not used for search
            primary_field_name="id",
            id_type=DataType.VARCHAR,
            max_length=256,
            vector_field_name="dummy_vector",
            auto_id=False,
        )

    def _make_id(self, account: str, folder: str) -> str:
        return f"{account}:{folder}"

    def load(self, account: str, folder: str) -> Optional[EmailCursor]:
        """Load checkpoint for account/folder."""
        checkpoint_id = self._make_id(account, folder)

        results = self.client.client.get(
            collection_name=COLLECTION_NAME,
            ids=[checkpoint_id],
            output_fields=["cursor_json"],
        )

        if not results:
            logger.debug(f"No checkpoint found for {checkpoint_id}")
            return None

        cursor_data = json.loads(results[0].get("cursor_json", "{}"))
        if not cursor_data:
            return None

        logger.debug(f"Loaded checkpoint for {checkpoint_id}: UID {cursor_data.get('last_uid')}")
        return EmailCursor(
            folder=cursor_data["folder"],
            uidvalidity=cursor_data["uidvalidity"],
            last_uid=cursor_data["last_uid"],
        )

    def save(self, account: str, cursor: EmailCursor) -> None:
        """Save checkpoint for account/folder."""
        checkpoint_id = self._make_id(account, cursor.folder)

        cursor_json = json.dumps({
            "folder": cursor.folder,
            "uidvalidity": cursor.uidvalidity,
            "last_uid": cursor.last_uid,
        })

        data = {
            "id": checkpoint_id,
            "dummy_vector": [0.0] * 8,
            "cursor_json": cursor_json,
        }

        self.client.client.upsert(
            collection_name=COLLECTION_NAME,
            data=[data],
        )
        logger.info(f"Saved checkpoint for {checkpoint_id}: UID {cursor.last_uid}")