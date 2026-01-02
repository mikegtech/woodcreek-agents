"""Milvus implementation of VectorStore for emails."""

from __future__ import annotations

from loguru import logger
from pymilvus import DataType

from dacribagents.application.ports.vector_store import VectorStore
from dacribagents.domain.entities.email_message import EmailMessage
from dacribagents.infrastructure.milvus_client import MilvusClientWrapper


COLLECTION_NAME = "emails"
EMBEDDING_DIM = 768  # nomic-ai/nomic-embed-text-v1.5


class MilvusEmailStore(VectorStore):
    """Store and retrieve email embeddings from Milvus."""

    def __init__(self, client: MilvusClientWrapper):
        self.client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create emails collection if it doesn't exist."""
        if self.client.client.has_collection(COLLECTION_NAME):
            logger.info(f"Collection {COLLECTION_NAME} already exists")
            return

        logger.info(f"Creating collection {COLLECTION_NAME}")
        self.client.client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=EMBEDDING_DIM,
            primary_field_name="id",
            id_type=DataType.VARCHAR,
            max_length=256,
            vector_field_name="embedding",
            metric_type="COSINE",
            auto_id=False,
        )

    def upsert_email(self, msg: EmailMessage, embedding: list[float]) -> None:
        """Insert or update an email with its embedding."""
        doc_id = f"{msg.account}:{msg.folder}:{msg.message_id}"

        data = {
            "id": doc_id,
            "embedding": embedding,
            "message_id": msg.message_id,
            "account": msg.account,
            "folder": msg.folder,
            "subject": msg.subject[:500] if msg.subject else "",
            "sender": msg.sender[:200] if msg.sender else "",
            "date": msg.date.isoformat(),
            "text": msg.text[:5000] if msg.text else "",  # Truncate for storage
        }

        self.client.client.upsert(
            collection_name=COLLECTION_NAME,
            data=[data],
        )
        logger.debug(f"Upserted email {doc_id}")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar emails."""
        results = self.client.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=top_k,
            output_fields=["message_id", "subject", "sender", "date", "text"],
        )

        return [
            {
                "id": hit["id"],
                "score": hit["distance"],
                "message_id": hit["entity"].get("message_id"),
                "subject": hit["entity"].get("subject"),
                "sender": hit["entity"].get("sender"),
                "date": hit["entity"].get("date"),
                "text": hit["entity"].get("text"),
            }
            for hit in results[0]
        ]