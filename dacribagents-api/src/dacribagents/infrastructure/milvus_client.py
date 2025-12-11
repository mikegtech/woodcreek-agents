"""Milvus vector database client for document embeddings and similarity search."""

from contextlib import asynccontextmanager
from typing import Any

from loguru import logger
from pymilvus import Collection, MilvusClient, connections, utility

from dacribagents.infrastructure.settings import Settings, get_settings


class MilvusClientWrapper:
    """Wrapper for Milvus vector database operations."""

    def __init__(self, settings: Settings | None = None):
        """Initialize Milvus client wrapper."""
        self.settings = settings or get_settings()
        self._client: MilvusClient | None = None

    def connect(self) -> MilvusClient:
        """Establish connection to Milvus."""
        if self._client is None:
            logger.info(f"Connecting to Milvus at {self.settings.milvus_uri}")
            self._client = MilvusClient(uri=self.settings.milvus_uri)
            logger.info("Milvus connection established")
        return self._client

    def disconnect(self) -> None:
        """Close Milvus connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Milvus connection closed")

    @property
    def client(self) -> MilvusClient:
        """Get or create Milvus client."""
        if self._client is None:
            return self.connect()
        return self._client

    def health_check(self) -> dict[str, Any]:
        """Check Milvus connection health."""
        try:
            # Use low-level connection for health check
            connections.connect(
                alias="health_check",
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
            )
            server_version = utility.get_server_version(using="health_check")
            connections.disconnect(alias="health_check")
            return {
                "status": "healthy",
                "uri": self.settings.milvus_uri,
                "server_version": server_version,
            }
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return {
                "status": "unhealthy",
                "uri": self.settings.milvus_uri,
                "error": str(e),
            }

    def ensure_collection(
        self,
        collection_name: str | None = None,
        dimension: int | None = None,
        recreate: bool = False,
    ) -> str:
        """Ensure collection exists, create if not."""
        name = collection_name or self.settings.milvus_collection_name
        dim = dimension or self.settings.embedding_dimension

        if self.client.has_collection(name):
            if recreate:
                logger.warning(f"Dropping existing collection: {name}")
                self.client.drop_collection(name)
            else:
                logger.info(f"Collection already exists: {name}")
                return name

        logger.info(f"Creating collection: {name} with dimension {dim}")
        self.client.create_collection(
            collection_name=name,
            dimension=dim,
            metric_type="COSINE",
            auto_id=True,
        )
        return name

    def insert_vectors(
        self,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
        collection_name: str | None = None,
    ) -> list[int]:
        """Insert vectors with metadata into collection."""
        name = collection_name or self.settings.milvus_collection_name
        data = [{"vector": v, **m} for v, m in zip(vectors, metadata, strict=True)]
        result = self.client.insert(collection_name=name, data=data)
        logger.debug(f"Inserted {len(vectors)} vectors into {name}")
        return result["ids"]

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        collection_name: str | None = None,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        name = collection_name or self.settings.milvus_collection_name
        results = self.client.search(
            collection_name=name,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr,
            output_fields=output_fields or ["*"],
        )
        return results[0] if results else []


# Singleton instance
_milvus_client: MilvusClientWrapper | None = None


def get_milvus_client() -> MilvusClientWrapper:
    """Get singleton Milvus client instance."""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClientWrapper()
    return _milvus_client


@asynccontextmanager
async def milvus_lifespan():
    """Async context manager for Milvus connection lifecycle."""
    client = get_milvus_client()
    client.connect()
    try:
        yield client
    finally:
        client.disconnect()