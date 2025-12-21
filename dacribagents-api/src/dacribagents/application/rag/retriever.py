"""RAG Retriever - Unified retrieval across emails, documents, and PDFs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from loguru import logger

from dacribagents.infrastructure.embeddings.factory import EmbeddingsFactory, Embedder
from dacribagents.infrastructure.milvus_client import MilvusClientWrapper, get_milvus_client


@dataclass
class RetrievedDocument:
    """A single retrieved document with metadata."""

    content: str
    score: float
    source: Literal["email", "document", "pdf"]
    metadata: dict = field(default_factory=dict)

    @property
    def snippet(self) -> str:
        """Get a truncated snippet of content (max 500 chars)."""
        if len(self.content) <= 500:
            return self.content
        return self.content[:497] + "..."

    def format_for_context(self) -> str:
        """Format this document for inclusion in agent context."""
        lines = []

        # Header with source info
        if self.source == "email":
            subject = self.metadata.get("subject", "No subject")
            sender = self.metadata.get("sender", "Unknown")
            date = self.metadata.get("date", "")
            if date:
                try:
                    dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
                    date = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            lines.append(f"[Email from {sender}, {date}]")
            lines.append(f"Subject: {subject}")
        elif self.source == "document":
            title = self.metadata.get("title", "Untitled")
            doc_type = self.metadata.get("doc_type", "document")
            lines.append(f"[{doc_type.title()}: {title}]")
        elif self.source == "pdf":
            filename = self.metadata.get("filename", "Unknown PDF")
            page = self.metadata.get("page", "")
            page_info = f", page {page}" if page else ""
            lines.append(f"[PDF: {filename}{page_info}]")

        # Content snippet
        lines.append(self.snippet)

        return "\n".join(lines)


class RAGRetriever:
    """
    Unified RAG retriever for searching across multiple collections.

    Supports:
    - emails: Ingested from woodcreek.me
    - documents: HOA docs, CC&Rs, manuals (future)
    - pdfs: Extracted PDF content (future)
    """

    # Collection configurations
    COLLECTIONS = {
        "emails": {
            "output_fields": ["message_id", "subject", "sender", "date", "text"],
            "content_field": "text",
            "source_type": "email",
        },
        "documents": {
            "output_fields": ["title", "doc_type", "content", "created_at"],
            "content_field": "content",
            "source_type": "document",
        },
        "pdfs": {
            "output_fields": ["filename", "page", "content", "extracted_at"],
            "content_field": "content",
            "source_type": "pdf",
        },
    }

    def __init__(
        self,
        milvus_client: MilvusClientWrapper | None = None,
        embedder: Embedder | None = None,
    ):
        self.milvus = milvus_client or get_milvus_client()
        self.embedder = embedder or EmbeddingsFactory.from_env()

    def _get_available_collections(self) -> list[str]:
        """Get list of collections that actually exist in Milvus."""
        available = []
        for name in self.COLLECTIONS:
            try:
                if self.milvus.client.has_collection(name):
                    available.append(name)
            except Exception as e:
                logger.debug(f"Collection {name} not available: {e}")
        return available

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collections: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedDocument]:
        """
        Retrieve relevant documents from Milvus.

        Args:
            query: The search query
            top_k: Number of results to return (per collection if multiple)
            collections: Which collections to search (default: all available)
            min_score: Minimum similarity score threshold (0-1 for cosine)

        Returns:
            List of RetrievedDocument sorted by score (highest first)
        """
        # Determine which collections to search
        available = self._get_available_collections()
        if not available:
            logger.warning("No collections available for RAG retrieval")
            return []

        if collections:
            search_collections = [c for c in collections if c in available]
        else:
            search_collections = available

        if not search_collections:
            logger.warning(f"Requested collections not available. Available: {available}")
            return []

        logger.info(f"RAG search: '{query[:50]}...' in {search_collections}")

        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Search each collection
        all_results: list[RetrievedDocument] = []

        for collection_name in search_collections:
            config = self.COLLECTIONS[collection_name]

            try:
                results = self.milvus.client.search(
                    collection_name=collection_name,
                    data=[query_embedding],
                    limit=top_k,
                    output_fields=config["output_fields"],
                )

                for hit in results[0]:
                    score = hit["distance"]  # Cosine similarity (0-1)

                    if score < min_score:
                        continue

                    entity = hit["entity"]
                    content = entity.get(config["content_field"], "")

                    # Build metadata from all fields except content
                    metadata = {
                        k: v for k, v in entity.items()
                        if k != config["content_field"]
                    }
                    metadata["collection"] = collection_name

                    doc = RetrievedDocument(
                        content=content,
                        score=score,
                        source=config["source_type"],
                        metadata=metadata,
                    )
                    all_results.append(doc)

            except Exception as e:
                logger.error(f"Error searching {collection_name}: {e}")
                continue

        # Sort by score (highest first) and limit to top_k total
        all_results.sort(key=lambda x: x.score, reverse=True)
        top_results = all_results[:top_k]

        logger.info(f"RAG retrieved {len(top_results)} documents (from {len(all_results)} total)")

        return top_results

    def retrieve_sync(
        self,
        query: str,
        top_k: int = 5,
        collections: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedDocument]:
        """Synchronous version of retrieve for non-async contexts."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, can't use run_until_complete
                # Fall back to direct implementation
                return self._retrieve_sync_impl(query, top_k, collections, min_score)
            return loop.run_until_complete(
                self.retrieve(query, top_k, collections, min_score)
            )
        except RuntimeError:
            return self._retrieve_sync_impl(query, top_k, collections, min_score)

    def _retrieve_sync_impl(
        self,
        query: str,
        top_k: int = 5,
        collections: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedDocument]:
        """Direct synchronous implementation."""
        available = self._get_available_collections()
        if not available:
            return []

        search_collections = (
            [c for c in collections if c in available]
            if collections
            else available
        )

        if not search_collections:
            return []

        query_embedding = self.embedder.embed(query)
        all_results: list[RetrievedDocument] = []

        for collection_name in search_collections:
            config = self.COLLECTIONS[collection_name]

            try:
                results = self.milvus.client.search(
                    collection_name=collection_name,
                    data=[query_embedding],
                    limit=top_k,
                    output_fields=config["output_fields"],
                )

                for hit in results[0]:
                    score = hit["distance"]
                    if score < min_score:
                        continue

                    entity = hit["entity"]
                    content = entity.get(config["content_field"], "")
                    metadata = {
                        k: v for k, v in entity.items()
                        if k != config["content_field"]
                    }
                    metadata["collection"] = collection_name

                    all_results.append(RetrievedDocument(
                        content=content,
                        score=score,
                        source=config["source_type"],
                        metadata=metadata,
                    ))

            except Exception as e:
                logger.error(f"Error searching {collection_name}: {e}")
                continue

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]


def build_context(documents: list[RetrievedDocument], max_chars: int = 4000) -> str:
    """
    Build a context string from retrieved documents for agent prompts.

    Args:
        documents: List of retrieved documents
        max_chars: Maximum total characters for context

    Returns:
        Formatted context string
    """
    if not documents:
        return ""

    lines = ["=== Relevant Context ===", ""]

    total_chars = 0
    for i, doc in enumerate(documents, 1):
        formatted = doc.format_for_context()

        # Check if adding this would exceed limit
        if total_chars + len(formatted) + 10 > max_chars:
            lines.append(f"... ({len(documents) - i + 1} more results truncated)")
            break

        lines.append(formatted)
        lines.append("")  # Blank line between docs
        total_chars += len(formatted) + 1

    lines.append("=== End Context ===")

    return "\n".join(lines)


# Singleton instance
_retriever: RAGRetriever | None = None


def get_rag_retriever() -> RAGRetriever:
    """Get or create RAG retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever