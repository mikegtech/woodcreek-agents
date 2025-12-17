"""Embeddings factory for creating embedding clients."""

from __future__ import annotations

import os
from typing import Protocol

from loguru import logger


class Embedder(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded, dimension: {self._dimension}")

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingsFactory:
    """Factory for creating embedder instances."""

    @staticmethod
    def from_env() -> Embedder:
        """Create embedder based on environment configuration."""
        provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        if provider == "sentence-transformers":
            return SentenceTransformerEmbedder(model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @staticmethod
    def create(provider: str = "sentence-transformers", model_name: str | None = None) -> Embedder:
        """Create embedder with explicit configuration."""
        if provider == "sentence-transformers":
            return SentenceTransformerEmbedder(
                model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")