"""Embeddings factory for creating embedding clients."""
from __future__ import annotations

import os
from typing import Protocol

import httpx
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


# =============================================================================
# HTTP-Based Embedders (Remote Services)
# =============================================================================


class HttpEmbedder:
    """
    HTTP client for remote embedding services.
    
    Works with FastAPI embedding endpoints that accept:
    POST /embed or /encode
    {
        "texts": ["text1", "text2"],
        "prefix": "query: "  (optional)
    }
    
    Returns:
    {
        "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    }
    """
    
    def __init__(
        self,
        endpoint: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        dimension: int = 768,
        timeout: float = 30.0,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._dimension = dimension
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        
        logger.info(f"HTTP Embedder initialized: {self._endpoint}, dim={dimension}")
    
    def _call_api(self, texts: list[str], prefix: str) -> list[list[float]]:
        """Make HTTP request to embedding service."""
        # Try common endpoint paths
        endpoints_to_try = [
            f"{self._endpoint}/embed",
            f"{self._endpoint}/encode",
            f"{self._endpoint}/embeddings",
            self._endpoint,  # Maybe endpoint already includes path
        ]
        
        payload = {"texts": texts}
        if prefix:
            payload["prefix"] = prefix
        
        last_error = None
        for url in endpoints_to_try:
            try:
                response = self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Handle different response formats
                if "embeddings" in data:
                    return data["embeddings"]
                elif "vectors" in data:
                    return data["vectors"]
                elif isinstance(data, list):
                    return data
                else:
                    logger.warning(f"Unexpected response format from {url}: {list(data.keys())}")
                    continue
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    continue  # Try next endpoint
                last_error = e
                logger.error(f"HTTP error from {url}: {e}")
            except Exception as e:
                last_error = e
                logger.debug(f"Failed to connect to {url}: {e}")
        
        raise RuntimeError(f"Failed to get embeddings from {self._endpoint}: {last_error}")
    
    def embed(self, text: str) -> list[float]:
        """Embed a single query text."""
        result = self._call_api([text], self._query_prefix)
        return result[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of query texts."""
        return self._call_api(texts, self._query_prefix)
    
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages/documents with passage prefix."""
        return self._call_api(texts, self._passage_prefix)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            self._client.close()


class AsyncHttpEmbedder:
    """Async version of HTTP embedder for use in async contexts."""
    
    def __init__(
        self,
        endpoint: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        dimension: int = 768,
        timeout: float = 30.0,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._dimension = dimension
        self._timeout = timeout
        
        logger.info(f"Async HTTP Embedder initialized: {self._endpoint}, dim={dimension}")
    
    async def _call_api(self, texts: list[str], prefix: str) -> list[list[float]]:
        """Make async HTTP request to embedding service."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            endpoints_to_try = [
                f"{self._endpoint}/embed",
                f"{self._endpoint}/encode", 
                f"{self._endpoint}/embeddings",
                self._endpoint,
            ]
            
            payload = {"texts": texts}
            if prefix:
                payload["prefix"] = prefix
            
            last_error = None
            for url in endpoints_to_try:
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    if "embeddings" in data:
                        return data["embeddings"]
                    elif "vectors" in data:
                        return data["vectors"]
                    elif isinstance(data, list):
                        return data
                        
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        continue
                    last_error = e
                except Exception as e:
                    last_error = e
            
            raise RuntimeError(f"Failed to get embeddings from {self._endpoint}: {last_error}")
    
    async def embed(self, text: str) -> list[float]:
        """Embed a single query text."""
        result = await self._call_api([text], self._query_prefix)
        return result[0]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of query texts."""
        return await self._call_api(texts, self._query_prefix)
    
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passages/documents with passage prefix."""
        return await self._call_api(texts, self._passage_prefix)
    
    @property
    def dimension(self) -> int:
        return self._dimension


# =============================================================================
# Local Embedders (Fallback)
# =============================================================================


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder (fallback if HTTP unavailable)."""

    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        trust_remote_code: bool = False,
        query_prefix: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        self._dimension = self.model.get_sentence_embedding_dimension()
        self._query_prefix = query_prefix
        logger.info(f"Local embedding model loaded, dimension: {self._dimension}")

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        if self._query_prefix:
            text = f"{self._query_prefix}{text}"
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if self._query_prefix:
            texts = [f"{self._query_prefix}{t}" for t in texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder:
    """OpenAI embeddings provider."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from langchain_openai import OpenAIEmbeddings
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        logger.info(f"Loading OpenAI embedding model: {model_name}")
        self._client = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
        
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = self._dimensions.get(model_name, 1536)
        logger.info(f"OpenAI embeddings ready, dimension: {self._dimension}")
    
    def embed(self, text: str) -> list[float]:
        return self._client.embed_query(text)
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)
    
    @property
    def dimension(self) -> int:
        return self._dimension


# =============================================================================
# Model Configurations
# =============================================================================

EMBEDDING_MODELS = {
    "nomic": {
        "provider": "http",
        "model_name": "nomic-ai/nomic-embed-text-v1.5",  # For reference
        "endpoint_env": "NOMIC_ENDPOINT",
        "endpoint_default": "http://dataops.trupryce.ai:8001",
        "query_prefix": "search_query: ",      # For queries
        "passage_prefix": "search_document: ", # For documents
        "dimension": 768,
        "max_tokens": 8192,
        "default_relevance_threshold": 0.35,
        "default_grounding_threshold": 0.50,
        "score_range": "0.3-0.6 typical",
    },
    "e5-base": {
        "provider": "http",
        "model_name": "intfloat/e5-base-v2",  # For reference
        "endpoint_env": "E5_ENDPOINT",
        "endpoint_default": "http://dataops.trupryce.ai:8000",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "dimension": 768,
        "max_tokens": 256,
        "default_relevance_threshold": 0.60,
        "default_grounding_threshold": 0.70,
        "score_range": "0.6-0.85 typical",
    },
    "e5-large": {
        "provider": "sentence-transformers",
        "model_name": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "dimension": 1024,
        "max_tokens": 512,
        "default_relevance_threshold": 0.60,
        "default_grounding_threshold": 0.70,
        "score_range": "0.6-0.85 typical",
    },
    "e5-small": {
        "provider": "sentence-transformers",
        "model_name": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
        "dimension": 384,
        "max_tokens": 512,
        "default_relevance_threshold": 0.55,
        "default_grounding_threshold": 0.65,
        "score_range": "0.5-0.8 typical",
    },
    "openai-small": {
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "max_tokens": 8191,
        "default_relevance_threshold": 0.70,
        "default_grounding_threshold": 0.75,
        "score_range": "0.7-0.95 typical",
    },
    "openai-large": {
        "provider": "openai",
        "model_name": "text-embedding-3-large",
        "dimension": 3072,
        "max_tokens": 8191,
        "default_relevance_threshold": 0.70,
        "default_grounding_threshold": 0.75,
        "score_range": "0.7-0.95 typical",
    },
    "minilm": {
        "provider": "sentence-transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_tokens": 256,
        "default_relevance_threshold": 0.50,
        "default_grounding_threshold": 0.60,
        "score_range": "0.5-0.8 typical",
    },
}


def get_model_config(model_key: str) -> dict:
    """Get configuration for an embedding model."""
    if model_key not in EMBEDDING_MODELS:
        logger.warning(f"Unknown model '{model_key}', defaulting to 'e5-base'")
        model_key = "e5-base"
    return EMBEDDING_MODELS[model_key]


# =============================================================================
# Factory
# =============================================================================

class EmbeddingsFactory:
    """Factory for creating embedder instances."""

    @staticmethod
    def from_env() -> Embedder:
        """Create embedder based on environment configuration."""
        provider = os.getenv("EMBEDDING_PROVIDER", "http")
        
        if provider == "http":
            # Default to E5
            endpoint = os.getenv("E5_ENDPOINT", "http://dataops.trupryce.ai:8000")
            return HttpEmbedder(
                endpoint=endpoint,
                query_prefix="query: ",
                passage_prefix="passage: ",
                dimension=768,
            )
        elif provider == "sentence-transformers":
            model_name = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
            trust_remote_code = "nomic" in model_name.lower()
            query_prefix = "query: " if "e5" in model_name.lower() else None
            return SentenceTransformerEmbedder(
                model_name, 
                trust_remote_code=trust_remote_code,
                query_prefix=query_prefix,
            )
        elif provider == "openai":
            model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            return OpenAIEmbedder(model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @staticmethod
    def create(
        provider: str = "http",
        model_name: str | None = None,
        **kwargs,
    ) -> Embedder:
        """Create embedder with explicit configuration."""
        if provider == "http":
            return HttpEmbedder(
                endpoint=kwargs.get("endpoint", "http://dataops.trupryce.ai:8000"),
                query_prefix=kwargs.get("query_prefix", "query: "),
                passage_prefix=kwargs.get("passage_prefix", "passage: "),
                dimension=kwargs.get("dimension", 768),
            )
        elif provider == "sentence-transformers":
            return SentenceTransformerEmbedder(
                model_name or "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=kwargs.get("trust_remote_code", False),
                query_prefix=kwargs.get("query_prefix"),
            )
        elif provider == "openai":
            return OpenAIEmbedder(model_name or "text-embedding-3-small")
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @staticmethod
    def from_model_key(model_key: str) -> Embedder:
        """
        Create embedder from a model key (e.g., 'nomic', 'e5-base', 'openai-small').
        
        This is the preferred method - uses preconfigured model settings.
        """
        config = get_model_config(model_key)
        provider = config["provider"]
        
        if provider == "http":
            # Get endpoint from env or use default
            endpoint = os.getenv(
                config.get("endpoint_env", ""),
                config.get("endpoint_default", "")
            )
            return HttpEmbedder(
                endpoint=endpoint,
                query_prefix=config.get("query_prefix", ""),
                passage_prefix=config.get("passage_prefix", ""),
                dimension=config["dimension"],
            )
        elif provider == "sentence-transformers":
            return SentenceTransformerEmbedder(
                config["model_name"],
                trust_remote_code=config.get("trust_remote_code", False),
                query_prefix=config.get("query_prefix"),
            )
        elif provider == "openai":
            return OpenAIEmbedder(config["model_name"])
        else:
            raise ValueError(f"Unknown provider in config: {provider}")
    
    @staticmethod
    def check_endpoint(endpoint: str, timeout: float = 5.0) -> bool:
        """Check if an embedding endpoint is reachable."""
        try:
            with httpx.Client(timeout=timeout) as client:
                # Try health check first
                for path in ["/health", "/", "/embed"]:
                    try:
                        url = f"{endpoint.rstrip('/')}{path}"
                        response = client.get(url)
                        if response.status_code < 500:
                            return True
                    except Exception:
                        continue
            return False
        except Exception as e:
            logger.debug(f"Endpoint check failed for {endpoint}: {e}")
            return False
    
    @staticmethod
    def check_e5_available() -> bool:
        """Check if E5 embedding service is available."""
        endpoint = os.getenv("E5_ENDPOINT", "http://dataops.trupryce.ai:8000")
        return EmbeddingsFactory.check_endpoint(endpoint)
    
    @staticmethod
    def check_nomic_available() -> bool:
        """Check if Nomic embedding service is available."""
        endpoint = os.getenv("NOMIC_ENDPOINT", "http://dataops.trupryce.ai:8001")
        return EmbeddingsFactory.check_endpoint(endpoint)