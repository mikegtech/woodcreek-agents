# src/dacribagents/infrastructure/__init__.py
"""Infrastructure layer - external services, databases, and configuration."""

from dacribagents.infrastructure.milvus_client import (
    MilvusClientWrapper,
    get_milvus_client,
    milvus_lifespan,
)
from dacribagents.infrastructure.postgres_client import (
    PostgresClientWrapper,
    get_postgres_client,
    postgres_lifespan,
)
from dacribagents.infrastructure.settings import Settings, get_settings

# Lattice integration (lazy import to avoid circular deps)
def get_lattice_retriever(*args, **kwargs):
    """Get Lattice email retriever (lazy import)."""
    from dacribagents.infrastructure.lattice.retriever import get_lattice_retriever as _get
    return _get(*args, **kwargs)


def get_lattice_publisher(*args, **kwargs):
    """Get Lattice Kafka publisher (lazy import)."""
    from dacribagents.infrastructure.lattice.publisher import get_lattice_publisher as _get
    return _get(*args, **kwargs)


__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # Milvus (local)
    "MilvusClientWrapper",
    "get_milvus_client",
    "milvus_lifespan",
    # Postgres (local)
    "PostgresClientWrapper",
    "get_postgres_client",
    "postgres_lifespan",
    # Lattice integration
    "get_lattice_retriever",
    "get_lattice_publisher",
]