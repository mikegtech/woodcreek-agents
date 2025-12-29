"""
Lattice integration module.

Provides:
- LatticeEmailRetriever: LangChain-compatible retriever for Lattice email chunks
- LatticeKafkaPublisher: Publishes raw emails to Lattice Kafka pipeline
"""

from dacribagents.infrastructure.lattice.retriever import (
    LatticeEmailRetriever,
    get_lattice_retriever,
)
from dacribagents.infrastructure.lattice.publisher import (
    LatticeKafkaPublisher,
    LatticeEnvelope,
    get_lattice_publisher,
)

__all__ = [
    # Retriever
    "LatticeEmailRetriever",
    "get_lattice_retriever",
    # Publisher
    "LatticeKafkaPublisher",
    "LatticeEnvelope",
    "get_lattice_publisher",
]