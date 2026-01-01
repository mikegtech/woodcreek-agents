"""
Agentic RAG Module for Woodcreek Agents.

This module provides advanced RAG patterns:
- Query Reformulation
- Self-Correction Loops
- Multi-hop Reasoning
- Relevance Checking
- Hallucination Detection

Two implementations are provided:
1. AgenticRAG - Direct implementation with iterative correction
2. AgenticRAGGraph - LangGraph-based state machine implementation

NVIDIA Certification Concepts:
- Corrective RAG (CRAG)
- Self-RAG
- Adaptive RAG
"""

from dacribagents.infrastructure.agentic_rag.agentic_rag import (
    AgenticRAG,
    QueryType,
    RAGResult,
    RetrievalStrategy,
    RetrievedChunk,
    get_agentic_rag,
)

from dacribagents.infrastructure.agentic_rag.langgraph_rag import (
    AgenticRAGGraph,
    RAGState,
    get_agentic_rag_graph,
)

__all__ = [
    # Core classes
    "AgenticRAG",
    "AgenticRAGGraph",
    
    # Data models
    "QueryType",
    "RAGResult",
    "RetrievalStrategy",
    "RetrievedChunk",
    "RAGState",
    
    # Factory functions
    "get_agentic_rag",
    "get_agentic_rag_graph",
]