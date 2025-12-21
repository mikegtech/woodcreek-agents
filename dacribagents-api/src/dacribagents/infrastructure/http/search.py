"""RAG Search API endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger


router = APIRouter(prefix="/search", tags=["search"])


# ============================================================================
# Request/Response Models
# ============================================================================


class SearchResult(BaseModel):
    """A single search result."""

    content: str = Field(description="Document content snippet")
    score: float = Field(description="Similarity score (0-1)")
    source: str = Field(description="Source type: email, document, pdf")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Search response with results and metadata."""

    query: str
    results: list[SearchResult]
    total: int
    collections_searched: list[str]


class ContextResponse(BaseModel):
    """Context response formatted for agent consumption."""

    query: str
    context: str
    result_count: int


# ============================================================================
# Endpoints
# ============================================================================


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
    collections: str | None = Query(
        default=None,
        description="Comma-separated collection names (emails,documents,pdfs)",
    ),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
) -> SearchResponse:
    """
    Search across knowledge base (emails, documents, PDFs).

    Returns ranked results with similarity scores.
    """
    try:
        from dacribagents.application.rag.retriever import get_rag_retriever

        retriever = get_rag_retriever()

        # Parse collections
        collection_list = None
        if collections:
            collection_list = [c.strip() for c in collections.split(",")]

        # Perform search
        results = await retriever.retrieve(
            query=q,
            top_k=top_k,
            collections=collection_list,
            min_score=min_score,
        )

        # Convert to response model
        search_results = [
            SearchResult(
                content=doc.snippet,
                score=doc.score,
                source=doc.source,
                metadata=doc.metadata,
            )
            for doc in results
        ]

        # Determine which collections were actually searched
        searched = retriever._get_available_collections()
        if collection_list:
            searched = [c for c in collection_list if c in searched]

        return SearchResponse(
            query=q,
            results=search_results,
            total=len(search_results),
            collections_searched=searched,
        )

    except Exception as e:
        logger.exception(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/context", response_model=ContextResponse)
async def get_context(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=10, description="Number of results"),
    max_chars: int = Query(default=4000, ge=500, le=10000, description="Max context length"),
) -> ContextResponse:
    """
    Get formatted context for agent consumption.

    Returns a pre-formatted context block ready for prompt injection.
    """
    try:
        from dacribagents.application.rag.retriever import (
            get_rag_retriever,
            build_context,
        )

        retriever = get_rag_retriever()

        results = await retriever.retrieve(query=q, top_k=top_k)
        context = build_context(results, max_chars=max_chars)

        return ContextResponse(
            query=q,
            context=context,
            result_count=len(results),
        )

    except Exception as e:
        logger.exception(f"Context generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Context generation failed: {str(e)}")


@router.get("/health")
async def search_health() -> dict:
    """Health check for RAG search subsystem."""
    try:
        from dacribagents.application.rag.retriever import get_rag_retriever

        retriever = get_rag_retriever()
        collections = retriever._get_available_collections()

        return {
            "status": "healthy",
            "available_collections": collections,
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        }

    except Exception as e:
        logger.error(f"Search health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }