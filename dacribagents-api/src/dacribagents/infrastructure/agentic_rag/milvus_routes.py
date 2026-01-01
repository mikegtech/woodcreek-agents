"""
Milvus Exploration API Routes.

Provides endpoints for exploring and understanding the Milvus vector database.
Useful for debugging, learning, and NVIDIA exam preparation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field


router = APIRouter(prefix="/milvus", tags=["Milvus Exploration"])


# =============================================================================
# Response Models
# =============================================================================


class CollectionInfo(BaseModel):
    """Collection information."""
    name: str
    description: str | None = None
    num_entities: int
    num_partitions: int
    loaded: bool
    schema_fields: list[dict[str, Any]]
    indexes: list[dict[str, Any]]


class CollectionStats(BaseModel):
    """Collection statistics."""
    name: str
    row_count: int
    data_size_mb: float | None = None
    index_size_mb: float | None = None


class VectorSearchRequest(BaseModel):
    """Vector search request."""
    collection: str = Field(default="email_chunks_v1")
    query_text: str = Field(..., description="Text to embed and search")
    top_k: int = Field(default=5, ge=1, le=100)
    filter_expr: str | None = Field(default=None, description="Milvus filter expression")
    output_fields: list[str] | None = Field(default=None, description="Fields to return")


class VectorSearchResult(BaseModel):
    """Vector search result."""
    query: str
    results: list[dict[str, Any]]
    total_found: int
    search_params: dict[str, Any]


class QueryRequest(BaseModel):
    """Direct query request."""
    collection: str = Field(default="email_chunks_v1")
    filter_expr: str = Field(..., description="Milvus filter expression (e.g., 'tenant_id == \"xyz\"')")
    output_fields: list[str] | None = Field(default=None)
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# =============================================================================
# Helper Functions
# =============================================================================


def get_milvus_client():
    """Get Milvus client."""
    try:
        from dacribagents.infrastructure.milvus_client import get_milvus_client
        return get_milvus_client()
    except Exception as e:
        logger.error(f"Failed to get Milvus client: {e}")
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {str(e)}")


def get_embedding_model():
    """Get embedding model for encoding queries."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise HTTPException(status_code=503, detail=f"Embedding model unavailable: {str(e)}")


# =============================================================================
# Collection Management Endpoints
# =============================================================================


@router.get("/health")
async def milvus_health() -> dict[str, Any]:
    """
    Check Milvus server health and connection status.
    
    Returns server version, connection status, and basic stats.
    """
    milvus = get_milvus_client()
    health = milvus.health_check()
    
    # Add collection count
    try:
        collections = milvus.client.list_collections()
        health["collection_count"] = len(collections)
        health["collections"] = collections
    except Exception as e:
        health["collection_error"] = str(e)
    
    return health


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    """
    List all Milvus collections with basic stats.
    
    Returns collection names and row counts.
    """
    milvus = get_milvus_client()
    collections = milvus.client.list_collections()
    
    result = []
    for name in collections:
        try:
            stats = milvus.client.get_collection_stats(name)
            result.append({
                "name": name,
                "row_count": stats.get("row_count", 0),
            })
        except Exception as e:
            result.append({
                "name": name,
                "error": str(e),
            })
    
    return {
        "collections": result,
        "total": len(collections),
    }


@router.get("/collections/{collection_name}")
async def get_collection_info(collection_name: str) -> dict[str, Any]:
    """
    Get detailed information about a specific collection.
    
    Returns schema, indexes, partitions, and statistics.
    """
    milvus = get_milvus_client()
    
    # Check collection exists
    if not milvus.client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        # Get collection info using describe_collection
        info = milvus.client.describe_collection(collection_name)
        stats = milvus.client.get_collection_stats(collection_name)
        
        # Get index info
        indexes = []
        try:
            index_info = milvus.client.list_indexes(collection_name)
            indexes = index_info if isinstance(index_info, list) else [index_info]
        except Exception:
            pass
        
        return {
            "name": collection_name,
            "description": info.get("description", ""),
            "schema": info,
            "stats": stats,
            "indexes": indexes,
            "row_count": stats.get("row_count", 0),
        }
    except Exception as e:
        logger.exception(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/schema")
async def get_collection_schema(collection_name: str) -> dict[str, Any]:
    """
    Get the schema (field definitions) for a collection.
    
    Shows field names, types, and constraints.
    Useful for understanding how to query and filter.
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        info = milvus.client.describe_collection(collection_name)
        
        # Parse fields into readable format
        fields = []
        for field in info.get("fields", []):
            field_info = {
                "name": field.get("name"),
                "type": field.get("type"),
                "description": field.get("description", ""),
                "is_primary": field.get("is_primary", False),
            }
            
            # Add dimension for vector fields
            if "params" in field:
                field_info["params"] = field["params"]
            if "max_length" in field:
                field_info["max_length"] = field["max_length"]
                
            fields.append(field_info)
        
        return {
            "collection": collection_name,
            "fields": fields,
            "field_count": len(fields),
        }
    except Exception as e:
        logger.exception(f"Failed to get schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/indexes")
async def get_collection_indexes(collection_name: str) -> dict[str, Any]:
    """
    Get index information for a collection.
    
    Shows index type, metric type, and parameters.
    Important for understanding search performance.
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        indexes = milvus.client.list_indexes(collection_name)
        
        # Get detailed index info for each
        index_details = []
        for idx in indexes if isinstance(indexes, list) else [indexes]:
            try:
                detail = milvus.client.describe_index(collection_name, idx)
                index_details.append(detail)
            except Exception:
                index_details.append({"name": idx})
        
        return {
            "collection": collection_name,
            "indexes": index_details,
        }
    except Exception as e:
        logger.exception(f"Failed to get indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Data Exploration Endpoints
# =============================================================================


@router.get("/collections/{collection_name}/sample")
async def sample_collection(
    collection_name: str,
    limit: int = Query(default=5, ge=1, le=100),
    output_fields: str = Query(default=None, description="Comma-separated field names"),
) -> dict[str, Any]:
    """
    Get sample records from a collection.
    
    Useful for understanding data structure and content.
    Does NOT return vector data by default (too large).
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        # Parse output fields
        fields = None
        if output_fields:
            fields = [f.strip() for f in output_fields.split(",")]
        else:
            # Get all non-vector fields by default
            info = milvus.client.describe_collection(collection_name)
            fields = [
                f["name"] for f in info.get("fields", [])
                if f.get("type") != 101  # 101 = FloatVector
            ]
        
        # Query with no filter to get sample
        results = milvus.client.query(
            collection_name=collection_name,
            filter="",  # No filter = random sample
            output_fields=fields,
            limit=limit,
        )
        
        return {
            "collection": collection_name,
            "sample_size": len(results),
            "fields_returned": fields,
            "data": results,
        }
    except Exception as e:
        logger.exception(f"Failed to sample collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_collection(request: QueryRequest) -> dict[str, Any]:
    """
    Execute a filter query against a collection.
    
    Uses Milvus expression syntax for filtering.
    
    Example filters:
    - `tenant_id == "woodcreek"`
    - `email_timestamp > 1704067200`
    - `section_type in ["body", "header"]`
    - `account_id == "acc123" and section_type == "body"`
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(request.collection):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection}' not found")
    
    try:
        # Get default fields if not specified
        fields = request.output_fields
        if not fields:
            info = milvus.client.describe_collection(request.collection)
            fields = [
                f["name"] for f in info.get("fields", [])
                if f.get("type") != 101  # Exclude vectors
            ]
        
        results = milvus.client.query(
            collection_name=request.collection,
            filter=request.filter_expr,
            output_fields=fields,
            limit=request.limit,
            offset=request.offset,
        )
        
        return {
            "collection": request.collection,
            "filter": request.filter_expr,
            "results": results,
            "count": len(results),
            "limit": request.limit,
            "offset": request.offset,
        }
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/count")
async def count_collection(
    collection_name: str,
    filter_expr: str = Query(default=None, description="Optional filter expression"),
) -> dict[str, Any]:
    """
    Count entities in a collection, optionally with a filter.
    
    Faster than querying all data when you just need counts.
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        if filter_expr:
            # Count with filter by querying PKs only
            results = milvus.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=["pk"],
                limit=16384,  # Max for count estimation
            )
            count = len(results)
            estimated = count >= 16384
        else:
            # Get total count from stats
            stats = milvus.client.get_collection_stats(collection_name)
            count = stats.get("row_count", 0)
            estimated = False
        
        return {
            "collection": collection_name,
            "filter": filter_expr,
            "count": count,
            "estimated": estimated,
        }
    except Exception as e:
        logger.exception(f"Count failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/distinct")
async def get_distinct_values(
    collection_name: str,
    field: str = Query(..., description="Field to get distinct values for"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """
    Get distinct values for a field in a collection.
    
    Useful for understanding data distribution.
    Only works for scalar fields, not vectors.
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        # Query the field
        results = milvus.client.query(
            collection_name=collection_name,
            filter="",
            output_fields=[field],
            limit=limit * 10,  # Over-fetch to find distinct
        )
        
        # Extract distinct values
        values = set()
        for row in results:
            if field in row:
                values.add(row[field])
            if len(values) >= limit:
                break
        
        return {
            "collection": collection_name,
            "field": field,
            "distinct_values": sorted(list(values)),
            "count": len(values),
            "truncated": len(values) >= limit,
        }
    except Exception as e:
        logger.exception(f"Distinct query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Vector Search Endpoints
# =============================================================================


@router.post("/search")
async def vector_search(request: VectorSearchRequest) -> VectorSearchResult:
    """
    Execute a vector similarity search.
    
    1. Embeds the query text using sentence-transformers
    2. Searches for similar vectors in Milvus
    3. Returns matching documents with similarity scores
    
    Supports optional filtering with Milvus expressions.
    """
    milvus = get_milvus_client()
    model = get_embedding_model()
    
    if not milvus.client.has_collection(request.collection):
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection}' not found")
    
    try:
        # Embed query
        query_vector = model.encode(request.query_text).tolist()
        
        # Get default output fields
        output_fields = request.output_fields
        if not output_fields:
            info = milvus.client.describe_collection(request.collection)
            output_fields = [
                f["name"] for f in info.get("fields", [])
                if f.get("type") != 101  # Exclude vectors
            ]
        
        # Search
        results = milvus.client.search(
            collection_name=request.collection,
            data=[query_vector],
            limit=request.top_k,
            filter=request.filter_expr,
            output_fields=output_fields,
        )
        
        # Format results
        formatted = []
        for hit in results[0] if results else []:
            formatted.append({
                "id": hit.get("id"),
                "distance": hit.get("distance"),
                "score": 1 - hit.get("distance", 0),  # COSINE: distance to similarity
                "entity": hit.get("entity", {}),
            })
        
        return VectorSearchResult(
            query=request.query_text,
            results=formatted,
            total_found=len(formatted),
            search_params={
                "collection": request.collection,
                "top_k": request.top_k,
                "filter": request.filter_expr,
                "metric": "COSINE",
            },
        )
    except Exception as e:
        logger.exception(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/by-vector")
async def search_by_vector(
    collection: str = Query(default="email_chunks_v1"),
    vector: list[float] = ...,
    top_k: int = Query(default=5, ge=1, le=100),
    filter_expr: str = Query(default=None),
) -> dict[str, Any]:
    """
    Search using a raw vector (no embedding).
    
    Use this when you already have a vector and want to find similar ones.
    Useful for finding similar documents to an existing document.
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(collection):
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
    
    try:
        # Get output fields
        info = milvus.client.describe_collection(collection)
        output_fields = [
            f["name"] for f in info.get("fields", [])
            if f.get("type") != 101
        ]
        
        results = milvus.client.search(
            collection_name=collection,
            data=[vector],
            limit=top_k,
            filter=filter_expr,
            output_fields=output_fields,
        )
        
        formatted = []
        for hit in results[0] if results else []:
            formatted.append({
                "id": hit.get("id"),
                "distance": hit.get("distance"),
                "score": 1 - hit.get("distance", 0),
                "entity": hit.get("entity", {}),
            })
        
        return {
            "results": formatted,
            "total_found": len(formatted),
            "vector_dimension": len(vector),
        }
    except Exception as e:
        logger.exception(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Utility Endpoints
# =============================================================================


@router.get("/collections/{collection_name}/entity/{pk}")
async def get_entity_by_pk(
    collection_name: str,
    pk: str,
    include_vector: bool = Query(default=False),
) -> dict[str, Any]:
    """
    Get a specific entity by its primary key.
    
    Optionally includes the vector data.
    """
    milvus = get_milvus_client()
    
    if not milvus.client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    
    try:
        # Get fields
        info = milvus.client.describe_collection(collection_name)
        output_fields = [
            f["name"] for f in info.get("fields", [])
            if include_vector or f.get("type") != 101
        ]
        
        results = milvus.client.query(
            collection_name=collection_name,
            filter=f'pk == "{pk}"',
            output_fields=output_fields,
            limit=1,
        )
        
        if not results:
            raise HTTPException(status_code=404, detail=f"Entity '{pk}' not found")
        
        return {
            "collection": collection_name,
            "pk": pk,
            "entity": results[0],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Get entity failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embedding-info")
async def get_embedding_info() -> dict[str, Any]:
    """
    Get information about the embedding model being used.
    
    Useful for understanding vector dimensions and model capabilities.
    """
    try:
        model = get_embedding_model()
        
        # Get model info
        test_embedding = model.encode("test")
        
        return {
            "model_name": "all-MiniLM-L6-v2",
            "dimension": len(test_embedding),
            "max_seq_length": model.max_seq_length,
            "normalize_embeddings": True,
            "metric_recommendation": "COSINE",
        }
    except Exception as e:
        logger.exception(f"Failed to get embedding info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed")
async def embed_text(
    text: str = Query(..., description="Text to embed"),
) -> dict[str, Any]:
    """
    Embed text and return the vector.
    
    Useful for testing embeddings and understanding vector representation.
    """
    try:
        model = get_embedding_model()
        vector = model.encode(text).tolist()
        
        return {
            "text": text,
            "text_length": len(text),
            "vector_dimension": len(vector),
            "vector": vector,
            "vector_preview": vector[:10],  # First 10 dimensions
        }
    except Exception as e:
        logger.exception(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))