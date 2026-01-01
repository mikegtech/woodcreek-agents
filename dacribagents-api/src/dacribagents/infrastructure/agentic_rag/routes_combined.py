"""
Woodcreek Agents API Routes.

Combines:
- Chat endpoints with supervisor routing
- Guardrails middleware
- Agentic RAG endpoints
- Health checks
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field


# =============================================================================
# Request/Response Models
# =============================================================================


class Message(BaseModel):
    """Chat message."""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Chat request with optional routing."""
    messages: list[Message]
    agent_type: str | None = Field(
        default=None,
        description="Force specific agent: 'hoa', 'solar', 'general'. If None, supervisor routes.",
    )
    use_guardrails: bool = Field(default=True, description="Enable NeMo Guardrails")
    use_rag: bool = Field(default=True, description="Enable Agentic RAG for supported agents")
    thread_id: str | None = Field(default=None, description="Conversation thread ID")


class ChatResponse(BaseModel):
    """Chat response with metadata."""
    response: str
    agent_type: str
    routed_by: str  # "supervisor" or "explicit"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryRequest(BaseModel):
    """Request for RAG query."""
    query: str = Field(..., description="The user's question")
    collection: str = Field(default="emails", description="Milvus collection to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max self-correction iterations")


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""
    query: str
    reformulated_query: str | None
    response: str
    is_grounded: bool
    confidence: float
    iterations: int
    strategy_used: str
    sources: list[dict[str, Any]]
    processing_time_ms: float


class QueryAnalysisRequest(BaseModel):
    """Request for query analysis."""
    query: str = Field(..., description="Query to analyze")


class QueryAnalysisResponse(BaseModel):
    """Response from query analysis."""
    query: str
    query_type: str
    entities: list[str]
    search_query: str
    sub_queries: list[str]
    suggested_strategy: str


class DocumentSearchRequest(BaseModel):
    """Request for document search."""
    query: str = Field(..., description="Search query")
    collection: str = Field(default="emails", description="Collection to search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum relevance score")


class DocumentSearchResponse(BaseModel):
    """Response from document search."""
    query: str
    results: list[dict[str, Any]]
    total_found: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    components: dict[str, str]
    timestamp: str


# =============================================================================
# Router Setup
# =============================================================================


router = APIRouter()


# =============================================================================
# Health Endpoints
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Comprehensive health check for all components.
    """
    from datetime import datetime, timezone
    from dacribagents.infrastructure.settings import get_settings
    
    settings = get_settings()
    components = {}
    
    # Check Milvus
    try:
        from dacribagents.infrastructure.milvus_client import get_milvus_client
        milvus = get_milvus_client()
        health = milvus.health_check()
        components["milvus"] = health.get("status", "unknown")
    except Exception as e:
        components["milvus"] = f"error: {str(e)[:50]}"
    
    # Check LLM
    components["llm"] = f"{settings.llm_provider}"
    if settings.llm_provider == "local":
        components["llm"] += f" ({settings.vllm_model_name})"
    
    # Check embedding model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        components["embeddings"] = "ok (all-MiniLM-L6-v2)"
    except Exception as e:
        components["embeddings"] = f"error: {str(e)[:50]}"
    
    # Check Guardrails
    try:
        from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
        components["guardrails"] = "available"
    except Exception:
        components["guardrails"] = "not configured"
    
    # Overall status
    critical_ok = components.get("milvus") == "healthy"
    status = "healthy" if critical_ok else "degraded"
    
    return HealthResponse(
        status=status,
        components=components,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# Chat Endpoints
# =============================================================================


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint with supervisor routing and guardrails.
    
    Features:
    - Automatic agent routing via supervisor
    - Optional explicit agent selection
    - NeMo Guardrails for safety
    - Agentic RAG for document-based answers
    
    Example:
    ```json
    POST /chat
    {
        "messages": [{"role": "user", "content": "What are the fence rules?"}],
        "use_guardrails": true,
        "use_rag": true
    }
    ```
    """
    from dacribagents.infrastructure.settings import get_settings
    
    settings = get_settings()
    start_time = time.time()
    
    # Extract user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    logger.info(f"Chat request: {user_message[:100]}...")
    
    metadata = {
        "guardrails_enabled": request.use_guardrails,
        "rag_enabled": request.use_rag,
    }
    
    # Input guardrails check
    if request.use_guardrails:
        try:
            from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
            guardrails = GuardrailsMiddleware(agent_type="general")
            allowed, reason = await guardrails.check_input(user_message)
            if not allowed:
                return ChatResponse(
                    response=reason,
                    agent_type="guardrails",
                    routed_by="guardrails",
                    metadata={"blocked": True, "reason": "input_filtered"},
                )
        except Exception as e:
            logger.warning(f"Guardrails check failed: {e}")
    
    # Determine agent
    agent_type = request.agent_type
    routed_by = "explicit" if agent_type else "supervisor"
    
    if not agent_type:
        # Use supervisor to route
        try:
            from dacribagents.application.agents.supervisor import get_supervisor
            supervisor = get_supervisor()
            routing = await supervisor.route(user_message)
            agent_type = routing.get("agent", "general")
            metadata["routing_confidence"] = routing.get("confidence", 0.0)
        except Exception as e:
            logger.warning(f"Supervisor routing failed: {e}, defaulting to general")
            agent_type = "general"
    
    # Get response from appropriate agent
    response_text = ""
    
    try:
        if agent_type == "hoa":
            if request.use_rag:
                from dacribagents.infrastructure.agentic_rag import get_agentic_rag
                rag = get_agentic_rag(collection_name="hoa_documents")
                result = await rag.query(user_message)
                response_text = result.response
                metadata["rag"] = {
                    "is_grounded": result.is_grounded,
                    "confidence": result.confidence,
                    "iterations": result.iterations,
                }
            else:
                from dacribagents.application.agents.hoa_agent import get_hoa_agent
                agent = get_hoa_agent()
                response_text = await agent.chat(user_message)
                
        elif agent_type == "solar":
            if request.use_rag:
                from dacribagents.infrastructure.agentic_rag import get_agentic_rag
                rag = get_agentic_rag(collection_name="solar_documents")
                result = await rag.query(user_message)
                response_text = result.response
                metadata["rag"] = {
                    "is_grounded": result.is_grounded,
                    "confidence": result.confidence,
                    "iterations": result.iterations,
                }
            else:
                from dacribagents.application.agents.solar_agent import get_solar_agent
                agent = get_solar_agent()
                response_text = await agent.chat(user_message)
                
        else:  # general
            from dacribagents.application.agents.general_assistant import get_general_assistant
            agent = get_general_assistant()
            response_text = await agent.chat(user_message)
            
    except Exception as e:
        logger.exception(f"Agent error: {e}")
        response_text = f"I apologize, but I encountered an error processing your request. Please try again."
        metadata["error"] = str(e)
    
    # Output guardrails check
    if request.use_guardrails and response_text:
        try:
            from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
            guardrails = GuardrailsMiddleware(agent_type=agent_type)
            allowed, filtered = await guardrails.check_output(response_text)
            if not allowed:
                response_text = filtered
                metadata["output_filtered"] = True
        except Exception as e:
            logger.warning(f"Output guardrails check failed: {e}")
    
    metadata["processing_time_ms"] = (time.time() - start_time) * 1000
    
    return ChatResponse(
        response=response_text,
        agent_type=agent_type,
        routed_by=routed_by,
        metadata=metadata,
    )


# =============================================================================
# RAG Endpoints
# =============================================================================


@router.post("/rag/query", response_model=RAGQueryResponse, tags=["Agentic RAG"])
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Execute an Agentic RAG query with self-correction.
    
    Features:
    - Query reformulation for better retrieval
    - Document relevance grading
    - Response grounding verification
    - Self-correction if not grounded
    
    Example:
    ```json
    POST /rag/query
    {
        "query": "What are the fence height limits?",
        "collection": "emails",
        "top_k": 5,
        "max_iterations": 3
    }
    ```
    """
    start_time = time.time()
    
    logger.info(f"RAG query: {request.query[:100]}...")
    
    try:
        from dacribagents.infrastructure.agentic_rag import get_agentic_rag
        
        rag = get_agentic_rag(
            collection_name=request.collection,
            max_iterations=request.max_iterations,
        )
        
        result = await rag.query(request.query, top_k=request.top_k)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGQueryResponse(
            query=result.query,
            reformulated_query=result.reformulated_query,
            response=result.response,
            is_grounded=result.is_grounded,
            confidence=result.confidence,
            iterations=result.iterations,
            strategy_used=result.strategy_used.value,
            sources=[
                {
                    "source": chunk.source,
                    "score": chunk.score,
                    "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                }
                for chunk in result.retrieved_chunks[:5]
            ],
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        logger.exception(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/analyze", response_model=QueryAnalysisResponse, tags=["Agentic RAG"])
async def analyze_query(request: QueryAnalysisRequest) -> QueryAnalysisResponse:
    """
    Analyze a query without executing retrieval.
    
    Returns query classification, entities, and suggested strategy.
    
    Example:
    ```json
    POST /rag/analyze
    {
        "query": "Can I build the same fence my neighbor has?"
    }
    ```
    """
    logger.info(f"Analyzing query: {request.query}")
    
    try:
        from dacribagents.infrastructure.agentic_rag import get_agentic_rag
        
        rag = get_agentic_rag()
        analysis = await rag.analyze_query(request.query)
        
        query_type = analysis.get("query_type")
        if hasattr(query_type, "value"):
            query_type_str = query_type.value
        else:
            query_type_str = str(query_type)
        
        strategy = "semantic"
        if query_type_str == "complex":
            strategy = "expanded"
        elif query_type_str == "multi_hop":
            strategy = "hybrid"
        
        return QueryAnalysisResponse(
            query=request.query,
            query_type=query_type_str,
            entities=analysis.get("entities", []),
            search_query=analysis.get("search_query", request.query),
            sub_queries=analysis.get("sub_queries", []),
            suggested_strategy=strategy,
        )
        
    except Exception as e:
        logger.exception(f"Query analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/search", response_model=DocumentSearchResponse, tags=["Agentic RAG"])
async def search_documents(request: DocumentSearchRequest) -> DocumentSearchResponse:
    """
    Search documents without RAG generation.
    
    Useful for testing retrieval quality and exploring collections.
    
    Example:
    ```json
    POST /rag/search
    {
        "query": "fence requirements",
        "collection": "emails",
        "top_k": 10,
        "min_score": 0.6
    }
    ```
    """
    logger.info(f"Document search: {request.query}")
    
    try:
        from dacribagents.infrastructure.agentic_rag import get_agentic_rag
        
        rag = get_agentic_rag(collection_name=request.collection)
        chunks = await rag.retrieve(request.query, top_k=request.top_k)
        filtered = [c for c in chunks if c.score >= request.min_score]
        
        return DocumentSearchResponse(
            query=request.query,
            results=[
                {
                    "source": chunk.source,
                    "score": round(chunk.score, 4),
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in filtered
            ],
            total_found=len(filtered),
        )
        
    except Exception as e:
        logger.exception(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/reformulate", tags=["Agentic RAG"])
async def reformulate_query(request: QueryAnalysisRequest) -> dict[str, Any]:
    """
    Reformulate a query for better retrieval.
    
    Returns original, reformulated, and expanded variations.
    
    Example:
    ```json
    POST /rag/reformulate
    {
        "query": "Can I have chickens?"
    }
    ```
    """
    logger.info(f"Reformulating: {request.query}")
    
    try:
        from dacribagents.infrastructure.agentic_rag import get_agentic_rag
        
        rag = get_agentic_rag()
        reformulated = await rag.reformulate_query(request.query)
        variations = await rag.expand_query(request.query)
        
        return {
            "original": request.query,
            "reformulated": reformulated,
            "variations": variations,
        }
        
    except Exception as e:
        logger.exception(f"Query reformulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/multi-hop", tags=["Agentic RAG"])
async def multi_hop_query(request: RAGQueryRequest) -> dict[str, Any]:
    """
    Execute a multi-hop RAG query for complex questions.
    
    Example:
    ```json
    POST /rag/multi-hop
    {
        "query": "What modifications need approval and how long does it take?"
    }
    ```
    """
    start_time = time.time()
    
    logger.info(f"Multi-hop query: {request.query}")
    
    try:
        from dacribagents.infrastructure.agentic_rag import get_agentic_rag
        
        rag = get_agentic_rag(
            collection_name=request.collection,
            max_iterations=request.max_iterations,
        )
        result = await rag.multi_hop_query(request.query, top_k=request.top_k)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "query": result.query,
            "sub_queries": result.sub_queries,
            "response": result.response,
            "is_grounded": result.is_grounded,
            "confidence": result.confidence,
            "documents_used": len(result.retrieved_chunks),
            "processing_time_ms": processing_time,
        }
        
    except Exception as e:
        logger.exception(f"Multi-hop query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/collections", tags=["Agentic RAG"])
async def list_collections() -> dict[str, Any]:
    """
    List available Milvus collections with stats.
    """
    try:
        from dacribagents.infrastructure.milvus_client import get_milvus_client
        
        milvus = get_milvus_client()
        collections = milvus.client.list_collections()
        
        collection_info = []
        for name in collections:
            try:
                stats = milvus.client.get_collection_stats(name)
                collection_info.append({
                    "name": name,
                    "row_count": stats.get("row_count", 0),
                })
            except Exception:
                collection_info.append({
                    "name": name,
                    "row_count": "unknown",
                })
        
        return {
            "collections": collection_info,
            "total": len(collections),
        }
        
    except Exception as e:
        logger.exception(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Guardrails Test Endpoints
# =============================================================================


@router.post("/guardrails/test-input", tags=["Guardrails"])
async def test_input_guardrails(
    text: str = Query(..., description="Text to check"),
    agent_type: str = Query(default="general", description="Agent context"),
) -> dict[str, Any]:
    """
    Test input guardrails without processing.
    
    Example:
    ```
    POST /guardrails/test-input?text=Hello&agent_type=general
    ```
    """
    try:
        from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
        
        guardrails = GuardrailsMiddleware(agent_type=agent_type)
        allowed, reason = await guardrails.check_input(text)
        
        return {
            "input": text,
            "allowed": allowed,
            "reason": reason if not allowed else None,
            "agent_type": agent_type,
        }
        
    except Exception as e:
        logger.exception(f"Guardrails test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/guardrails/test-output", tags=["Guardrails"])
async def test_output_guardrails(
    text: str = Query(..., description="Text to check"),
    agent_type: str = Query(default="general", description="Agent context"),
) -> dict[str, Any]:
    """
    Test output guardrails without actual agent processing.
    
    Example:
    ```
    POST /guardrails/test-output?text=Here is some advice&agent_type=hoa
    ```
    """
    try:
        from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
        
        guardrails = GuardrailsMiddleware(agent_type=agent_type)
        allowed, filtered = await guardrails.check_output(text)
        
        return {
            "input": text,
            "allowed": allowed,
            "filtered": filtered if not allowed else None,
            "agent_type": agent_type,
        }
        
    except Exception as e:
        logger.exception(f"Guardrails test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))