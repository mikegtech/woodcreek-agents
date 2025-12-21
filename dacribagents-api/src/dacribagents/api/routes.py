"""API routes for the Woodcreek Agents service."""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from dacribagents.application.agents import (
    get_general_assistant,
    get_supervisor,
)
from dacribagents.domain import AgentType, Message, MessageRole
from dacribagents.infrastructure import (
    get_milvus_client,
    get_postgres_client,
    get_settings,
)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    agent_type: str | None = Field(None, description="Specific agent to use (optional, bypasses routing)")
    thread_id: str | None = Field(None, description="Conversation thread ID for continuity")
    use_routing: bool = Field(True, description="Use supervisor routing (default: True)")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    message: ChatMessage
    agent_type: str
    routed_by: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness check response with service status."""

    status: str
    timestamp: str
    services: dict[str, str]


class AgentInfo(BaseModel):
    """Information about an available agent."""

    name: str
    type: str
    description: str
    capabilities: list[str]


# ============================================================================
# Health Endpoints
# ============================================================================


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.app_version,
    )


@router.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
async def readiness_check() -> ReadinessResponse:
    """Readiness check with service connectivity status."""
    services: dict[str, str] = {}

    # Check Milvus
    try:
        milvus = get_milvus_client()
        health = milvus.health_check()
        services["milvus"] = health.get("status", "unknown")
    except Exception as e:
        logger.warning(f"Milvus health check failed: {e}")
        services["milvus"] = f"error: {str(e)[:50]}"

    # Check PostgreSQL
    try:
        postgres = get_postgres_client()
        health = postgres.health_check()
        services["postgres"] = health.get("status", "unknown")
    except Exception as e:
        logger.warning(f"PostgreSQL health check failed: {e}")
        services["postgres"] = f"error: {str(e)[:50]}"

    # Determine overall status
    all_healthy = all(v == "healthy" for v in services.values())
    status = "ready" if all_healthy else "degraded"

    return ReadinessResponse(
        status=status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
    )


@router.get("/health/live", tags=["health"])
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


# ============================================================================
# Chat Endpoint with Supervisor Routing
# ============================================================================


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message and get a response from an agent.

    Routing behavior:
    - If use_routing=True (default): Supervisor classifies intent and routes to appropriate agent
    - If use_routing=False or agent_type is specified: Uses the specified agent directly
    """
    try:
        # Convert request messages to domain messages
        messages: list[Message] = []
        for msg in request.messages:
            try:
                role = MessageRole(msg.role)
            except ValueError:
                role = MessageRole.USER
            messages.append(Message(role=role, content=msg.content))

        # Determine which agent to use
        if request.agent_type and not request.use_routing:
            # Direct agent access (bypass routing)
            try:
                agent_type = AgentType(request.agent_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid agent_type: {request.agent_type}. "
                           f"Valid types: {[a.value for a in AgentType]}",
                )

            # Get the specific agent
            if agent_type == AgentType.GENERAL_ASSISTANT:
                from dacribagents.application.agents import get_general_assistant
                agent = get_general_assistant()
            elif agent_type == AgentType.HOA_COMPLIANCE:
                from dacribagents.application.agents import get_hoa_agent
                agent = get_hoa_agent()
            elif agent_type == AgentType.SOLAR:
                from dacribagents.application.agents import get_solar_agent
                agent = get_solar_agent()
            else:
                agent = get_general_assistant()

            logger.info(f"Direct agent access: {agent_type.value}")
            response = await agent.process(messages)

            return ChatResponse(
                message=ChatMessage(role=response.role.value, content=response.content),
                agent_type=agent.agent_type.value,
                routed_by=None,
                thread_id=request.thread_id,
                metadata=getattr(agent, "model_info", {}),
            )

        else:
            # Use Supervisor routing
            supervisor = get_supervisor()

            logger.info(f"Routing request through Supervisor")
            logger.info(f"Available agents: {supervisor.registered_agents}")

            response = await supervisor.process(messages)

            handled_by = response.metadata.get("handled_by", "unknown")

            return ChatResponse(
                message=ChatMessage(role=response.role.value, content=response.content),
                agent_type=handled_by,
                routed_by="supervisor",
                thread_id=request.thread_id,
                metadata={"routed": True, "handled_by": handled_by},
            )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error in chat: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Agent Information
# ============================================================================


@router.get("/agents", response_model=list[AgentInfo], tags=["agents"])
async def list_agents() -> list[AgentInfo]:
    """List all available agents and their capabilities."""
    return [
        AgentInfo(
            name="General Assistant",
            type=AgentType.GENERAL_ASSISTANT.value,
            description="General purpose assistant for everyday questions and tasks",
            capabilities=["general_qa", "recommendations", "explanations"],
        ),
        AgentInfo(
            name="HOA Compliance",
            type=AgentType.HOA_COMPLIANCE.value,
            description="Expert on HOA rules, CC&Rs, and compliance requirements",
            capabilities=["rule_lookup", "violation_check", "approval_process", "townsq_help"],
        ),
        AgentInfo(
            name="Solar Installation",
            type=AgentType.SOLAR.value,
            description="Expert on solar panel installation, permits, and energy systems",
            capabilities=["installation_status", "permit_info", "energy_monitoring", "financing"],
        ),
        AgentInfo(
            name="Home Maintenance",
            type=AgentType.HOME_MAINTENANCE.value,
            description="Manages maintenance schedules, vendor coordination, and repairs",
            capabilities=["scheduling", "vendor_lookup", "maintenance_tracking"],
        ),
        AgentInfo(
            name="Security & Cameras",
            type=AgentType.SECURITY_CAMERAS.value,
            description="Monitors security systems, cameras, and alerts",
            capabilities=["camera_status", "alert_management", "event_history"],
        ),
        AgentInfo(
            name="Supervisor",
            type=AgentType.SUPERVISOR.value,
            description="Routes requests to the appropriate specialized agent",
            capabilities=["intent_classification", "routing", "delegation"],
        ),
    ]


@router.get("/agents/registered", tags=["agents"])
async def list_registered_agents() -> dict[str, Any]:
    """List currently registered and active agents."""
    supervisor = get_supervisor()
    return {
        "registered_agents": supervisor.registered_agents,
        "count": len(supervisor.registered_agents),
    }


@router.get("/config", tags=["config"])
async def get_config() -> dict[str, Any]:
    """Get current LLM configuration (non-sensitive)."""
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "vllm_base_url": settings.vllm_base_url if settings.llm_provider == "local" else None,
        "vllm_model_name": settings.vllm_model_name if settings.llm_provider == "local" else None,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
    }