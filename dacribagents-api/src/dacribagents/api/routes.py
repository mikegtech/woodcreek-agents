"""API routes for Woodcreek Agents."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dacribagents.domain import AgentType
from dacribagents.infrastructure import (
    get_milvus_client,
    get_postgres_client,
    get_settings,
)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    environment: str
    services: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(..., min_length=1, max_length=10000)
    thread_id: str | None = None
    agent_type: AgentType | None = None


class ChatResponse(BaseModel):
    """Chat response payload."""

    response: str
    thread_id: str
    agent_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Health Endpoints
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/health/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check() -> HealthResponse:
    """Readiness check with service status."""
    settings = get_settings()
    milvus = get_milvus_client()
    postgres = get_postgres_client()

    services = {
        "milvus": milvus.health_check(),
        "postgres": postgres.health_check(),
    }

    # Determine overall status
    all_healthy = all(svc.get("status") == "healthy" for svc in services.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status,
        version=settings.app_version,
        environment=settings.environment,
        services=services,
    )


@router.get("/health/live", tags=["Health"])
async def liveness_check() -> dict[str, str]:
    """Simple liveness probe for Kubernetes."""
    return {"status": "alive"}


# =============================================================================
# Chat Endpoints
# =============================================================================


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the agent system."""
    import uuid

    # Generate thread_id if not provided
    thread_id = request.thread_id or str(uuid.uuid4())

    # TODO: Integrate with actual agent system
    # For now, return a placeholder response
    return ChatResponse(
        response=f"Received: {request.message[:100]}... (Agent integration pending)",
        thread_id=thread_id,
        agent_type=request.agent_type.value if request.agent_type else "supervisor",
        metadata={"placeholder": True},
    )


@router.get("/agents", tags=["Agents"])
async def list_agents() -> dict[str, Any]:
    """List available agents."""
    return {
        "agents": [
            {
                "type": agent.value,
                "name": agent.name.replace("_", " ").title(),
                "description": _get_agent_description(agent),
            }
            for agent in AgentType
        ]
    }


def _get_agent_description(agent_type: AgentType) -> str:
    """Get description for an agent type."""
    descriptions = {
        AgentType.HOA_COMPLIANCE: "Handles HOA rules, CC&Rs, compliance questions, and violation concerns",
        AgentType.HOME_MAINTENANCE: "Handles maintenance tasks, landscaping, repairs, and vendor coordination",
        AgentType.SECURITY_CAMERAS: "Handles security cameras, monitoring, alerts, and perimeter security",
        AgentType.SUPERVISOR: "Routes requests to appropriate specialized agents",
        AgentType.GENERAL_ASSISTANT: "Handles general questions and tasks",
    }
    return descriptions.get(agent_type, "No description available")