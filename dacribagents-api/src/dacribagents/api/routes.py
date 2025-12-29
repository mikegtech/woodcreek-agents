"""
API routes for the Woodcreek Agents service.

Updated with NeMo Guardrails integration for safe, on-topic conversations.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from dacribagents.application.agents import get_general_assistant
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
    agent_type: str | None = Field(None, description="Specific agent to use (optional)")
    thread_id: str | None = Field(None, description="Conversation thread ID for continuity")
    use_guardrails: bool = Field(True, description="Enable NeMo Guardrails (default: True)")


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
# Guardrails Helper
# ============================================================================


async def check_input_guardrails(message: str) -> tuple[bool, str]:
    """
    Check input message against guardrails.
    
    Returns:
        (allowed, reason) - True if allowed, False with reason if blocked
    """
    try:
        from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
        
        middleware = GuardrailsMiddleware()
        return await middleware.check_input(message)
    except ImportError:
        logger.warning("Guardrails not available, skipping input check")
        return True, ""
    except Exception as e:
        logger.error(f"Guardrails input check failed: {e}")
        return True, ""  # Fail open - allow if guardrails error


async def check_output_guardrails(message: str) -> tuple[bool, str]:
    """
    Check output message against guardrails.
    
    Returns:
        (allowed, filtered_message) - True with original or filtered message
    """
    try:
        from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
        
        middleware = GuardrailsMiddleware()
        return await middleware.check_output(message)
    except ImportError:
        logger.warning("Guardrails not available, skipping output check")
        return True, message
    except Exception as e:
        logger.error(f"Guardrails output check failed: {e}")
        return True, message  # Fail open


def get_agent_for_type(agent_type: str | None):
    """Get the appropriate agent (with or without guardrails)."""
    try:
        if agent_type == "hoa" or agent_type == "hoa_compliance":
            from dacribagents.infrastructure.guardrails import get_guarded_hoa_agent
            return get_guarded_hoa_agent(), "hoa_compliance"
        elif agent_type == "solar":
            from dacribagents.infrastructure.guardrails import get_guarded_solar_agent
            return get_guarded_solar_agent(), "solar"
        else:
            from dacribagents.infrastructure.guardrails import get_guarded_general_assistant
            return get_guarded_general_assistant(), "general_assistant"
    except ImportError:
        logger.warning("Guardrails not available, using unguarded agent")
        return get_general_assistant(), agent_type or "general_assistant"


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

    # Check Guardrails
    try:
        from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
        services["guardrails"] = "available"
    except ImportError:
        services["guardrails"] = "not_configured"

    # Determine overall status
    critical_services = ["milvus", "postgres"]
    critical_healthy = all(services.get(s) == "healthy" for s in critical_services if s in services)
    status = "ready" if critical_healthy else "degraded"

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
# Chat Endpoint (with Guardrails)
# ============================================================================


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message and get a response from an agent.

    Features:
    - NeMo Guardrails for input/output safety (enabled by default)
    - Agent routing based on agent_type
    - Jailbreak and toxicity prevention
    - Topic-specific guardrails for HOA and Solar agents
    """
    try:
        # Get the last user message for guardrails check
        last_user_message = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.role == "user":
                    last_user_message = msg.content
                    break
        
        # === INPUT GUARDRAILS ===
        if request.use_guardrails and last_user_message:
            allowed, reason = await check_input_guardrails(last_user_message)
            if not allowed:
                logger.warning(f"Input blocked by guardrails: {reason}")
                return ChatResponse(
                    message=ChatMessage(role="assistant", content=reason),
                    agent_type="guardrails",
                    metadata={"blocked": True, "reason": "input_guardrails"},
                )
        
        # Convert request messages to domain messages
        messages: list[Message] = []
        for msg in request.messages:
            try:
                role = MessageRole(msg.role)
            except ValueError:
                role = MessageRole.USER
            messages.append(Message(role=role, content=msg.content))

        # Get appropriate agent (with guardrails if available)
        if request.use_guardrails:
            agent, agent_type_str = get_agent_for_type(request.agent_type)
        else:
            agent = get_general_assistant()
            agent_type_str = "general_assistant"

        logger.info(f"Processing chat with {len(messages)} messages using {agent_type_str}")
        logger.info(f"Guardrails: {'enabled' if request.use_guardrails else 'disabled'}")

        # Process the messages
        response = await agent.process(messages)

        # === OUTPUT GUARDRAILS ===
        response_content = response.content
        if request.use_guardrails:
            allowed, filtered_content = await check_output_guardrails(response_content)
            if not allowed:
                logger.warning("Output modified by guardrails")
                response_content = filtered_content

        # Build metadata
        metadata = getattr(agent, "model_info", {})
        metadata["guardrails_enabled"] = request.use_guardrails

        return ChatResponse(
            message=ChatMessage(role=response.role.value, content=response_content),
            agent_type=agent_type_str,
            thread_id=request.thread_id,
            metadata=metadata,
        )

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
            description="General purpose assistant for everyday questions and tasks (with guardrails)",
            capabilities=["general_qa", "recommendations", "explanations", "routing"],
        ),
        AgentInfo(
            name="HOA Compliance",
            type=AgentType.HOA_COMPLIANCE.value,
            description="Expert on HOA rules, CC&Rs, and compliance requirements (with guardrails)",
            capabilities=["rule_lookup", "violation_help", "arc_guidance", "appeal_drafting"],
        ),
        AgentInfo(
            name="Solar Specialist",
            type="solar",
            description="Solar panel installation, costs, incentives, and Texas regulations (with guardrails)",
            capabilities=["cost_estimation", "incentive_info", "hoa_solar_rights", "provider_guidance"],
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
    ]


@router.get("/config", tags=["config"])
async def get_config() -> dict[str, Any]:
    """Get current LLM and guardrails configuration (non-sensitive)."""
    settings = get_settings()
    
    config = {
        "llm_provider": settings.llm_provider,
        "vllm_base_url": settings.vllm_base_url if settings.llm_provider == "local" else None,
        "vllm_model_name": settings.vllm_model_name if settings.llm_provider == "local" else None,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
    }
    
    # Add guardrails status
    try:
        from dacribagents.infrastructure.guardrails import GuardrailedAgent
        config["guardrails"] = {
            "available": True,
            "agents": ["general", "hoa", "solar"],
        }
    except ImportError:
        config["guardrails"] = {
            "available": False,
            "reason": "not_installed",
        }
    
    return config