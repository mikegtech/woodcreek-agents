"""Model registry for per-agent LLM configuration.

Adapted from TruPryce for Woodcreek household orchestration.
Woodcreek-specific agents: router, reminder_classifier, general,
hoa, solar, maintenance, tool_planner.

Tool Binding Strategy:
- Models have capabilities (supports_tools, supports_json_schema)
- Agents have enable_tools flag to opt-in to tool usage
- Effective tool binding requires: global enable + agent enable + model supports
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from dacribagents.infrastructure.settings import Settings


class ModelCapabilities(BaseModel):
    """Capabilities of a model."""

    supports_tools: bool = False
    supports_json_schema: bool = False
    supports_structured_output: bool = False
    context_window: int = 4096
    is_chat_model: bool = True


class ModelSpec(BaseModel):
    """Specification for an available LLM model."""

    name: str
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    description: str = ""


class AgentModelConfig(BaseModel):
    """Model configuration for a specific agent."""

    model_name: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int | None = None
    enable_tools: bool = False

    def to_llm_kwargs(self) -> dict:
        """Convert to kwargs for ChatOpenAI constructor."""
        return {"model": self.model_name, "temperature": self.temperature}


# ── Default models (Tailscale network vLLM instances) ───────────────────────

DEFAULT_MODELS: dict[str, ModelSpec] = {
    "nvcr-gpt-oss-20b": ModelSpec(
        name="nvcr-gpt-oss-20b",
        capabilities=ModelCapabilities(context_window=8192),
        description="NVIDIA 20B GPT — general inference",
    ),
    "gpt-oss-20b": ModelSpec(
        name="gpt-oss-20b",
        capabilities=ModelCapabilities(context_window=32768),
        description="GPT-OSS 20B reasoning model",
    ),
    "vllm-mistral-7b": ModelSpec(
        name="vllm-mistral-7b",
        capabilities=ModelCapabilities(context_window=8192),
        description="Mistral 7B — fast routing/classification",
    ),
    "mistral-7b-instruct-v0_3": ModelSpec(
        name="mistral-7b-instruct-v0_3",
        capabilities=ModelCapabilities(context_window=8192),
        description="Mistral 7B Instruct v0.3",
    ),
    "nvcr-gpt-oss-8b": ModelSpec(
        name="nvcr-gpt-oss-8b",
        capabilities=ModelCapabilities(context_window=4096),
        description="NVIDIA 8B GPT — smaller, faster",
    ),
    "qwen2.5-14b-instruct": ModelSpec(
        name="qwen2.5-14b-instruct",
        capabilities=ModelCapabilities(
            supports_tools=True, supports_json_schema=True, context_window=32768,
        ),
        description="Qwen2.5 14B — tool calling for planner role",
    ),
}

# ── Default Woodcreek agent configurations ──────────────────────────────────

DEFAULT_AGENT_CONFIGS: dict[str, AgentModelConfig] = {
    "router": AgentModelConfig(
        model_name="vllm-mistral-7b", temperature=0.0, max_tokens=256,
    ),
    "reminder_classifier": AgentModelConfig(
        model_name="vllm-mistral-7b", temperature=0.0, max_tokens=512,
    ),
    "general": AgentModelConfig(
        model_name="nvcr-gpt-oss-20b", temperature=0.3,
    ),
    "hoa": AgentModelConfig(
        model_name="nvcr-gpt-oss-20b", temperature=0.1,
    ),
    "solar": AgentModelConfig(
        model_name="nvcr-gpt-oss-20b", temperature=0.1,
    ),
    "maintenance": AgentModelConfig(
        model_name="nvcr-gpt-oss-20b", temperature=0.1,
    ),
    "tool_planner": AgentModelConfig(
        model_name="qwen2.5-14b-instruct", temperature=0.0, max_tokens=512,
        enable_tools=True,
    ),
}


def build_model_registry(settings: Settings) -> dict[str, ModelSpec]:
    """Build model registry from settings."""
    registry = dict(DEFAULT_MODELS)

    for model_name in [settings.vllm_model_name, settings.vllm_router_model_name]:
        if model_name and model_name not in registry:
            registry[model_name] = ModelSpec(name=model_name, description=f"Custom: {model_name}")

    tool_model = settings.vllm_tool_planner_model_name
    if tool_model and tool_model not in registry:
        registry[tool_model] = ModelSpec(
            name=tool_model,
            capabilities=ModelCapabilities(supports_tools=True, supports_json_schema=True),
            description=f"Tool planner: {tool_model}",
        )

    return registry


def build_agent_model_configs(settings: Settings) -> dict[str, AgentModelConfig]:
    """Build agent configs from settings with overrides."""
    configs = dict(DEFAULT_AGENT_CONFIGS)

    configs["router"] = AgentModelConfig(
        model_name=settings.vllm_router_model_name,
        temperature=settings.vllm_router_temperature,
        max_tokens=settings.vllm_router_max_tokens,
    )
    configs["reminder_classifier"] = AgentModelConfig(
        model_name=settings.vllm_router_model_name,
        temperature=0.0, max_tokens=512,
    )

    specialist = AgentModelConfig(
        model_name=settings.vllm_model_name,
        temperature=settings.agent_temperature,
    )
    for agent in ["hoa", "solar", "maintenance"]:
        configs[agent] = specialist

    configs["general"] = AgentModelConfig(
        model_name=settings.vllm_model_name,
        temperature=min(settings.agent_temperature + 0.2, 1.0),
    )

    configs["tool_planner"] = AgentModelConfig(
        model_name=settings.vllm_tool_planner_model_name,
        temperature=settings.tool_planner_temperature,
        max_tokens=settings.tool_planner_max_tokens,
        enable_tools=True,
    )

    return configs
