"""LLM infrastructure — model registry, factory, and vLLM client.

Adapted from TruPryce agent LLM infrastructure for Woodcreek household
orchestration.  Provides per-agent model selection, multi-server routing,
and three-level tool binding control.
"""

from dacribagents.infrastructure.llm.factory import LLMFactory, create_llm_factory, normalize_base_url
from dacribagents.infrastructure.llm.model_registry import (
    AgentModelConfig,
    ModelCapabilities,
    ModelSpec,
    build_agent_model_configs,
    build_model_registry,
)

__all__ = [
    "AgentModelConfig",
    "LLMFactory",
    "ModelCapabilities",
    "ModelSpec",
    "build_agent_model_configs",
    "build_model_registry",
    "create_llm_factory",
    "normalize_base_url",
]
