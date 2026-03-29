"""LLM factory — per-agent ChatOpenAI instances with URL normalization and caching.

Adapted from TruPryce. Uses loguru instead of structlog to match Woodcreek patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from loguru import logger

from dacribagents.infrastructure.llm.model_registry import (
    AgentModelConfig,
    ModelSpec,
    build_agent_model_configs,
    build_model_registry,
)

if TYPE_CHECKING:
    from dacribagents.infrastructure.settings import Settings


def normalize_base_url(url: str) -> str:
    """Normalize vLLM base URL to canonical form (WITH /v1 suffix).

    langchain_openai does NOT append /v1 automatically. vLLM serves at
    /v1/chat/completions, so the base_url MUST include /v1.
    """
    url = url.rstrip("/")
    protocol_end = url.find("://")
    if protocol_end > 0:
        protocol = url[: protocol_end + 3]
        path = url[protocol_end + 3 :]
        while "//" in path:
            path = path.replace("//", "/")
        url = protocol + path
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


class LLMFactory:
    """Factory for creating and caching per-agent LLM instances.

    Supports separate base URLs for router, tool planner, and specialist agents.
    """

    def __init__(
        self,
        base_url: str,
        model_registry: dict[str, ModelSpec],
        agent_configs: dict[str, AgentModelConfig],
        router_base_url: str | None = None,
        tool_planner_base_url: str | None = None,
    ) -> None:
        self._base_url = normalize_base_url(base_url)
        self._router_base_url = normalize_base_url(router_base_url) if router_base_url else self._base_url
        self._tool_planner_base_url = (
            normalize_base_url(tool_planner_base_url) if tool_planner_base_url else self._base_url
        )
        self._model_registry = model_registry
        self._agent_configs = agent_configs
        self._cache: dict[str, ChatOpenAI] = {}

        logger.info(
            f"LLMFactory initialized: base={self._base_url} router={self._router_base_url} "
            f"planner={self._tool_planner_base_url} agents={list(agent_configs.keys())}"
        )

    def _get_base_url_for_agent(self, agent_name: str) -> str:
        if agent_name in {"router", "reminder_classifier"}:
            return self._router_base_url
        if agent_name == "tool_planner":
            return self._tool_planner_base_url
        return self._base_url

    def get_llm_for_agent(self, agent_name: str) -> ChatOpenAI:
        """Get or create a cached LLM for a specific agent."""
        if agent_name not in self._agent_configs:
            raise KeyError(f"No config for agent '{agent_name}'. Available: {list(self._agent_configs.keys())}")

        config = self._agent_configs[agent_name]
        agent_url = self._get_base_url_for_agent(agent_name)
        cache_key = f"{agent_url}:{config.model_name}:{config.temperature}:{config.max_tokens}"

        if cache_key not in self._cache:
            llm_kwargs: dict = {
                "base_url": agent_url,
                "api_key": "not-needed",
                "model": config.model_name,
                "temperature": config.temperature,
                "stop": [],
            }
            if config.max_tokens is not None:
                llm_kwargs["max_tokens"] = config.max_tokens
            if agent_name in ("tool_planner", "general"):
                llm_kwargs["streaming"] = False

            self._cache[cache_key] = ChatOpenAI(**llm_kwargs)
            logger.debug(f"Created LLM: agent={agent_name} model={config.model_name} url={agent_url}")

        return self._cache[cache_key]

    def get_agent_config(self, agent_name: str) -> AgentModelConfig:
        """Get the config for an agent."""
        return self._agent_configs[agent_name]

    def get_model_spec(self, model_name: str) -> ModelSpec | None:
        """Get model spec by name."""
        return self._model_registry.get(model_name)

    def get_model_supports_tools(self, agent_name: str) -> bool:
        """Check if the agent's model supports tool calling."""
        config = self._agent_configs.get(agent_name)
        if not config:
            return False
        spec = self._model_registry.get(config.model_name)
        return spec.capabilities.supports_tools if spec else False

    @property
    def available_models(self) -> list[str]:  # noqa: D102
        return list(self._model_registry.keys())

    @property
    def configured_agents(self) -> list[str]:  # noqa: D102
        return list(self._agent_configs.keys())


# ── Singleton ───────────────────────────────────────────────────────────────

_factory: LLMFactory | None = None


def create_llm_factory(settings: Settings) -> LLMFactory:
    """Create an LLMFactory from settings."""
    registry = build_model_registry(settings)
    configs = build_agent_model_configs(settings)
    return LLMFactory(
        base_url=settings.vllm_base_url,
        model_registry=registry,
        agent_configs=configs,
        router_base_url=settings.vllm_router_base_url,
        tool_planner_base_url=settings.vllm_tool_planner_base_url,
    )


def get_llm_factory() -> LLMFactory:
    """Get or create the singleton LLMFactory."""
    global _factory  # noqa: PLW0603
    if _factory is None:
        from dacribagents.infrastructure.settings import get_settings  # noqa: PLC0415

        _factory = create_llm_factory(get_settings())
    return _factory
