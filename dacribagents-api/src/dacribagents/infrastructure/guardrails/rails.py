"""
NeMo Guardrails Integration for Woodcreek Agents.

This module provides a wrapper that adds guardrails to existing agents,
ensuring safe and on-topic conversations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

from dacribagents.domain.models import AgentType, Message, MessageRole


# Path to guardrails config directory
GUARDRAILS_CONFIG_PATH = Path(__file__).parent / "config"


class GuardrailedAgent:
    """
    Wrapper that adds NeMo Guardrails to any agent.
    
    Usage:
        agent = GeneralAssistantAgent()
        guarded = GuardrailedAgent(agent, agent_type="general")
        response = await guarded.process(messages)
    """
    
    def __init__(
        self,
        agent: Any,
        agent_type: str = "general",
        config_path: Optional[Path] = None,
    ):
        """
        Initialize guardrailed agent.
        
        Args:
            agent: The underlying agent to wrap
            agent_type: Type of agent ("general", "hoa", "solar")
            config_path: Path to guardrails config (uses default if not provided)
        """
        self.agent = agent
        self.agent_type = agent_type
        self.config_path = config_path or GUARDRAILS_CONFIG_PATH / agent_type
        
        self._rails: Optional[LLMRails] = None
        
        logger.info(f"GuardrailedAgent initialized for {agent_type}")
    
    @property
    def rails(self) -> LLMRails:
        """Lazily initialize the guardrails."""
        if self._rails is None:
            self._rails = self._create_rails()
        return self._rails
    
    def _create_rails(self) -> LLMRails:
        """Create and configure the NeMo Guardrails instance."""
        # Load configuration
        config = RailsConfig.from_path(str(self.config_path))
        
        # Override model config based on environment
        self._configure_model(config)
        
        # Create rails instance
        rails = LLMRails(config)
        
        # Register custom actions
        self._register_actions(rails)
        
        logger.info(f"NeMo Guardrails loaded from {self.config_path}")
        return rails
    
    def _configure_model(self, config: RailsConfig) -> None:
        """Configure the LLM model based on environment settings."""
        from dacribagents.infrastructure.settings import get_settings
        
        settings = get_settings()
        
        # Update model configuration based on provider
        if settings.llm_provider == "local":
            # Use local vLLM
            config.models[0].engine = "openai"
            config.models[0].model = settings.vllm_model_name
            config.models[0].parameters = {
                "base_url": settings.vllm_base_url,
                "api_key": "not-needed",
            }
        elif settings.llm_provider == "groq":
            config.models[0].engine = "openai"  # Groq is OpenAI-compatible
            config.models[0].model = "llama-3.3-70b-versatile"
            config.models[0].parameters = {
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": settings.groq_api_key.get_secret_value() if settings.groq_api_key else "",
            }
        elif settings.llm_provider == "openai":
            config.models[0].engine = "openai"
            config.models[0].model = "gpt-4o"
            config.models[0].parameters = {
                "api_key": settings.openai_api_key.get_secret_value() if settings.openai_api_key else "",
            }
    
    def _register_actions(self, rails: LLMRails) -> None:
        """Register custom actions with the rails instance."""
        from dacribagents.infrastructure.guardrails import actions
        
        # Register all actions from the actions module
        rails.register_action(actions.check_hoa_topic, name="check_hoa_topic")
        rails.register_action(actions.check_solar_topic, name="check_solar_topic")
        rails.register_action(actions.check_general_topic, name="check_general_topic")
        rails.register_action(actions.check_needs_hoa_disclaimer, name="check_needs_hoa_disclaimer")
        rails.register_action(actions.check_mentions_financial, name="check_mentions_financial")
        rails.register_action(actions.check_jailbreak, name="check_jailbreak")
        rails.register_action(actions.check_toxicity, name="check_toxicity")
        rails.register_action(actions.self_check_input, name="self_check_input")
        rails.register_action(actions.self_check_output, name="self_check_output")
        rails.register_action(actions.check_relevance, name="check_relevance")
        
        # Register the underlying agent as an action for delegation
        @action(name="call_agent")
        async def call_agent(context: dict) -> str:
            """Call the underlying agent."""
            messages = context.get("messages", [])
            # Convert to domain messages
            domain_messages = [
                Message(
                    role=MessageRole(m.get("role", "user")),
                    content=m.get("content", "")
                )
                for m in messages
            ]
            response = await self.agent.process(domain_messages)
            return response.content
        
        rails.register_action(call_agent, name="call_agent")
        
        logger.debug(f"Registered {10} custom actions")
    
    async def process(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
    ) -> Message:
        """
        Process messages through guardrails and the underlying agent.
        
        Args:
            messages: List of conversation messages
            context: Optional context for RAG
            
        Returns:
            Response message from the agent (after guardrails)
        """
        # Convert messages to format expected by NeMo Guardrails
        rails_messages = [
            {"role": m.role.value, "content": m.content}
            for m in messages
        ]
        
        try:
            # Run through guardrails
            response = await self.rails.generate_async(
                messages=rails_messages,
                options={
                    "output_vars": True,  # Get internal state for debugging
                }
            )
            
            # Extract the response
            if isinstance(response, dict):
                content = response.get("content", response.get("response", ""))
            else:
                content = str(response)
            
            logger.debug(f"Guardrails response: {content[:100]}...")
            
            return Message(
                role=MessageRole.ASSISTANT,
                content=content,
            )
            
        except Exception as e:
            logger.error(f"Guardrails error: {e}")
            # Fall back to underlying agent
            return await self.agent.process(messages, context)
    
    @property
    def model_info(self) -> dict[str, str]:
        """Get model info including guardrails status."""
        base_info = getattr(self.agent, "model_info", {})
        return {
            **base_info,
            "guardrails": "enabled",
            "guardrails_type": self.agent_type,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def get_guarded_general_assistant() -> GuardrailedAgent:
    """Get a guardrailed General Assistant agent."""
    from dacribagents.application.agents import get_general_assistant
    
    agent = get_general_assistant()
    return GuardrailedAgent(agent, agent_type="general")


def get_guarded_hoa_agent() -> GuardrailedAgent:
    """Get a guardrailed HOA Compliance agent."""
    from dacribagents.application.agents import get_general_assistant
    
    # For now, use general assistant with HOA rails
    # TODO: Create dedicated HOA agent with RAG
    agent = get_general_assistant()
    return GuardrailedAgent(agent, agent_type="hoa")


def get_guarded_solar_agent() -> GuardrailedAgent:
    """Get a guardrailed Solar agent."""
    from dacribagents.application.agents import get_general_assistant
    
    # For now, use general assistant with Solar rails
    # TODO: Create dedicated Solar agent with RAG
    agent = get_general_assistant()
    return GuardrailedAgent(agent, agent_type="solar")


# =============================================================================
# Integration with Existing Chat Endpoint
# =============================================================================


class GuardrailsMiddleware:
    """
    Middleware that can be added to the chat endpoint.
    
    Provides input validation and output filtering without
    modifying the underlying agent.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or GUARDRAILS_CONFIG_PATH / "general"
        self._rails: Optional[LLMRails] = None
    
    @property
    def rails(self) -> LLMRails:
        if self._rails is None:
            config = RailsConfig.from_path(str(self.config_path))
            self._rails = LLMRails(config)
        return self._rails
    
    async def check_input(self, message: str) -> tuple[bool, str]:
        """
        Check if input message should be allowed.
        
        Returns:
            (allowed, reason) - True if allowed, False with reason if blocked
        """
        from dacribagents.infrastructure.guardrails import actions
        
        context = {"last_user_message": message}
        
        # Run safety checks
        if not await actions.check_jailbreak(context):
            return False, "This appears to be an attempt to manipulate the system."
        
        if not await actions.check_toxicity(context):
            return False, "Please keep the conversation respectful and appropriate."
        
        if not await actions.check_general_topic(context):
            return False, "I can't help with that request. Let's focus on home-related topics."
        
        return True, ""
    
    async def check_output(self, message: str) -> tuple[bool, str]:
        """
        Check if output message should be allowed.
        
        Returns:
            (allowed, filtered_message) - True with original or filtered message
        """
        from dacribagents.infrastructure.guardrails import actions
        
        context = {"last_bot_message": message}
        
        if not await actions.self_check_output(context, None):
            return False, "I apologize, but I need to reconsider my response."
        
        return True, message
