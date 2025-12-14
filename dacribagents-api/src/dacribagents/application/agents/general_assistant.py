"""General Assistant Agent - fallback agent for general queries."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, StateGraph
from loguru import logger

from dacribagents.domain.models import AgentType, Message, MessageRole
from dacribagents.infrastructure.settings import Settings, get_settings

from .base import AgentState, BaseAgent

SYSTEM_PROMPT = """You are a helpful home assistant for a homeowner in the Woodcreek community in Fate, Texas.

You help with general questions and tasks. For specialized topics, you may suggest consulting the appropriate specialist:
- HOA rules, CC&Rs, and compliance questions → HOA Compliance Agent
- Home repairs, maintenance schedules, landscaping → Home Maintenance Agent  
- Security cameras, alerts, and perimeter monitoring → Security & Cameras Agent

Be friendly, concise, and helpful. If you don't know something, say so rather than making things up.

Current context:
- Community: Woodcreek (Fate, TX)
- HOA Management: TownSq platform
"""


def create_llm(settings: Settings) -> BaseChatModel:
    """Create the appropriate LLM based on settings."""
    provider = settings.llm_provider

    if provider == "local":
        from langchain_openai import ChatOpenAI

        logger.info(f"Initializing local vLLM at {settings.vllm_base_url} with model {settings.vllm_model_name}")
        return ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key="not-needed",
            model_name=settings.vllm_model_name,
            temperature=0.7,
            max_tokens=1024,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when llm_provider=groq")

        logger.info("Initializing Groq LLM with model llama-3.3-70b-versatile")
        return ChatGroq(
            api_key=settings.groq_api_key.get_secret_value(),
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when llm_provider=openai")

        logger.info("Initializing OpenAI LLM with model gpt-4o")
        return ChatOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=1024,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when llm_provider=anthropic")

        logger.info("Initializing Anthropic LLM with model claude-sonnet-4-20250514")
        return ChatAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model_name="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=1024,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


class GeneralAssistantAgent(BaseAgent):
    """General purpose assistant agent with configurable LLM backend."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(AgentType.GENERAL_ASSISTANT)
        self.settings = settings or get_settings()
        self._llm: BaseChatModel | None = None

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize the LLM based on settings."""
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    @property
    def model_info(self) -> dict[str, str]:
        """Get information about the current model configuration."""
        provider = self.settings.llm_provider
        if provider == "local":
            return {
                "provider": "local_vllm",
                "model": self.settings.vllm_model_name,
                "base_url": self.settings.vllm_base_url,
            }
        elif provider == "groq":
            return {"provider": "groq", "model": "llama-3.3-70b-versatile"}
        elif provider == "openai":
            return {"provider": "openai", "model": "gpt-4o"}
        elif provider == "anthropic":
            return {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        return {"provider": provider, "model": "unknown"}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("generate", self._generate_response)

        # Set entry point
        graph.set_entry_point("generate")

        # Add edge to end
        graph.add_edge("generate", END)

        return graph.compile()

    async def _generate_response(self, state: AgentState) -> dict[str, Any]:
        """Generate a response using the LLM."""
        messages = state["messages"]

        # Ensure system prompt is first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        # Call the LLM
        response = await self.llm.ainvoke(messages)

        logger.debug(f"Generated response: {response.content[:100]}...")

        return {"messages": messages + [response]}

    async def process(self, messages: list[Message], context: dict[str, Any] | None = None) -> Message:
        """Process messages and return a response."""
        # Convert to LangChain messages
        lc_messages = self._convert_to_langchain_messages(messages)

        # Build initial state
        initial_state: AgentState = {
            "messages": lc_messages,
            "context": context or {},
            "agent_type": self.agent_type.value,
            "thread_id": None,
        }

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        # Extract the last message (the response)
        response_messages = result["messages"]
        if response_messages:
            last_message = response_messages[-1]
            return self._convert_from_langchain_message(last_message)

        # Fallback if no response
        return Message(
            role=MessageRole.ASSISTANT,
            content="I apologize, but I wasn't able to generate a response. Please try again.",
        )


# Singleton instance
_general_assistant: GeneralAssistantAgent | None = None


def get_general_assistant(settings: Settings | None = None) -> GeneralAssistantAgent:
    """Get or create the General Assistant agent singleton."""
    global _general_assistant
    if _general_assistant is None:
        _general_assistant = GeneralAssistantAgent(settings)
    return _general_assistant