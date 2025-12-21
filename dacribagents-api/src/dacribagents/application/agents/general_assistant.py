"""General Assistant Agent - fallback agent for general queries with RAG support."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
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

RAG_SYSTEM_PROMPT = """You are a helpful home assistant for a homeowner in the Woodcreek community in Fate, Texas.

You help with general questions and tasks. You have access to a knowledge base of emails, documents, and information about the community.

When answering questions:
1. Use the provided context when it's relevant to the question
2. If the context contains the answer, reference it naturally (e.g., "Based on the HOA email from December...")
3. If the context doesn't help, answer from your general knowledge
4. Be friendly, concise, and helpful
5. If you don't know something, say so rather than making things up

Current context:
- Community: Woodcreek (Fate, TX)
- HOA Management: TownSq platform

{rag_context}
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
    """General purpose assistant agent with configurable LLM backend and RAG support."""

    def __init__(self, settings: Settings | None = None, enable_rag: bool = True):
        super().__init__(AgentType.GENERAL_ASSISTANT)
        self.settings = settings or get_settings()
        self._llm: BaseChatModel | None = None
        self.enable_rag = enable_rag
        self._retriever = None

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize the LLM based on settings."""
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    @property
    def retriever(self):
        """Lazily initialize the RAG retriever."""
        if self._retriever is None and self.enable_rag:
            try:
                from dacribagents.application.rag.retriever import get_rag_retriever
                self._retriever = get_rag_retriever()
                logger.info("RAG retriever initialized for General Assistant")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG retriever: {e}")
                self._retriever = None
        return self._retriever

    @property
    def model_info(self) -> dict[str, str]:
        """Get information about the current model configuration."""
        provider = self.settings.llm_provider
        info = {"rag_enabled": str(self.enable_rag)}

        if provider == "local":
            info.update({
                "provider": "local_vllm",
                "model": self.settings.vllm_model_name,
                "base_url": self.settings.vllm_base_url,
            })
        elif provider == "groq":
            info.update({"provider": "groq", "model": "llama-3.3-70b-versatile"})
        elif provider == "openai":
            info.update({"provider": "openai", "model": "gpt-4o"})
        elif provider == "anthropic":
            info.update({"provider": "anthropic", "model": "claude-sonnet-4-20250514"})
        else:
            info.update({"provider": provider, "model": "unknown"})

        return info

    def _get_rag_context(self, query: str) -> str:
        """Retrieve relevant context for the query."""
        if not self.enable_rag or not self.retriever:
            return ""

        try:
            from dacribagents.application.rag.retriever import build_context

            # Use sync version since we're in a potentially mixed context
            results = self.retriever.retrieve_sync(
                query=query,
                top_k=5,
                min_score=0.3,  # Only include reasonably relevant results
            )

            if results:
                context = build_context(results, max_chars=3000)
                logger.info(f"RAG: Retrieved {len(results)} documents for context")
                return context
            else:
                logger.debug("RAG: No relevant documents found")
                return ""

        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return ""

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
        """Generate a response using the LLM with RAG context."""
        messages = state["messages"]

        # Extract the latest user message for RAG query
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # Get RAG context
        rag_context = ""
        if user_query and self.enable_rag:
            rag_context = self._get_rag_context(user_query)

        # Build system prompt with or without RAG context
        if rag_context:
            system_prompt = RAG_SYSTEM_PROMPT.format(rag_context=rag_context)
        else:
            system_prompt = SYSTEM_PROMPT

        # Ensure system prompt is first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        else:
            # Replace existing system prompt with RAG-enhanced one
            messages = [SystemMessage(content=system_prompt)] + list(messages[1:])

        # Call the LLM
        response = await self.llm.ainvoke(messages)

        logger.debug(f"Generated response: {response.content[:100]}...")

        return {"messages": messages + [response]}

    def _convert_to_langchain_messages(self, messages: list[Message]) -> list[BaseMessage]:
        """Convert domain messages to LangChain messages."""
        lc_messages = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))
        return lc_messages

    def _convert_from_langchain_message(self, msg: BaseMessage) -> Message:
        """Convert LangChain message to domain message."""
        if isinstance(msg, HumanMessage):
            role = MessageRole.USER
        elif isinstance(msg, AIMessage):
            role = MessageRole.ASSISTANT
        elif isinstance(msg, SystemMessage):
            role = MessageRole.SYSTEM
        else:
            role = MessageRole.ASSISTANT

        return Message(role=role, content=msg.content)

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