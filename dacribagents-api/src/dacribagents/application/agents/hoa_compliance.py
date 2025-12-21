"""HOA Compliance Agent - Expert on HOA rules, CC&Rs, and community guidelines."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from loguru import logger

from dacribagents.domain.models import AgentType, Message, MessageRole
from dacribagents.infrastructure.settings import Settings, get_settings

from .base import AgentState, BaseAgent
from .general_assistant import create_llm

SYSTEM_PROMPT = """You are the HOA Compliance Agent for Woodcreek community in Fate, Texas.

Your expertise includes:
- HOA rules and regulations
- CC&Rs (Covenants, Conditions & Restrictions)
- Architectural guidelines and approval processes
- Violation notices and appeals
- Community standards and enforcement
- TownSq platform usage for HOA matters

When answering questions:
1. Reference specific rules or guidelines when available from the context
2. Be clear about what requires HOA approval vs. what is allowed
3. Explain the process for submissions or appeals
4. Be helpful but accurate - if unsure, recommend contacting the HOA directly

Contact: hoa@woodcreek.me
HOA Management Platform: TownSq

{rag_context}
"""

SYSTEM_PROMPT_NO_RAG = """You are the HOA Compliance Agent for Woodcreek community in Fate, Texas.

Your expertise includes:
- HOA rules and regulations
- CC&Rs (Covenants, Conditions & Restrictions)
- Architectural guidelines and approval processes
- Violation notices and appeals
- Community standards and enforcement
- TownSq platform usage for HOA matters

Be helpful but accurate - if you don't have specific information about a rule, recommend the homeowner contact the HOA directly or check TownSq.

Contact: hoa@woodcreek.me
HOA Management Platform: TownSq
"""


class HOAComplianceAgent(BaseAgent):
    """HOA Compliance specialist with RAG support."""

    def __init__(self, settings: Settings | None = None, enable_rag: bool = True):
        super().__init__(AgentType.HOA_COMPLIANCE)
        self.settings = settings or get_settings()
        self._llm: BaseChatModel | None = None
        self.enable_rag = enable_rag
        self._retriever = None

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize the LLM."""
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
                logger.info("RAG retriever initialized for HOA Agent")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG retriever: {e}")
        return self._retriever

    @property
    def model_info(self) -> dict[str, str]:
        """Get model configuration info."""
        return {
            "agent": "hoa_compliance",
            "rag_enabled": str(self.enable_rag),
            "provider": self.settings.llm_provider,
        }

    def _get_rag_context(self, query: str) -> str:
        """Retrieve HOA-relevant context."""
        if not self.enable_rag or not self.retriever:
            return ""

        try:
            from dacribagents.application.rag.retriever import build_context

            # Search with HOA-related query enhancement
            enhanced_query = f"HOA rules compliance {query}"
            results = self.retriever.retrieve_sync(
                query=enhanced_query,
                top_k=5,
                min_score=0.25,
            )

            if results:
                context = build_context(results, max_chars=3000)
                logger.info(f"HOA RAG: Retrieved {len(results)} documents")
                return context
            return ""

        except Exception as e:
            logger.warning(f"HOA RAG retrieval failed: {e}")
            return ""

    def _build_graph(self) -> StateGraph:
        """Build the agent graph."""
        graph = StateGraph(AgentState)
        graph.add_node("generate", self._generate_response)
        graph.set_entry_point("generate")
        graph.add_edge("generate", END)
        return graph.compile()

    async def _generate_response(self, state: AgentState) -> dict[str, Any]:
        """Generate response with RAG context."""
        messages = state["messages"]

        # Extract user query
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # Get RAG context
        rag_context = self._get_rag_context(user_query) if user_query else ""

        # Build system prompt
        if rag_context:
            system_prompt = SYSTEM_PROMPT.format(rag_context=rag_context)
        else:
            system_prompt = SYSTEM_PROMPT_NO_RAG

        # Ensure system prompt is first
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        else:
            messages = [SystemMessage(content=system_prompt)] + list(messages[1:])

        response = await self.llm.ainvoke(messages)
        logger.debug(f"HOA Agent response: {response.content[:100]}...")

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
        if isinstance(msg, AIMessage):
            role = MessageRole.ASSISTANT
        elif isinstance(msg, HumanMessage):
            role = MessageRole.USER
        else:
            role = MessageRole.ASSISTANT
        return Message(role=role, content=msg.content)

    async def process(self, messages: list[Message], context: dict[str, Any] | None = None) -> Message:
        """Process messages and return a response."""
        lc_messages = self._convert_to_langchain_messages(messages)

        initial_state: AgentState = {
            "messages": lc_messages,
            "context": context or {},
            "agent_type": self.agent_type.value,
            "thread_id": None,
        }

        result = await self.graph.ainvoke(initial_state)

        if result["messages"]:
            return self._convert_from_langchain_message(result["messages"][-1])

        return Message(
            role=MessageRole.ASSISTANT,
            content="I apologize, but I couldn't generate a response. Please contact hoa@woodcreek.me directly.",
        )


# Singleton
_hoa_agent: HOAComplianceAgent | None = None


def get_hoa_agent(settings: Settings | None = None) -> HOAComplianceAgent:
    """Get or create HOA agent singleton."""
    global _hoa_agent
    if _hoa_agent is None:
        _hoa_agent = HOAComplianceAgent(settings)
    return _hoa_agent