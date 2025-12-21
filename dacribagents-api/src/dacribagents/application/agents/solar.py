"""Solar Agent - Expert on solar panel installation, permits, and energy systems."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from loguru import logger

from dacribagents.domain.models import AgentType, Message, MessageRole
from dacribagents.infrastructure.settings import Settings, get_settings

from .base import AgentState, BaseAgent
from .general_assistant import create_llm

SYSTEM_PROMPT = """You are the Solar Installation Agent for a homeowner in Woodcreek community, Fate, Texas.

Your expertise includes:
- Solar panel installation process and timeline
- Permits and inspections required in Texas/Rockwall County
- HOA approval requirements for solar installations
- Net metering and utility interconnection (Oncor territory)
- Solar financing options (loans, leases, PPAs)
- Equipment selection (panels, inverters, batteries)
- Installation contractor coordination
- Monitoring and maintenance

Current installation context:
- Location: Woodcreek community, Fate, TX
- Utility: Oncor (transmission), various REPs for retail
- HOA: Requires architectural approval for exterior modifications

When answering:
1. Reference any relevant emails or documents from the context
2. Provide specific, actionable information about the solar installation process
3. Remind about HOA approval requirements when relevant
4. Be helpful with timeline expectations and next steps

Contact: solar@woodcreek.me

{rag_context}
"""

SYSTEM_PROMPT_NO_RAG = """You are the Solar Installation Agent for a homeowner in Woodcreek community, Fate, Texas.

Your expertise includes:
- Solar panel installation process and timeline
- Permits and inspections required in Texas/Rockwall County
- HOA approval requirements for solar installations
- Net metering and utility interconnection (Oncor territory)
- Solar financing options (loans, leases, PPAs)
- Equipment selection (panels, inverters, batteries)
- Installation contractor coordination
- Monitoring and maintenance

Current installation context:
- Location: Woodcreek community, Fate, TX
- Utility: Oncor (transmission), various REPs for retail
- HOA: Requires architectural approval for exterior modifications

Be helpful with general solar information. For specific details about your installation, check your emails or contact solar@woodcreek.me.
"""


class SolarAgent(BaseAgent):
    """Solar installation specialist with RAG support."""

    def __init__(self, settings: Settings | None = None, enable_rag: bool = True):
        super().__init__(AgentType.SOLAR)
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
                logger.info("RAG retriever initialized for Solar Agent")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG retriever: {e}")
        return self._retriever

    @property
    def model_info(self) -> dict[str, str]:
        """Get model configuration info."""
        return {
            "agent": "solar",
            "rag_enabled": str(self.enable_rag),
            "provider": self.settings.llm_provider,
        }

    def _get_rag_context(self, query: str) -> str:
        """Retrieve solar-relevant context."""
        if not self.enable_rag or not self.retriever:
            return ""

        try:
            from dacribagents.application.rag.retriever import build_context

            # Search with solar-related query enhancement
            enhanced_query = f"solar panel installation energy {query}"
            results = self.retriever.retrieve_sync(
                query=enhanced_query,
                top_k=5,
                min_score=0.25,
            )

            if results:
                context = build_context(results, max_chars=3000)
                logger.info(f"Solar RAG: Retrieved {len(results)} documents")
                return context
            return ""

        except Exception as e:
            logger.warning(f"Solar RAG retrieval failed: {e}")
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
        logger.debug(f"Solar Agent response: {response.content[:100]}...")

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
            content="I apologize, but I couldn't generate a response. Please contact solar@woodcreek.me directly.",
        )


# Singleton
_solar_agent: SolarAgent | None = None


def get_solar_agent(settings: Settings | None = None) -> SolarAgent:
    """Get or create Solar agent singleton."""
    global _solar_agent
    if _solar_agent is None:
        _solar_agent = SolarAgent(settings)
    return _solar_agent