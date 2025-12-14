"""Base agent class for all Woodcreek agents."""

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from loguru import logger

from dacribagents.domain.models import AgentType, Message, MessageRole


class AgentState(TypedDict):
    """State passed through the agent graph."""

    messages: list[BaseMessage]
    context: dict[str, Any]
    agent_type: str
    thread_id: str | None


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self._graph: StateGraph | None = None

    @property
    def graph(self) -> StateGraph:
        """Lazily build and return the agent graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for this agent."""
        pass

    @abstractmethod
    async def process(self, messages: list[Message], context: dict[str, Any] | None = None) -> Message:
        """Process messages and return a response."""
        pass

    def _convert_to_langchain_messages(self, messages: list[Message]) -> list[BaseMessage]:
        """Convert domain messages to LangChain message format."""
        lc_messages: list[BaseMessage] = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))
            else:
                logger.warning(f"Unknown message role: {msg.role}, treating as human message")
                lc_messages.append(HumanMessage(content=msg.content))
        return lc_messages

    def _convert_from_langchain_message(self, message: BaseMessage) -> Message:
        """Convert a LangChain message back to domain message format."""
        if isinstance(message, AIMessage):
            role = MessageRole.ASSISTANT
        elif isinstance(message, HumanMessage):
            role = MessageRole.USER
        elif isinstance(message, SystemMessage):
            role = MessageRole.SYSTEM
        else:
            role = MessageRole.ASSISTANT

        content = message.content if isinstance(message.content, str) else str(message.content)
        return Message(role=role, content=content)