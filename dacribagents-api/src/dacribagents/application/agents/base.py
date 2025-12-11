"""Base agent class for LangGraph-based agents."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from dacribagents.domain.models import AgentState, AgentType


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    agent_type: AgentType
    system_prompt: str = "You are a helpful assistant."

    def __init__(self):
        """Initialize the base agent."""
        self._graph: CompiledStateGraph | None = None
        logger.info(f"Initializing {self.agent_type.value} agent")

    @property
    def graph(self) -> CompiledStateGraph:
        """Get or create the compiled state graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph state graph. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """Process the current state. Must be implemented by subclasses."""
        pass

    def _create_messages(self, state: AgentState) -> list[Any]:
        """Convert state messages to LangChain message objects."""
        messages = [SystemMessage(content=self.system_prompt)]
        for msg in state.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        return messages

    async def invoke(self, user_message: str, thread_id: str | None = None) -> dict[str, Any]:
        """Invoke the agent with a user message."""
        initial_state = AgentState(
            messages=[{"role": "user", "content": user_message}],
            current_agent=self.agent_type,
        )

        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        result = await self.graph.ainvoke(initial_state.model_dump(), config=config)
        return result


class SimpleReActAgent(BaseAgent):
    """A simple ReAct-style agent implementation."""

    def __init__(self, llm: Any, tools: list[Any] | None = None):
        """Initialize with LLM and optional tools."""
        self.llm = llm
        self.tools = tools or []
        super().__init__()

    def _build_graph(self) -> CompiledStateGraph:
        """Build a simple reasoning graph."""

        def should_continue(state: dict) -> Literal["continue", "end"]:
            """Determine if agent should continue or end."""
            if not state.get("should_continue", True):
                return "end"
            messages = state.get("messages", [])
            if messages and messages[-1].get("role") == "assistant":
                return "end"
            return "continue"

        async def agent_node(state: dict) -> dict:
            """Main agent reasoning node."""
            result = await self.process(AgentState(**state))
            return result.model_dump()

        # Build the graph
        workflow = StateGraph(dict)
        workflow.add_node("agent", agent_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "agent", "end": END},
        )

        return workflow.compile()

    async def process(self, state: AgentState) -> AgentState:
        """Process state through LLM."""
        messages = self._create_messages(state)

        if self.tools:
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self.llm.ainvoke(messages)

        state.messages.append({"role": "assistant", "content": response.content})
        state.should_continue = False
        return state