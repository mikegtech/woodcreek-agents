"""Supervisor agent for routing requests to specialized agents."""

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from dacribagents.application.agents.base import BaseAgent
from dacribagents.domain.models import AgentState, AgentType

SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor agent for the Woodcreek home management system.
Your job is to route user requests to the appropriate specialized agent.

Available agents:
- hoa_compliance: Handles HOA rules, CC&Rs, compliance questions, and violation concerns
- home_maintenance: Handles maintenance tasks, landscaping, repairs, and vendor coordination
- security_cameras: Handles security cameras, monitoring, alerts, and perimeter security
- general_assistant: Handles general questions that don't fit other categories

Analyze the user's request and determine which agent should handle it.
Respond with ONLY the agent name (one of: hoa_compliance, home_maintenance, security_cameras, general_assistant).
"""


class SupervisorAgent(BaseAgent):
    """Supervisor agent that routes to specialized agents."""

    agent_type = AgentType.SUPERVISOR
    system_prompt = SUPERVISOR_SYSTEM_PROMPT

    def __init__(self, llm: Any, agents: dict[AgentType, BaseAgent] | None = None):
        """Initialize supervisor with LLM and agent registry."""
        self.llm = llm
        self.agents = agents or {}
        super().__init__()

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialized agent."""
        self.agents[agent.agent_type] = agent
        logger.info(f"Registered agent: {agent.agent_type.value}")

    def _build_graph(self) -> CompiledStateGraph:
        """Build the supervisor routing graph."""

        async def route_node(state: dict) -> dict:
            """Determine which agent should handle the request."""
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=state["messages"][-1]["content"]),
            ]

            response = await self.llm.ainvoke(messages)
            route_to = response.content.strip().lower()

            # Map response to AgentType
            agent_mapping = {
                "hoa_compliance": AgentType.HOA_COMPLIANCE,
                "home_maintenance": AgentType.HOME_MAINTENANCE,
                "security_cameras": AgentType.SECURITY_CAMERAS,
                "general_assistant": AgentType.GENERAL_ASSISTANT,
            }

            target_agent = agent_mapping.get(route_to, AgentType.GENERAL_ASSISTANT)
            logger.info(f"Routing to: {target_agent.value}")

            return {**state, "current_agent": target_agent.value, "route_decision": route_to}

        async def delegate_node(state: dict) -> dict:
            """Delegate to the selected agent."""
            agent_type_str = state.get("current_agent", "general_assistant")

            # Convert string back to AgentType
            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                agent_type = AgentType.GENERAL_ASSISTANT

            agent = self.agents.get(agent_type)

            if agent is None:
                logger.warning(f"No agent registered for {agent_type}, using fallback response")
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I'd route this to the {agent_type.value} agent, but it's not currently available.",
                })
                return state

            # Invoke the specialized agent
            result = await agent.invoke(
                state["messages"][-1]["content"] if state["messages"] else "",
                thread_id=state.get("thread_id"),
            )
            return {**state, **result}

        def should_delegate(state: dict) -> Literal["delegate", "end"]:
            """Check if we should delegate or end."""
            if state.get("current_agent"):
                return "delegate"
            return "end"

        # Build the graph
        workflow = StateGraph(dict)
        workflow.add_node("route", route_node)
        workflow.add_node("delegate", delegate_node)

        workflow.set_entry_point("route")
        workflow.add_conditional_edges(
            "route",
            should_delegate,
            {"delegate": "delegate", "end": END},
        )
        workflow.add_edge("delegate", END)

        return workflow.compile()

    async def process(self, state: AgentState) -> AgentState:
        """Process through the supervisor graph."""
        result = await self.graph.ainvoke(state.model_dump())
        return AgentState(**result)