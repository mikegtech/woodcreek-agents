"""Supervisor Agent - Routes requests to specialized agents based on intent."""

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from dacribagents.domain.models import AgentType, Message, MessageRole
from dacribagents.infrastructure.settings import Settings, get_settings

from .base import AgentState, BaseAgent
from .general_assistant import create_llm

ROUTING_SYSTEM_PROMPT = """You are a routing agent for the Woodcreek home management system.
Your ONLY job is to classify user requests and return the appropriate agent name.

Available agents and their responsibilities:

1. **solar** - Solar panel installation, solar energy, panels, inverters, batteries, 
   net metering, Oncor interconnection, solar permits, solar financing, solar monitoring.
   Keywords: solar, panels, energy, kilowatt, kWh, inverter, battery, sunlight, roof installation

2. **hoa_compliance** - HOA rules, CC&Rs, architectural guidelines, violations, 
   community standards, TownSq, HOA fees, property modifications requiring approval.
   Keywords: HOA, rules, violation, CC&R, architectural, approval, TownSq, community standards

3. **home_maintenance** - Home repairs, HVAC, plumbing, electrical, landscaping, 
   appliances, contractors, maintenance schedules.
   Keywords: repair, fix, broken, maintenance, HVAC, plumbing, electrical, landscaping, contractor

4. **security_cameras** - Security systems, cameras, monitoring, alerts, motion detection,
   Ring, Nest, surveillance.
   Keywords: camera, security, surveillance, motion, alert, Ring, Nest, monitor

5. **general_assistant** - General questions, casual conversation, anything that doesn't 
   clearly fit the above categories.

RULES:
- Respond with ONLY the agent name (one word, lowercase)
- If the request mentions solar/panels/energy → solar
- If the request mentions HOA/rules/violations/approval → hoa_compliance
- If unclear between two agents, prefer the more specific one
- Default to general_assistant only if truly unclear

Examples:
- "When will my solar panels be installed?" → solar
- "Do I need HOA approval for solar panels?" → solar (solar is the primary topic)
- "What are the rules about fences?" → hoa_compliance
- "My AC is broken" → home_maintenance
- "What's the weather today?" → general_assistant
"""


class SupervisorAgent(BaseAgent):
    """Supervisor that routes to specialized agents using LLM classification."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(AgentType.SUPERVISOR)
        self.settings = settings or get_settings()
        self._llm = None
        self._agents: dict[AgentType, BaseAgent] = {}

    @property
    def llm(self):
        """Lazily initialize the LLM."""
        if self._llm is None:
            self._llm = create_llm(self.settings)
        return self._llm

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialized agent."""
        self._agents[agent.agent_type] = agent
        logger.info(f"Supervisor: Registered {agent.agent_type.value}")

    def get_agent(self, agent_type: AgentType) -> BaseAgent | None:
        """Get a registered agent by type."""
        return self._agents.get(agent_type)

    @property
    def registered_agents(self) -> list[str]:
        """List registered agent names."""
        return [a.value for a in self._agents.keys()]

    async def classify_intent(self, query: str) -> AgentType:
        """Classify user intent and return target agent type."""
        messages = [
            SystemMessage(content=ROUTING_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            route_to = response.content.strip().lower().replace("_", "_")

            # Clean up response (LLM might add extra text)
            route_to = route_to.split()[0] if route_to else "general_assistant"
            route_to = route_to.strip(".,!?\"'")

            logger.info(f"Intent classification: '{query[:50]}...' → {route_to}")

            # Map to AgentType
            agent_mapping = {
                "solar": AgentType.SOLAR,
                "hoa_compliance": AgentType.HOA_COMPLIANCE,
                "hoa": AgentType.HOA_COMPLIANCE,
                "home_maintenance": AgentType.HOME_MAINTENANCE,
                "maintenance": AgentType.HOME_MAINTENANCE,
                "security_cameras": AgentType.SECURITY_CAMERAS,
                "security": AgentType.SECURITY_CAMERAS,
                "general_assistant": AgentType.GENERAL_ASSISTANT,
                "general": AgentType.GENERAL_ASSISTANT,
            }

            return agent_mapping.get(route_to, AgentType.GENERAL_ASSISTANT)

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return AgentType.GENERAL_ASSISTANT

    def _build_graph(self) -> CompiledStateGraph:
        """Build the supervisor routing graph."""

        async def route_node(state: dict) -> dict:
            """Classify intent and set target agent."""
            messages = state.get("messages", [])
            if not messages:
                return {**state, "current_agent": AgentType.GENERAL_ASSISTANT.value}

            # Get the last user message
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                query = last_message.content
            elif isinstance(last_message, dict):
                query = last_message.get("content", "")
            else:
                query = str(last_message)

            # Classify intent
            target_agent = await self.classify_intent(query)

            return {
                **state,
                "current_agent": target_agent.value,
                "route_decision": target_agent.value,
            }

        async def delegate_node(state: dict) -> dict:
            """Delegate to the selected agent."""
            agent_type_str = state.get("current_agent", "general_assistant")

            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                agent_type = AgentType.GENERAL_ASSISTANT

            agent = self._agents.get(agent_type)

            if agent is None:
                logger.warning(f"Agent {agent_type.value} not registered, using fallback")
                # Try general assistant as fallback
                agent = self._agents.get(AgentType.GENERAL_ASSISTANT)

            if agent is None:
                logger.error("No agents available!")
                state["messages"].append(
                    AIMessage(content="I'm sorry, no agents are currently available to help.")
                )
                return state

            # Get user messages for the agent
            messages = state.get("messages", [])

            # Convert to domain messages
            domain_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    domain_messages.append(Message(role=MessageRole.USER, content=msg.content))
                elif isinstance(msg, AIMessage):
                    domain_messages.append(Message(role=MessageRole.ASSISTANT, content=msg.content))
                elif isinstance(msg, dict):
                    role = MessageRole(msg.get("role", "user"))
                    domain_messages.append(Message(role=role, content=msg.get("content", "")))

            # Process with the selected agent
            response = await agent.process(domain_messages)

            # Add response to state
            state["messages"].append(AIMessage(content=response.content))
            state["response"] = response.content
            state["handled_by"] = agent_type.value

            logger.info(f"Request handled by: {agent_type.value}")

            return state

        def should_delegate(state: dict) -> Literal["delegate", "end"]:
            """Check if we should delegate."""
            if state.get("current_agent"):
                return "delegate"
            return "end"

        # Build graph
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

    async def process(self, messages: list[Message], context: dict[str, Any] | None = None) -> Message:
        """Process through supervisor routing."""
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))

        initial_state = {
            "messages": lc_messages,
            "context": context or {},
            "current_agent": None,
        }

        result = await self.graph.ainvoke(initial_state)

        # Extract response
        if result.get("response"):
            return Message(
                role=MessageRole.ASSISTANT,
                content=result["response"],
                metadata={"handled_by": result.get("handled_by", "unknown")},
            )

        # Fallback
        return Message(
            role=MessageRole.ASSISTANT,
            content="I'm sorry, I couldn't process your request.",
        )


# Singleton with lazy initialization
_supervisor: SupervisorAgent | None = None


def get_supervisor(settings: Settings | None = None) -> SupervisorAgent:
    """Get or create Supervisor singleton with all agents registered."""
    global _supervisor
    if _supervisor is None:
        _supervisor = SupervisorAgent(settings)

        # Register all available agents
        from .general_assistant import get_general_assistant
        from .hoa_compliance import get_hoa_agent
        from .solar import get_solar_agent

        _supervisor.register_agent(get_general_assistant(settings))
        _supervisor.register_agent(get_hoa_agent(settings))
        _supervisor.register_agent(get_solar_agent(settings))

        logger.info(f"Supervisor initialized with agents: {_supervisor.registered_agents}")

    return _supervisor