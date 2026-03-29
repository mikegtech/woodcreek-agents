"""LangGraph interaction graph for Slack/operator requests.

Graph shape:
    classify → validate → route_or_clarify → END

Each node is bounded and purposeful:
- classify: LLM call → ParsedIntent
- validate: deterministic Python → validated ParsedIntent
- route_or_clarify: if valid → call service, render response; if ambiguous → return clarification

This graph is for the INTERACTION LAYER, not the lifecycle engine.
"""

from __future__ import annotations

from typing import TypedDict
from uuid import UUID

from langgraph.graph import END, StateGraph
from loguru import logger

from dacribagents.application.interaction.classifier import classify_intent
from dacribagents.application.interaction.router import RouteResult, route
from dacribagents.application.interaction.validator import validate
from dacribagents.application.ports.reminder_store import ReminderStore


class InteractionState(TypedDict):
    """State for the interaction graph."""

    user_text: str
    household_id: str
    actor_id: str
    response_text: str
    intent_type: str
    confidence: float


def build_interaction_graph(
    store: ReminderStore,
    llm: object | None = None,
) -> object:
    """Build a compiled LangGraph for Slack interaction."""

    def classify_node(state: InteractionState) -> InteractionState:
        """LLM intent classification + entity extraction."""
        intent = classify_intent(state["user_text"], llm=llm)
        logger.info(f"Intent: {intent.intent.value} (confidence={intent.confidence:.2f}) subject={intent.subject!r}")

        # Store the full intent as a module-level cache for routing
        # (LangGraph state is dict-based, so we serialize key fields)
        _intent_cache[state["user_text"]] = intent

        return {
            **state,
            "intent_type": intent.intent.value,
            "confidence": intent.confidence,
        }

    def validate_node(state: InteractionState) -> InteractionState:
        """Deterministic validation of extracted entities."""
        intent = _intent_cache.get(state["user_text"])
        if intent is None:
            return {**state, "response_text": "I couldn't understand that. Try again?"}

        household_id = UUID(state["household_id"])
        validated = validate(intent, store, household_id)
        _intent_cache[state["user_text"]] = validated

        if validated.needs_clarification:
            return {**state, "response_text": validated.clarification_question}

        return state

    def route_node(state: InteractionState) -> InteractionState:
        """Route validated intent to deterministic service."""
        if state.get("response_text"):
            return state  # clarification already set

        intent = _intent_cache.get(state["user_text"])
        if intent is None:
            return {**state, "response_text": "Something went wrong. Try again?"}

        household_id = UUID(state["household_id"])
        actor_id = UUID(state["actor_id"])

        result: RouteResult = route(intent, store, household_id, actor_id)
        return {**state, "response_text": result.message}

    # Build graph
    graph = StateGraph(InteractionState)
    graph.add_node("classify", classify_node)
    graph.add_node("validate", validate_node)
    graph.add_node("route", route_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "validate")
    graph.add_edge("validate", "route")
    graph.add_edge("route", END)

    return graph.compile()


# Module-level cache for passing ParsedIntent between nodes
# (LangGraph state is dict-only; this avoids serialization complexity)
_intent_cache: dict = {}


def run_interaction(
    text: str,
    store: ReminderStore,
    household_id: UUID,
    actor_id: UUID,
    llm: object | None = None,
) -> str:
    """Run the full interaction graph and return the response text."""
    graph = build_interaction_graph(store, llm=llm)

    state = InteractionState(
        user_text=text,
        household_id=str(household_id),
        actor_id=str(actor_id),
        response_text="",
        intent_type="",
        confidence=0.0,
    )

    config = {"configurable": {"thread_id": f"interaction-{hash(text)}"}}
    result = graph.invoke(state, config)
    return result.get("response_text", "I couldn't process that request.")
