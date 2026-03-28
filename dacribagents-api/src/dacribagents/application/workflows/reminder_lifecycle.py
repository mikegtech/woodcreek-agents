"""LangGraph durable workflow for per-reminder lifecycle orchestration.

Graph shape::

    evaluate_state → route
        ├── ready_to_deliver → attempt_delivery → route
        │       ├── delivery_failed → retry_decision → route
        │       │       ├── retry → attempt_delivery
        │       │       └── give up → END
        │       └── delivered → END (wait for ack; re-invoke on next tick)
        └── all other states → END (wait; re-invoke on next tick)

Each invoke drives the graph through immediate transitions, then exits
at wait states.  The control loop re-invokes on the next tick.

Invariants:
- PostgreSQL reminder tables are the canonical state.
- Graph state is an orchestration envelope (reminder_id + step metadata).
- Nodes read the store for current state — re-invocation always sees updates.
"""

from __future__ import annotations

from typing import TypedDict
from uuid import UUID

from langgraph.graph import END, StateGraph

from dacribagents.application.ports.delivery_channel import DeliveryChannelAdapter
from dacribagents.application.ports.event_publisher import EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services.delivery import DeliveryDispatcher
from dacribagents.domain.reminders.enums import DeliveryChannel, ReminderState
from dacribagents.domain.reminders.lifecycle import is_terminal


class ReminderWorkflowState(TypedDict):
    reminder_id: str
    household_id: str
    outcome: str
    delivery_attempted: bool
    retry_count: int


def initial_state(reminder_id: UUID, household_id: UUID) -> ReminderWorkflowState:
    return ReminderWorkflowState(
        reminder_id=str(reminder_id),
        household_id=str(household_id),
        outcome="",
        delivery_attempted=False,
        retry_count=0,
    )


def build_reminder_graph(
    store: ReminderStore,
    adapters: dict[DeliveryChannel, DeliveryChannelAdapter] | None = None,
    events: EventPublisher | None = None,
) -> StateGraph:
    _adapters = adapters or {}
    _events = events or NoOpEventPublisher()

    def _get(state):
        return store.get_reminder(UUID(state["reminder_id"]))

    def evaluate_state(state: ReminderWorkflowState) -> ReminderWorkflowState:
        r = _get(state)
        if r is None:
            return {**state, "outcome": "not_found"}
        if is_terminal(r.state):
            return {**state, "outcome": r.state.value}
        if r.state == ReminderState.PENDING_DELIVERY:
            return {**state, "outcome": "ready_to_deliver"}
        return {**state, "outcome": "waiting"}

    def attempt_delivery(state: ReminderWorkflowState) -> ReminderWorkflowState:
        r = _get(state)
        if r is None or r.state != ReminderState.PENDING_DELIVERY:
            return {**state, "outcome": "not_ready"}
        DeliveryDispatcher(store=store, adapters=_adapters, events=_events).dispatch_one(r.id)
        updated = store.get_reminder(r.id)
        if updated and updated.state == ReminderState.DELIVERED:
            return {**state, "outcome": "delivered", "delivery_attempted": True}
        if updated and updated.state == ReminderState.FAILED:
            return {**state, "outcome": "delivery_failed", "delivery_attempted": True}
        return {**state, "outcome": "delivered", "delivery_attempted": True}

    def retry_decision(state: ReminderWorkflowState) -> ReminderWorkflowState:
        if state["retry_count"] >= 3:
            return {**state, "outcome": "failed"}
        return {**state, "outcome": "retry", "retry_count": state["retry_count"] + 1}

    def route_after_eval(state: ReminderWorkflowState) -> str:
        return "attempt_delivery" if state["outcome"] == "ready_to_deliver" else END

    def route_after_delivery(state: ReminderWorkflowState) -> str:
        return "retry_decision" if state["outcome"] == "delivery_failed" else END

    def route_after_retry(state: ReminderWorkflowState) -> str:
        return "attempt_delivery" if state["outcome"] == "retry" else END

    graph = StateGraph(ReminderWorkflowState)
    graph.add_node("evaluate_state", evaluate_state)
    graph.add_node("attempt_delivery", attempt_delivery)
    graph.add_node("retry_decision", retry_decision)
    graph.set_entry_point("evaluate_state")
    graph.add_conditional_edges("evaluate_state", route_after_eval, {
        "attempt_delivery": "attempt_delivery", END: END,
    })
    graph.add_conditional_edges("attempt_delivery", route_after_delivery, {
        "retry_decision": "retry_decision", END: END,
    })
    graph.add_conditional_edges("retry_decision", route_after_retry, {
        "attempt_delivery": "attempt_delivery", END: END,
    })
    return graph.compile()
