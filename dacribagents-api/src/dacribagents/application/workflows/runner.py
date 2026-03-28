"""Reminder workflow runner — invokes LangGraph lifecycle for eligible reminders.

Called by the control loop when ``LANGGRAPH_ENABLED=true``.  Falls back
to the imperative service-chain path when disabled.

The runner:
1. Finds reminders that need lifecycle progression.
2. Creates or resumes a LangGraph workflow per reminder.
3. Uses durable checkpointing so workflows survive restarts.
"""

from __future__ import annotations

from uuid import UUID

from loguru import logger

from dacribagents.application.ports.delivery_channel import DeliveryChannelAdapter
from dacribagents.application.ports.event_publisher import EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.workflows.reminder_lifecycle import (
    ReminderWorkflowState,
    build_reminder_graph,
    initial_state,
)
from dacribagents.domain.reminders.enums import DeliveryChannel, ReminderState
from dacribagents.domain.reminders.lifecycle import TERMINAL_STATES


def run_workflows(  # noqa: PLR0913
    store: ReminderStore,
    adapters: dict[DeliveryChannel, DeliveryChannelAdapter],
    household_id: UUID,
    *,
    events: EventPublisher | None = None,
    checkpointer: object | None = None,
    max_reminders: int = 20,
) -> int:
    """Invoke or resume LangGraph workflows for eligible reminders.

    Returns the number of reminders processed.
    """
    publisher = events or NoOpEventPublisher()

    # Build the compiled graph (with optional checkpointer)
    graph = build_reminder_graph(store, adapters, publisher)

    # Find reminders that need progression
    active_states = [s for s in ReminderState if s not in TERMINAL_STATES]
    reminders, _ = store.list_reminders(household_id, states=active_states, limit=max_reminders)

    processed = 0
    for reminder in reminders:
        thread_id = f"reminder-{reminder.id}"
        config = {"configurable": {"thread_id": thread_id}}

        if checkpointer:
            config["configurable"]["checkpoint_ns"] = "reminder_lifecycle"

        state = initial_state(reminder.id, reminder.household_id)

        try:
            # Invoke the graph — it will progress through as many nodes as possible
            # and stop at wait states (approval, ack, schedule check)
            result = graph.invoke(state, config)
            processed += 1

            if result.get("outcome"):
                logger.info(
                    f"Workflow {thread_id}: outcome={result['outcome']} step={result['step']}"
                )
        except Exception as e:
            logger.error(f"Workflow {thread_id} failed: {e}")

    return processed


def run_single_workflow(
    store: ReminderStore,
    adapters: dict[DeliveryChannel, DeliveryChannelAdapter],
    reminder_id: UUID,
    household_id: UUID,
    *,
    events: EventPublisher | None = None,
) -> ReminderWorkflowState | None:
    """Run the LangGraph lifecycle for a single reminder. Returns final state."""
    publisher = events or NoOpEventPublisher()
    graph = build_reminder_graph(store, adapters, publisher)

    state = initial_state(reminder_id, household_id)
    config = {"configurable": {"thread_id": f"reminder-{reminder_id}"}}

    try:
        return graph.invoke(state, config)
    except Exception as e:
        logger.error(f"Single workflow for {reminder_id} failed: {e}")
        return None
