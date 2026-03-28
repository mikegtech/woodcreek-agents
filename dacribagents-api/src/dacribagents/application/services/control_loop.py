"""Reminder control-loop driver.

Runs the multi-phase reminder cycle on each tick:
1. **Schedule**: advance due SCHEDULED reminders to PENDING_DELIVERY.
2. **Dispatch**: send PENDING_DELIVERY reminders via channel adapters.
3. **Escalate**: retry failed deliveries and escalate unacknowledged ones.
4. **Digest**: batch eligible low-priority reminders.
5. **LangGraph** (optional): invoke/resume per-reminder lifecycle workflows.

The ``run_cycle()`` function is a pure operation — it can be driven by
a FastAPI background task, an external cron, or a scheduled Lambda.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from loguru import logger

from dacribagents.application.ports.delivery_channel import DeliveryChannelAdapter
from dacribagents.application.ports.event_publisher import EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services import digest, escalation
from dacribagents.application.services.delivery import DeliveryDispatcher
from dacribagents.application.services.scheduler import tick
from dacribagents.domain.reminders.enums import DeliveryChannel


@dataclass(frozen=True)
class CycleResult:
    """Summary of a single control-loop cycle."""

    scheduled: int = 0
    dispatched: int = 0
    retried: int = 0
    escalated: int = 0
    digested: int = 0
    graph_processed: int = 0


def run_cycle(  # noqa: PLR0913
    store: ReminderStore,
    adapters: dict[DeliveryChannel, DeliveryChannelAdapter],
    household_id: UUID,
    *,
    now: datetime | None = None,
    events: EventPublisher | None = None,
    langgraph_enabled: bool = False,
) -> CycleResult:
    """Execute one full control-loop cycle.

    Safe to call frequently — idempotent within a tick window.
    When ``langgraph_enabled`` is True, Phase 5 runs per-reminder
    LangGraph workflows in addition to the imperative phases.
    """
    publisher = events or NoOpEventPublisher()

    # Phase 1: Schedule — advance due reminders
    executions = tick(store, now)
    logger.info(f"Control loop: {len(executions)} reminders scheduled")

    # Phase 2: Dispatch — send pending deliveries
    dispatcher = DeliveryDispatcher(store=store, adapters=adapters, events=publisher)
    deliveries = dispatcher.dispatch_pending(household_id)
    logger.info(f"Control loop: {len(deliveries)} deliveries dispatched")

    # Phase 3: Retry + Escalate
    retried = escalation.check_retries(store, adapters, household_id, now)
    escalated = escalation.check_escalations(store, adapters, household_id, now, publisher)
    if retried:
        logger.info(f"Control loop: {len(retried)} deliveries retried")
    if escalated:
        logger.info(f"Control loop: {len(escalated)} reminders escalated")

    # Phase 4: Digest — batch eligible low-priority reminders
    email_adapter = adapters.get(DeliveryChannel.EMAIL)
    digested = 0
    if email_adapter:
        batch = digest.generate_and_deliver(store, household_id, email_adapter, events=publisher, now=now)
        if batch:
            digested = len(batch.reminder_ids)
            logger.info(f"Control loop: digest generated with {digested} items")

    # Phase 5: LangGraph workflows (optional, feature-gated)
    graph_processed = 0
    if langgraph_enabled:
        try:
            from dacribagents.application.workflows.runner import run_workflows  # noqa: PLC0415

            graph_processed = run_workflows(store, adapters, household_id, events=publisher)
            if graph_processed:
                logger.info(f"Control loop: {graph_processed} LangGraph workflows processed")
        except Exception as e:
            logger.error(f"LangGraph workflow phase failed (non-fatal): {e}")

    return CycleResult(
        scheduled=len(executions),
        dispatched=len(deliveries),
        retried=len(retried),
        escalated=len(escalated),
        digested=digested,
        graph_processed=graph_processed,
    )
