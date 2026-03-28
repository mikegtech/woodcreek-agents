"""Acknowledgment intake service — marks deliveries as acknowledged.

This is the channel-neutral seam for ack handling.  SMS reply webhooks,
email action links, Slack buttons, and push notification actions all
converge here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from dacribagents.application.ports.event_publisher import DomainEvent, EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.domain.reminders.entities import ReminderAcknowledgement
from dacribagents.domain.reminders.enums import AckMethod, ReminderState
from dacribagents.domain.reminders.lifecycle import can_transition


def acknowledge_delivery(  # noqa: PLR0913
    store: ReminderStore,
    *,
    reminder_id: UUID,
    member_id: UUID,
    method: AckMethod,
    delivery_id: UUID | None = None,
    note: str | None = None,
    events: EventPublisher | None = None,
) -> ReminderAcknowledgement | None:
    """Record an acknowledgment and transition the reminder if appropriate."""
    reminder = store.get_reminder(reminder_id)
    if reminder is None:
        return None

    ack = ReminderAcknowledgement(
        id=uuid4(),
        delivery_id=delivery_id or uuid4(),
        member_id=member_id,
        method=method,
        acknowledged_at=datetime.now(UTC),
        note=note,
    )
    ack = store.create_acknowledgement(ack)

    if can_transition(reminder.state, ReminderState.ACKNOWLEDGED):
        store.update_reminder_state(reminder_id, ReminderState.ACKNOWLEDGED)
        publisher = events or NoOpEventPublisher()
        publisher.publish(DomainEvent(
            event_type="reminder.acknowledged",
            reminder_id=reminder_id,
            household_id=reminder.household_id,
            timestamp=datetime.now(UTC),
            payload={"method": method.value, "member_id": str(member_id)},
        ))

    return ack
