"""Event-to-reminder intake service.

Deterministic mapping of upstream Kafka events into reminder creation
commands.  This is the application boundary between the event bus and
the reminder domain.

Supported event types (MVP):
- ``warranty.expiring``       → reminder, normal
- ``hoa.deadline.approaching`` → reminder, urgent
- ``camera.offline``          → alert, critical
- ``maintenance.due``         → reminder, normal

Unknown event types are logged and ignored.  All event-created reminders
require approval unless the mapping explicitly marks them otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from loguru import logger

from dacribagents.application.ports.event_publisher import DomainEvent, EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.use_cases.reminder_workflows import (
    DuplicateReminderError,
    ingest_event,
)
from dacribagents.domain.reminders.entities import Reminder
from dacribagents.domain.reminders.enums import (
    NotificationIntent,
    ReminderSource,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.models import (
    EventIntakeRequest,
    ReminderTargetInput,
)

# ── Upstream event model ────────────────────────────────────────────────────


@dataclass(frozen=True)
class UpstreamEvent:
    """Normalized upstream event from Kafka or other sources."""

    event_type: str
    event_id: str
    household_id: UUID
    timestamp: datetime
    subject: str
    body: str = ""
    severity: str = "normal"
    source_service: str = ""
    metadata: dict = field(default_factory=dict)


# ── Event type mapping ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EventMapping:
    """How an event type maps to a reminder."""

    source: ReminderSource
    intent: NotificationIntent
    urgency: UrgencyLevel
    requires_approval: bool = True


_EVENT_MAPPINGS: dict[str, EventMapping] = {
    "warranty.expiring": EventMapping(
        source=ReminderSource.WARRANTY,
        intent=NotificationIntent.REMINDER,
        urgency=UrgencyLevel.NORMAL,
    ),
    "hoa.deadline.approaching": EventMapping(
        source=ReminderSource.HOA,
        intent=NotificationIntent.REMINDER,
        urgency=UrgencyLevel.URGENT,
    ),
    "camera.offline": EventMapping(
        source=ReminderSource.TELEMETRY,
        intent=NotificationIntent.ALERT,
        urgency=UrgencyLevel.CRITICAL,
    ),
    "maintenance.due": EventMapping(
        source=ReminderSource.MAINTENANCE,
        intent=NotificationIntent.REMINDER,
        urgency=UrgencyLevel.NORMAL,
    ),
}

# Severity string from upstream → UrgencyLevel override
_SEVERITY_MAP: dict[str, UrgencyLevel] = {
    "critical": UrgencyLevel.CRITICAL,
    "high": UrgencyLevel.URGENT,
    "urgent": UrgencyLevel.URGENT,
    "normal": UrgencyLevel.NORMAL,
    "low": UrgencyLevel.LOW,
}


def get_supported_event_types() -> list[str]:
    """Return the list of event types this service handles."""
    return list(_EVENT_MAPPINGS.keys())


# ── Intake result ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IntakeResult:
    """Outcome of processing an upstream event."""

    event_id: str
    status: str  # "created", "duplicate", "ignored"
    reminder: Reminder | None = None
    reason: str | None = None


# ── Main intake function ────────────────────────────────────────────────────


def process_upstream_event(
    store: ReminderStore,
    event: UpstreamEvent,
    events: EventPublisher | None = None,
) -> IntakeResult:
    """Map an upstream event to a reminder and ingest it.

    Returns an IntakeResult indicating what happened.
    """
    publisher = events or NoOpEventPublisher()

    mapping = _EVENT_MAPPINGS.get(event.event_type)
    if mapping is None:
        logger.debug(f"Ignoring unsupported event type: {event.event_type}")
        return IntakeResult(event_id=event.event_id, status="ignored", reason=f"Unsupported event type: {event.event_type}")

    # Severity override: use upstream severity only when explicitly set
    # and different from default "normal" (which would downgrade urgent mappings).
    urgency = mapping.urgency
    if event.severity and event.severity != "normal":
        urgency = _SEVERITY_MAP.get(event.severity, mapping.urgency)

    dedupe_key = f"{event.event_type}:{event.event_id}"

    request = EventIntakeRequest(
        source=mapping.source,
        source_event_id=event.event_id,
        dedupe_key=dedupe_key,
        household_id=event.household_id,
        subject=event.subject,
        body=event.body,
        urgency=urgency,
        intent=mapping.intent,
        source_agent=event.source_service or None,
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
    )

    try:
        reminder = ingest_event(store, request)
    except DuplicateReminderError as e:
        logger.info(f"Duplicate event suppressed: {dedupe_key} (existing: {e.existing_id})")
        publisher.publish(DomainEvent(
            event_type="reminder.duplicate_ignored",
            reminder_id=e.existing_id,
            household_id=event.household_id,
            timestamp=event.timestamp,
            payload={"source_event_id": event.event_id, "dedupe_key": dedupe_key},
        ))
        return IntakeResult(event_id=event.event_id, status="duplicate", reason=f"Existing reminder: {e.existing_id}")

    logger.info(f"Reminder created from event {event.event_type}: {reminder.id}")
    publisher.publish(DomainEvent(
        event_type="reminder.created_from_event",
        reminder_id=reminder.id,
        household_id=event.household_id,
        timestamp=event.timestamp,
        payload={
            "source_event_type": event.event_type,
            "source_event_id": event.event_id,
            "intent": mapping.intent.value,
            "urgency": urgency.value,
        },
    ))

    return IntakeResult(event_id=event.event_id, status="created", reminder=reminder)
