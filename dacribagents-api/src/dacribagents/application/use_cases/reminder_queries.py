"""Read-only reminder intelligence queries.

These are pure query functions — no state mutations.  They build on top of
the ``ReminderStore`` port and expose the read-model that Slack and other
operator surfaces consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from dacribagents.application.ports.calendar_adapter import CalendarEvent, DateRange
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.domain.reminders.entities import (
    Reminder,
    ReminderDelivery,
    ReminderSchedule,
    ReminderTarget,
)
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    NotificationIntent,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.lifecycle import TERMINAL_STATES

# ── Result types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReminderExplanation:
    """Human-readable explanation of a reminder's state and metadata."""

    reminder: Reminder
    targets: list[ReminderTarget]
    schedule: ReminderSchedule | None
    approval_reason: str | None
    source_description: str
    lifecycle_summary: str


@dataclass(frozen=True)
class DraftPreview:
    """Preview of what a reminder would look like if created."""

    subject: str
    body: str
    intent: NotificationIntent
    urgency: UrgencyLevel
    source: ReminderSource
    target_description: str
    schedule_description: str | None
    likely_channels: list[DeliveryChannel]
    requires_approval: bool
    approval_reason: str | None
    notes: list[str]


@dataclass(frozen=True)
class ScheduleSummary:
    """Combined view of reminders and calendar events for a time window."""

    reminders: list[Reminder]
    calendar_events: list[CalendarEvent]
    date_range: DateRange


# ── Query functions ─────────────────────────────────────────────────────────


def list_pending_approval(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return reminders that are awaiting approval.

    Includes DRAFT reminders flagged for approval and those in PENDING_APPROVAL.
    """
    reminders, _ = store.list_reminders(
        household_id, states=[ReminderState.DRAFT, ReminderState.PENDING_APPROVAL],
    )
    return [r for r in reminders if r.requires_approval or r.state == ReminderState.PENDING_APPROVAL]


def list_active_alerts(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return non-terminal reminders with alert intent."""
    active_states = [s for s in ReminderState if s not in TERMINAL_STATES]
    reminders, _ = store.list_reminders(household_id, states=active_states)
    return [r for r in reminders if r.intent == NotificationIntent.ALERT]


def list_by_states(
    store: ReminderStore,
    household_id: UUID,
    states: list[ReminderState],
) -> list[Reminder]:
    """Return reminders in the given states."""
    reminders, _ = store.list_reminders(household_id, states=states)
    return reminders


def list_active(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return all non-terminal reminders."""
    active_states = [s for s in ReminderState if s not in TERMINAL_STATES]
    reminders, _ = store.list_reminders(household_id, states=active_states)
    return reminders


def list_event_originated(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return non-terminal reminders that were created from upstream events."""
    active_states = [s for s in ReminderState if s not in TERMINAL_STATES]
    reminders, _ = store.list_reminders(household_id, states=active_states)
    return [r for r in reminders if r.source_event_id is not None]


def list_delivered(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return reminders in DELIVERED state."""
    reminders, _ = store.list_reminders(household_id, states=[ReminderState.DELIVERED])
    return reminders


def list_failed_deliveries(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return reminders in FAILED state."""
    reminders, _ = store.list_reminders(household_id, states=[ReminderState.FAILED])
    return reminders


def get_delivery_details(
    store: ReminderStore,
    reminder_id: UUID,
) -> list[ReminderDelivery]:
    """Return all delivery attempts for a reminder's executions."""
    # Collect deliveries across all executions for this reminder
    all_deliveries: list[ReminderDelivery] = []
    # The store tracks deliveries by execution_id, so we look at all executions
    for exec_id, execution in getattr(store, "executions", {}).items():
        if execution.reminder_id == reminder_id:
            all_deliveries.extend(store.get_deliveries_for_execution(exec_id))
    return all_deliveries


def explain_reminder(
    store: ReminderStore,
    reminder_id: UUID,
) -> ReminderExplanation | None:
    """Build a human-readable explanation of a reminder."""
    reminder = store.get_reminder(reminder_id)
    if reminder is None:
        return None

    targets = store.get_targets(reminder_id)
    schedule = store.get_schedule(reminder_id)

    approval_reason = _explain_approval(reminder) if reminder.requires_approval else None
    source_desc = _describe_source(reminder)
    lifecycle_desc = _describe_lifecycle(reminder)

    return ReminderExplanation(
        reminder=reminder,
        targets=targets,
        schedule=schedule,
        approval_reason=approval_reason,
        source_description=source_desc,
        lifecycle_summary=lifecycle_desc,
    )


def preview_draft(  # noqa: PLR0913
    subject: str,
    body: str = "",
    intent: NotificationIntent = NotificationIntent.REMINDER,
    urgency: UrgencyLevel = UrgencyLevel.NORMAL,
    source: ReminderSource = ReminderSource.USER,
    target_type: TargetType = TargetType.HOUSEHOLD,
    schedule_description: str | None = None,
) -> DraftPreview:
    """Preview what a reminder would look like without persisting anything."""
    requires_approval = source not in {ReminderSource.USER, ReminderSource.HOUSEHOLD_ROUTINE}
    approval_reason = (
        f"Reminders from {source.value!r} sources require human approval before delivery"
        if requires_approval
        else None
    )

    likely_channels = _estimate_channels(urgency, intent)
    target_desc = _describe_target_type(target_type)

    notes: list[str] = []
    if not body:
        notes.append("No body text — consider adding details for the recipient.")
    if urgency == UrgencyLevel.CRITICAL:
        notes.append("Critical urgency will bypass quiet hours for delivery.")
    if intent == NotificationIntent.DIGEST:
        notes.append("Digest intent — may be batched with other low-priority items.")

    return DraftPreview(
        subject=subject,
        body=body,
        intent=intent,
        urgency=urgency,
        source=source,
        target_description=target_desc,
        schedule_description=schedule_description,
        likely_channels=likely_channels,
        requires_approval=requires_approval,
        approval_reason=approval_reason,
        notes=notes,
    )


def list_scheduled_next(
    store: ReminderStore,
    household_id: UUID,
    limit: int = 10,
) -> list[tuple[Reminder, ReminderSchedule | None]]:
    """Return the next N scheduled reminders ordered by next_fire_at."""
    reminders, _ = store.list_reminders(household_id, states=[ReminderState.SCHEDULED])
    pairs = []
    for r in reminders:
        sched = store.get_schedule(r.id)
        pairs.append((r, sched))
    # Sort by next_fire_at, nulls last
    pairs.sort(key=lambda p: p[1].next_fire_at if p[1] and p[1].next_fire_at else datetime.max)
    return pairs[:limit]


def list_recurring(
    store: ReminderStore,
    household_id: UUID,
) -> list[tuple[Reminder, ReminderSchedule]]:
    """Return all active recurring reminders with their schedules."""
    reminders, _ = store.list_reminders(household_id, states=[ReminderState.SCHEDULED])
    result = []
    for r in reminders:
        sched = store.get_schedule(r.id)
        if sched and sched.schedule_type == ScheduleType.RECURRING:
            result.append((r, sched))
    return result


def list_pending_delivery(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return reminders that have reached pending_delivery state."""
    reminders, _ = store.list_reminders(household_id, states=[ReminderState.PENDING_DELIVERY])
    return reminders


def build_schedule_summary(
    store: ReminderStore,
    household_id: UUID,
    date_range: DateRange,
    calendar_events: list[CalendarEvent] | None = None,
) -> ScheduleSummary:
    """Build a combined view of reminders and calendar events for a window."""
    active_states = [s for s in ReminderState if s not in TERMINAL_STATES]
    reminders, _ = store.list_reminders(household_id, states=active_states)
    return ScheduleSummary(
        reminders=reminders,
        calendar_events=calendar_events or [],
        date_range=date_range,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

_SOURCE_LABELS: dict[ReminderSource, str] = {
    ReminderSource.USER: "Created manually by a household member",
    ReminderSource.AGENT: "Generated by an agent",
    ReminderSource.CALENDAR: "Derived from a calendar event",
    ReminderSource.EMAIL: "Extracted from an email",
    ReminderSource.MAINTENANCE: "Generated by the maintenance agent",
    ReminderSource.HOA: "Generated by the HOA compliance agent",
    ReminderSource.WARRANTY: "Generated by the warranty tracker",
    ReminderSource.TELEMETRY: "Triggered by device/sensor telemetry",
    ReminderSource.HOUSEHOLD_ROUTINE: "Recurring household routine",
    ReminderSource.EVENT_BUS: "Received via the event bus",
}


def _describe_source(reminder: Reminder) -> str:
    label = _SOURCE_LABELS.get(reminder.source, f"Source: {reminder.source.value}")
    if reminder.source_agent:
        label += f" ({reminder.source_agent})"
    if reminder.source_event_id:
        label += f" [event: {reminder.source_event_id}]"
    return label


def _explain_approval(reminder: Reminder) -> str:
    if reminder.source == ReminderSource.AGENT:
        return "Agent-authored reminders require human approval before delivery."
    if reminder.source in {
        ReminderSource.MAINTENANCE,
        ReminderSource.HOA,
        ReminderSource.WARRANTY,
        ReminderSource.TELEMETRY,
    }:
        return f"Reminders from {reminder.source.value!r} sources require approval — no autonomous delivery in this phase."
    if reminder.source == ReminderSource.EVENT_BUS:
        return "Event-bus reminders require approval for traceability."
    return "This reminder was flagged for approval."


def _describe_lifecycle(reminder: Reminder) -> str:
    if reminder.state in TERMINAL_STATES:
        return f"Terminal state: {reminder.state.value}. No further transitions."
    return f"Current state: {reminder.state.value}. Active — awaiting next lifecycle step."


def _describe_target_type(target_type: TargetType) -> str:
    if target_type == TargetType.HOUSEHOLD:
        return "Targets the entire household"
    if target_type == TargetType.INDIVIDUAL:
        return "Targets a specific household member"
    return "Targets members by role"


def _estimate_channels(urgency: UrgencyLevel, intent: NotificationIntent) -> list[DeliveryChannel]:
    if intent == NotificationIntent.DIGEST:
        return [DeliveryChannel.EMAIL]
    if urgency in {UrgencyLevel.CRITICAL, UrgencyLevel.URGENT}:
        return [DeliveryChannel.SMS, DeliveryChannel.SLACK]
    return [DeliveryChannel.SLACK, DeliveryChannel.EMAIL]
