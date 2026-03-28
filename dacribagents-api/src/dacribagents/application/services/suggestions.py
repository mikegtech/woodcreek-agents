"""Suggestion draft engine — generates reminder suggestions from context.

Produces **draft suggestions only**, not autonomous sends.  All
suggestions enter the existing draft/approval/governance path via
the Slack/operator surface.

Input signals:
- Calendar events (upcoming deadlines, travel prep windows)
- Email-derived dates (if extraction seam exists)
- Recurring household patterns (maintenance cadence, HOA dues)
- Event-originated context already stored in the system

Output: ``SuggestionDraft`` records for operator review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from dacribagents.application.ports.calendar_adapter import CalendarEvent
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.domain.reminders.enums import (
    NotificationIntent,
    ReminderSource,
    ReminderState,
    UrgencyLevel,
)
from dacribagents.domain.reminders.lifecycle import TERMINAL_STATES


@dataclass(frozen=True)
class SuggestionDraft:
    """A suggested reminder for operator review."""

    id: UUID
    household_id: UUID
    subject: str
    body: str
    urgency: UrgencyLevel
    intent: NotificationIntent
    source: ReminderSource
    reason: str  # why this suggestion was generated
    context_sources: list[str]  # what data contributed
    suggested_fire_at: datetime | None = None
    metadata: dict = field(default_factory=dict)


def generate_suggestions(
    store: ReminderStore,
    household_id: UUID,
    calendar_events: list[CalendarEvent] | None = None,
    now: datetime | None = None,
) -> list[SuggestionDraft]:
    """Generate bounded, deterministic reminder suggestions.

    Rules:
    1. Calendar events in the next 3 days with no matching reminder → suggest.
    2. HOA dues pattern: if no HOA reminder exists and we're near a known deadline window → suggest.
    3. Maintenance cadence: if last maintenance reminder was >90 days ago → suggest.
    """
    from datetime import timezone as _tz  # noqa: PLC0415

    now = now or datetime.now(_tz.utc)
    # Ensure now is tz-aware for comparison with tz-aware event datetimes
    if now.tzinfo is None:
        now = now.replace(tzinfo=_tz.utc)
    suggestions: list[SuggestionDraft] = []
    events = calendar_events or []

    # Rule 1: Calendar events without matching reminders
    active_states = [s for s in ReminderState if s not in TERMINAL_STATES]
    reminders, _ = store.list_reminders(household_id, states=active_states)
    reminder_subjects = {r.subject.lower() for r in reminders}

    for event in events:
        if event.start <= now + timedelta(days=3) and event.start > now:
            subject_lower = event.title.lower()
            if not any(subject_lower in rs or rs in subject_lower for rs in reminder_subjects):
                suggestions.append(SuggestionDraft(
                    id=uuid4(),
                    household_id=household_id,
                    subject=f"Upcoming: {event.title}",
                    body=f"Calendar event '{event.title}' is in {(event.start - now).days} day(s).",
                    urgency=UrgencyLevel.NORMAL,
                    intent=NotificationIntent.REMINDER,
                    source=ReminderSource.CALENDAR,
                    reason="Calendar event within 3 days with no matching reminder",
                    context_sources=[f"calendar:{event.id}"],
                    suggested_fire_at=event.start - timedelta(hours=1),
                ))

    # Rule 2: Recurring maintenance check
    maintenance_reminders = [r for r in reminders if r.source == ReminderSource.MAINTENANCE]
    if maintenance_reminders:
        latest = max(maintenance_reminders, key=lambda r: r.created_at)
        created_naive = latest.created_at.replace(tzinfo=None) if latest.created_at.tzinfo else latest.created_at
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        if (now_naive - created_naive).days > 90:
            suggestions.append(SuggestionDraft(
                id=uuid4(),
                household_id=household_id,
                subject="Quarterly maintenance check may be due",
                body="Last maintenance reminder was over 90 days ago.",
                urgency=UrgencyLevel.LOW,
                intent=NotificationIntent.REMINDER,
                source=ReminderSource.MAINTENANCE,
                reason="No maintenance reminder in >90 days",
                context_sources=["maintenance_cadence"],
            ))
    elif reminders:
        # No maintenance reminders at all but household has activity
        suggestions.append(SuggestionDraft(
            id=uuid4(),
            household_id=household_id,
            subject="Consider scheduling a home maintenance check",
            body="No maintenance reminders found — consider seasonal maintenance review.",
            urgency=UrgencyLevel.LOW,
            intent=NotificationIntent.DIGEST,
            source=ReminderSource.MAINTENANCE,
            reason="No maintenance reminders exist for this household",
            context_sources=["household_pattern"],
        ))

    return suggestions
