"""Calendar query service — free/busy, conflicts, and context enrichment.

Provides application-level calendar intelligence on top of the
provider-neutral ``CalendarAdapter``.  Used by Slack/operator queries
and reminder context enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID

from dacribagents.application.ports.calendar_adapter import CalendarEvent, DateRange
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.domain.reminders.entities import CalendarIdentity, HouseholdMember, Reminder
from dacribagents.domain.reminders.enums import ReminderState
from dacribagents.domain.reminders.lifecycle import TERMINAL_STATES

# ── Result types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BusySlot:
    """A time window where a member is busy."""

    start: datetime
    end: datetime
    title: str
    event_id: str


@dataclass(frozen=True)
class FreeSlot:
    """A time window where a member is free."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class FreeBusySummary:
    """Free/busy summary for a member in a time window."""

    member_name: str
    date_range: DateRange
    busy: list[BusySlot] = field(default_factory=list)
    free: list[FreeSlot] = field(default_factory=list)


@dataclass(frozen=True)
class ConflictReport:
    """Calendar conflicts for a group of members in a time window."""

    date_range: DateRange
    member_conflicts: dict[str, list[BusySlot]] = field(default_factory=dict)


@dataclass(frozen=True)
class ReminderOverlap:
    """A reminder that overlaps with a calendar event."""

    reminder: Reminder
    overlapping_events: list[CalendarEvent]


# ── Identity resolution ────────────────────────────────────────────────────


def resolve_identity(
    store: ReminderStore,
    member: HouseholdMember,
) -> CalendarIdentity | None:
    """Find the active CalendarIdentity for a member.

    Scans the store's calendar_identities (when available on the store
    instance) for a matching member_id + active status.
    """
    identities = getattr(store, "calendar_identities", {})
    for identity in identities.values():
        if identity.member_id == member.id and identity.active:
            return identity
    return None


# ── Free/busy ───────────────────────────────────────────────────────────────


def compute_free_busy(
    events: list[CalendarEvent],
    date_range: DateRange,
    member_name: str = "",
) -> FreeBusySummary:
    """Compute free/busy slots from a list of calendar events."""
    busy = [
        BusySlot(start=e.start, end=e.end, title=e.title, event_id=e.id)
        for e in sorted(events, key=lambda e: e.start)
        if not e.all_day
    ]

    # Compute free windows between busy slots
    free: list[FreeSlot] = []
    cursor = date_range.start
    for b in busy:
        if b.start > cursor:
            free.append(FreeSlot(start=cursor, end=b.start))
        cursor = max(cursor, b.end)
    if cursor < date_range.end:
        free.append(FreeSlot(start=cursor, end=date_range.end))

    return FreeBusySummary(member_name=member_name, date_range=date_range, busy=busy, free=free)


def find_free_window(
    summary: FreeBusySummary,
    duration_minutes: int,
) -> FreeSlot | None:
    """Find the first free window of at least *duration_minutes*."""
    needed = timedelta(minutes=duration_minutes)
    for slot in summary.free:
        if slot.end - slot.start >= needed:
            return FreeSlot(start=slot.start, end=slot.start + needed)
    return None


# ── Conflicts ───────────────────────────────────────────────────────────────


def compute_conflicts(
    adapter: object,
    store: ReminderStore,
    household_id: UUID,
    date_range: DateRange,
) -> ConflictReport:
    """Check all household members for calendar conflicts in a window."""
    members = store.get_household_members(household_id)
    member_conflicts: dict[str, list[BusySlot]] = {}

    for m in members:
        identity = resolve_identity(store, m)
        if identity is None:
            continue
        events = adapter.list_events(identity.id, date_range)
        if events:
            member_conflicts[m.name] = [
                BusySlot(start=e.start, end=e.end, title=e.title, event_id=e.id)
                for e in events
                if not e.all_day
            ]

    return ConflictReport(date_range=date_range, member_conflicts=member_conflicts)


# ── Reminder/event overlap ──────────────────────────────────────────────────


def find_reminder_overlaps(
    store: ReminderStore,
    household_id: UUID,
    events: list[CalendarEvent],
) -> list[ReminderOverlap]:
    """Find active reminders that overlap with calendar events."""
    active_states = [s for s in ReminderState if s not in TERMINAL_STATES]
    reminders, _ = store.list_reminders(household_id, states=active_states)

    overlaps: list[ReminderOverlap] = []
    for r in reminders:
        schedule = store.get_schedule(r.id)
        if not schedule or not schedule.next_fire_at:
            continue
        fire_start = schedule.next_fire_at
        fire_end = fire_start + timedelta(minutes=15)  # assume 15min reminder window

        matching = [
            e for e in events
            if e.start < fire_end and e.end > fire_start and not e.all_day
        ]
        if matching:
            overlaps.append(ReminderOverlap(reminder=r, overlapping_events=matching))

    return overlaps


# ── Context enrichment ──────────────────────────────────────────────────────


def enrich_reminder_context(
    store: ReminderStore,
    reminder_id: UUID,
    events: list[CalendarEvent],
) -> dict:
    """Add calendar context to a reminder explanation.

    Returns metadata about nearby events, conflicts, and timing.
    """
    reminder = store.get_reminder(reminder_id)
    if reminder is None:
        return {"error": "Reminder not found"}

    schedule = store.get_schedule(reminder_id)
    if not schedule or not schedule.next_fire_at:
        return {"note": "No schedule — calendar context not applicable"}

    fire = schedule.next_fire_at
    window = DateRange(start=fire - timedelta(hours=1), end=fire + timedelta(hours=1))

    nearby = [e for e in events if e.start < window.end and e.end > window.start]
    overlapping = [e for e in nearby if e.start < fire + timedelta(minutes=15) and e.end > fire]

    return {
        "fire_time": fire.isoformat(),
        "nearby_events": len(nearby),
        "overlapping_events": len(overlapping),
        "overlap_titles": [e.title for e in overlapping],
        "nearby_titles": [e.title for e in nearby],
        "note": (
            f"Reminder fires during: {', '.join(e.title for e in overlapping)}"
            if overlapping
            else "No calendar conflicts at fire time"
        ),
    }
