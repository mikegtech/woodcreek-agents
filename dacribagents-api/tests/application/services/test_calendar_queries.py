"""Tests for calendar query service — free/busy, conflicts, overlap, identity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from dacribagents.application.ports.calendar_adapter import CalendarEvent, DateRange
from dacribagents.application.services.calendar_queries import (
    compute_conflicts,
    compute_free_busy,
    enrich_reminder_context,
    find_free_window,
    find_reminder_overlaps,
    resolve_identity,
)
from dacribagents.application.use_cases.reminder_workflows import create_reminder, schedule_reminder
from dacribagents.domain.reminders.entities import CalendarIdentity, Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    CalendarProviderType,
    MemberRole,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.models import (
    CreateReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    ScheduleReminderRequest,
)
from dacribagents.infrastructure.calendar.mock_adapter import MockCalendarAdapter
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore

_TZ = timezone.utc


@pytest.fixture()
def household_id():
    return uuid4()


@pytest.fixture()
def member_id():
    return uuid4()


@pytest.fixture()
def identity_id():
    return uuid4()


@pytest.fixture()
def store(household_id, member_id, identity_id):
    s = InMemoryReminderStore()
    s.households[household_id] = Household(id=household_id, name="Woodcreek", created_at=datetime(2026, 1, 1))
    s.members[member_id] = HouseholdMember(
        id=member_id, household_id=household_id, name="Mike",
        role=MemberRole.ADMIN, timezone="America/Chicago",
        created_at=datetime(2026, 1, 1), email="mike@woodcreek.me",
    )
    s.calendar_identities[identity_id] = CalendarIdentity(
        id=identity_id, household_id=household_id, member_id=member_id,
        provider=CalendarProviderType.WORKMAIL_EWS,
        provider_account_id="mike@woodcreek.awsapps.com",
        display_name="Mike's WorkMail", active=True,
        created_at=datetime(2026, 1, 1),
    )
    return s


@pytest.fixture()
def calendar(identity_id):
    cal = MockCalendarAdapter()
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    cal.add_event(CalendarEvent(
        id="evt-1", title="Morning standup",
        start=tomorrow.replace(hour=9), end=tomorrow.replace(hour=9, minute=30),
        source=CalendarProviderType.WORKMAIL_EWS, owner_identity_id=identity_id,
    ))
    cal.add_event(CalendarEvent(
        id="evt-2", title="Lunch with client",
        start=tomorrow.replace(hour=12), end=tomorrow.replace(hour=13),
        source=CalendarProviderType.WORKMAIL_EWS, owner_identity_id=identity_id,
    ))
    cal.add_event(CalendarEvent(
        id="evt-3", title="Team retro",
        start=tomorrow.replace(hour=15), end=tomorrow.replace(hour=16),
        source=CalendarProviderType.WORKMAIL_EWS, owner_identity_id=identity_id,
    ))
    return cal


# ── Identity resolution ────────────────────────────────────────────────────


def test_resolve_identity(store, member_id):
    member = store.get_member(member_id)
    identity = resolve_identity(store, member)
    assert identity is not None
    assert identity.provider == CalendarProviderType.WORKMAIL_EWS


def test_resolve_identity_missing(store, household_id):
    # Create a member without a calendar identity
    no_cal = HouseholdMember(
        id=uuid4(), household_id=household_id, name="Kid",
        role=MemberRole.CHILD, timezone="America/Chicago",
        created_at=datetime(2026, 1, 1),
    )
    store.members[no_cal.id] = no_cal
    assert resolve_identity(store, no_cal) is None


# ── Free/busy ───────────────────────────────────────────────────────────────


def test_compute_free_busy(calendar, identity_id):
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    dr = DateRange(start=tomorrow.replace(hour=8), end=tomorrow.replace(hour=17))
    events = calendar.list_events(identity_id, dr)

    summary = compute_free_busy(events, dr, "Mike")
    assert summary.member_name == "Mike"
    assert len(summary.busy) == 3
    assert len(summary.free) >= 2  # 8-9, 9:30-12, 13-15, 16-17


def test_find_free_window_found(calendar, identity_id):
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    dr = DateRange(start=tomorrow.replace(hour=8), end=tomorrow.replace(hour=17))
    events = calendar.list_events(identity_id, dr)

    summary = compute_free_busy(events, dr, "Mike")
    slot = find_free_window(summary, 30)
    assert slot is not None
    assert slot.end - slot.start >= timedelta(minutes=30)


def test_find_free_window_not_found(calendar, identity_id):
    """Very short window search in a packed schedule."""
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    # Tight range: only during the standup
    dr = DateRange(start=tomorrow.replace(hour=9), end=tomorrow.replace(hour=9, minute=30))
    events = calendar.list_events(identity_id, dr)

    summary = compute_free_busy(events, dr, "Mike")
    slot = find_free_window(summary, 60)
    assert slot is None


# ── Conflicts ───────────────────────────────────────────────────────────────


def test_compute_conflicts(calendar, store, household_id):
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    dr = DateRange(start=tomorrow.replace(hour=8), end=tomorrow.replace(hour=17))

    report = compute_conflicts(calendar, store, household_id, dr)
    assert "Mike" in report.member_conflicts
    assert len(report.member_conflicts["Mike"]) == 3


def test_conflicts_empty_for_no_identities(store, household_id):
    """Members without calendar identities don't appear in conflict report."""
    kid = HouseholdMember(
        id=uuid4(), household_id=household_id, name="Kid",
        role=MemberRole.CHILD, timezone="America/Chicago",
        created_at=datetime(2026, 1, 1),
    )
    store.members[kid.id] = kid

    cal = MockCalendarAdapter()
    dr = DateRange(start=datetime(2026, 4, 2, tzinfo=_TZ), end=datetime(2026, 4, 3, tzinfo=_TZ))
    report = compute_conflicts(cal, store, household_id, dr)
    assert "Kid" not in report.member_conflicts


# ── Reminder/event overlap ──────────────────────────────────────────────────


def test_find_reminder_overlaps(store, calendar, household_id, member_id, identity_id):
    # Create a reminder scheduled at 9:15 — overlaps with morning standup
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Check email",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    schedule_reminder(
        store, r.id,
        ScheduleReminderRequest(schedule=ReminderScheduleInput(
            schedule_type=ScheduleType.ONE_SHOT,
            fire_at=tomorrow.replace(hour=9, minute=15),
        )),
    )

    dr = DateRange(start=tomorrow.replace(hour=8), end=tomorrow.replace(hour=17))
    events = calendar.list_events(identity_id, dr)
    overlaps = find_reminder_overlaps(store, household_id, events)

    assert len(overlaps) == 1
    assert overlaps[0].reminder.subject == "Check email"
    assert any("standup" in e.title.lower() for e in overlaps[0].overlapping_events)


def test_no_overlaps(store, calendar, household_id, member_id, identity_id):
    # Reminder at 10am — no events at that time
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Break time",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    schedule_reminder(
        store, r.id,
        ScheduleReminderRequest(schedule=ReminderScheduleInput(
            schedule_type=ScheduleType.ONE_SHOT,
            fire_at=tomorrow.replace(hour=10),
        )),
    )

    dr = DateRange(start=tomorrow.replace(hour=8), end=tomorrow.replace(hour=17))
    events = calendar.list_events(identity_id, dr)
    overlaps = find_reminder_overlaps(store, household_id, events)
    assert len(overlaps) == 0


# ── Context enrichment ──────────────────────────────────────────────────────


def test_enrich_reminder_context(store, calendar, household_id, member_id, identity_id):
    tomorrow = datetime(2026, 4, 2, tzinfo=_TZ)
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Check email",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    schedule_reminder(
        store, r.id,
        ScheduleReminderRequest(schedule=ReminderScheduleInput(
            schedule_type=ScheduleType.ONE_SHOT,
            fire_at=tomorrow.replace(hour=9, minute=15),
        )),
    )

    dr = DateRange(start=tomorrow.replace(hour=8), end=tomorrow.replace(hour=17))
    events = calendar.list_events(identity_id, dr)

    ctx = enrich_reminder_context(store, r.id, events)
    assert ctx["overlapping_events"] >= 1
    assert "standup" in str(ctx["overlap_titles"]).lower()


def test_enrich_no_schedule(store, household_id, member_id):
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Draft only",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    ctx = enrich_reminder_context(store, r.id, [])
    assert "not applicable" in ctx.get("note", "").lower()
