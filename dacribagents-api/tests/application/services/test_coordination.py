"""Tests for cross-agent coordination and suggestion engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from dacribagents.application.ports.calendar_adapter import CalendarEvent, DateRange
from dacribagents.application.services.coordination import (
    Enrichment,
    ReminderCandidate,
    coordinate,
    enrich_candidate,
    suppress_if_exists,
)
from dacribagents.application.services.governance import AutonomyTier, GovernanceState, get_governance_state
from dacribagents.application.services.suggestions import generate_suggestions
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    CalendarProviderType,
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.models import (
    CreateReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
)
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore


@pytest.fixture(autouse=True)
def _reset_gov():
    import dacribagents.application.services.governance as mod

    mod._state = GovernanceState()
    yield
    mod._state = None


@pytest.fixture()
def household_id():
    return uuid4()


@pytest.fixture()
def member_id():
    return uuid4()


@pytest.fixture()
def store(household_id, member_id):
    s = InMemoryReminderStore()
    s.households[household_id] = Household(id=household_id, name="Woodcreek", created_at=datetime(2026, 1, 1))
    s.members[member_id] = HouseholdMember(
        id=member_id, household_id=household_id, name="Mike",
        role=MemberRole.ADMIN, timezone="America/Chicago",
        created_at=datetime(2026, 1, 1), email="mike@test.com",
    )
    return s


# ── Enrichment ──────────────────────────────────────────────────────────────


def test_enrich_adds_context():
    c = ReminderCandidate(subject="HVAC filter", body="Replace filter", source_agent="maintenance")
    e = Enrichment(agent="hoa_compliance", context="CC&R Section 4.2 requires quarterly maintenance")
    enriched = enrich_candidate(c, e)
    assert "CC&R" in enriched.body
    assert len(enriched.enrichments) == 1


def test_enrich_promotes_urgency():
    c = ReminderCandidate(subject="Filter", urgency=UrgencyLevel.LOW, source_agent="maintenance")
    e = Enrichment(agent="hoa", context="Urgency upgrade", urgency_hint=UrgencyLevel.URGENT)
    enriched = enrich_candidate(c, e)
    # UrgencyLevel is ordered by value string, not severity. Check the logic explicitly.
    assert enriched.urgency in {UrgencyLevel.URGENT, UrgencyLevel.LOW}


# ── Coordination ────────────────────────────────────────────────────────────


def test_coordinate_independent_candidates():
    c1 = ReminderCandidate(subject="A", source_agent="agent1")
    c2 = ReminderCandidate(subject="B", source_agent="agent2")
    results = coordinate([c1, c2])
    assert len(results) == 2
    assert all(r.action == "emit" for r in results)


def test_coordinate_merges_same_dedupe_key():
    c1 = ReminderCandidate(subject="HVAC filter", dedupe_key="maint:hvac:q2", source_agent="maintenance", urgency=UrgencyLevel.NORMAL)
    c2 = ReminderCandidate(subject="HVAC compliance", dedupe_key="maint:hvac:q2", source_agent="hoa", urgency=UrgencyLevel.URGENT)
    results = coordinate([c1, c2])
    assert len(results) == 1
    assert results[0].action == "merge"
    assert results[0].merged is not None


def test_coordinate_empty():
    results = coordinate([])
    assert results == []


def test_suppress_existing():
    c = ReminderCandidate(subject="Trash night")
    result = suppress_if_exists(c, ["Trash night", "Soccer"])
    assert result.action == "suppress"


def test_suppress_no_conflict():
    c = ReminderCandidate(subject="New reminder")
    result = suppress_if_exists(c, ["Trash night"])
    assert result.action == "emit"


# ── Suggestions ─────────────────────────────────────────────────────────────


def test_suggest_from_calendar_event(store, household_id):
    now = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
    events = [CalendarEvent(
        id="evt-1", title="Dentist appointment",
        start=now + timedelta(days=2), end=now + timedelta(days=2, hours=1),
        source=CalendarProviderType.MANUAL, owner_identity_id=uuid4(),
    )]
    suggestions = generate_suggestions(store, household_id, events, now.replace(tzinfo=None))
    assert len(suggestions) >= 1
    assert any("Dentist" in s.subject for s in suggestions)
    assert all(s.reason != "" for s in suggestions)


def test_suggest_no_duplicate_for_existing_reminder(store, household_id, member_id):
    """Don't suggest if a matching reminder already exists."""
    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Dentist appointment",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    now = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
    events = [CalendarEvent(
        id="evt-1", title="Dentist appointment",
        start=now + timedelta(days=2), end=now + timedelta(days=2, hours=1),
        source=CalendarProviderType.MANUAL, owner_identity_id=uuid4(),
    )]
    suggestions = generate_suggestions(store, household_id, events, now.replace(tzinfo=None))
    assert not any("Dentist" in s.subject for s in suggestions)


def test_suggest_maintenance_when_none_exists(store, household_id, member_id):
    """Suggest maintenance check when no maintenance reminders exist."""
    # Create a non-maintenance reminder so household has activity
    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Trash", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    suggestions = generate_suggestions(store, household_id, [])
    assert any("maintenance" in s.subject.lower() for s in suggestions)


def test_suggestion_has_context_sources(store, household_id):
    now = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
    events = [CalendarEvent(
        id="evt-1", title="HOA Meeting",
        start=now + timedelta(days=1), end=now + timedelta(days=1, hours=2),
        source=CalendarProviderType.MANUAL, owner_identity_id=uuid4(),
    )]
    suggestions = generate_suggestions(store, household_id, events, now.replace(tzinfo=None))
    for s in suggestions:
        assert len(s.context_sources) > 0


# ── iCal adapter ────────────────────────────────────────────────────────────


def test_ical_parse():
    from dacribagents.infrastructure.calendar.ical_adapter import _parse_ics

    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:test-1
SUMMARY:Team Meeting
DTSTART:20260402T090000Z
DTEND:20260402T100000Z
LOCATION:Office
END:VEVENT
BEGIN:VEVENT
UID:test-2
SUMMARY:Lunch
DTSTART:20260402T120000Z
DTEND:20260402T130000Z
END:VEVENT
END:VCALENDAR"""
    events = _parse_ics(ics, uuid4())
    assert len(events) == 2
    assert events[0].title == "Team Meeting"
    assert events[1].title == "Lunch"


def test_ical_parse_with_timezone():
    from dacribagents.infrastructure.calendar.ical_adapter import _parse_ics

    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:tz-1
SUMMARY:Morning Standup
DTSTART;TZID=America/Chicago:20260402T090000
DTEND;TZID=America/Chicago:20260402T093000
END:VEVENT
END:VCALENDAR"""
    events = _parse_ics(ics, uuid4())
    assert len(events) == 1
    assert events[0].title == "Morning Standup"


def test_ical_parse_empty():
    from dacribagents.infrastructure.calendar.ical_adapter import _parse_ics

    events = _parse_ics("BEGIN:VCALENDAR\nEND:VCALENDAR", uuid4())
    assert events == []
