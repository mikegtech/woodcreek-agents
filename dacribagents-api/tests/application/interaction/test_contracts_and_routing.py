"""Tests for interaction contracts, validation, routing, and schedule parsing."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from uuid import uuid4

import pytest

from dacribagents.application.interaction.contracts import IntentType, ParsedIntent, TargetScope
from dacribagents.application.interaction.router import RouteResult, route
from dacribagents.application.interaction.validator import parse_schedule_to_datetime, validate
from dacribagents.application.services.governance import AutonomyTier, GovernanceState, get_governance_state
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import MemberRole, ReminderState, ScheduleType, TargetType
from dacribagents.domain.reminders.models import CreateReminderRequest, ReminderTargetInput
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
        created_at=datetime(2026, 1, 1), email="mike@test.com", phone="+15125551234",
    )
    return s


# ── Schedule parsing ────────────────────────────────────────────────────────


def test_parse_at_5pm():
    dt = parse_schedule_to_datetime("at 5pm")
    assert dt is not None
    assert dt.hour == 17


def test_parse_at_5():
    """Bare 'at 5' should default to PM for small numbers."""
    dt = parse_schedule_to_datetime("at 5")
    assert dt is not None
    assert dt.hour == 17


def test_parse_at_3_30pm():
    dt = parse_schedule_to_datetime("at 3:30pm")
    assert dt is not None
    assert dt.hour == 15
    assert dt.minute == 30


def test_parse_tomorrow_morning():
    dt = parse_schedule_to_datetime("tomorrow morning")
    assert dt is not None
    assert dt.hour == 8


def test_parse_tomorrow_at_5pm():
    dt = parse_schedule_to_datetime("tomorrow at 5pm")
    assert dt is not None
    assert dt.hour == 17


def test_parse_in_30_minutes():
    before = datetime.now(tz=timezone(timedelta(hours=-5)))
    dt = parse_schedule_to_datetime("in 30 minutes")
    assert dt is not None
    assert dt > before


def test_parse_empty():
    assert parse_schedule_to_datetime("") is None
    assert parse_schedule_to_datetime("no time here") is None


# ── Validation ──────────────────────────────────────────────────────────────


def test_validate_create_missing_subject(store, household_id):
    intent = ParsedIntent(intent=IntentType.CREATE_REMINDER, subject="")
    result = validate(intent, store, household_id)
    assert result.needs_clarification is True
    assert "about" in result.clarification_question.lower()


def test_validate_create_unknown_member(store, household_id):
    intent = ParsedIntent(
        intent=IntentType.CREATE_REMINDER, subject="groceries",
        target_scope=TargetScope.INDIVIDUAL, target_member_name="Nonexistent",
    )
    result = validate(intent, store, household_id)
    assert result.needs_clarification is True
    assert "Mike" in result.clarification_question  # suggests available members


def test_validate_create_valid(store, household_id):
    intent = ParsedIntent(
        intent=IntentType.CREATE_REMINDER, subject="pick up groceries",
        target_scope=TargetScope.INDIVIDUAL, target_member_name="Mike",
    )
    result = validate(intent, store, household_id)
    assert result.needs_clarification is False


def test_validate_action_missing_id(store, household_id):
    intent = ParsedIntent(intent=IntentType.CANCEL_REMINDER, reminder_short_id="")
    result = validate(intent, store, household_id)
    assert result.needs_clarification is True
    assert "short ID" in result.clarification_question


# ── Routing ─────────────────────────────────────────────────────────────────


def test_route_create_reminder(store, household_id, member_id):
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)
    intent = ParsedIntent(
        intent=IntentType.CREATE_REMINDER,
        subject="pick up groceries",
        target_scope=TargetScope.INDIVIDUAL,
        target_member_name="Mike",
        schedule_text="at 5pm",
    )
    result = route(intent, store, household_id, member_id)
    assert result.success is True
    assert "created" in result.message.lower()
    assert "pick up groceries" in result.message


def test_route_create_household(store, household_id, member_id):
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)
    intent = ParsedIntent(
        intent=IntentType.CREATE_REMINDER,
        subject="trash night",
        target_scope=TargetScope.HOUSEHOLD,
        schedule_text="at 8pm",
    )
    result = route(intent, store, household_id, member_id)
    assert result.success is True
    assert "trash night" in result.message


def test_route_reminder_query_active(store, household_id, member_id):
    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(subject="Test", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)]),
    )
    intent = ParsedIntent(intent=IntentType.REMINDER_QUERY, query_subject="active reminders")
    result = route(intent, store, household_id, member_id)
    assert result.success is True
    assert "Test" in result.message


def test_route_governance_status(store, household_id, member_id):
    intent = ParsedIntent(intent=IntentType.GOVERNANCE_QUERY, query_subject="status")
    result = route(intent, store, household_id, member_id)
    assert result.success is True
    assert "Tier" in result.message


def test_route_unknown_intent(store, household_id, member_id):
    intent = ParsedIntent(intent=IntentType.UNKNOWN)
    result = route(intent, store, household_id, member_id)
    assert result.success is False


def test_route_clarification_passes_through(store, household_id, member_id):
    intent = ParsedIntent(
        intent=IntentType.CREATE_REMINDER,
        needs_clarification=True,
        clarification_question="What should the reminder be about?",
    )
    result = route(intent, store, household_id, member_id)
    assert result.success is False
    assert "about" in result.message.lower()
