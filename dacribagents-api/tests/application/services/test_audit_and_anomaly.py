"""Tests for unified audit timeline, anomaly review, and health endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dacribagents.application.services.anomaly_review import check_anomalies
from dacribagents.application.services.audit_timeline import (
    get_household_timeline,
    get_reminder_timeline,
)
from dacribagents.application.services.governance import (
    AutonomyTier,
    GovernanceState,
    activate_kill_switch,
    deactivate_kill_switch,
    evaluate,
    get_governance_state,
)
from dacribagents.application.use_cases.reminder_workflows import create_reminder, submit_for_approval
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    MemberRole,
    ReminderSource,
    ScheduleType,
    TargetType,
)
from dacribagents.domain.reminders.models import (
    CreateReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    SubmitForApprovalRequest,
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
        created_at=datetime(2026, 1, 1), email="mike@woodcreek.me",
    )
    return s


# ── Reminder timeline ───────────────────────────────────────────────────────


def test_reminder_timeline_creation(store, household_id, member_id):
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Trash day",
            source=ReminderSource.HOA, source_event_id="hoa:test",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    timeline = get_reminder_timeline(store, r.id)
    assert len(timeline) >= 1
    assert timeline[0].event_type == "reminder.created"
    assert "Trash day" in timeline[0].summary


def test_reminder_timeline_with_approval(store, household_id, member_id):
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="HOA dues",
            source=ReminderSource.HOA, source_event_id="hoa:dues",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    submit_for_approval(store, r.id, SubmitForApprovalRequest(actor_id=member_id))

    timeline = get_reminder_timeline(store, r.id)
    types = [e.event_type for e in timeline]
    assert "reminder.created" in types
    assert "approval.submitted" in types


def test_reminder_timeline_nonexistent(store):
    timeline = get_reminder_timeline(store, uuid4())
    assert timeline == []


# ── Household timeline ──────────────────────────────────────────────────────


def test_household_timeline(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)
    evaluate(household_id=household_id, action_type="auto_escalate", requires_tier=AutonomyTier.TIER_3)

    timeline = get_household_timeline(None, household_id)  # store not needed for governance-only
    assert len(timeline) == 2


# ── Anomaly detection ───────────────────────────────────────────────────────


def test_no_anomalies_by_default(household_id):
    flags = check_anomalies(household_id)
    assert flags == []


def test_budget_exhaustion_anomaly(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_3)
    state.daily_budget = 5

    for _ in range(5):
        evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)

    flags = check_anomalies(household_id)
    rules = [f.rule for f in flags]
    assert "budget_exhaustion" in rules


def test_kill_switch_churn_anomaly(household_id):
    for _ in range(3):
        activate_kill_switch("test")
        deactivate_kill_switch("test")

    flags = check_anomalies(household_id)
    rules = [f.rule for f in flags]
    assert "kill_switch_churn" in rules


def test_blocked_spike_anomaly(household_id):
    state = get_governance_state()
    # Tier 0 — everything is blocked
    for _ in range(12):
        evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)

    flags = check_anomalies(household_id)
    # approval_required, not "blocked" — let's check
    # At Tier 0, decisions are "approval_required", not "blocked"
    # blocked_spike only triggers for "blocked" decisions (kill switch, budget)
    # So we need actual blocked decisions
    state.set_tier(household_id, AutonomyTier.TIER_3)
    activate_kill_switch("test")
    for _ in range(12):
        evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)

    flags = check_anomalies(household_id)
    rules = [f.rule for f in flags]
    assert "blocked_spike" in rules


def test_escalation_concentration_anomaly(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_3)

    for _ in range(4):
        evaluate(household_id=household_id, action_type="auto_escalate", requires_tier=AutonomyTier.TIER_3)

    flags = check_anomalies(household_id)
    rules = [f.rule for f in flags]
    assert "escalation_concentration" in rules


# ── Health endpoint (unit test) ─────────────────────────────────────────────


def test_governance_health_check():
    from dacribagents.infrastructure.http.health import _check_governance  # noqa: PLC0415

    result = _check_governance()
    assert result["status"] == "healthy"
    assert result["kill_switch_active"] is False
