"""Tests for governance integration — dispatch gating, Tier 2/3, mute, review."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, time
from uuid import uuid4

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.services.delivery import DeliveryDispatcher
from dacribagents.application.services.escalation import check_escalations
from dacribagents.application.services.governance import (
    AutonomyTier,
    GovernanceState,
    generate_review_summary,
    get_governance_state,
    is_member_muted,
    is_tier2_auto_eligible,
    list_muted_members,
    mute_member,
    unmute_member,
)
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ReminderState,
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


class StubAdapter:
    def __init__(self, ch: DeliveryChannel, succeed: bool = True):
        self.channel = ch
        self.calls: list[dict] = []
        self._succeed = succeed

    def send(self, **kwargs) -> DeliveryResult:
        self.calls.append(kwargs)
        if self._succeed:
            return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id=f"id-{len(self.calls)}")
        return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason="Request timeout")


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
        created_at=datetime(2026, 1, 1),
        email="mike@woodcreek.me", phone="+15125551234",
    )
    return s


def _create_pending(store, hid, mid, source=ReminderSource.USER, urgency=UrgencyLevel.NORMAL):
    r = create_reminder(
        store, hid, mid,
        CreateReminderRequest(
            subject="Test", source=source, urgency=urgency,
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=datetime(2026, 4, 1, tzinfo=UTC)),
        ),
    )
    store.update_reminder_state(r.id, ReminderState.PENDING_DELIVERY)
    return store.get_reminder(r.id)


# ── Tier 2: auto-send eligibility ──────────────────────────────────────────


def test_household_routine_is_tier2_eligible():
    assert is_tier2_auto_eligible(ReminderSource.HOUSEHOLD_ROUTINE) is True


def test_user_source_not_tier2_eligible():
    assert is_tier2_auto_eligible(ReminderSource.USER) is False


def test_agent_source_not_tier2_eligible():
    assert is_tier2_auto_eligible(ReminderSource.AGENT) is False


# ── Governance gating in dispatch ───────────────────────────────────────────


def test_tier0_blocks_autonomous_dispatch(store, household_id, member_id):
    """At Tier 0, dispatch is blocked by governance (requires approval)."""
    reminder = _create_pending(store, household_id, member_id)
    adapter = StubAdapter(DeliveryChannel.EMAIL)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    # At Tier 0, governance blocks — no deliveries
    assert len(deliveries) == 0
    assert len(adapter.calls) == 0


def test_tier1_allows_approved_dispatch(store, household_id, member_id):
    """At Tier 1, dispatch proceeds (TIER_1 >= required TIER_1 for USER source)."""
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_1)

    reminder = _create_pending(store, household_id, member_id)
    adapter = StubAdapter(DeliveryChannel.EMAIL)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1


def test_tier2_auto_sends_household_routine(store, household_id, member_id):
    """Tier 2 allows auto-send for HOUSEHOLD_ROUTINE source."""
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    reminder = _create_pending(store, household_id, member_id, source=ReminderSource.HOUSEHOLD_ROUTINE)
    adapter = StubAdapter(DeliveryChannel.EMAIL)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1
    assert adapter.calls[0]["subject"] == "Test"


def test_tier2_blocks_non_routine_auto_send(store, household_id, member_id):
    """Tier 2 still requires Tier 1 for non-routine sources — allowed since Tier 2 >= Tier 1."""
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    reminder = _create_pending(store, household_id, member_id, source=ReminderSource.HOA)
    adapter = StubAdapter(DeliveryChannel.EMAIL)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    # HOA requires Tier 1, household is at Tier 2, so this is allowed
    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1


# ── Governance gating in escalation ─────────────────────────────────────────


def test_tier2_blocks_autonomous_escalation(store, household_id, member_id):
    """Escalation requires Tier 3 — blocked at Tier 2."""
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    reminder = _create_pending(store, household_id, member_id, urgency=UrgencyLevel.URGENT)
    adapter = StubAdapter(DeliveryChannel.SMS)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: adapter})
    dispatcher.dispatch_one(reminder.id)

    # Age the delivery to trigger escalation
    r = store.get_reminder(reminder.id)
    store.reminders[r.id] = replace(r, updated_at=datetime(2020, 1, 1, tzinfo=UTC))

    email = StubAdapter(DeliveryChannel.EMAIL)
    escalated = check_escalations(
        store, {DeliveryChannel.SMS: adapter, DeliveryChannel.EMAIL: email},
        household_id, datetime(2026, 4, 1, tzinfo=UTC),
    )
    assert len(escalated) == 0  # blocked by governance (Tier 2 < required Tier 3)


def test_tier3_allows_autonomous_escalation(store, household_id, member_id):
    """Escalation proceeds at Tier 3."""
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_3)

    reminder = _create_pending(store, household_id, member_id, urgency=UrgencyLevel.URGENT)
    adapter = StubAdapter(DeliveryChannel.SMS)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: adapter})
    dispatcher.dispatch_one(reminder.id)

    r = store.get_reminder(reminder.id)
    store.reminders[r.id] = replace(r, updated_at=datetime(2020, 1, 1, tzinfo=UTC))

    email = StubAdapter(DeliveryChannel.EMAIL)
    escalated = check_escalations(
        store, {DeliveryChannel.SMS: adapter, DeliveryChannel.EMAIL: email},
        household_id, datetime(2026, 4, 1, tzinfo=UTC),
    )
    assert len(escalated) >= 1


# ── Member mute / opt-out ──────────────────────────────────────────────────


def test_mute_member(member_id):
    mute_member(member_id, "Testing")
    assert is_member_muted(member_id) is True


def test_unmute_member(member_id):
    mute_member(member_id, "Testing")
    unmute_member(member_id)
    assert is_member_muted(member_id) is False


def test_muted_member_skipped_in_delivery(store, household_id, member_id):
    """Muted member is skipped for non-critical delivery."""
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_1)
    mute_member(member_id, "Muted by test")

    reminder = _create_pending(store, household_id, member_id)
    adapter = StubAdapter(DeliveryChannel.EMAIL)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(adapter.calls) == 0  # skipped due to mute


def test_muted_member_receives_critical(store, household_id, member_id):
    """Critical reminders bypass mute."""
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_1)
    mute_member(member_id, "Muted by test")

    reminder = _create_pending(store, household_id, member_id, urgency=UrgencyLevel.CRITICAL)
    adapter = StubAdapter(DeliveryChannel.SMS)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(adapter.calls) == 1  # critical bypasses mute


def test_list_muted_members(member_id):
    mute_member(member_id, "Testing")
    assert member_id in list_muted_members()


# ── Governance review summary ───────────────────────────────────────────────


def test_governance_review(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    from dacribagents.application.services.governance import evaluate

    evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)
    evaluate(household_id=household_id, action_type="auto_escalate", requires_tier=AutonomyTier.TIER_3)

    review = generate_review_summary(household_id)
    assert review["allowed"] == 1
    assert review["approval_required"] == 1
    assert review["total_entries"] == 2
    assert "auto_send" in review["action_type_counts"]
