"""Tests for escalation, retry, preference rules, quiet hours, and control loop."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from uuid import uuid4

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.ports.event_publisher import DomainEvent
from dacribagents.application.services.channel_policy import select_channel, _in_quiet_hours
from dacribagents.application.services.control_loop import run_cycle
from dacribagents.application.services.delivery import DeliveryDispatcher
from dacribagents.application.services.escalation import (
    MAX_RETRIES,
    check_escalations,
    check_retries,
    compute_backoff,
    is_retriable,
)
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.entities import (
    Household,
    HouseholdMember,
    PreferenceRule,
    ReminderDelivery,
)
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


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubAdapter:
    def __init__(self, ch: DeliveryChannel, succeed: bool = True, failure_reason: str = "Request timeout"):
        self.channel = ch
        self.calls: list[dict] = []
        self._succeed = succeed
        self._failure_reason = failure_reason

    def send(self, **kwargs) -> DeliveryResult:
        self.calls.append(kwargs)
        if self._succeed:
            return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id=f"id-{len(self.calls)}")
        return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason=self._failure_reason)


class CollectingPublisher:
    def __init__(self):
        self.events: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _setup_governance(household_id):
    """Reset governance and set Tier 3 so escalation tests work."""
    from dacribagents.application.services.governance import GovernanceState, AutonomyTier, get_governance_state
    import dacribagents.application.services.governance as gov_mod

    gov_mod._state = GovernanceState()
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_3)
    yield
    gov_mod._state = None


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
        quiet_hours_start=time(22, 0), quiet_hours_end=time(7, 0),
    )
    return s


def _create_pending(store, household_id, member_id, urgency=UrgencyLevel.NORMAL):
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Test", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            urgency=urgency,
            schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=datetime(2026, 4, 1, tzinfo=UTC)),
        ),
    )
    store.update_reminder_state(r.id, ReminderState.PENDING_DELIVERY)
    return store.get_reminder(r.id)


# ── Backoff computation ─────────────────────────────────────────────────────


def test_backoff_exponential():
    d0 = compute_backoff(0)
    d1 = compute_backoff(1)
    d2 = compute_backoff(2)
    assert d1 == d0 * 2
    assert d2 == d0 * 4


def test_backoff_capped():
    d10 = compute_backoff(10)
    assert d10 == timedelta(minutes=30)


# ── Retriable detection ─────────────────────────────────────────────────────


def test_retriable_timeout():
    assert is_retriable("Request timeout") is True


def test_retriable_rate_limit():
    assert is_retriable("HTTP 429 rate limit exceeded") is True


def test_non_retriable():
    assert is_retriable("SMTP authentication failed") is False
    assert is_retriable(None) is False


# ── Quiet hours ─────────────────────────────────────────────────────────────


def test_quiet_hours_active():
    assert _in_quiet_hours(time(22, 0), time(7, 0), time(23, 0)) is True
    assert _in_quiet_hours(time(22, 0), time(7, 0), time(3, 0)) is True


def test_quiet_hours_inactive():
    assert _in_quiet_hours(time(22, 0), time(7, 0), time(12, 0)) is False


def test_quiet_hours_none():
    assert _in_quiet_hours(None, None, time(23, 0)) is False


def test_quiet_hours_suppresses_normal(member_id):
    sel = select_channel(
        member_id=member_id, phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.NORMAL, intent=NotificationIntent.REMINDER,
        quiet_hours_start=time(22, 0), quiet_hours_end=time(7, 0),
        current_time=time(23, 0),
    )
    assert sel.suppressed is True
    assert "quiet hours" in sel.suppression_reason.lower()


def test_quiet_hours_critical_bypasses(member_id):
    sel = select_channel(
        member_id=member_id, phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.CRITICAL, intent=NotificationIntent.ALERT,
        quiet_hours_start=time(22, 0), quiet_hours_end=time(7, 0),
        current_time=time(23, 0),
    )
    assert sel.suppressed is False


# ── PreferenceRule integration ──────────────────────────────────────────────


def test_preference_rule_overrides_default(member_id, household_id):
    """Member prefers SMS for normal urgency (default would be email)."""
    rule = PreferenceRule(
        id=uuid4(), household_id=household_id, member_id=member_id,
        preferred_channel=DeliveryChannel.SMS, fallback_channel=DeliveryChannel.EMAIL,
        active=True, created_at=datetime(2026, 1, 1), urgency=UrgencyLevel.NORMAL,
    )
    sel = select_channel(
        member_id=member_id, phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.NORMAL, intent=NotificationIntent.REMINDER,
        preference_rules=[rule],
    )
    assert sel.primary_channel == DeliveryChannel.SMS
    assert sel.fallback_channel == DeliveryChannel.EMAIL


def test_household_default_rule(member_id, household_id):
    """Household-wide default (member_id=None)."""
    rule = PreferenceRule(
        id=uuid4(), household_id=household_id, member_id=None,
        preferred_channel=DeliveryChannel.SMS, active=True,
        created_at=datetime(2026, 1, 1),
    )
    sel = select_channel(
        member_id=member_id, phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.NORMAL, intent=NotificationIntent.REMINDER,
        preference_rules=[rule],
    )
    assert sel.primary_channel == DeliveryChannel.SMS


def test_preference_quiet_hours_override(member_id, household_id):
    """Rule with quiet_hours_override=True bypasses quiet hours."""
    rule = PreferenceRule(
        id=uuid4(), household_id=household_id, member_id=member_id,
        preferred_channel=DeliveryChannel.SMS, active=True,
        created_at=datetime(2026, 1, 1), quiet_hours_override=True,
    )
    sel = select_channel(
        member_id=member_id, phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.NORMAL, intent=NotificationIntent.REMINDER,
        preference_rules=[rule],
        quiet_hours_start=time(22, 0), quiet_hours_end=time(7, 0),
        current_time=time(23, 0),
    )
    assert sel.suppressed is False
    assert sel.primary_channel == DeliveryChannel.SMS


# ── Retry behavior ──────────────────────────────────────────────────────────


def test_retry_transient_failure(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id, UrgencyLevel.URGENT)
    fail_adapter = StubAdapter(DeliveryChannel.SMS, succeed=False)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: fail_adapter})
    dispatcher.dispatch_one(reminder.id)  # creates FAILED delivery

    # Now retry with a succeeding adapter
    ok_adapter = StubAdapter(DeliveryChannel.SMS, succeed=True)
    far_future = datetime(2099, 1, 1, tzinfo=UTC)
    retried = check_retries(store, {DeliveryChannel.SMS: ok_adapter}, household_id, far_future)
    assert len(retried) == 1
    assert retried[0].status == DeliveryStatus.DELIVERED


def test_retry_respects_backoff(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id, UrgencyLevel.URGENT)
    fail_adapter = StubAdapter(DeliveryChannel.SMS, succeed=False)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: fail_adapter})
    dispatcher.dispatch_one(reminder.id)

    # Immediately after failure — backoff not elapsed
    ok_adapter = StubAdapter(DeliveryChannel.SMS, succeed=True)
    now = datetime.now(UTC)
    retried = check_retries(store, {DeliveryChannel.SMS: ok_adapter}, household_id, now)
    assert len(retried) == 0  # too soon


def test_non_retriable_not_retried(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id)
    # Create a delivery that failed with a non-retriable reason
    from dacribagents.application.services.delivery import DeliveryDispatcher

    class AuthFailAdapter:
        channel = DeliveryChannel.EMAIL
        def send(self, **kw):
            return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason="SMTP authentication failed")

    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: AuthFailAdapter()})
    dispatcher.dispatch_one(reminder.id)

    ok = StubAdapter(DeliveryChannel.EMAIL, succeed=True)
    retried = check_retries(store, {DeliveryChannel.EMAIL: ok}, household_id, datetime(2099, 1, 1, tzinfo=UTC))
    assert len(retried) == 0


# ── Escalation behavior ────────────────────────────────────────────────────


def test_escalation_after_timeout(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id, UrgencyLevel.URGENT)
    ok = StubAdapter(DeliveryChannel.SMS, succeed=True)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: ok})
    dispatcher.dispatch_one(reminder.id)

    # Simulate time passing beyond ack timeout
    r = store.get_reminder(reminder.id)
    store.reminders[r.id] = replace(r, updated_at=datetime(2020, 1, 1, tzinfo=UTC))

    email = StubAdapter(DeliveryChannel.EMAIL, succeed=True)
    publisher = CollectingPublisher()
    escalated = check_escalations(
        store, {DeliveryChannel.SMS: ok, DeliveryChannel.EMAIL: email},
        household_id, datetime(2026, 4, 1, tzinfo=UTC), publisher,
    )
    assert len(escalated) >= 1
    assert any(e.event_type == "reminder.escalation_triggered" for e in publisher.events)


def test_no_escalation_before_timeout(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id)
    ok = StubAdapter(DeliveryChannel.EMAIL, succeed=True)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: ok})
    dispatcher.dispatch_one(reminder.id)

    # Check immediately — timeout not reached
    escalated = check_escalations(store, {DeliveryChannel.EMAIL: ok}, household_id)
    assert len(escalated) == 0


# ── Control loop ────────────────────────────────────────────────────────────


def test_control_loop_full_cycle(store, household_id, member_id):
    # Create a scheduled reminder
    create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Loop test", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=datetime(2026, 4, 1, tzinfo=UTC)),
        ),
    )

    ok = StubAdapter(DeliveryChannel.EMAIL, succeed=True)
    result = run_cycle(
        store, {DeliveryChannel.EMAIL: ok}, household_id,
        now=datetime(2026, 4, 1, 0, 1, tzinfo=UTC),
    )
    assert result.scheduled == 1
    assert result.dispatched == 1


def test_control_loop_empty_store(store, household_id):
    result = run_cycle(store, {}, household_id)
    assert result.scheduled == 0
    assert result.dispatched == 0
    assert result.retried == 0
    assert result.escalated == 0
