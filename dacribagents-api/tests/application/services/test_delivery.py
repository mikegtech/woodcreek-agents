"""Tests for the delivery dispatcher, channel policy, and acknowledgment."""

from __future__ import annotations

from datetime import UTC, datetime, time
from uuid import UUID, uuid4

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.ports.event_publisher import DomainEvent, NoOpEventPublisher
from dacribagents.application.services.acknowledgment import acknowledge_delivery
from dacribagents.application.services.channel_policy import select_channel
from dacribagents.application.services.delivery import DeliveryDispatcher
from dacribagents.application.use_cases.reminder_workflows import create_reminder, schedule_reminder
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    AckMethod,
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
    ScheduleReminderRequest,
)
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore


# ── Stub adapter ────────────────────────────────────────────────────────────


class StubSmsAdapter:
    """Test adapter that always succeeds."""

    channel = DeliveryChannel.SMS

    def __init__(self, succeed: bool = True) -> None:
        self.calls: list[dict] = []
        self._succeed = succeed

    def send(self, **kwargs) -> DeliveryResult:
        self.calls.append(kwargs)
        if self._succeed:
            return DeliveryResult(status=DeliveryStatus.SENT, provider_message_id=f"telnyx-{uuid4().hex[:8]}")
        return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason="Simulated failure")


class StubEmailAdapter:
    """Test adapter that always succeeds."""

    channel = DeliveryChannel.EMAIL

    def __init__(self, succeed: bool = True) -> None:
        self.calls: list[dict] = []
        self._succeed = succeed

    def send(self, **kwargs) -> DeliveryResult:
        self.calls.append(kwargs)
        if self._succeed:
            return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id=f"<test-{uuid4().hex[:8]}>")
        return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason="Simulated failure")


class CollectingPublisher:
    """Collects published events for assertions."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


# ── Fixtures ────────────────────────────────────────────────────────────────


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
        id=member_id,
        household_id=household_id,
        name="Mike",
        role=MemberRole.ADMIN,
        timezone="America/Chicago",
        created_at=datetime(2026, 1, 1),
        email="mike@woodcreek.me",
        phone="+15125551234",
        slack_id="U1234MIKE",
    )
    return s


def _create_pending(store, household_id, member_id, urgency=UrgencyLevel.NORMAL, intent=NotificationIntent.REMINDER):
    """Create a reminder and advance to PENDING_DELIVERY."""
    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Test reminder",
            body="Test body",
            urgency=urgency,
            intent=intent,
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(
                schedule_type=ScheduleType.ONE_SHOT,
                fire_at=datetime(2026, 4, 2, 6, 30, tzinfo=UTC),
            ),
        ),
    )
    # Advance to PENDING_DELIVERY
    store.update_reminder_state(reminder.id, ReminderState.PENDING_DELIVERY)
    return store.get_reminder(reminder.id)


# ── Channel selection policy ────────────────────────────────────────────────


def test_urgent_prefers_sms():
    sel = select_channel(
        member_id=uuid4(), phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.URGENT, intent=NotificationIntent.ALERT,
    )
    assert sel.primary_channel == DeliveryChannel.SMS
    assert sel.fallback_channel == DeliveryChannel.EMAIL


def test_normal_prefers_email():
    sel = select_channel(
        member_id=uuid4(), phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.NORMAL, intent=NotificationIntent.REMINDER,
    )
    assert sel.primary_channel == DeliveryChannel.EMAIL
    assert sel.fallback_channel == DeliveryChannel.SMS


def test_digest_always_email():
    sel = select_channel(
        member_id=uuid4(), phone="+1555", email="a@b.com", slack_id=None,
        urgency=UrgencyLevel.LOW, intent=NotificationIntent.DIGEST,
    )
    assert sel.primary_channel == DeliveryChannel.EMAIL
    assert sel.fallback_channel is None


def test_no_contact_suppressed():
    sel = select_channel(
        member_id=uuid4(), phone=None, email=None, slack_id=None,
        urgency=UrgencyLevel.NORMAL, intent=NotificationIntent.REMINDER,
    )
    assert sel.suppressed is True


def test_phone_only_urgent():
    sel = select_channel(
        member_id=uuid4(), phone="+1555", email=None, slack_id=None,
        urgency=UrgencyLevel.URGENT, intent=NotificationIntent.ALERT,
    )
    assert sel.primary_channel == DeliveryChannel.SMS
    assert sel.fallback_channel is None


# ── Delivery dispatcher ────────────────────────────────────────────────────


def test_dispatch_sms_success(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id, urgency=UrgencyLevel.URGENT)
    sms = StubSmsAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: sms})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1
    assert deliveries[0].status in {DeliveryStatus.SENT, DeliveryStatus.DELIVERED}
    assert sms.calls[0]["recipient_address"] == "+15125551234"

    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.DELIVERED


def test_dispatch_email_success(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id)
    email = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: email})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1
    assert deliveries[0].status == DeliveryStatus.DELIVERED
    assert email.calls[0]["recipient_address"] == "mike@woodcreek.me"


def test_dispatch_failure_transitions_to_failed(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id, urgency=UrgencyLevel.URGENT)
    sms = StubSmsAdapter(succeed=False)
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: sms})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1
    assert deliveries[0].status == DeliveryStatus.FAILED

    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.FAILED


def test_dispatch_pending_batch(store, household_id, member_id):
    _create_pending(store, household_id, member_id)
    _create_pending(store, household_id, member_id)
    email = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: email})

    deliveries = dispatcher.dispatch_pending(household_id)
    assert len(deliveries) == 2


def test_dispatch_no_adapter_fails(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id)
    dispatcher = DeliveryDispatcher(store=store, adapters={})  # no adapters

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1
    assert deliveries[0].status == DeliveryStatus.FAILED
    assert "No adapter" in (deliveries[0].failure_reason or "")


def test_dispatch_ignores_non_pending(store, household_id, member_id):
    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(subject="Draft only", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)]),
    )
    email = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: email})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 0


def test_dispatch_publishes_events(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id)
    email = StubEmailAdapter()
    publisher = CollectingPublisher()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: email}, events=publisher)

    dispatcher.dispatch_one(reminder.id)
    assert len(publisher.events) == 1
    assert publisher.events[0].event_type == "reminder.delivered"


def test_dispatch_failure_publishes_event(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id, urgency=UrgencyLevel.URGENT)
    sms = StubSmsAdapter(succeed=False)
    publisher = CollectingPublisher()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: sms}, events=publisher)

    dispatcher.dispatch_one(reminder.id)
    assert len(publisher.events) == 1
    assert publisher.events[0].event_type == "reminder.delivery_failed"


# ── Acknowledgment ──────────────────────────────────────────────────────────


def test_acknowledge_delivered_reminder(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id)
    email = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: email})
    dispatcher.dispatch_one(reminder.id)

    ack = acknowledge_delivery(
        store, reminder_id=reminder.id, member_id=member_id,
        method=AckMethod.SMS_REPLY, note="Got it",
    )
    assert ack is not None
    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.ACKNOWLEDGED


def test_acknowledge_publishes_event(store, household_id, member_id):
    reminder = _create_pending(store, household_id, member_id)
    email = StubEmailAdapter()
    DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: email}).dispatch_one(reminder.id)

    publisher = CollectingPublisher()
    acknowledge_delivery(
        store, reminder_id=reminder.id, member_id=member_id,
        method=AckMethod.SLACK_BUTTON, events=publisher,
    )
    assert len(publisher.events) == 1
    assert publisher.events[0].event_type == "reminder.acknowledged"


def test_acknowledge_non_delivered_no_transition(store, household_id, member_id):
    """Ack a DRAFT reminder — records the ack but doesn't transition."""
    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(subject="Draft", targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)]),
    )
    ack = acknowledge_delivery(
        store, reminder_id=reminder.id, member_id=member_id, method=AckMethod.DASHBOARD,
    )
    assert ack is not None
    assert store.get_reminder(reminder.id).state == ReminderState.DRAFT  # unchanged
