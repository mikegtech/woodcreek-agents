"""Tests for digest aggregation, eligibility, scheduling, and delivery."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.ports.event_publisher import DomainEvent
from dacribagents.application.services.digest import (
    DigestBatch,
    collect_eligible,
    generate_and_deliver,
    is_digest_eligible,
)
from dacribagents.application.use_cases.reminder_queries import list_digest_eligible
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.entities import Household, HouseholdMember, Reminder
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


class StubEmailAdapter:
    channel = DeliveryChannel.EMAIL

    def __init__(self, succeed: bool = True):
        self.calls: list[dict] = []
        self._succeed = succeed

    def send(self, **kwargs) -> DeliveryResult:
        self.calls.append(kwargs)
        if self._succeed:
            return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id=f"digest-{uuid4().hex[:8]}")
        return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason="SMTP error")


class CollectingPublisher:
    def __init__(self):
        self.events: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


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


def _create_pending(store, hid, mid, urgency=UrgencyLevel.LOW, intent=NotificationIntent.REMINDER):
    r = create_reminder(
        store, hid, mid,
        CreateReminderRequest(
            subject=f"Test {urgency.value} {intent.value}",
            urgency=urgency, intent=intent,
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=datetime(2026, 4, 1, tzinfo=UTC)),
        ),
    )
    store.update_reminder_state(r.id, ReminderState.PENDING_DELIVERY)
    return store.get_reminder(r.id)


# ── Eligibility ─────────────────────────────────────────────────────────────


def test_digest_intent_eligible(store, household_id, member_id):
    r = _create_pending(store, household_id, member_id, UrgencyLevel.NORMAL, NotificationIntent.DIGEST)
    assert is_digest_eligible(r) is True


def test_low_urgency_eligible(store, household_id, member_id):
    r = _create_pending(store, household_id, member_id, UrgencyLevel.LOW, NotificationIntent.REMINDER)
    assert is_digest_eligible(r) is True


def test_normal_urgency_not_eligible(store, household_id, member_id):
    r = _create_pending(store, household_id, member_id, UrgencyLevel.NORMAL, NotificationIntent.REMINDER)
    assert is_digest_eligible(r) is False


def test_urgent_not_eligible(store, household_id, member_id):
    r = _create_pending(store, household_id, member_id, UrgencyLevel.URGENT, NotificationIntent.ALERT)
    assert is_digest_eligible(r) is False


def test_critical_not_eligible(store, household_id, member_id):
    r = _create_pending(store, household_id, member_id, UrgencyLevel.CRITICAL, NotificationIntent.ALERT)
    assert is_digest_eligible(r) is False


def test_low_alert_not_eligible(store, household_id, member_id):
    """LOW urgency alerts are NOT eligible — alerts should not be digested."""
    r = _create_pending(store, household_id, member_id, UrgencyLevel.LOW, NotificationIntent.ALERT)
    assert is_digest_eligible(r) is False


# ── Collection ──────────────────────────────────────────────────────────────


def test_collect_eligible(store, household_id, member_id):
    _create_pending(store, household_id, member_id, UrgencyLevel.LOW)
    _create_pending(store, household_id, member_id, UrgencyLevel.NORMAL, NotificationIntent.DIGEST)
    _create_pending(store, household_id, member_id, UrgencyLevel.URGENT)  # not eligible

    eligible = collect_eligible(store, household_id)
    assert len(eligible) == 2


def test_collect_excludes_non_pending(store, household_id, member_id):
    """Only PENDING_DELIVERY reminders are collected."""
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Draft only", urgency=UrgencyLevel.LOW,
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    # DRAFT state — not pending delivery
    eligible = collect_eligible(store, household_id)
    assert len(eligible) == 0


# ── Digest generation and delivery ──────────────────────────────────────────


def test_generate_delivers_email(store, household_id, member_id):
    _create_pending(store, household_id, member_id, UrgencyLevel.LOW)
    _create_pending(store, household_id, member_id, UrgencyLevel.NORMAL, NotificationIntent.DIGEST)

    adapter = StubEmailAdapter()
    batch = generate_and_deliver(store, household_id, adapter)

    assert batch is not None
    assert batch.delivered is True
    assert len(batch.reminder_ids) == 2
    assert len(adapter.calls) == 1  # one email to Mike
    assert "Digest" in adapter.calls[0]["subject"]


def test_generate_marks_delivered(store, household_id, member_id):
    r1 = _create_pending(store, household_id, member_id, UrgencyLevel.LOW)
    r2 = _create_pending(store, household_id, member_id, UrgencyLevel.NORMAL, NotificationIntent.DIGEST)

    adapter = StubEmailAdapter()
    generate_and_deliver(store, household_id, adapter)

    assert store.get_reminder(r1.id).state == ReminderState.DELIVERED
    assert store.get_reminder(r2.id).state == ReminderState.DELIVERED


def test_generate_no_eligible(store, household_id):
    adapter = StubEmailAdapter()
    batch = generate_and_deliver(store, household_id, adapter)
    assert batch is None


def test_generate_no_emails(store, household_id, member_id):
    _create_pending(store, household_id, member_id, UrgencyLevel.LOW)
    # Remove email from member
    from dataclasses import replace
    m = store.members[member_id]
    store.members[member_id] = replace(m, email=None)

    adapter = StubEmailAdapter()
    batch = generate_and_deliver(store, household_id, adapter)
    assert batch is None


def test_generate_failed_delivery(store, household_id, member_id):
    _create_pending(store, household_id, member_id, UrgencyLevel.LOW)

    adapter = StubEmailAdapter(succeed=False)
    batch = generate_and_deliver(store, household_id, adapter)

    assert batch is not None
    assert batch.delivered is False
    assert batch.delivery_error is not None


def test_generate_publishes_event(store, household_id, member_id):
    _create_pending(store, household_id, member_id, UrgencyLevel.LOW)
    adapter = StubEmailAdapter()
    publisher = CollectingPublisher()

    generate_and_deliver(store, household_id, adapter, events=publisher)

    assert len(publisher.events) == 1
    assert publisher.events[0].event_type == "digest.delivered"


def test_no_duplicate_digesting(store, household_id, member_id):
    """Reminders included in a digest are marked DELIVERED and not re-collected."""
    _create_pending(store, household_id, member_id, UrgencyLevel.LOW)
    adapter = StubEmailAdapter()

    batch1 = generate_and_deliver(store, household_id, adapter)
    assert batch1 is not None

    batch2 = generate_and_deliver(store, household_id, adapter)
    assert batch2 is None  # no more eligible reminders


# ── Query visibility ────────────────────────────────────────────────────────


def test_list_digest_eligible_query(store, household_id, member_id):
    _create_pending(store, household_id, member_id, UrgencyLevel.LOW)
    _create_pending(store, household_id, member_id, UrgencyLevel.URGENT)

    eligible = list_digest_eligible(store, household_id)
    assert len(eligible) == 1
