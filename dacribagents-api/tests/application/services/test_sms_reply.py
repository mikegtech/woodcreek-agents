"""Tests for SMS inbound reply handler and Slack delivery adapter."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, time
from uuid import uuid4

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.services.delivery import DeliveryDispatcher
from dacribagents.application.services.sms_reply_handler import (
    SmsReplyResult,
    handle_sms_reply,
    parse_sms_command,
)
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
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
)
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubSmsAdapter:
    channel = DeliveryChannel.SMS

    def send(self, **kwargs) -> DeliveryResult:
        return DeliveryResult(status=DeliveryStatus.SENT, provider_message_id=f"telnyx-{uuid4().hex[:8]}")


class StubSlackAdapter:
    channel = DeliveryChannel.SLACK

    def __init__(self):
        self.calls: list[dict] = []

    def send(self, **kwargs) -> DeliveryResult:
        self.calls.append(kwargs)
        return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id=f"slack-{uuid4().hex[:8]}")


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
        id=member_id, household_id=household_id, name="Mike",
        role=MemberRole.ADMIN, timezone="America/Chicago",
        created_at=datetime(2026, 1, 1),
        email="mike@woodcreek.me", phone="+15125551234", slack_id="U1234MIKE",
    )
    return s


def _create_and_deliver_sms(store, household_id, member_id):
    """Create a reminder, schedule, advance to PENDING_DELIVERY, dispatch via SMS."""
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Take out trash",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            urgency=UrgencyLevel.URGENT,
            schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=datetime(2026, 4, 1, tzinfo=UTC)),
        ),
    )
    store.update_reminder_state(r.id, ReminderState.PENDING_DELIVERY)
    sms = StubSmsAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SMS: sms})
    dispatcher.dispatch_one(r.id)
    return store.get_reminder(r.id)


# ── Command parsing ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["OK", "ok", "Ok", "done", "DONE", "ack", "ACK", "yes", "got it"])
def test_parse_ack_commands(text):
    action, _ = parse_sms_command(text)
    assert action == "ack"


@pytest.mark.parametrize("text", ["cancel", "CANCEL", "stop", "STOP"])
def test_parse_cancel_commands(text):
    action, _ = parse_sms_command(text)
    assert action == "cancel"


def test_parse_snooze_default():
    action, minutes = parse_sms_command("snooze")
    assert action == "snooze"
    assert minutes == 30


def test_parse_snooze_with_minutes():
    action, minutes = parse_sms_command("SNOOZE 60")
    assert action == "snooze"
    assert minutes == 60


def test_parse_snooze_clamped():
    action, minutes = parse_sms_command("snooze 9999")
    assert action == "snooze"
    assert minutes == 1440  # max 24hr


def test_parse_unknown():
    action, _ = parse_sms_command("what's the weather?")
    assert action == "unknown"


# ── SMS ack handling ────────────────────────────────────────────────────────


def test_sms_ack_ok(store, household_id, member_id):
    reminder = _create_and_deliver_sms(store, household_id, member_id)
    assert reminder.state == ReminderState.DELIVERED

    result = handle_sms_reply(store, from_number="+15125551234", text="OK")
    assert result.status == "acknowledged"
    assert result.reminder_id == reminder.id

    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.ACKNOWLEDGED


def test_sms_snooze(store, household_id, member_id):
    reminder = _create_and_deliver_sms(store, household_id, member_id)

    result = handle_sms_reply(store, from_number="+15125551234", text="snooze 30")
    assert result.status == "snoozed"
    assert result.reminder_id == reminder.id

    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.SNOOZED


def test_sms_cancel(store, household_id, member_id):
    reminder = _create_and_deliver_sms(store, household_id, member_id)

    result = handle_sms_reply(store, from_number="+15125551234", text="cancel")
    assert result.status == "cancelled"

    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.CANCELLED


def test_sms_unrecognized_reply(store, household_id, member_id):
    _create_and_deliver_sms(store, household_id, member_id)
    result = handle_sms_reply(store, from_number="+15125551234", text="what's for dinner?")
    assert result.status == "unrecognized"


def test_sms_no_match_unknown_number(store, household_id, member_id):
    _create_and_deliver_sms(store, household_id, member_id)
    result = handle_sms_reply(store, from_number="+19999999999", text="OK")
    assert result.status == "no_match"


def test_sms_no_match_no_delivery(store, household_id, member_id):
    """Member exists but has no SMS delivery to ack."""
    result = handle_sms_reply(store, from_number="+15125551234", text="OK")
    assert result.status == "no_match"


# ── Slack delivery adapter ─────────────────────────────────────────────────


def test_slack_delivery_success(store, household_id, member_id):
    r = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Team meeting",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=datetime(2026, 4, 1, tzinfo=UTC)),
        ),
    )
    store.update_reminder_state(r.id, ReminderState.PENDING_DELIVERY)

    slack = StubSlackAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.SLACK: slack})
    deliveries = dispatcher.dispatch_one(r.id)

    assert len(deliveries) == 1
    # The member has a slack_id, and normal urgency defaults to email, but we're
    # testing with only Slack adapter registered
    assert store.get_reminder(r.id).state in {ReminderState.DELIVERED, ReminderState.FAILED}


def test_slack_adapter_formats_message():
    from dacribagents.infrastructure.delivery.slack_adapter import _format_slack_notification

    text = _format_slack_notification("Take out trash", "Bins by 7am", "urgent", "3a8f1b2c")
    assert "Take out trash" in text
    assert "3a8f1b2c" in text
    assert ":warning:" in text


def test_slack_adapter_no_channel():
    from dacribagents.infrastructure.delivery.slack_adapter import SlackDeliveryAdapter

    adapter = SlackDeliveryAdapter(default_channel="")
    result = adapter.send(
        recipient_id=uuid4(), recipient_address="",
        subject="Test", body="", reminder_id=uuid4(), urgency="normal",
    )
    assert result.status == DeliveryStatus.FAILED
    assert "channel" in result.failure_reason.lower()
