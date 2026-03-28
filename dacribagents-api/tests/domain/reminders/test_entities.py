"""Tests for reminder domain entities — targeting, schedules, immutability."""

from __future__ import annotations

from datetime import datetime, time
from uuid import uuid4

import pytest

from dacribagents.domain.reminders.entities import (
    CalendarAccessPolicy,
    CalendarIdentity,
    Household,
    HouseholdMember,
    MemberDevice,
    PreferenceRule,
    Reminder,
    ReminderAcknowledgement,
    ReminderDelivery,
    ReminderSchedule,
    ReminderTarget,
)
from dacribagents.domain.reminders.enums import (
    AckMethod,
    CalendarAccessLevel,
    CalendarProviderType,
    DeliveryChannel,
    DeliveryStatus,
    DevicePlatform,
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)


# ── Immutability ────────────────────────────────────────────────────────────


def test_household_is_frozen(household):
    with pytest.raises(AttributeError):
        household.name = "Changed"


def test_reminder_is_frozen(draft_reminder):
    with pytest.raises(AttributeError):
        draft_reminder.state = ReminderState.SCHEDULED


def test_member_is_frozen(admin_member):
    with pytest.raises(AttributeError):
        admin_member.phone = "+10000000000"


# ── Household member construction ───────────────────────────────────────────


def test_member_with_all_contact_info(admin_member):
    assert admin_member.email == "mike@woodcreek.me"
    assert admin_member.phone == "+15125551234"
    assert admin_member.slack_id == "U1234MIKE"
    assert admin_member.quiet_hours_start == time(22, 0)
    assert admin_member.quiet_hours_end == time(7, 0)


def test_member_with_minimal_info(child_member):
    assert child_member.email is None
    assert child_member.phone is None
    assert child_member.slack_id is None
    assert child_member.role == MemberRole.CHILD


# ── Targeting ───────────────────────────────────────────────────────────────


def test_household_target(household_target):
    assert household_target.target_type == TargetType.HOUSEHOLD
    assert household_target.member_id is None
    assert household_target.role is None


def test_individual_target(individual_target, admin_member):
    assert individual_target.target_type == TargetType.INDIVIDUAL
    assert individual_target.member_id == admin_member.id


def test_role_target():
    target = ReminderTarget(
        id=uuid4(),
        reminder_id=uuid4(),
        target_type=TargetType.ROLE,
        role=MemberRole.ADMIN,
    )
    assert target.role == MemberRole.ADMIN
    assert target.member_id is None


# ── Schedule types ──────────────────────────────────────────────────────────


def test_one_shot_schedule(one_shot_schedule):
    assert one_shot_schedule.schedule_type == ScheduleType.ONE_SHOT
    assert one_shot_schedule.fire_at is not None
    assert one_shot_schedule.cron_expression is None


def test_recurring_schedule():
    s = ReminderSchedule(
        id=uuid4(),
        reminder_id=uuid4(),
        schedule_type=ScheduleType.RECURRING,
        timezone="America/Chicago",
        cron_expression="0 7 * * WED",
    )
    assert s.cron_expression == "0 7 * * WED"
    assert s.fire_at is None


def test_relative_schedule():
    s = ReminderSchedule(
        id=uuid4(),
        reminder_id=uuid4(),
        schedule_type=ScheduleType.RELATIVE,
        timezone="America/Chicago",
        relative_to="hoa_dues_deadline_2026Q2",
        relative_offset_minutes=-4320,  # 3 days before
    )
    assert s.relative_offset_minutes == -4320
    assert s.relative_to is not None


# ── Delivery and acknowledgement ────────────────────────────────────────────


def test_delivery_entity():
    d = ReminderDelivery(
        id=uuid4(),
        execution_id=uuid4(),
        member_id=uuid4(),
        channel=DeliveryChannel.SMS,
        status=DeliveryStatus.QUEUED,
        created_at=datetime.utcnow(),
    )
    assert d.escalation_step == 0
    assert d.provider_message_id is None


def test_delivery_with_escalation():
    d = ReminderDelivery(
        id=uuid4(),
        execution_id=uuid4(),
        member_id=uuid4(),
        channel=DeliveryChannel.EMAIL,
        status=DeliveryStatus.SENT,
        created_at=datetime.utcnow(),
        escalation_step=2,
        provider_message_id="<abc@woodcreek.me>",
        sent_at=datetime.utcnow(),
    )
    assert d.escalation_step == 2
    assert d.provider_message_id is not None


def test_acknowledgement():
    ack = ReminderAcknowledgement(
        id=uuid4(),
        delivery_id=uuid4(),
        member_id=uuid4(),
        method=AckMethod.SLACK_BUTTON,
        acknowledged_at=datetime.utcnow(),
        note="Got it",
    )
    assert ack.snoozed_until is None
    assert ack.note == "Got it"


def test_snooze_acknowledgement():
    snooze_time = datetime(2026, 3, 28, 14, 0)
    ack = ReminderAcknowledgement(
        id=uuid4(),
        delivery_id=uuid4(),
        member_id=uuid4(),
        method=AckMethod.SMS_REPLY,
        acknowledged_at=datetime.utcnow(),
        snoozed_until=snooze_time,
    )
    assert ack.snoozed_until == snooze_time


# ── Event-source fields ─────────────────────────────────────────────────────


def test_reminder_defaults_intent_to_reminder(draft_reminder):
    assert draft_reminder.intent == NotificationIntent.REMINDER
    assert draft_reminder.source_event_id is None
    assert draft_reminder.dedupe_key is None


def test_reminder_with_event_fields():
    r = Reminder(
        id=uuid4(),
        household_id=uuid4(),
        subject="Inverter temperature warning",
        body="Panel zone 3 inverter at 85°C",
        urgency=UrgencyLevel.URGENT,
        intent=NotificationIntent.ALERT,
        source=ReminderSource.TELEMETRY,
        source_event_id="solar-monitor:inv3:temp:2026-03-28T14:00Z",
        dedupe_key="telemetry:inverter3:temp_warn",
        state=ReminderState.DRAFT,
        created_by=uuid4(),
        created_at=datetime(2026, 3, 28, 14, 0),
        updated_at=datetime(2026, 3, 28, 14, 0),
    )
    assert r.intent == NotificationIntent.ALERT
    assert r.source == ReminderSource.TELEMETRY
    assert r.source_event_id == "solar-monitor:inv3:temp:2026-03-28T14:00Z"
    assert r.dedupe_key == "telemetry:inverter3:temp_warn"


def test_domain_specific_sources_exist():
    """Confirm all expected upstream producer sources are defined."""
    expected = {"maintenance", "hoa", "warranty", "telemetry", "household_routine"}
    actual = {s.value for s in ReminderSource}
    assert expected.issubset(actual)


def test_notification_intent_values():
    assert {i.value for i in NotificationIntent} == {"reminder", "alert", "digest"}


# ── Preference rules ────────────────────────────────────────────────────────


def test_household_wide_preference():
    rule = PreferenceRule(
        id=uuid4(),
        household_id=uuid4(),
        preferred_channel=DeliveryChannel.SMS,
        active=True,
        created_at=datetime.utcnow(),
        urgency=UrgencyLevel.URGENT,
    )
    assert rule.member_id is None
    assert rule.urgency == UrgencyLevel.URGENT


def test_member_specific_preference():
    member_id = uuid4()
    rule = PreferenceRule(
        id=uuid4(),
        household_id=uuid4(),
        preferred_channel=DeliveryChannel.EMAIL,
        active=True,
        created_at=datetime.utcnow(),
        member_id=member_id,
        fallback_channel=DeliveryChannel.SMS,
    )
    assert rule.member_id == member_id
    assert rule.fallback_channel == DeliveryChannel.SMS


# ── Calendar entities ───────────────────────────────────────────────────────


def test_calendar_identity():
    ci = CalendarIdentity(
        id=uuid4(),
        household_id=uuid4(),
        provider=CalendarProviderType.WORKMAIL_EWS,
        provider_account_id="arn:aws:workmail:us-east-1:123:org/woodcreek",
        display_name="Woodcreek WorkMail",
        active=True,
        created_at=datetime.utcnow(),
    )
    assert ci.provider == CalendarProviderType.WORKMAIL_EWS
    assert ci.member_id is None  # household-level


def test_calendar_access_policy_defaults_read_only():
    policy = CalendarAccessPolicy(
        id=uuid4(),
        calendar_identity_id=uuid4(),
        access_level=CalendarAccessLevel.READ_ONLY,
        sync_frequency_minutes=15,
        created_at=datetime.utcnow(),
    )
    assert policy.write_back_enabled is False
    assert policy.visible_calendars == ()


# ── Member device (future push) ────────────────────────────────────────────


def test_member_device():
    device = MemberDevice(
        id=uuid4(),
        member_id=uuid4(),
        platform=DevicePlatform.IOS,
        device_token="abc123apns",
        active=True,
        registered_at=datetime.utcnow(),
    )
    assert device.platform == DevicePlatform.IOS
    assert device.display_name is None
