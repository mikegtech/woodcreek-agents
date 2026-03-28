"""Tests for Pydantic API contract models — validation, targeting, scheduling."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dacribagents.domain.reminders.enums import (
    AckMethod,
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.models import (
    AcknowledgeReminderRequest,
    CreateReminderRequest,
    EventIntakeRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    ScheduleReminderRequest,
    SnoozeReminderRequest,
    UpdateReminderRequest,
)


# ── ReminderTargetInput validation ──────────────────────────────────────────


def test_individual_target_requires_member_id():
    with pytest.raises(ValidationError, match="member_id is required"):
        ReminderTargetInput(target_type=TargetType.INDIVIDUAL)


def test_individual_target_with_member_id():
    t = ReminderTargetInput(target_type=TargetType.INDIVIDUAL, member_id=uuid4())
    assert t.member_id is not None


def test_role_target_requires_role():
    with pytest.raises(ValidationError, match="role is required"):
        ReminderTargetInput(target_type=TargetType.ROLE)


def test_role_target_with_role():
    t = ReminderTargetInput(target_type=TargetType.ROLE, role=MemberRole.ADMIN)
    assert t.role == MemberRole.ADMIN


def test_household_target_rejects_member_id():
    with pytest.raises(ValidationError, match="must be None"):
        ReminderTargetInput(target_type=TargetType.HOUSEHOLD, member_id=uuid4())


def test_household_target_clean():
    t = ReminderTargetInput(target_type=TargetType.HOUSEHOLD)
    assert t.member_id is None
    assert t.role is None


# ── ReminderScheduleInput validation ────────────────────────────────────────


def test_one_shot_requires_fire_at():
    with pytest.raises(ValidationError, match="fire_at is required"):
        ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT)


def test_one_shot_with_fire_at():
    s = ReminderScheduleInput(
        schedule_type=ScheduleType.ONE_SHOT,
        fire_at=datetime(2026, 4, 1, 9, 0),
    )
    assert s.fire_at is not None


def test_relative_requires_both_fields():
    with pytest.raises(ValidationError, match="relative_to and relative_offset_minutes"):
        ReminderScheduleInput(schedule_type=ScheduleType.RELATIVE, relative_to="event_123")


def test_relative_with_both_fields():
    s = ReminderScheduleInput(
        schedule_type=ScheduleType.RELATIVE,
        relative_to="hoa_dues",
        relative_offset_minutes=-4320,
    )
    assert s.relative_offset_minutes == -4320


def test_recurring_requires_cron():
    with pytest.raises(ValidationError, match="cron_expression is required"):
        ReminderScheduleInput(schedule_type=ScheduleType.RECURRING)


def test_recurring_with_cron():
    s = ReminderScheduleInput(
        schedule_type=ScheduleType.RECURRING,
        cron_expression="0 7 * * WED",
    )
    assert s.cron_expression == "0 7 * * WED"


# ── CreateReminderRequest ───────────────────────────────────────────────────


def test_create_requires_at_least_one_target():
    with pytest.raises(ValidationError, match="at least"):
        CreateReminderRequest(
            subject="Test",
            targets=[],
        )


def test_create_requires_subject():
    with pytest.raises(ValidationError):
        CreateReminderRequest(
            subject="",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        )


def test_create_agent_source_requires_source_agent():
    with pytest.raises(ValidationError, match="source_agent is required"):
        CreateReminderRequest(
            subject="Test",
            source=ReminderSource.AGENT,
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        )


def test_create_agent_source_with_agent():
    r = CreateReminderRequest(
        subject="HOA deadline approaching",
        source=ReminderSource.AGENT,
        source_agent="hoa_compliance",
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
    )
    assert r.source_agent == "hoa_compliance"


def test_create_user_source_no_agent_required():
    r = CreateReminderRequest(
        subject="Take out trash",
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
    )
    assert r.source == ReminderSource.USER
    assert r.source_agent is None


def test_create_with_schedule():
    r = CreateReminderRequest(
        subject="Mow lawn",
        targets=[ReminderTargetInput(target_type=TargetType.INDIVIDUAL, member_id=uuid4())],
        schedule=ReminderScheduleInput(
            schedule_type=ScheduleType.ONE_SHOT,
            fire_at=datetime(2026, 4, 5, 8, 0),
        ),
    )
    assert r.schedule is not None


def test_create_defaults():
    r = CreateReminderRequest(
        subject="Test",
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
    )
    assert r.urgency == UrgencyLevel.NORMAL
    assert r.source == ReminderSource.USER
    assert r.requires_approval is False


# ── UpdateReminderRequest ───────────────────────────────────────────────────


def test_update_all_none_is_valid():
    u = UpdateReminderRequest()
    assert u.subject is None
    assert u.urgency is None


def test_update_partial():
    u = UpdateReminderRequest(subject="Updated subject", urgency=UrgencyLevel.URGENT)
    assert u.subject == "Updated subject"
    assert u.body is None  # not changed


# ── AcknowledgeReminderRequest ──────────────────────────────────────────────


def test_ack_request():
    a = AcknowledgeReminderRequest(
        member_id=uuid4(),
        method=AckMethod.SLACK_BUTTON,
        note="Done",
    )
    assert a.note == "Done"


# ── SnoozeReminderRequest ──────────────────────────────────────────────────


def test_snooze_request():
    s = SnoozeReminderRequest(
        member_id=uuid4(),
        method=AckMethod.SMS_REPLY,
        snooze_until=datetime(2026, 3, 28, 15, 0),
    )
    assert s.snooze_until is not None


# ── EventIntakeRequest ──────────────────────────────────────────────────────


def test_event_intake_telemetry():
    r = EventIntakeRequest(
        source=ReminderSource.TELEMETRY,
        source_event_id="solar-monitor:inv3:temp:2026-03-28T14:00Z",
        dedupe_key="telemetry:inverter3:temp_warn",
        household_id=uuid4(),
        subject="Inverter temperature warning",
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        urgency=UrgencyLevel.URGENT,
        intent=NotificationIntent.ALERT,
    )
    assert r.source == ReminderSource.TELEMETRY
    assert r.intent == NotificationIntent.ALERT
    assert r.dedupe_key is not None


def test_event_intake_maintenance():
    r = EventIntakeRequest(
        source=ReminderSource.MAINTENANCE,
        source_event_id="maint:hvac-filter:2026Q2",
        household_id=uuid4(),
        subject="Replace HVAC filter",
        body="Quarterly HVAC filter replacement is due",
        targets=[ReminderTargetInput(target_type=TargetType.ROLE, role=MemberRole.ADMIN)],
    )
    assert r.source == ReminderSource.MAINTENANCE
    assert r.intent == NotificationIntent.ALERT  # default for event intake


def test_event_intake_requires_source_event_id():
    with pytest.raises(ValidationError, match="source_event_id"):
        EventIntakeRequest(
            source=ReminderSource.HOA,
            source_event_id="",
            household_id=uuid4(),
            subject="Test",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        )


def test_event_intake_warranty():
    r = EventIntakeRequest(
        source=ReminderSource.WARRANTY,
        source_event_id="warranty:homepro-vac:expiring-30d",
        dedupe_key="warranty:homepro-vac:expiring-30d",
        household_id=uuid4(),
        subject="HomePro central vac warranty expires in 30 days",
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        intent=NotificationIntent.REMINDER,
    )
    assert r.source == ReminderSource.WARRANTY
    assert r.intent == NotificationIntent.REMINDER


def test_create_request_intent_defaults_to_reminder():
    r = CreateReminderRequest(
        subject="Test",
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
    )
    assert r.intent == NotificationIntent.REMINDER


def test_create_request_with_event_fields():
    r = CreateReminderRequest(
        subject="HOA deadline approaching",
        source=ReminderSource.HOA,
        source_event_id="hoa:dues:2026Q2",
        dedupe_key="hoa:dues:2026Q2:reminder",
        intent=NotificationIntent.REMINDER,
        targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
    )
    assert r.source_event_id == "hoa:dues:2026Q2"
    assert r.dedupe_key == "hoa:dues:2026Q2:reminder"
