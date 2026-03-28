"""Tests for read-only reminder intelligence queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from dacribagents.application.ports.calendar_adapter import CalendarEvent, DateRange
from dacribagents.application.use_cases import reminder_queries as rq
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.enums import (
    CalendarProviderType,
    DeliveryChannel,
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


@pytest.fixture()
def store():
    return InMemoryReminderStore()


@pytest.fixture()
def household_id():
    return uuid4()


@pytest.fixture()
def member_id():
    return uuid4()


def _target() -> ReminderTargetInput:
    return ReminderTargetInput(target_type=TargetType.HOUSEHOLD)


def _seed(store, household_id, member_id):
    # User reminder — auto-scheduled
    create_reminder(
        store,
        household_id,
        member_id,
        CreateReminderRequest(
            subject="Trash day",
            targets=[_target()],
            schedule=ReminderScheduleInput(
                schedule_type=ScheduleType.ONE_SHOT,
                fire_at=datetime(2026, 4, 2, 6, 30),
            ),
        ),
    )
    # HOA reminder — needs approval
    create_reminder(
        store,
        household_id,
        member_id,
        CreateReminderRequest(
            subject="HOA dues",
            source=ReminderSource.HOA,
            source_event_id="hoa:dues:2026Q2",
            targets=[_target()],
        ),
    )
    # Telemetry alert — needs approval
    create_reminder(
        store,
        household_id,
        member_id,
        CreateReminderRequest(
            subject="Inverter temp",
            source=ReminderSource.TELEMETRY,
            source_event_id="solar:inv3",
            intent=NotificationIntent.ALERT,
            urgency=UrgencyLevel.URGENT,
            targets=[_target()],
        ),
    )


# ── list_pending_approval ───────────────────────────────────────────────────


def test_pending_approval_returns_approval_required(store, household_id, member_id):
    _seed(store, household_id, member_id)
    pending = rq.list_pending_approval(store, household_id)
    assert len(pending) == 2
    assert all(r.requires_approval for r in pending)
    subjects = {r.subject for r in pending}
    assert "HOA dues" in subjects
    assert "Inverter temp" in subjects


def test_pending_approval_empty(store, household_id):
    assert rq.list_pending_approval(store, household_id) == []


# ── list_active_alerts ──────────────────────────────────────────────────────


def test_active_alerts(store, household_id, member_id):
    _seed(store, household_id, member_id)
    alerts = rq.list_active_alerts(store, household_id)
    assert len(alerts) == 1
    assert alerts[0].subject == "Inverter temp"
    assert alerts[0].intent == NotificationIntent.ALERT


# ── list_active ─────────────────────────────────────────────────────────────


def test_list_active(store, household_id, member_id):
    _seed(store, household_id, member_id)
    active = rq.list_active(store, household_id)
    assert len(active) == 3


# ── explain_reminder ────────────────────────────────────────────────────────


def test_explain_reminder(store, household_id, member_id):
    _seed(store, household_id, member_id)
    pending = rq.list_pending_approval(store, household_id)
    expl = rq.explain_reminder(store, pending[0].id)
    assert expl is not None
    assert expl.approval_reason is not None
    assert expl.source_description != ""


def test_explain_nonexistent(store):
    assert rq.explain_reminder(store, uuid4()) is None


# ── preview_draft ───────────────────────────────────────────────────────────


def test_preview_user_draft():
    preview = rq.preview_draft(subject="Soccer at 5pm", target_type=TargetType.HOUSEHOLD)
    assert preview.requires_approval is False
    assert preview.target_description == "Targets the entire household"
    assert DeliveryChannel.SLACK in preview.likely_channels


def test_preview_agent_draft():
    preview = rq.preview_draft(
        subject="HVAC filter replacement",
        source=ReminderSource.MAINTENANCE,
        urgency=UrgencyLevel.URGENT,
    )
    assert preview.requires_approval is True
    assert preview.approval_reason is not None
    assert DeliveryChannel.SMS in preview.likely_channels


def test_preview_critical_note():
    preview = rq.preview_draft(subject="Emergency", urgency=UrgencyLevel.CRITICAL)
    assert any("quiet hours" in n.lower() for n in preview.notes)


def test_preview_empty_body_note():
    preview = rq.preview_draft(subject="No body")
    assert any("body" in n.lower() for n in preview.notes)


def test_preview_digest():
    preview = rq.preview_draft(subject="Weekly summary", intent=NotificationIntent.DIGEST)
    assert DeliveryChannel.EMAIL in preview.likely_channels
    assert any("digest" in n.lower() for n in preview.notes)


# ── build_schedule_summary ──────────────────────────────────────────────────


def test_schedule_summary(store, household_id, member_id):
    _seed(store, household_id, member_id)
    tz = timezone(timedelta(hours=-5))
    now = datetime.now(tz=tz)
    dr = DateRange(start=now, end=now + timedelta(days=1))
    events = [
        CalendarEvent(
            id="e1",
            title="Soccer",
            start=now + timedelta(hours=2),
            end=now + timedelta(hours=3),
            source=CalendarProviderType.MANUAL,
            owner_identity_id=uuid4(),
        )
    ]
    summary = rq.build_schedule_summary(store, household_id, dr, events)
    assert len(summary.calendar_events) == 1
    assert len(summary.reminders) == 3
