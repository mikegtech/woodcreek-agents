"""Tests for Slack command handler — routing, parsing, and responses."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from dacribagents.application.ports.calendar_adapter import CalendarEvent, DateRange
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.enums import (
    CalendarProviderType,
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
from dacribagents.infrastructure.calendar.mock_adapter import MockCalendarAdapter
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore
from dacribagents.infrastructure.slack.handler import SlackCommandHandler, _parse_draft_intent, _strip_mention


@pytest.fixture()
def household_id():
    return uuid4()


@pytest.fixture()
def member_id():
    return uuid4()


@pytest.fixture()
def store():
    return InMemoryReminderStore()


@pytest.fixture()
def calendar():
    return MockCalendarAdapter()


@pytest.fixture()
def handler(store, calendar, household_id):
    return SlackCommandHandler(store=store, calendar=calendar, household_id=household_id)


def _target() -> ReminderTargetInput:
    return ReminderTargetInput(target_type=TargetType.HOUSEHOLD)


def _seed_reminders(store, household_id, member_id):
    """Seed a few reminders for testing queries."""
    # User reminder (no approval needed)
    create_reminder(
        store,
        household_id,
        member_id,
        CreateReminderRequest(
            subject="Take out trash",
            targets=[_target()],
            schedule=ReminderScheduleInput(
                schedule_type=ScheduleType.ONE_SHOT,
                fire_at=datetime(2026, 4, 2, 6, 30),
            ),
        ),
    )

    # Agent reminder (needs approval)
    create_reminder(
        store,
        household_id,
        member_id,
        CreateReminderRequest(
            subject="HOA dues deadline",
            source=ReminderSource.HOA,
            source_event_id="hoa:dues:2026Q2",
            intent=NotificationIntent.REMINDER,
            targets=[_target()],
        ),
    )

    # Alert from telemetry
    create_reminder(
        store,
        household_id,
        member_id,
        CreateReminderRequest(
            subject="Inverter temp warning",
            source=ReminderSource.TELEMETRY,
            source_event_id="solar:inv3:temp",
            intent=NotificationIntent.ALERT,
            urgency=UrgencyLevel.URGENT,
            targets=[_target()],
        ),
    )


# ── Mention stripping ──────────────────────────────────────────────────────


def test_strip_mention():
    assert _strip_mention("<@U123ABC> hello world") == "hello world"


def test_strip_mention_no_mention():
    assert _strip_mention("hello world") == "hello world"


# ── Draft intent parsing ───────────────────────────────────────────────────


def test_parse_draft_soccer():
    result = _parse_draft_intent("<@U123> draft a reminder for the family about soccer at 5")
    assert result["subject"] == "soccer"
    assert result["household"] is True
    assert result["schedule"] is not None
    assert "5" in result["schedule"]


def test_parse_draft_simple():
    result = _parse_draft_intent("draft a reminder about dentist appointment")
    assert result["subject"] == "dentist appointment"


# ── Command routing ─────────────────────────────────────────────────────────


def test_pending_approval(handler, store, household_id, member_id):
    _seed_reminders(store, household_id, member_id)
    resp = handler.handle("<@U123> show reminders waiting for approval")
    assert "approval" in resp.text.lower()
    assert "HOA dues deadline" in resp.text


def test_active_alerts(handler, store, household_id, member_id):
    _seed_reminders(store, household_id, member_id)
    resp = handler.handle("<@U123> list active alerts")
    assert "Alert" in resp.text or "Inverter" in resp.text


def test_active_reminders(handler, store, household_id, member_id):
    _seed_reminders(store, household_id, member_id)
    resp = handler.handle("<@U123> what reminders are pending?")
    assert "Active Reminders" in resp.text or "items" in resp.text


def test_schedule_query_tomorrow(handler, store, household_id, member_id, calendar):
    _seed_reminders(store, household_id, member_id)
    # Add a calendar event for tomorrow
    tz = timezone(timedelta(hours=-5))
    tomorrow = datetime.now(tz=tz).replace(hour=0, minute=0, second=0) + timedelta(days=1)
    calendar.add_event(
        CalendarEvent(
            id="evt-1",
            title="Soccer practice",
            start=tomorrow.replace(hour=17),
            end=tomorrow.replace(hour=18, minute=30),
            source=CalendarProviderType.MANUAL,
            owner_identity_id=uuid4(),
        )
    )
    resp = handler.handle("<@U123> what does tomorrow look like?")
    assert "Soccer practice" in resp.text or "Schedule" in resp.text


def test_conflicts_query(handler, calendar):
    tz = timezone(timedelta(hours=-5))
    now = datetime.now(tz=tz)
    # Use Thursday of next week
    days_until_thu = (3 - now.weekday()) % 7 or 7
    thursday = now.replace(hour=0, minute=0, second=0) + timedelta(days=days_until_thu)
    calendar.add_event(
        CalendarEvent(
            id="evt-2",
            title="Band rehearsal",
            start=thursday.replace(hour=19),
            end=thursday.replace(hour=21),
            source=CalendarProviderType.MANUAL,
            owner_identity_id=uuid4(),
        )
    )
    resp = handler.handle("<@U123> who has conflicts Thursday night?")
    assert "Band rehearsal" in resp.text or "Thursday" in resp.text


def test_draft_preview(handler):
    resp = handler.handle("<@U123> draft a reminder for the family about soccer at 5")
    assert "Draft Preview" in resp.text
    assert "soccer" in resp.text.lower()
    assert "Targets the entire household" in resp.text


def test_explain_approval(handler, store, household_id, member_id):
    _seed_reminders(store, household_id, member_id)
    resp = handler.handle("<@U123> explain why this reminder needs approval")
    assert "approval" in resp.text.lower() or "Approval" in resp.text


def test_unknown_command(handler):
    resp = handler.handle("<@U123> something completely unrelated")
    assert "help" in resp.text.lower() or "Try:" in resp.text


def test_empty_store_approval(handler):
    resp = handler.handle("<@U123> show reminders waiting for approval")
    assert "clear" in resp.text.lower() or "nothing" in resp.text.lower()


def test_empty_store_alerts(handler):
    resp = handler.handle("<@U123> list active alerts")
    assert "No items" in resp.text
