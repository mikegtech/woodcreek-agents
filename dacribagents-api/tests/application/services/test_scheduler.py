"""Tests for the reminder scheduling engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from dacribagents.application.services.scheduler import compute_next_fire, tick
from dacribagents.application.use_cases.reminder_workflows import (
    create_reminder,
    schedule_reminder,
)
from dacribagents.domain.reminders.enums import (
    ReminderState,
    ScheduleType,
    TargetType,
)
from dacribagents.domain.reminders.models import (
    CreateReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    ScheduleReminderRequest,
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


def _target():
    return ReminderTargetInput(target_type=TargetType.HOUSEHOLD)


# ── One-shot scheduling ────────────────────────────────────────────────────


def test_one_shot_fires_at_time(store, household_id, member_id):
    fire_at = datetime(2026, 4, 2, 6, 30, tzinfo=UTC)
    request = CreateReminderRequest(
        subject="Trash day",
        targets=[_target()],
        schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=fire_at),
    )
    reminder = create_reminder(store, household_id, member_id, request)
    assert reminder.state == ReminderState.SCHEDULED

    # Tick before fire time — nothing happens
    execs = tick(store, fire_at - timedelta(minutes=1))
    assert len(execs) == 0

    # Tick at fire time — transitions to PENDING_DELIVERY
    execs = tick(store, fire_at)
    assert len(execs) == 1
    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.PENDING_DELIVERY


def test_one_shot_does_not_refire(store, household_id, member_id):
    fire_at = datetime(2026, 4, 2, 6, 30, tzinfo=UTC)
    request = CreateReminderRequest(
        subject="One time only",
        targets=[_target()],
        schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=fire_at),
    )
    create_reminder(store, household_id, member_id, request)

    # Fire once
    tick(store, fire_at)
    # Tick again — should not fire again
    execs = tick(store, fire_at + timedelta(minutes=5))
    assert len(execs) == 0


# ── Recurring scheduling ───────────────────────────────────────────────────


def test_recurring_creates_execution_and_stays_scheduled(store, household_id, member_id):
    # Every Wednesday at 7am
    now = datetime(2026, 4, 1, 7, 0, tzinfo=UTC)  # a Wednesday
    request = CreateReminderRequest(
        subject="Trash day",
        targets=[_target()],
    )
    reminder = create_reminder(store, household_id, member_id, request)
    schedule_reminder(
        store,
        reminder.id,
        ScheduleReminderRequest(
            schedule=ReminderScheduleInput(
                schedule_type=ScheduleType.RECURRING,
                cron_expression="0 7 * * WED",
                fire_at=now,  # first occurrence for next_fire_at seeding
            ),
        ),
    )
    # Manually set next_fire_at (normally done by schedule creation)
    sched = store.get_schedule(reminder.id)
    from dataclasses import replace

    store.update_schedule(replace(sched, next_fire_at=now))

    # Tick at fire time
    execs = tick(store, now)
    assert len(execs) == 1

    # Reminder stays SCHEDULED (not PENDING_DELIVERY)
    updated = store.get_reminder(reminder.id)
    assert updated.state == ReminderState.SCHEDULED

    # next_fire_at was updated to next Wednesday
    new_sched = store.get_schedule(reminder.id)
    assert new_sched.next_fire_at > now


def test_recurring_fires_multiple_occurrences(store, household_id, member_id):
    now = datetime(2026, 4, 1, 7, 0, tzinfo=UTC)
    request = CreateReminderRequest(subject="Daily check", targets=[_target()])
    reminder = create_reminder(store, household_id, member_id, request)
    schedule_reminder(
        store,
        reminder.id,
        ScheduleReminderRequest(
            schedule=ReminderScheduleInput(
                schedule_type=ScheduleType.RECURRING,
                cron_expression="0 7 * * *",  # daily at 7am
            ),
        ),
    )
    from dataclasses import replace

    sched = store.get_schedule(reminder.id)
    store.update_schedule(replace(sched, next_fire_at=now))

    # First tick
    execs1 = tick(store, now)
    assert len(execs1) == 1

    # Get updated next_fire_at
    sched2 = store.get_schedule(reminder.id)
    assert sched2.next_fire_at is not None

    # Second tick at new fire time
    execs2 = tick(store, sched2.next_fire_at)
    assert len(execs2) == 1

    # Total executions: 2
    assert len(store.executions) == 2


# ── compute_next_fire ──────────────────────────────────────────────────────


def test_compute_next_fire_daily():
    base = datetime(2026, 4, 1, 7, 0)
    result = compute_next_fire("0 7 * * *", base)
    assert result > base
    assert result.hour == 7
    assert result.minute == 0


def test_compute_next_fire_weekly():
    # Wednesday at 7am
    base = datetime(2026, 4, 1, 7, 0)  # This is a Wednesday
    result = compute_next_fire("0 7 * * WED", base)
    assert result > base
    assert result.weekday() == 2  # Wednesday


def test_compute_next_fire_monthly():
    base = datetime(2026, 4, 1, 9, 0)
    result = compute_next_fire("0 9 1 * *", base)
    assert result > base
    assert result.day == 1
    assert result.month == 5


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_tick_empty_store(store):
    execs = tick(store, datetime.now(UTC))
    assert execs == []


def test_tick_ignores_non_scheduled(store, household_id, member_id):
    request = CreateReminderRequest(subject="Draft only", targets=[_target()])
    create_reminder(store, household_id, member_id, request)
    # Reminder is DRAFT, not SCHEDULED — should not fire
    execs = tick(store, datetime.now(UTC))
    assert execs == []


def test_tick_ignores_future_fire_time(store, household_id, member_id):
    future = datetime(2099, 1, 1, tzinfo=UTC)
    request = CreateReminderRequest(
        subject="Far future",
        targets=[_target()],
        schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=future),
    )
    create_reminder(store, household_id, member_id, request)
    execs = tick(store, datetime(2026, 4, 1, tzinfo=UTC))
    assert execs == []
