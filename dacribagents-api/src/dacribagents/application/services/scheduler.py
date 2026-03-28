"""Reminder scheduling engine.

Drives time-based state progression for reminders:

- **One-shot**: SCHEDULED → PENDING_DELIVERY when ``next_fire_at <= now``.
  Creates a single ``ReminderExecution``. The reminder does not re-enter
  the queue.

- **Recurring**: Stays SCHEDULED. Each tick creates a ``ReminderExecution``
  for the current occurrence and computes the next ``next_fire_at``.

The ``tick()`` function is a pure operation that can be driven by any
mechanism — FastAPI background task, cron job, Lambda trigger, or a
LangGraph durable workflow in Phase 4+.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from croniter import croniter

from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.domain.reminders.entities import (
    ReminderExecution,
    ReminderSchedule,
)
from dacribagents.domain.reminders.enums import ReminderState, ScheduleType


def tick(store: ReminderStore, now: datetime | None = None) -> list[ReminderExecution]:
    """Find all due SCHEDULED reminders and fire them.

    Returns the list of ``ReminderExecution`` records created.
    """
    now = now or datetime.now(UTC)
    due = store.list_due_reminders(now)
    executions: list[ReminderExecution] = []

    for reminder, schedule in due:
        execution = _fire(store, reminder.id, schedule, now)
        executions.append(execution)

    return executions


def compute_next_fire(cron_expression: str, after: datetime, tz: str = "America/Chicago") -> datetime:
    """Compute the next fire time for a cron expression after *after*."""
    cron = croniter(cron_expression, after)
    return cron.get_next(datetime)


def _fire(
    store: ReminderStore,
    reminder_id: object,
    schedule: ReminderSchedule,
    now: datetime,
) -> ReminderExecution:
    """Fire a single scheduled reminder occurrence."""
    execution = ReminderExecution(
        id=uuid4(),
        reminder_id=reminder_id,
        schedule_id=schedule.id,
        fired_at=now,
        created_at=now,
    )
    store.create_execution(execution)

    if schedule.schedule_type == ScheduleType.RECURRING:
        # Recurring: stay SCHEDULED, compute next fire time.
        next_fire = compute_next_fire(schedule.cron_expression, now)
        updated = replace(schedule, next_fire_at=next_fire)
        store.update_schedule(updated)
        # Reminder stays SCHEDULED — executions track individual occurrences.
    else:
        # One-shot / relative: transition to PENDING_DELIVERY.
        store.update_reminder_state(reminder_id, ReminderState.PENDING_DELIVERY)

    return execution
