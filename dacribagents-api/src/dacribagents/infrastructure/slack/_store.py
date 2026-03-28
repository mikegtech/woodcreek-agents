"""Household store provider — bridges Slack/HTTP endpoints to the reminder runtime.

In production, returns the PostgresReminderStore from the runtime container.
In dev/test, falls back to InMemoryReminderStore.
"""

from __future__ import annotations

from uuid import UUID

from dacribagents.infrastructure.reminders.runtime import get_runtime

_DEFAULT_HOUSEHOLD_ID = UUID("00000000-0000-0000-0000-000000000001")


def get_household_store() -> tuple:
    """Return (store, calendar, household_id) from the runtime container."""
    rt = get_runtime()
    if rt.store is None:
        # Runtime not initialized yet — create a dev fallback
        from dacribagents.infrastructure.calendar.mock_adapter import MockCalendarAdapter  # noqa: PLC0415
        from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore  # noqa: PLC0415

        rt.store = InMemoryReminderStore()
        rt.calendar = MockCalendarAdapter()
        rt.household_id = _DEFAULT_HOUSEHOLD_ID

    return rt.store, rt.calendar, rt.household_id
