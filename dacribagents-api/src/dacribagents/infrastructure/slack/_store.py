"""Household store provider for Slack integration.

In Phase 1 this returns a placeholder in-memory store and mock calendar.
When a PostgreSQL-backed ReminderStore is implemented, replace this with
a real provider wired through the application container.
"""

from __future__ import annotations

from uuid import UUID

from dacribagents.infrastructure.calendar.mock_adapter import MockCalendarAdapter
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore

_DEFAULT_HOUSEHOLD_ID = UUID("00000000-0000-0000-0000-000000000001")

_store: InMemoryReminderStore | None = None
_calendar: MockCalendarAdapter | None = None


def get_household_store() -> tuple[InMemoryReminderStore, MockCalendarAdapter, UUID]:
    """Return (store, calendar, household_id) for the Slack handler.

    Returns singleton in-memory instances.  The HTTP endpoint will
    eventually source these from the application container / DI root.
    """
    global _store, _calendar  # noqa: PLW0603
    if _store is None:
        _store = InMemoryReminderStore()
    if _calendar is None:
        _calendar = MockCalendarAdapter()
    return _store, _calendar, _DEFAULT_HOUSEHOLD_ID
