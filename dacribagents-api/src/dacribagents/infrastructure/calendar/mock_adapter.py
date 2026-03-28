"""In-memory calendar adapter for development and testing.

Implements the ``CalendarAdapter`` protocol with a pre-seeded event store.
Use ``add_event()`` to populate events, then query with ``list_events()``
and ``get_event()``.
"""

from __future__ import annotations

from uuid import UUID

from dacribagents.application.ports.calendar_adapter import (
    CalendarEvent,
    DateRange,
)
from dacribagents.domain.reminders.enums import CalendarProviderType


class MockCalendarAdapter:
    """In-memory calendar adapter for dev/test."""

    def __init__(self) -> None:
        self._events: list[CalendarEvent] = []

    @property
    def provider_type(self) -> CalendarProviderType:  # noqa: D102
        return CalendarProviderType.MANUAL

    def add_event(self, event: CalendarEvent) -> None:
        """Seed an event into the mock store."""
        self._events.append(event)

    def list_events(
        self,
        identity_id: UUID,
        date_range: DateRange,
    ) -> list[CalendarEvent]:
        """Return events within *date_range* for the given identity."""
        return [
            e
            for e in self._events
            if e.owner_identity_id == identity_id and e.start < date_range.end and e.end > date_range.start
        ]

    def get_event(
        self,
        identity_id: UUID,
        event_id: str,
    ) -> CalendarEvent | None:
        """Return a single event by ID."""
        for e in self._events:
            if e.id == event_id and e.owner_identity_id == identity_id:
                return e
        return None

    def list_all_events(self, date_range: DateRange) -> list[CalendarEvent]:
        """Return all events in range regardless of identity (household view)."""
        return [e for e in self._events if e.start < date_range.end and e.end > date_range.start]
