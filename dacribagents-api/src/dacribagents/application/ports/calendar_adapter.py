"""Port for provider-neutral calendar adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol
from uuid import UUID

from dacribagents.domain.reminders.enums import CalendarProviderType


@dataclass(frozen=True)
class DateRange:
    """An inclusive date range for calendar queries."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class CalendarEvent:
    """A provider-neutral calendar event."""

    id: str
    title: str
    start: datetime
    end: datetime
    source: CalendarProviderType
    owner_identity_id: UUID
    description: str = ""
    all_day: bool = False
    recurrence_rule: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


class CalendarAdapter(Protocol):
    """Read (and eventually write) interface for external calendars.

    Phase 1: list_events, get_event (read-only).
    Phase 4+: create_event, delete_event (write-back, behind CalendarAccessPolicy).
    """

    @property
    def provider_type(self) -> CalendarProviderType:
        """Which calendar provider this adapter serves."""
        ...

    def list_events(
        self,
        identity_id: UUID,
        date_range: DateRange,
    ) -> list[CalendarEvent]:
        """Return events visible to *identity_id* within *date_range*."""
        ...

    def get_event(
        self,
        identity_id: UUID,
        event_id: str,
    ) -> CalendarEvent | None:
        """Return a single event by ID, or None if not found."""
        ...

    # ── Write operations (Phase 8C+, behind CalendarAccessPolicy) ───

    def create_event(  # noqa: D102, PLR0913
        self,
        identity_id: UUID,
        title: str,
        start: datetime,
        end: datetime,
        body: str = "",
        metadata: dict | None = None,
    ) -> str | None:
        """Create a calendar event. Returns the external event ID, or None on failure."""
        ...

    def delete_event(  # noqa: D102
        self,
        identity_id: UUID,
        event_id: str,
    ) -> bool:
        """Delete a calendar event by ID. Returns True on success."""
        ...
