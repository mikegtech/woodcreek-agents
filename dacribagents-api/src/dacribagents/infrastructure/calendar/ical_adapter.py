"""Read-only iCal/ICS feed adapter.

Fetches and parses a published iCal feed URL, normalizes events into
the provider-neutral ``CalendarEvent`` model.

Usage::

    adapter = ICalFeedAdapter("https://example.com/calendar.ics")
    events = adapter.list_events(identity_id, date_range)
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import httpx
from loguru import logger

from dacribagents.application.ports.calendar_adapter import (
    CalendarEvent,
    DateRange,
)
from dacribagents.domain.reminders.enums import CalendarProviderType


class ICalFeedAdapter:
    """Read-only calendar adapter backed by a published iCal/ICS URL."""

    def __init__(self, feed_url: str, timeout: int = 30) -> None:
        self._url = feed_url
        self._timeout = timeout
        self._cached_events: list[CalendarEvent] | None = None
        self._cache_time: datetime | None = None

    @property
    def provider_type(self) -> CalendarProviderType:  # noqa: D102
        return CalendarProviderType.ICAL_FEED

    def list_events(
        self,
        identity_id: UUID,
        date_range: DateRange,
    ) -> list[CalendarEvent]:
        """Return events within *date_range* from the iCal feed."""
        events = self._load_events(identity_id)
        return [e for e in events if e.start < date_range.end and e.end > date_range.start]

    def get_event(
        self,
        identity_id: UUID,
        event_id: str,
    ) -> CalendarEvent | None:
        """Return a single event by UID."""
        for e in self._load_events(identity_id):
            if e.id == event_id:
                return e
        return None

    def list_all_events(self, date_range: DateRange) -> list[CalendarEvent]:
        """Household-wide view — same as list_events for a single feed."""
        return self.list_events(UUID(int=0), date_range)

    def _load_events(self, identity_id: UUID) -> list[CalendarEvent]:
        """Fetch and parse the iCal feed (with basic caching)."""
        now = datetime.now(timezone.utc)
        if self._cached_events is not None and self._cache_time and (now - self._cache_time).seconds < 300:
            return self._cached_events

        try:
            resp = httpx.get(self._url, timeout=self._timeout, follow_redirects=True)
            resp.raise_for_status()
            events = _parse_ics(resp.text, identity_id)
            self._cached_events = events
            self._cache_time = now
            logger.info(f"iCal feed loaded: {len(events)} events from {self._url}")
            return events
        except Exception as e:
            logger.error(f"iCal feed fetch failed: {e}")
            return self._cached_events or []


def _parse_ics(ics_text: str, identity_id: UUID) -> list[CalendarEvent]:
    """Minimal iCal parser — extracts VEVENT components.

    This is a lightweight parser for common iCal patterns.  For full
    RFC 5545 compliance, ``icalendar`` library would be needed.
    """
    events: list[CalendarEvent] = []
    in_event = False
    current: dict = {}

    for line in ics_text.splitlines():
        line = line.strip()
        if line == "BEGIN:VEVENT":
            in_event = True
            current = {}
        elif line == "END:VEVENT" and in_event:
            in_event = False
            event = _build_event(current, identity_id)
            if event:
                events.append(event)
        elif in_event and ":" in line:
            key, _, value = line.partition(":")
            # Handle properties with parameters (e.g., DTSTART;TZID=America/Chicago:20260401T090000)
            key = key.split(";")[0]
            current[key] = value

    return events


def _build_event(props: dict, identity_id: UUID) -> CalendarEvent | None:
    """Convert parsed iCal properties to a CalendarEvent."""
    uid = props.get("UID", "")
    summary = props.get("SUMMARY", "")
    dtstart = _parse_ical_datetime(props.get("DTSTART", ""))
    dtend = _parse_ical_datetime(props.get("DTEND", ""))

    if not dtstart:
        return None

    if not dtend:
        dtend = dtstart

    return CalendarEvent(
        id=uid,
        title=summary or "(No title)",
        start=dtstart,
        end=dtend,
        source=CalendarProviderType.ICAL_FEED,
        owner_identity_id=identity_id,
        description=props.get("DESCRIPTION", ""),
        all_day=len(props.get("DTSTART", "")) == 8,  # date-only = all-day
        metadata={"location": props.get("LOCATION", "")},
    )


def _parse_ical_datetime(value: str) -> datetime | None:
    """Parse an iCal datetime value (basic format)."""
    if not value:
        return None
    value = value.rstrip("Z")
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
