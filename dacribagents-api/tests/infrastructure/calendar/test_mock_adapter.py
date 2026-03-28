"""Tests for in-memory mock calendar adapter."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from dacribagents.application.ports.calendar_adapter import CalendarEvent, DateRange
from dacribagents.domain.reminders.enums import CalendarProviderType
from dacribagents.infrastructure.calendar.mock_adapter import MockCalendarAdapter


def _make_event(
    title: str = "Test Event",
    start_offset_hours: int = 0,
    duration_hours: int = 1,
    identity_id=None,
) -> CalendarEvent:
    base = datetime(2026, 4, 1, 9, 0)
    return CalendarEvent(
        id=f"evt-{title.lower().replace(' ', '-')}",
        title=title,
        start=base + timedelta(hours=start_offset_hours),
        end=base + timedelta(hours=start_offset_hours + duration_hours),
        source=CalendarProviderType.MANUAL,
        owner_identity_id=identity_id or uuid4(),
    )


def test_empty_adapter():
    adapter = MockCalendarAdapter()
    dr = DateRange(start=datetime(2026, 4, 1), end=datetime(2026, 4, 2))
    assert adapter.list_events(uuid4(), dr) == []


def test_add_and_list():
    adapter = MockCalendarAdapter()
    identity = uuid4()
    event = _make_event("Soccer", identity_id=identity)
    adapter.add_event(event)

    dr = DateRange(start=datetime(2026, 4, 1), end=datetime(2026, 4, 2))
    result = adapter.list_events(identity, dr)
    assert len(result) == 1
    assert result[0].title == "Soccer"


def test_filter_by_identity():
    adapter = MockCalendarAdapter()
    id1, id2 = uuid4(), uuid4()
    adapter.add_event(_make_event("Event A", identity_id=id1))
    adapter.add_event(_make_event("Event B", identity_id=id2))

    dr = DateRange(start=datetime(2026, 4, 1), end=datetime(2026, 4, 2))
    assert len(adapter.list_events(id1, dr)) == 1
    assert len(adapter.list_events(id2, dr)) == 1


def test_filter_by_date_range():
    adapter = MockCalendarAdapter()
    identity = uuid4()
    adapter.add_event(_make_event("Morning", start_offset_hours=0, identity_id=identity))
    adapter.add_event(_make_event("Afternoon", start_offset_hours=6, identity_id=identity))

    morning_range = DateRange(
        start=datetime(2026, 4, 1, 8, 0),
        end=datetime(2026, 4, 1, 11, 0),
    )
    assert len(adapter.list_events(identity, morning_range)) == 1


def test_get_event():
    adapter = MockCalendarAdapter()
    identity = uuid4()
    event = _make_event("Dentist", identity_id=identity)
    adapter.add_event(event)

    found = adapter.get_event(identity, event.id)
    assert found is not None
    assert found.title == "Dentist"


def test_get_event_not_found():
    adapter = MockCalendarAdapter()
    assert adapter.get_event(uuid4(), "nonexistent") is None


def test_list_all_events():
    adapter = MockCalendarAdapter()
    adapter.add_event(_make_event("A", identity_id=uuid4()))
    adapter.add_event(_make_event("B", identity_id=uuid4()))

    dr = DateRange(start=datetime(2026, 4, 1), end=datetime(2026, 4, 2))
    assert len(adapter.list_all_events(dr)) == 2


def test_provider_type():
    adapter = MockCalendarAdapter()
    assert adapter.provider_type == CalendarProviderType.MANUAL
