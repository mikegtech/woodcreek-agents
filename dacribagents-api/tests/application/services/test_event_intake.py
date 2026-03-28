"""Tests for event-to-reminder intake: mapping, dedup, lifecycle entry."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dacribagents.application.ports.event_publisher import DomainEvent
from dacribagents.application.services.event_intake import (
    IntakeResult,
    UpstreamEvent,
    get_supported_event_types,
    process_upstream_event,
)
from dacribagents.domain.reminders.entities import Household, HouseholdMember
from dacribagents.domain.reminders.enums import (
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ReminderState,
    UrgencyLevel,
)
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore


class CollectingPublisher:
    def __init__(self):
        self.events: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


@pytest.fixture()
def household_id():
    return uuid4()


@pytest.fixture()
def store(household_id):
    s = InMemoryReminderStore()
    s.households[household_id] = Household(id=household_id, name="Test", created_at=datetime(2026, 1, 1))
    mike = HouseholdMember(
        id=uuid4(), household_id=household_id, name="Mike",
        role=MemberRole.ADMIN, timezone="America/Chicago",
        created_at=datetime(2026, 1, 1), email="mike@test.com",
    )
    s.members[mike.id] = mike
    return s


def _event(event_type: str, household_id, **kwargs):
    defaults = {
        "event_id": f"{event_type}:{uuid4().hex[:8]}",
        "household_id": household_id,
        "timestamp": datetime(2026, 4, 1, 9, 0, tzinfo=UTC),
        "subject": f"Test {event_type}",
    }
    defaults.update(kwargs)
    return UpstreamEvent(event_type=event_type, **defaults)


# ── Supported event types ───────────────────────────────────────────────────


def test_supported_event_types():
    types = get_supported_event_types()
    assert "warranty.expiring" in types
    assert "hoa.deadline.approaching" in types
    assert "camera.offline" in types
    assert "maintenance.due" in types


# ── Event-to-reminder mapping ───────────────────────────────────────────────


def test_warranty_creates_reminder(store, household_id):
    event = _event("warranty.expiring", household_id, subject="Warranty expires in 30 days")
    result = process_upstream_event(store, event)

    assert result.status == "created"
    assert result.reminder is not None
    assert result.reminder.source == ReminderSource.WARRANTY
    assert result.reminder.intent == NotificationIntent.REMINDER
    assert result.reminder.urgency == UrgencyLevel.NORMAL
    assert result.reminder.requires_approval is True
    assert result.reminder.state == ReminderState.DRAFT


def test_hoa_creates_urgent_reminder(store, household_id):
    event = _event("hoa.deadline.approaching", household_id, subject="HOA dues due in 7 days")
    result = process_upstream_event(store, event)

    assert result.status == "created"
    assert result.reminder.source == ReminderSource.HOA
    assert result.reminder.urgency == UrgencyLevel.URGENT


def test_camera_creates_critical_alert(store, household_id):
    event = _event("camera.offline", household_id, subject="Front camera offline")
    result = process_upstream_event(store, event)

    assert result.status == "created"
    assert result.reminder.source == ReminderSource.TELEMETRY
    assert result.reminder.intent == NotificationIntent.ALERT
    assert result.reminder.urgency == UrgencyLevel.CRITICAL


def test_maintenance_creates_reminder(store, household_id):
    event = _event("maintenance.due", household_id, subject="HVAC filter replacement due")
    result = process_upstream_event(store, event)

    assert result.status == "created"
    assert result.reminder.source == ReminderSource.MAINTENANCE


# ── Severity override ───────────────────────────────────────────────────────


def test_severity_overrides_default_urgency(store, household_id):
    """Upstream severity 'critical' should override the default mapping urgency."""
    event = _event("warranty.expiring", household_id, severity="critical")
    result = process_upstream_event(store, event)

    assert result.reminder.urgency == UrgencyLevel.CRITICAL


# ── Unknown event types ─────────────────────────────────────────────────────


def test_unknown_event_type_ignored(store, household_id):
    event = _event("unknown.thing.happened", household_id)
    result = process_upstream_event(store, event)

    assert result.status == "ignored"
    assert result.reminder is None
    assert "Unsupported" in result.reason


# ── Deduplication ───────────────────────────────────────────────────────────


def test_duplicate_event_suppressed(store, household_id):
    event = _event("warranty.expiring", household_id, event_id="warranty:vac:exp-30d")
    r1 = process_upstream_event(store, event)
    assert r1.status == "created"

    r2 = process_upstream_event(store, event)
    assert r2.status == "duplicate"
    assert r2.reminder is None


def test_duplicate_after_cancel_allowed(store, household_id):
    event = _event("warranty.expiring", household_id, event_id="warranty:vac:exp-30d")
    r1 = process_upstream_event(store, event)
    assert r1.status == "created"

    # Cancel the first reminder
    from dacribagents.application.use_cases.reminder_workflows import cancel_reminder
    cancel_reminder(store, r1.reminder.id)

    # Same event should create a new reminder now
    r2 = process_upstream_event(store, event)
    assert r2.status == "created"
    assert r2.reminder.id != r1.reminder.id


def test_replay_safe_idempotent(store, household_id):
    """Multiple identical events produce exactly one active reminder."""
    event = _event("camera.offline", household_id, event_id="cam:front:offline:20260401")
    results = [process_upstream_event(store, event) for _ in range(5)]

    created = [r for r in results if r.status == "created"]
    duplicates = [r for r in results if r.status == "duplicate"]
    assert len(created) == 1
    assert len(duplicates) == 4


# ── Event publishing ────────────────────────────────────────────────────────


def test_created_event_published(store, household_id):
    publisher = CollectingPublisher()
    event = _event("warranty.expiring", household_id)
    process_upstream_event(store, event, events=publisher)

    assert len(publisher.events) == 1
    assert publisher.events[0].event_type == "reminder.created_from_event"


def test_duplicate_event_published(store, household_id):
    publisher = CollectingPublisher()
    event = _event("warranty.expiring", household_id, event_id="dup-test")
    process_upstream_event(store, event, events=publisher)
    process_upstream_event(store, event, events=publisher)

    types = [e.event_type for e in publisher.events]
    assert "reminder.created_from_event" in types
    assert "reminder.duplicate_ignored" in types


# ── Source metadata ─────────────────────────────────────────────────────────


def test_source_event_id_preserved(store, household_id):
    event = _event("hoa.deadline.approaching", household_id, event_id="hoa:dues:2026Q2")
    result = process_upstream_event(store, event)

    assert result.reminder.source_event_id == "hoa:dues:2026Q2"


def test_dedupe_key_includes_event_type(store, household_id):
    event = _event("warranty.expiring", household_id, event_id="vac:exp")
    result = process_upstream_event(store, event)

    assert result.reminder.dedupe_key == "warranty.expiring:vac:exp"


def test_source_service_maps_to_source_agent(store, household_id):
    event = _event("maintenance.due", household_id, source_service="home-maintenance-agent")
    result = process_upstream_event(store, event)

    assert result.reminder.source_agent == "home-maintenance-agent"
