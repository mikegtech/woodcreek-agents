"""Event-driven reminder creation integration test."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dacribagents.application.services.event_intake import UpstreamEvent, process_upstream_event
from dacribagents.domain.reminders.enums import ReminderSource, ReminderState, UrgencyLevel


@pytest.mark.integration
def test_event_creates_reminder(seeded_store):
    """Kafka-shaped event payload → reminder creation with correct metadata."""
    store, household_id, member_id = seeded_store

    event = UpstreamEvent(
        event_type="warranty.expiring",
        event_id="warranty:homepro-vac:exp-30d",
        household_id=household_id,
        timestamp=datetime(2026, 4, 1, 9, 0, tzinfo=UTC),
        subject="HomePro warranty expires in 30 days",
        body="Central vacuum warranty expires 2026-05-01",
        severity="normal",
        source_service="warranty-tracker",
    )
    result = process_upstream_event(store, event)

    assert result.status == "created"
    assert result.reminder is not None
    assert result.reminder.source == ReminderSource.WARRANTY
    assert result.reminder.source_event_id == "warranty:homepro-vac:exp-30d"
    assert result.reminder.requires_approval is True
    assert result.reminder.state == ReminderState.DRAFT


@pytest.mark.integration
def test_event_dedupe(seeded_store):
    """Same event ID produces exactly one active reminder."""
    store, household_id, _ = seeded_store

    event = UpstreamEvent(
        event_type="camera.offline",
        event_id="cam:front:offline:20260401",
        household_id=household_id,
        timestamp=datetime(2026, 4, 1, 14, 0, tzinfo=UTC),
        subject="Front camera offline",
    )

    r1 = process_upstream_event(store, event)
    r2 = process_upstream_event(store, event)

    assert r1.status == "created"
    assert r2.status == "duplicate"


@pytest.mark.integration
def test_event_severity_override(seeded_store):
    """Upstream severity overrides default mapping urgency."""
    store, household_id, _ = seeded_store

    event = UpstreamEvent(
        event_type="maintenance.due",
        event_id="maint:hvac:2026Q2",
        household_id=household_id,
        timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        subject="HVAC filter",
        severity="critical",
    )
    result = process_upstream_event(store, event)
    assert result.reminder.urgency == UrgencyLevel.CRITICAL
