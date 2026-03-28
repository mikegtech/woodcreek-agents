"""Tests for reminder runtime container, store selection, and production wiring."""

from __future__ import annotations

from uuid import uuid4

import pytest

from dacribagents.infrastructure.logging import get_correlation_id, reminder_log_context
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore
from dacribagents.infrastructure.reminders.runtime import ReminderRuntime, get_runtime, reset_runtime


@pytest.fixture(autouse=True)
def _reset():
    reset_runtime()
    yield
    reset_runtime()


# ── Runtime store selection ─────────────────────────────────────────────────


def test_runtime_defaults_to_memory_store():
    """In dev without Postgres, runtime creates InMemoryReminderStore."""
    rt = ReminderRuntime()
    # Don't call initialize() (it would try to connect Postgres)
    # Instead, test the _create_memory_store path directly
    store = rt._create_memory_store()
    assert isinstance(store, InMemoryReminderStore)


def test_runtime_singleton():
    rt1 = get_runtime()
    rt2 = get_runtime()
    assert rt1 is rt2


def test_runtime_reset():
    rt1 = get_runtime()
    reset_runtime()
    rt2 = get_runtime()
    assert rt1 is not rt2


def test_runtime_noop_publisher_default():
    from dacribagents.application.ports.event_publisher import NoOpEventPublisher

    rt = ReminderRuntime()
    assert isinstance(rt.event_publisher, NoOpEventPublisher)


# ── Store bridge (Slack _store.py) ──────────────────────────────────────────


def test_store_bridge_returns_tuple():
    from dacribagents.infrastructure.slack._store import get_household_store

    store, calendar, hid = get_household_store()
    assert store is not None
    assert calendar is not None
    assert hid is not None


# ── Structured logging ──────────────────────────────────────────────────────


def test_correlation_id_context():
    assert get_correlation_id() == ""
    with reminder_log_context(reminder_id=uuid4(), operation="test") as cid:
        assert len(cid) == 12
        assert get_correlation_id() == cid
    assert get_correlation_id() == ""


def test_correlation_id_nested():
    with reminder_log_context(operation="outer") as outer_id:
        assert get_correlation_id() == outer_id
        with reminder_log_context(operation="inner") as inner_id:
            assert get_correlation_id() == inner_id
        # After inner exits, outer is restored
        assert get_correlation_id() == outer_id


# ── Scheduler task ──────────────────────────────────────────────────────────


def test_scheduler_tick_skips_when_no_store():
    """_run_tick should not crash when runtime is not initialized."""
    from dacribagents.infrastructure.reminders.scheduler_task import _run_tick

    reset_runtime()
    _run_tick()  # should not raise


# ── Governance store ────────────────────────────────────────────────────────


def test_governance_store_import():
    """PostgresGovernanceStore should be importable and constructable pattern."""
    from dacribagents.infrastructure.reminders.governance_store import PostgresGovernanceStore

    # Just verify the class is importable — actual DB tests require integration setup
    assert PostgresGovernanceStore is not None
