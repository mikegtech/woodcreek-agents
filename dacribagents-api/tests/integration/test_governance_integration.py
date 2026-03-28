"""Governance integration test — tier gating, kill switch, mute against real Postgres."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.services.delivery import DeliveryDispatcher
from dacribagents.application.services.governance import (
    AutonomyTier,
    activate_kill_switch,
    get_governance_state,
    get_governance_summary,
    mute_member,
)
from dacribagents.application.use_cases.reminder_workflows import create_reminder
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
)
from dacribagents.domain.reminders.models import (
    CreateReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
)


class StubEmailAdapter:
    channel = DeliveryChannel.EMAIL

    def __init__(self):
        self.calls = []

    def send(self, **kw) -> DeliveryResult:
        self.calls.append(kw)
        return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id="stub")


def _create_pending(store, hid, mid, source=ReminderSource.USER):
    r = create_reminder(
        store, hid, mid,
        CreateReminderRequest(
            subject="Test", source=source,
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=datetime(2026, 4, 1, tzinfo=UTC)),
        ),
    )
    store.update_reminder_state(r.id, ReminderState.PENDING_DELIVERY)
    return store.get_reminder(r.id)


@pytest.mark.integration
def test_tier1_allows_dispatch(seeded_store):
    store, hid, mid = seeded_store
    get_governance_state().set_tier(hid, AutonomyTier.TIER_1)

    reminder = _create_pending(store, hid, mid)
    adapter = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 1
    assert store.get_reminder(reminder.id).state == ReminderState.DELIVERED


@pytest.mark.integration
def test_tier0_blocks_dispatch(seeded_store):
    store, hid, mid = seeded_store
    # Default tier 0

    reminder = _create_pending(store, hid, mid)
    adapter = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 0


@pytest.mark.integration
def test_kill_switch_blocks_autonomous(seeded_store):
    store, hid, mid = seeded_store
    get_governance_state().set_tier(hid, AutonomyTier.TIER_2)
    activate_kill_switch("integration_test")

    reminder = _create_pending(store, hid, mid, source=ReminderSource.HOUSEHOLD_ROUTINE)
    adapter = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(deliveries) == 0


@pytest.mark.integration
def test_muted_member_skipped(seeded_store):
    store, hid, mid = seeded_store
    get_governance_state().set_tier(hid, AutonomyTier.TIER_1)
    mute_member(mid, "Test mute")

    reminder = _create_pending(store, hid, mid)
    adapter = StubEmailAdapter()
    dispatcher = DeliveryDispatcher(store=store, adapters={DeliveryChannel.EMAIL: adapter})

    deliveries = dispatcher.dispatch_one(reminder.id)
    assert len(adapter.calls) == 0


@pytest.mark.integration
def test_governance_summary_integration(seeded_store):
    store, hid, mid = seeded_store
    get_governance_state().set_tier(hid, AutonomyTier.TIER_2)

    summary = get_governance_summary(hid)
    assert summary["tier"] == 2
    assert summary["kill_switch_active"] is False
