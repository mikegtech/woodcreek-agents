"""End-to-end reminder lifecycle integration test against real Postgres.

Flow: create → approve → schedule → tick → dispatch (stub) → ack via SMS reply.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.application.services.control_loop import run_cycle
from dacribagents.application.services.governance import AutonomyTier, get_governance_state
from dacribagents.application.services.sms_reply_handler import handle_sms_reply
from dacribagents.application.use_cases.reminder_workflows import (
    approve_reminder,
    create_reminder,
    schedule_reminder,
    submit_for_approval,
)
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
)
from dacribagents.domain.reminders.models import (
    ApproveReminderRequest,
    CreateReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    ScheduleReminderRequest,
    SubmitForApprovalRequest,
)


class StubSmsAdapter:
    channel = DeliveryChannel.SMS

    def send(self, **kw) -> DeliveryResult:
        return DeliveryResult(status=DeliveryStatus.SENT, provider_message_id=f"telnyx-{uuid4().hex[:8]}")


class StubEmailAdapter:
    channel = DeliveryChannel.EMAIL

    def send(self, **kw) -> DeliveryResult:
        return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id=f"email-{uuid4().hex[:8]}")


@pytest.mark.integration
def test_full_lifecycle_e2e(seeded_store):
    """Create → submit → approve → schedule → tick → dispatch → ack."""
    store, household_id, member_id = seeded_store

    # Set governance tier so dispatch works
    get_governance_state().set_tier(household_id, AutonomyTier.TIER_1)

    # 1. Create reminder (HOA source → requires approval)
    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="HOA Dues Deadline",
            body="Semi-annual dues due by April 15",
            source=ReminderSource.HOA,
            source_event_id="hoa:dues:2026Q2",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
        ),
    )
    assert reminder.state == ReminderState.DRAFT
    assert reminder.requires_approval is True

    # 2. Submit for approval
    submitted = submit_for_approval(store, reminder.id, SubmitForApprovalRequest(actor_id=member_id))
    assert submitted.state == ReminderState.PENDING_APPROVAL

    # 3. Approve
    approved = approve_reminder(store, reminder.id, ApproveReminderRequest(actor_id=member_id))
    assert approved.state == ReminderState.APPROVED

    # 4. Schedule
    fire_at = datetime(2026, 4, 10, 8, 0, tzinfo=UTC)
    scheduled = schedule_reminder(
        store, reminder.id,
        ScheduleReminderRequest(schedule=ReminderScheduleInput(
            schedule_type=ScheduleType.ONE_SHOT, fire_at=fire_at,
        )),
    )
    assert scheduled.state == ReminderState.SCHEDULED

    # 5. Tick — advance to PENDING_DELIVERY
    adapters = {DeliveryChannel.SMS: StubSmsAdapter(), DeliveryChannel.EMAIL: StubEmailAdapter()}
    result = run_cycle(store, adapters, household_id, now=fire_at)
    assert result.scheduled == 1

    # 6. Dispatch — tick already triggered dispatch in the same cycle
    updated = store.get_reminder(reminder.id)
    assert updated.state in {ReminderState.PENDING_DELIVERY, ReminderState.DELIVERED}

    # If still PENDING_DELIVERY, run another cycle to dispatch
    if updated.state == ReminderState.PENDING_DELIVERY:
        result2 = run_cycle(store, adapters, household_id)
        assert result2.dispatched >= 1
        updated = store.get_reminder(reminder.id)
        assert updated.state == ReminderState.DELIVERED

    # 7. Acknowledge via SMS reply
    ack_result = handle_sms_reply(store, from_number="+15125551234", text="OK")
    assert ack_result.status == "acknowledged"

    final = store.get_reminder(reminder.id)
    assert final.state == ReminderState.ACKNOWLEDGED


@pytest.mark.integration
def test_user_reminder_auto_schedules(seeded_store):
    """USER source with schedule auto-transitions to SCHEDULED."""
    store, household_id, member_id = seeded_store

    reminder = create_reminder(
        store, household_id, member_id,
        CreateReminderRequest(
            subject="Trash Night",
            targets=[ReminderTargetInput(target_type=TargetType.HOUSEHOLD)],
            schedule=ReminderScheduleInput(
                schedule_type=ScheduleType.ONE_SHOT,
                fire_at=datetime(2026, 4, 2, 18, 0, tzinfo=UTC),
            ),
        ),
    )
    assert reminder.state == ReminderState.SCHEDULED

    schedule = store.get_schedule(reminder.id)
    assert schedule is not None
    assert schedule.next_fire_at is not None
