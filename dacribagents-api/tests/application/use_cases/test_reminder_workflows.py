"""Tests for reminder workflow use cases including approval flows."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

import pytest

from dacribagents.application.use_cases.reminder_workflows import (
    ApprovalRequiredError,
    DuplicateReminderError,
    ReminderNotFoundError,
    ReminderStateError,
    acknowledge_reminder,
    approve_reminder,
    cancel_reminder,
    create_reminder,
    get_approval_history,
    ingest_event,
    list_reminders,
    reject_reminder,
    schedule_reminder,
    snooze_reminder,
    submit_for_approval,
    update_reminder,
)
from dacribagents.domain.reminders.enums import (
    AckMethod,
    ApprovalAction,
    NotificationIntent,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.lifecycle import InvalidTransitionError
from dacribagents.domain.reminders.models import (
    AcknowledgeReminderRequest,
    ApproveReminderRequest,
    CreateReminderRequest,
    EventIntakeRequest,
    RejectReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    ScheduleReminderRequest,
    SnoozeReminderRequest,
    SubmitForApprovalRequest,
    UpdateReminderRequest,
)
from dacribagents.infrastructure.reminders.in_memory_store import InMemoryReminderStore


@pytest.fixture()
def store() -> InMemoryReminderStore:
    return InMemoryReminderStore()


@pytest.fixture()
def household_id() -> UUID:
    return uuid4()


@pytest.fixture()
def member_id() -> UUID:
    return uuid4()


def _household_target() -> ReminderTargetInput:
    return ReminderTargetInput(target_type=TargetType.HOUSEHOLD)


def _one_shot_schedule() -> ReminderScheduleInput:
    return ReminderScheduleInput(
        schedule_type=ScheduleType.ONE_SHOT,
        fire_at=datetime(2026, 4, 2, 6, 30),
    )


# ── create_reminder ─────────────────────────────────────────────────────────


def test_create_draft(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Take out trash",
        targets=[_household_target()],
    )
    reminder = create_reminder(store, household_id, member_id, request)

    assert reminder.state == ReminderState.DRAFT
    assert reminder.subject == "Take out trash"
    assert reminder.requires_approval is False
    assert reminder.id in store.reminders
    assert len(store.targets[reminder.id]) == 1


def test_create_with_schedule_auto_schedules(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Mow lawn",
        targets=[_household_target()],
        schedule=_one_shot_schedule(),
    )
    reminder = create_reminder(store, household_id, member_id, request)

    assert reminder.state == ReminderState.SCHEDULED
    assert reminder.id in store.schedules


def test_create_agent_reminder_requires_approval(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="HOA deadline",
        source=ReminderSource.AGENT,
        source_agent="hoa_compliance",
        targets=[_household_target()],
        schedule=_one_shot_schedule(),
    )
    reminder = create_reminder(store, household_id, member_id, request)

    # Agent-sourced requires approval, so stays DRAFT even with schedule
    assert reminder.state == ReminderState.DRAFT
    assert reminder.requires_approval is True


def test_create_telemetry_requires_approval(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Inverter temp",
        source=ReminderSource.TELEMETRY,
        source_event_id="solar:inv3",
        intent=NotificationIntent.ALERT,
        targets=[_household_target()],
    )
    reminder = create_reminder(store, household_id, member_id, request)
    assert reminder.requires_approval is True


def test_create_household_routine_no_approval(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Trash day",
        source=ReminderSource.HOUSEHOLD_ROUTINE,
        targets=[_household_target()],
        schedule=_one_shot_schedule(),
    )
    reminder = create_reminder(store, household_id, member_id, request)
    assert reminder.requires_approval is False
    assert reminder.state == ReminderState.SCHEDULED


# ── update_reminder ─────────────────────────────────────────────────────────


def test_update_draft(store, household_id, member_id):
    request = CreateReminderRequest(subject="Old", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    updated = update_reminder(store, reminder.id, UpdateReminderRequest(subject="New", urgency=UrgencyLevel.URGENT))
    assert updated.subject == "New"
    assert updated.urgency == UrgencyLevel.URGENT


def test_update_non_draft_raises(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Test",
        targets=[_household_target()],
        schedule=_one_shot_schedule(),
    )
    reminder = create_reminder(store, household_id, member_id, request)
    assert reminder.state == ReminderState.SCHEDULED

    with pytest.raises(ReminderStateError, match="DRAFT"):
        update_reminder(store, reminder.id, UpdateReminderRequest(subject="Nope"))


# ── submit_for_approval ─────────────────────────────────────────────────────


def test_submit_for_approval(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="HOA dues",
        source=ReminderSource.HOA,
        source_event_id="hoa:dues",
        targets=[_household_target()],
    )
    reminder = create_reminder(store, household_id, member_id, request)
    assert reminder.state == ReminderState.DRAFT

    submitted = submit_for_approval(
        store, reminder.id,
        SubmitForApprovalRequest(actor_id=member_id, reason="Needs review"),
    )
    assert submitted.state == ReminderState.PENDING_APPROVAL

    records = store.get_approval_records(reminder.id)
    assert len(records) == 1
    assert records[0].action == ApprovalAction.SUBMITTED
    assert records[0].actor_id == member_id


def test_submit_already_scheduled_raises(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Test",
        targets=[_household_target()],
        schedule=_one_shot_schedule(),
    )
    reminder = create_reminder(store, household_id, member_id, request)
    assert reminder.state == ReminderState.SCHEDULED

    with pytest.raises(InvalidTransitionError):
        submit_for_approval(store, reminder.id, SubmitForApprovalRequest(actor_id=member_id))


# ── approve_reminder ────────────────────────────────────────────────────────


def test_approve_pending(store, household_id, member_id):
    reminder = _create_and_submit(store, household_id, member_id)
    approver = uuid4()

    approved = approve_reminder(
        store, reminder.id,
        ApproveReminderRequest(actor_id=approver, reason="Looks good"),
    )
    assert approved.state == ReminderState.APPROVED

    records = store.get_approval_records(reminder.id)
    assert len(records) == 2  # submitted + approved
    assert records[1].action == ApprovalAction.APPROVED
    assert records[1].actor_id == approver


def test_approve_non_pending_raises(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    with pytest.raises(InvalidTransitionError):
        approve_reminder(store, reminder.id, ApproveReminderRequest(actor_id=member_id))


# ── reject_reminder ─────────────────────────────────────────────────────────


def test_reject_pending(store, household_id, member_id):
    reminder = _create_and_submit(store, household_id, member_id)

    rejected = reject_reminder(
        store, reminder.id,
        RejectReminderRequest(actor_id=member_id, reason="Too late"),
    )
    assert rejected.state == ReminderState.REJECTED

    records = store.get_approval_records(reminder.id)
    assert records[-1].action == ApprovalAction.REJECTED
    assert records[-1].reason == "Too late"


def test_reject_non_pending_raises(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    with pytest.raises(InvalidTransitionError):
        reject_reminder(store, reminder.id, RejectReminderRequest(actor_id=member_id, reason="No"))


# ── schedule_reminder ───────────────────────────────────────────────────────


def test_schedule_draft_no_approval(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    scheduled = schedule_reminder(store, reminder.id, ScheduleReminderRequest(schedule=_one_shot_schedule()))
    assert scheduled.state == ReminderState.SCHEDULED


def test_schedule_after_approval(store, household_id, member_id):
    reminder = _create_and_submit(store, household_id, member_id)
    approve_reminder(store, reminder.id, ApproveReminderRequest(actor_id=member_id))

    scheduled = schedule_reminder(store, reminder.id, ScheduleReminderRequest(schedule=_one_shot_schedule()))
    assert scheduled.state == ReminderState.SCHEDULED


def test_schedule_draft_with_approval_required_raises(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Test",
        source=ReminderSource.HOA,
        source_event_id="hoa:test",
        targets=[_household_target()],
    )
    reminder = create_reminder(store, household_id, member_id, request)
    assert reminder.requires_approval is True

    with pytest.raises(ApprovalRequiredError):
        schedule_reminder(store, reminder.id, ScheduleReminderRequest(schedule=_one_shot_schedule()))


def test_schedule_pending_approval_raises(store, household_id, member_id):
    reminder = _create_and_submit(store, household_id, member_id)

    with pytest.raises(InvalidTransitionError):
        schedule_reminder(store, reminder.id, ScheduleReminderRequest(schedule=_one_shot_schedule()))


# ── cancel_reminder ─────────────────────────────────────────────────────────


def test_cancel_draft(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)
    assert cancel_reminder(store, reminder.id).state == ReminderState.CANCELLED


def test_cancel_pending_approval(store, household_id, member_id):
    reminder = _create_and_submit(store, household_id, member_id)
    assert cancel_reminder(store, reminder.id).state == ReminderState.CANCELLED


def test_cancel_terminal_raises(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)
    cancel_reminder(store, reminder.id)

    with pytest.raises(InvalidTransitionError):
        cancel_reminder(store, reminder.id)


# ── acknowledge_reminder ────────────────────────────────────────────────────


def test_acknowledge_delivered(store, household_id, member_id):
    reminder = _advance_to_delivered(store, household_id, member_id)

    ack_req = AcknowledgeReminderRequest(member_id=member_id, method=AckMethod.SLACK_BUTTON)
    result = acknowledge_reminder(store, reminder.id, ack_req)
    assert result.state == ReminderState.ACKNOWLEDGED


def test_acknowledge_non_delivered_raises(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    with pytest.raises(InvalidTransitionError):
        acknowledge_reminder(
            store, reminder.id,
            AcknowledgeReminderRequest(member_id=member_id, method=AckMethod.SMS_REPLY),
        )


# ── snooze_reminder ─────────────────────────────────────────────────────────


def test_snooze_delivered(store, household_id, member_id):
    reminder = _advance_to_delivered(store, household_id, member_id)

    snooze_req = SnoozeReminderRequest(
        member_id=member_id,
        method=AckMethod.SMS_REPLY,
        snooze_until=datetime(2026, 3, 28, 15, 0),
    )
    result = snooze_reminder(store, reminder.id, snooze_req)
    assert result.state == ReminderState.SNOOZED


# ── list_reminders ──────────────────────────────────────────────────────────


def test_list_empty(store, household_id):
    results, total = list_reminders(store, household_id)
    assert results == []
    assert total == 0


def test_list_filters_by_state(store, household_id, member_id):
    for subject in ("A", "B", "C"):
        req = CreateReminderRequest(subject=subject, targets=[_household_target()])
        create_reminder(store, household_id, member_id, req)

    # Schedule B
    reminders_list, _ = list_reminders(store, household_id)
    schedule_reminder(
        store,
        reminders_list[1].id,
        ScheduleReminderRequest(schedule=_one_shot_schedule()),
    )

    drafts, total_drafts = list_reminders(store, household_id, states=[ReminderState.DRAFT])
    assert total_drafts == 2

    scheduled, total_scheduled = list_reminders(store, household_id, states=[ReminderState.SCHEDULED])
    assert total_scheduled == 1


# ── get_approval_history ────────────────────────────────────────────────────


def test_approval_history_full_cycle(store, household_id, member_id):
    reminder = _create_and_submit(store, household_id, member_id)
    approve_reminder(store, reminder.id, ApproveReminderRequest(actor_id=uuid4(), reason="OK"))

    history = get_approval_history(store, reminder.id)
    assert len(history) == 2
    assert history[0].action == ApprovalAction.SUBMITTED
    assert history[1].action == ApprovalAction.APPROVED


def test_approval_history_not_found_raises(store):
    with pytest.raises(ReminderNotFoundError):
        get_approval_history(store, uuid4())


# ── ingest_event ────────────────────────────────────────────────────────────


def test_ingest_telemetry_event(store, household_id):
    request = EventIntakeRequest(
        source=ReminderSource.TELEMETRY,
        source_event_id="solar:inv3:temp:2026-03-28T14:00Z",
        dedupe_key="telemetry:inverter3:temp_warn",
        household_id=household_id,
        subject="Inverter temperature warning",
        urgency=UrgencyLevel.URGENT,
        intent=NotificationIntent.ALERT,
        targets=[_household_target()],
    )
    reminder = ingest_event(store, request)
    assert reminder.state == ReminderState.DRAFT
    assert reminder.requires_approval is True


def test_ingest_dedupe_blocks_duplicate(store, household_id):
    request = EventIntakeRequest(
        source=ReminderSource.WARRANTY,
        source_event_id="warranty:vac:exp-30d",
        dedupe_key="warranty:vac:expiring",
        household_id=household_id,
        subject="Warranty expiring soon",
        targets=[_household_target()],
    )
    ingest_event(store, request)

    with pytest.raises(DuplicateReminderError):
        ingest_event(store, request)


def test_ingest_dedupe_allows_after_cancel(store, household_id):
    request = EventIntakeRequest(
        source=ReminderSource.WARRANTY,
        source_event_id="warranty:vac:exp-30d",
        dedupe_key="warranty:vac:expiring",
        household_id=household_id,
        subject="Warranty expiring soon",
        targets=[_household_target()],
    )
    first = ingest_event(store, request)
    cancel_reminder(store, first.id)

    second = ingest_event(store, request)
    assert second.id != first.id


# ── Error cases ─────────────────────────────────────────────────────────────


def test_not_found_raises(store):
    with pytest.raises(ReminderNotFoundError):
        cancel_reminder(store, uuid4())


# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_and_submit(store, household_id, member_id) -> object:
    """Create an HOA reminder and submit it for approval."""
    request = CreateReminderRequest(
        subject="HOA dues deadline",
        source=ReminderSource.HOA,
        source_event_id="hoa:dues:2026Q2",
        targets=[_household_target()],
    )
    reminder = create_reminder(store, household_id, member_id, request)
    return submit_for_approval(
        store, reminder.id,
        SubmitForApprovalRequest(actor_id=member_id),
    )


def _advance_to_delivered(store, household_id, member_id) -> object:
    """Create a user reminder and advance it to DELIVERED."""
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)
    store.update_reminder_state(reminder.id, ReminderState.SCHEDULED)
    store.update_reminder_state(reminder.id, ReminderState.PENDING_DELIVERY)
    return store.update_reminder_state(reminder.id, ReminderState.DELIVERED)
