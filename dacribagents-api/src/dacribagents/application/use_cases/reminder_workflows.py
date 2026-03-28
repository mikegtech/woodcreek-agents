"""Use cases for reminder lifecycle workflows.

These orchestrate domain logic and ports. No infrastructure dependencies.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services import approval_policy
from dacribagents.domain.reminders.entities import (
    ApprovalRecord,
    Reminder,
    ReminderAcknowledgement,
    ReminderSchedule,
    ReminderTarget,
)
from dacribagents.domain.reminders.enums import (
    ApprovalAction,
    ReminderState,
)
from dacribagents.domain.reminders.lifecycle import (
    validate_transition,
)
from dacribagents.domain.reminders.models import (
    AcknowledgeReminderRequest,
    ApproveReminderRequest,
    CreateReminderRequest,
    EventIntakeRequest,
    RejectReminderRequest,
    ScheduleReminderRequest,
    SnoozeReminderRequest,
    SubmitForApprovalRequest,
    UpdateReminderRequest,
)


class ReminderNotFoundError(Exception):
    """Raised when a reminder ID does not exist."""


class ReminderStateError(Exception):
    """Raised when an operation is invalid for the current reminder state."""


class DuplicateReminderError(Exception):
    """Raised when an event intake matches an existing dedupe key."""

    def __init__(self, dedupe_key: str, existing_id: UUID) -> None:
        self.dedupe_key = dedupe_key
        self.existing_id = existing_id
        super().__init__(f"Reminder with dedupe_key {dedupe_key!r} already exists: {existing_id}")


class ApprovalRequiredError(Exception):
    """Raised when trying to schedule a reminder that needs approval first."""

    def __init__(self, reminder_id: UUID) -> None:
        self.reminder_id = reminder_id
        super().__init__(f"Reminder {reminder_id} requires approval before scheduling")


# ── Workflow functions ──────────────────────────────────────────────────────


def create_reminder(
    store: ReminderStore,
    household_id: UUID,
    created_by: UUID,
    request: CreateReminderRequest,
) -> Reminder:
    """Create a new reminder draft with targets and optional schedule.

    The approval policy determines whether the reminder needs approval.
    If no approval is needed and a schedule is provided, transitions
    directly to SCHEDULED.
    """
    now = datetime.now(UTC)
    reminder_id = uuid4()

    decision = approval_policy.evaluate(
        source=request.source,
        urgency=request.urgency,
        requires_approval_override=request.requires_approval,
        source_agent=request.source_agent,
    )

    reminder = Reminder(
        id=reminder_id,
        household_id=household_id,
        subject=request.subject,
        body=request.body,
        urgency=request.urgency,
        intent=request.intent,
        source=request.source,
        source_agent=request.source_agent,
        source_event_id=request.source_event_id,
        dedupe_key=request.dedupe_key,
        state=ReminderState.DRAFT,
        requires_approval=decision.required,
        created_by=created_by,
        created_at=now,
        updated_at=now,
    )
    reminder = store.create_reminder(reminder)

    targets = [
        ReminderTarget(
            id=uuid4(),
            reminder_id=reminder_id,
            target_type=t.target_type,
            member_id=t.member_id,
            role=t.role,
        )
        for t in request.targets
    ]
    store.set_targets(reminder_id, targets)

    # If a schedule is provided and no approval needed, auto-schedule.
    if request.schedule and not decision.required:
        schedule = _build_schedule(reminder_id, request.schedule)
        store.set_schedule(schedule)
        reminder = store.update_reminder_state(reminder_id, ReminderState.SCHEDULED)

    return reminder


def update_reminder(
    store: ReminderStore,
    reminder_id: UUID,
    request: UpdateReminderRequest,
) -> Reminder:
    """Update a draft reminder's mutable fields. Only allowed in DRAFT state."""
    reminder = _get_or_raise(store, reminder_id)
    if reminder.state != ReminderState.DRAFT:
        raise ReminderStateError(f"Cannot update reminder in {reminder.state.value!r} state; must be DRAFT")

    now = datetime.now(UTC)
    updated = replace(
        reminder,
        subject=request.subject if request.subject is not None else reminder.subject,
        body=request.body if request.body is not None else reminder.body,
        urgency=request.urgency if request.urgency is not None else reminder.urgency,
        updated_at=now,
    )
    updated = store.update_reminder(updated)

    if request.targets is not None:
        targets = [
            ReminderTarget(
                id=uuid4(),
                reminder_id=reminder_id,
                target_type=t.target_type,
                member_id=t.member_id,
                role=t.role,
            )
            for t in request.targets
        ]
        store.set_targets(reminder_id, targets)

    if request.schedule is not None:
        schedule = _build_schedule(reminder_id, request.schedule)
        store.set_schedule(schedule)

    return updated


def submit_for_approval(
    store: ReminderStore,
    reminder_id: UUID,
    request: SubmitForApprovalRequest,
) -> Reminder:
    """Submit a draft reminder for human approval. DRAFT -> PENDING_APPROVAL."""
    reminder = _get_or_raise(store, reminder_id)
    validate_transition(reminder.state, ReminderState.PENDING_APPROVAL)

    store.create_approval_record(ApprovalRecord(
        id=uuid4(),
        reminder_id=reminder_id,
        action=ApprovalAction.SUBMITTED,
        actor_id=request.actor_id,
        created_at=datetime.now(UTC),
        reason=request.reason,
    ))
    return store.update_reminder_state(reminder_id, ReminderState.PENDING_APPROVAL)


def approve_reminder(
    store: ReminderStore,
    reminder_id: UUID,
    request: ApproveReminderRequest,
) -> Reminder:
    """Approve a pending-approval reminder. PENDING_APPROVAL -> APPROVED."""
    reminder = _get_or_raise(store, reminder_id)
    validate_transition(reminder.state, ReminderState.APPROVED)

    store.create_approval_record(ApprovalRecord(
        id=uuid4(),
        reminder_id=reminder_id,
        action=ApprovalAction.APPROVED,
        actor_id=request.actor_id,
        created_at=datetime.now(UTC),
        reason=request.reason,
    ))
    return store.update_reminder_state(reminder_id, ReminderState.APPROVED)


def reject_reminder(
    store: ReminderStore,
    reminder_id: UUID,
    request: RejectReminderRequest,
) -> Reminder:
    """Reject a pending-approval reminder. PENDING_APPROVAL -> REJECTED (terminal)."""
    reminder = _get_or_raise(store, reminder_id)
    validate_transition(reminder.state, ReminderState.REJECTED)

    store.create_approval_record(ApprovalRecord(
        id=uuid4(),
        reminder_id=reminder_id,
        action=ApprovalAction.REJECTED,
        actor_id=request.actor_id,
        created_at=datetime.now(UTC),
        reason=request.reason,
    ))
    return store.update_reminder_state(reminder_id, ReminderState.REJECTED)


def schedule_reminder(
    store: ReminderStore,
    reminder_id: UUID,
    request: ScheduleReminderRequest,
) -> Reminder:
    """Attach a schedule and transition to SCHEDULED.

    Valid from DRAFT (if no approval needed) or APPROVED (after approval).
    Raises ApprovalRequiredError if the reminder needs approval but is still
    in DRAFT state.
    """
    reminder = _get_or_raise(store, reminder_id)

    # Block: DRAFT → SCHEDULED when approval is required
    if reminder.state == ReminderState.DRAFT and reminder.requires_approval:
        raise ApprovalRequiredError(reminder_id)

    validate_transition(reminder.state, ReminderState.SCHEDULED)

    schedule = _build_schedule(reminder_id, request.schedule)
    store.set_schedule(schedule)
    return store.update_reminder_state(reminder_id, ReminderState.SCHEDULED)


def cancel_reminder(
    store: ReminderStore,
    reminder_id: UUID,
) -> Reminder:
    """Cancel a reminder from any non-terminal state."""
    reminder = _get_or_raise(store, reminder_id)
    validate_transition(reminder.state, ReminderState.CANCELLED)
    return store.update_reminder_state(reminder_id, ReminderState.CANCELLED)


def acknowledge_reminder(
    store: ReminderStore,
    reminder_id: UUID,
    request: AcknowledgeReminderRequest,
) -> Reminder:
    """Acknowledge a delivered reminder. DELIVERED -> ACKNOWLEDGED."""
    reminder = _get_or_raise(store, reminder_id)
    validate_transition(reminder.state, ReminderState.ACKNOWLEDGED)

    ack = ReminderAcknowledgement(
        id=uuid4(),
        delivery_id=uuid4(),  # resolved by store in production
        member_id=request.member_id,
        method=request.method,
        acknowledged_at=datetime.now(UTC),
        note=request.note,
    )
    store.create_acknowledgement(ack)
    return store.update_reminder_state(reminder_id, ReminderState.ACKNOWLEDGED)


def snooze_reminder(
    store: ReminderStore,
    reminder_id: UUID,
    request: SnoozeReminderRequest,
) -> Reminder:
    """Snooze a delivered reminder. DELIVERED -> SNOOZED."""
    reminder = _get_or_raise(store, reminder_id)
    validate_transition(reminder.state, ReminderState.SNOOZED)

    ack = ReminderAcknowledgement(
        id=uuid4(),
        delivery_id=uuid4(),  # resolved by store in production
        member_id=request.member_id,
        method=request.method,
        acknowledged_at=datetime.now(UTC),
        note=request.note,
        snoozed_until=request.snooze_until,
    )
    store.create_acknowledgement(ack)
    return store.update_reminder_state(reminder_id, ReminderState.SNOOZED)


def list_reminders(  # noqa: PLR0913
    store: ReminderStore,
    household_id: UUID,
    *,
    states: list[ReminderState] | None = None,
    member_id: UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Reminder], int]:
    """List reminders with optional filters."""
    return store.list_reminders(
        household_id,
        states=states,
        member_id=member_id,
        limit=limit,
        offset=offset,
    )


def get_approval_history(
    store: ReminderStore,
    reminder_id: UUID,
) -> list[ApprovalRecord]:
    """Return approval audit records for a reminder."""
    _get_or_raise(store, reminder_id)
    return store.get_approval_records(reminder_id)


def ingest_event(
    store: ReminderStore,
    request: EventIntakeRequest,
) -> Reminder:
    """Intake a reminder from an upstream event source.

    Checks dedupe, then creates as a draft requiring approval.
    """
    if request.dedupe_key:
        existing = store.find_by_dedupe_key(request.household_id, request.dedupe_key)
        if existing is not None:
            raise DuplicateReminderError(request.dedupe_key, existing.id)

    create_req = CreateReminderRequest(
        subject=request.subject,
        body=request.body,
        urgency=request.urgency,
        intent=request.intent,
        source=request.source,
        source_agent=request.source_agent,
        source_event_id=request.source_event_id,
        dedupe_key=request.dedupe_key,
        targets=request.targets,
        schedule=request.schedule,
        requires_approval=True,
    )
    return create_reminder(store, request.household_id, request.household_id, create_req)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_or_raise(store: ReminderStore, reminder_id: UUID) -> Reminder:
    reminder = store.get_reminder(reminder_id)
    if reminder is None:
        raise ReminderNotFoundError(f"Reminder {reminder_id} not found")
    return reminder


def _build_schedule(reminder_id: UUID, schedule_input: object) -> ReminderSchedule:
    return ReminderSchedule(
        id=uuid4(),
        reminder_id=reminder_id,
        schedule_type=schedule_input.schedule_type,
        timezone=schedule_input.timezone,
        fire_at=schedule_input.fire_at,
        relative_to=schedule_input.relative_to,
        relative_offset_minutes=schedule_input.relative_offset_minutes,
        cron_expression=schedule_input.cron_expression,
        next_fire_at=schedule_input.fire_at,  # for ONE_SHOT, next == fire_at
    )
