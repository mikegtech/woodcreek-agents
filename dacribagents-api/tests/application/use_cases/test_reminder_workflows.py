"""Tests for reminder workflow use cases with an in-memory store stub."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID, uuid4

import pytest

from dacribagents.application.use_cases.reminder_workflows import (
    DuplicateReminderError,
    ReminderNotFoundError,
    ReminderStateError,
    acknowledge_reminder,
    cancel_reminder,
    create_reminder,
    ingest_event,
    list_reminders,
    schedule_reminder,
    snooze_reminder,
    update_reminder,
)
from dacribagents.domain.reminders.entities import (
    Household,
    HouseholdMember,
    PreferenceRule,
    Reminder,
    ReminderAcknowledgement,
    ReminderDelivery,
    ReminderExecution,
    ReminderSchedule,
    ReminderTarget,
)
from dacribagents.domain.reminders.enums import (
    AckMethod,
    MemberRole,
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
    CreateReminderRequest,
    EventIntakeRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    ScheduleReminderRequest,
    SnoozeReminderRequest,
    UpdateReminderRequest,
)


# ── In-memory store stub ────────────────────────────────────────────────────


class InMemoryReminderStore:
    """Minimal in-memory implementation of ReminderStore for testing."""

    def __init__(self) -> None:
        self.reminders: dict[UUID, Reminder] = {}
        self.targets: dict[UUID, list[ReminderTarget]] = {}
        self.schedules: dict[UUID, ReminderSchedule] = {}
        self.executions: dict[UUID, ReminderExecution] = {}
        self.deliveries: dict[UUID, list[ReminderDelivery]] = {}
        self.acknowledgements: dict[UUID, ReminderAcknowledgement] = {}
        self.households: dict[UUID, Household] = {}
        self.members: dict[UUID, HouseholdMember] = {}
        self.preference_rules: list[PreferenceRule] = []

    def get_household(self, household_id: UUID) -> Household | None:
        return self.households.get(household_id)

    def get_household_members(self, household_id: UUID) -> list[HouseholdMember]:
        return [m for m in self.members.values() if m.household_id == household_id]

    def get_member(self, member_id: UUID) -> HouseholdMember | None:
        return self.members.get(member_id)

    def create_reminder(self, reminder: Reminder) -> Reminder:
        self.reminders[reminder.id] = reminder
        return reminder

    def get_reminder(self, reminder_id: UUID) -> Reminder | None:
        return self.reminders.get(reminder_id)

    def update_reminder(self, reminder: Reminder) -> Reminder:
        self.reminders[reminder.id] = reminder
        return reminder

    def update_reminder_state(self, reminder_id: UUID, new_state: ReminderState) -> Reminder:
        old = self.reminders[reminder_id]
        updated = replace(old, state=new_state, updated_at=datetime.utcnow())
        self.reminders[reminder_id] = updated
        return updated

    def list_reminders(
        self,
        household_id: UUID,
        *,
        states: list[ReminderState] | None = None,
        member_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Reminder], int]:
        results = [r for r in self.reminders.values() if r.household_id == household_id]
        if states:
            results = [r for r in results if r.state in states]
        total = len(results)
        return results[offset : offset + limit], total

    def find_by_dedupe_key(self, household_id: UUID, dedupe_key: str) -> Reminder | None:
        terminal = {ReminderState.CANCELLED, ReminderState.FAILED, ReminderState.ACKNOWLEDGED}
        for r in self.reminders.values():
            if r.household_id == household_id and r.dedupe_key == dedupe_key and r.state not in terminal:
                return r
        return None

    def set_targets(self, reminder_id: UUID, targets: list[ReminderTarget]) -> list[ReminderTarget]:
        self.targets[reminder_id] = targets
        return targets

    def get_targets(self, reminder_id: UUID) -> list[ReminderTarget]:
        return self.targets.get(reminder_id, [])

    def set_schedule(self, schedule: ReminderSchedule) -> ReminderSchedule:
        self.schedules[schedule.reminder_id] = schedule
        return schedule

    def get_schedule(self, reminder_id: UUID) -> ReminderSchedule | None:
        return self.schedules.get(reminder_id)

    def create_execution(self, execution: ReminderExecution) -> ReminderExecution:
        self.executions[execution.id] = execution
        return execution

    def create_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:
        self.deliveries.setdefault(delivery.execution_id, []).append(delivery)
        return delivery

    def update_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:
        return delivery

    def get_deliveries_for_execution(self, execution_id: UUID) -> list[ReminderDelivery]:
        return self.deliveries.get(execution_id, [])

    def create_acknowledgement(self, ack: ReminderAcknowledgement) -> ReminderAcknowledgement:
        self.acknowledgements[ack.delivery_id] = ack
        return ack

    def get_acknowledgement(self, delivery_id: UUID) -> ReminderAcknowledgement | None:
        return self.acknowledgements.get(delivery_id)

    def get_preference_rules(
        self,
        household_id: UUID,
        member_id: UUID | None = None,
    ) -> list[PreferenceRule]:
        return [
            r
            for r in self.preference_rules
            if r.household_id == household_id and (member_id is None or r.member_id == member_id)
        ]


# ── Fixtures ────────────────────────────────────────────────────────────────


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
    assert reminder.household_id == household_id
    assert reminder.created_by == member_id
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

    # Agent-sourced reminders always require approval, so stay DRAFT even with schedule
    assert reminder.state == ReminderState.DRAFT
    assert reminder.requires_approval is True


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


# ── schedule_reminder ───────────────────────────────────────────────────────


def test_schedule_draft(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    scheduled = schedule_reminder(store, reminder.id, ScheduleReminderRequest(schedule=_one_shot_schedule()))
    assert scheduled.state == ReminderState.SCHEDULED
    assert reminder.id in store.schedules


def test_schedule_non_draft_raises(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Test",
        targets=[_household_target()],
        schedule=_one_shot_schedule(),
    )
    reminder = create_reminder(store, household_id, member_id, request)
    # Already SCHEDULED

    with pytest.raises(InvalidTransitionError):
        schedule_reminder(store, reminder.id, ScheduleReminderRequest(schedule=_one_shot_schedule()))


# ── cancel_reminder ─────────────────────────────────────────────────────────


def test_cancel_draft(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    cancelled = cancel_reminder(store, reminder.id)
    assert cancelled.state == ReminderState.CANCELLED


def test_cancel_scheduled(store, household_id, member_id):
    request = CreateReminderRequest(
        subject="Test",
        targets=[_household_target()],
        schedule=_one_shot_schedule(),
    )
    reminder = create_reminder(store, household_id, member_id, request)

    cancelled = cancel_reminder(store, reminder.id)
    assert cancelled.state == ReminderState.CANCELLED


def test_cancel_terminal_raises(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)
    cancel_reminder(store, reminder.id)

    with pytest.raises(InvalidTransitionError):
        cancel_reminder(store, reminder.id)


# ── acknowledge_reminder ────────────────────────────────────────────────────


def test_acknowledge_delivered(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)
    # Manually advance to DELIVERED for test
    store.update_reminder_state(reminder.id, ReminderState.SCHEDULED)
    store.update_reminder_state(reminder.id, ReminderState.PENDING_DELIVERY)
    store.update_reminder_state(reminder.id, ReminderState.DELIVERED)

    ack_req = AcknowledgeReminderRequest(member_id=member_id, method=AckMethod.SLACK_BUTTON)
    result = acknowledge_reminder(store, reminder.id, ack_req)
    assert result.state == ReminderState.ACKNOWLEDGED


def test_acknowledge_non_delivered_raises(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    with pytest.raises(InvalidTransitionError):
        acknowledge_reminder(
            store,
            reminder.id,
            AcknowledgeReminderRequest(member_id=member_id, method=AckMethod.SMS_REPLY),
        )


# ── snooze_reminder ─────────────────────────────────────────────────────────


def test_snooze_delivered(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)
    store.update_reminder_state(reminder.id, ReminderState.SCHEDULED)
    store.update_reminder_state(reminder.id, ReminderState.PENDING_DELIVERY)
    store.update_reminder_state(reminder.id, ReminderState.DELIVERED)

    snooze_req = SnoozeReminderRequest(
        member_id=member_id,
        method=AckMethod.SMS_REPLY,
        snooze_until=datetime(2026, 3, 28, 15, 0),
    )
    result = snooze_reminder(store, reminder.id, snooze_req)
    assert result.state == ReminderState.SNOOZED


def test_snooze_non_delivered_raises(store, household_id, member_id):
    request = CreateReminderRequest(subject="Test", targets=[_household_target()])
    reminder = create_reminder(store, household_id, member_id, request)

    with pytest.raises(InvalidTransitionError):
        snooze_reminder(
            store,
            reminder.id,
            SnoozeReminderRequest(
                member_id=member_id,
                method=AckMethod.SLACK_BUTTON,
                snooze_until=datetime(2026, 3, 28, 15, 0),
            ),
        )


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


# ── Error cases ─────────────────────────────────────────────────────────────


def test_not_found_raises(store):
    with pytest.raises(ReminderNotFoundError):
        cancel_reminder(store, uuid4())


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
    assert reminder.source == ReminderSource.TELEMETRY
    assert reminder.intent == NotificationIntent.ALERT
    assert reminder.source_event_id == "solar:inv3:temp:2026-03-28T14:00Z"
    assert reminder.dedupe_key == "telemetry:inverter3:temp_warn"


def test_ingest_maintenance_event(store, household_id):
    request = EventIntakeRequest(
        source=ReminderSource.MAINTENANCE,
        source_event_id="maint:hvac-filter:2026Q2",
        household_id=household_id,
        subject="Replace HVAC filter",
        targets=[_household_target()],
    )
    reminder = ingest_event(store, request)

    assert reminder.source == ReminderSource.MAINTENANCE
    assert reminder.requires_approval is True


def test_ingest_hoa_event(store, household_id):
    request = EventIntakeRequest(
        source=ReminderSource.HOA,
        source_event_id="hoa:dues:2026Q2",
        dedupe_key="hoa:dues:2026Q2",
        household_id=household_id,
        subject="HOA dues deadline approaching",
        intent=NotificationIntent.REMINDER,
        targets=[_household_target()],
    )
    reminder = ingest_event(store, request)

    assert reminder.source == ReminderSource.HOA
    assert reminder.intent == NotificationIntent.REMINDER


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

    with pytest.raises(DuplicateReminderError, match="warranty:vac:expiring"):
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

    # Same dedupe_key should work again because the first was cancelled
    second = ingest_event(store, request)
    assert second.id != first.id


def test_ingest_without_dedupe_key_allows_duplicates(store, household_id):
    request = EventIntakeRequest(
        source=ReminderSource.TELEMETRY,
        source_event_id="solar:inv3:temp:event1",
        household_id=household_id,
        subject="Temperature warning",
        targets=[_household_target()],
    )
    first = ingest_event(store, request)
    # Change event ID but no dedupe key — both should succeed
    request2 = EventIntakeRequest(
        source=ReminderSource.TELEMETRY,
        source_event_id="solar:inv3:temp:event2",
        household_id=household_id,
        subject="Temperature warning again",
        targets=[_household_target()],
    )
    second = ingest_event(store, request2)
    assert first.id != second.id
