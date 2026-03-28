"""In-memory implementation of ReminderStore for development and testing.

This is a real implementation of the ``ReminderStore`` protocol that stores
everything in dicts.  Useful for:
- Phase 1 Slack integration before the PostgreSQL store is wired
- Unit/integration tests
- Local development without a database

All public methods implement the ReminderStore protocol — docstrings on
the protocol definition in ``application/ports/reminder_store.py``.
"""
# ruff: noqa: D102

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from dacribagents.domain.reminders.entities import (
    ApprovalRecord,
    CalendarIdentity,
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
from dacribagents.domain.reminders.enums import ReminderState


class InMemoryReminderStore:
    """Dict-backed ReminderStore for dev and test."""

    def __init__(self) -> None:
        self.reminders: dict[UUID, Reminder] = {}
        self.targets: dict[UUID, list[ReminderTarget]] = {}
        self.schedules: dict[UUID, ReminderSchedule] = {}
        self.executions: dict[UUID, ReminderExecution] = {}
        self.deliveries: dict[UUID, list[ReminderDelivery]] = {}
        self.acknowledgements: dict[UUID, ReminderAcknowledgement] = {}
        self.approval_records: dict[UUID, list[ApprovalRecord]] = {}
        self.households: dict[UUID, Household] = {}
        self.members: dict[UUID, HouseholdMember] = {}
        self.calendar_identities: dict[UUID, CalendarIdentity] = {}
        self.preference_rules: list[PreferenceRule] = []

    # ── Household ───────────────────────────────────────────────────────

    def get_household(self, household_id: UUID) -> Household | None:
        return self.households.get(household_id)

    def get_household_members(self, household_id: UUID) -> list[HouseholdMember]:
        return [m for m in self.members.values() if m.household_id == household_id]

    def get_member(self, member_id: UUID) -> HouseholdMember | None:
        return self.members.get(member_id)

    # ── Reminders ───────────────────────────────────────────────────────

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
        updated = replace(old, state=new_state, updated_at=datetime.now(UTC))
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

    # ── Targets ─────────────────────────────────────────────────────────

    def set_targets(self, reminder_id: UUID, targets: list[ReminderTarget]) -> list[ReminderTarget]:
        self.targets[reminder_id] = targets
        return targets

    def get_targets(self, reminder_id: UUID) -> list[ReminderTarget]:
        return self.targets.get(reminder_id, [])

    # ── Schedules ───────────────────────────────────────────────────────

    def set_schedule(self, schedule: ReminderSchedule) -> ReminderSchedule:
        self.schedules[schedule.reminder_id] = schedule
        return schedule

    def get_schedule(self, reminder_id: UUID) -> ReminderSchedule | None:
        return self.schedules.get(reminder_id)

    def update_schedule(self, schedule: ReminderSchedule) -> ReminderSchedule:
        self.schedules[schedule.reminder_id] = schedule
        return schedule

    def list_due_reminders(self, now: datetime) -> list[tuple]:
        results = []
        for r in self.reminders.values():
            if r.state != ReminderState.SCHEDULED:
                continue
            sched = self.schedules.get(r.id)
            if sched and sched.next_fire_at and sched.next_fire_at <= now:
                results.append((r, sched))
        return results

    # ── Executions & Deliveries ─────────────────────────────────────────

    def create_execution(self, execution: ReminderExecution) -> ReminderExecution:
        self.executions[execution.id] = execution
        return execution

    def create_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:
        self.deliveries.setdefault(delivery.execution_id, []).append(delivery)
        return delivery

    def update_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:
        exec_deliveries = self.deliveries.get(delivery.execution_id, [])
        for i, d in enumerate(exec_deliveries):
            if d.id == delivery.id:
                exec_deliveries[i] = delivery
                break
        return delivery

    def get_deliveries_for_execution(self, execution_id: UUID) -> list[ReminderDelivery]:
        return self.deliveries.get(execution_id, [])

    # ── Acknowledgements ────────────────────────────────────────────────

    def create_acknowledgement(self, ack: ReminderAcknowledgement) -> ReminderAcknowledgement:
        self.acknowledgements[ack.delivery_id] = ack
        return ack

    def get_acknowledgement(self, delivery_id: UUID) -> ReminderAcknowledgement | None:
        return self.acknowledgements.get(delivery_id)

    # ── Approval Records ───────────────────────────────────────────────

    def create_approval_record(self, record: ApprovalRecord) -> ApprovalRecord:
        self.approval_records.setdefault(record.reminder_id, []).append(record)
        return record

    def get_approval_records(self, reminder_id: UUID) -> list[ApprovalRecord]:
        return self.approval_records.get(reminder_id, [])

    # ── Preferences ─────────────────────────────────────────────────────

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
