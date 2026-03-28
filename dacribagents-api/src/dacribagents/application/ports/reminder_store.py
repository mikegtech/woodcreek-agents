"""Port for reminder persistence operations."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

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
from dacribagents.domain.reminders.enums import ReminderState


class ReminderStore(Protocol):
    """Persistence port for all reminder domain entities.

    Method signatures are the contract; implementations live in infrastructure.
    """

    # ── Household ───────────────────────────────────────────────────────

    def get_household(self, household_id: UUID) -> Household | None:  # noqa: D102
        ...

    def get_household_members(self, household_id: UUID) -> list[HouseholdMember]:  # noqa: D102
        ...

    def get_member(self, member_id: UUID) -> HouseholdMember | None:  # noqa: D102
        ...

    # ── Reminders ───────────────────────────────────────────────────────

    def create_reminder(self, reminder: Reminder) -> Reminder:  # noqa: D102
        ...

    def get_reminder(self, reminder_id: UUID) -> Reminder | None:  # noqa: D102
        ...

    def update_reminder(self, reminder: Reminder) -> Reminder:  # noqa: D102
        ...

    def update_reminder_state(self, reminder_id: UUID, new_state: ReminderState) -> Reminder:  # noqa: D102
        ...

    def list_reminders(  # noqa: D102
        self,
        household_id: UUID,
        *,
        states: list[ReminderState] | None = None,
        member_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Reminder], int]: ...

    def find_by_dedupe_key(self, household_id: UUID, dedupe_key: str) -> Reminder | None:  # noqa: D102
        ...

    # ── Targets ─────────────────────────────────────────────────────────

    def set_targets(self, reminder_id: UUID, targets: list[ReminderTarget]) -> list[ReminderTarget]:  # noqa: D102
        ...

    def get_targets(self, reminder_id: UUID) -> list[ReminderTarget]:  # noqa: D102
        ...

    # ── Schedules ───────────────────────────────────────────────────────

    def set_schedule(self, schedule: ReminderSchedule) -> ReminderSchedule:  # noqa: D102
        ...

    def get_schedule(self, reminder_id: UUID) -> ReminderSchedule | None:  # noqa: D102
        ...

    # ── Executions & Deliveries ─────────────────────────────────────────

    def create_execution(self, execution: ReminderExecution) -> ReminderExecution:  # noqa: D102
        ...

    def create_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:  # noqa: D102
        ...

    def update_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:  # noqa: D102
        ...

    def get_deliveries_for_execution(self, execution_id: UUID) -> list[ReminderDelivery]:  # noqa: D102
        ...

    # ── Acknowledgements ────────────────────────────────────────────────

    def create_acknowledgement(self, ack: ReminderAcknowledgement) -> ReminderAcknowledgement:  # noqa: D102
        ...

    def get_acknowledgement(self, delivery_id: UUID) -> ReminderAcknowledgement | None:  # noqa: D102
        ...

    # ── Preferences ─────────────────────────────────────────────────────

    def get_preference_rules(  # noqa: D102
        self,
        household_id: UUID,
        member_id: UUID | None = None,
    ) -> list[PreferenceRule]: ...
