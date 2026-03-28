"""Calendar mirror service — constrained reminder-to-calendar write-back.

Mirrors eligible approved+scheduled reminders as calendar events via the
``CalendarAdapter.create_event()`` interface.  One-way only: reminder
system is authoritative.

Eligibility (per ADR-007):
- SCHEDULED or APPROVED state
- INDIVIDUAL target (not HOUSEHOLD)
- Target has active CalendarIdentity
- Intent ≠ DIGEST
- Not already mirrored
- Governance allows

See ``docs/ADRs/ADR-007.md`` for full decision record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from loguru import logger

from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services import governance
from dacribagents.application.services.calendar_queries import resolve_identity
from dacribagents.domain.reminders.entities import Reminder
from dacribagents.domain.reminders.enums import (
    NotificationIntent,
    ReminderState,
    TargetType,
)


@dataclass(frozen=True)
class CalendarMirrorRecord:
    """Linkage between a reminder and its mirrored calendar event."""

    id: UUID
    reminder_id: UUID
    calendar_identity_id: UUID
    provider: str
    external_event_id: str
    status: str  # "active", "deleted", "failed"
    created_at: datetime
    deleted_at: datetime | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class MirrorResult:
    """Outcome of a mirror attempt."""

    reminder_id: UUID
    status: str  # "mirrored", "ineligible", "already_mirrored", "failed", "governance_blocked"
    external_event_id: str | None = None
    reason: str = ""


class CalendarMirrorService:
    """Mirrors eligible reminders into external calendars."""

    def __init__(
        self,
        store: ReminderStore,
        calendar_adapter: object,
    ) -> None:
        self.store = store
        self.adapter = calendar_adapter
        self._mirrors: dict[UUID, CalendarMirrorRecord] = {}  # in-memory for MVP

    def mirror_if_eligible(self, reminder_id: UUID) -> MirrorResult:  # noqa: PLR0911
        """Check eligibility and mirror a reminder to calendar if appropriate."""
        reminder = self.store.get_reminder(reminder_id)
        if reminder is None:
            return MirrorResult(reminder_id=reminder_id, status="ineligible", reason="Reminder not found")

        # Already mirrored?
        if reminder_id in self._mirrors and self._mirrors[reminder_id].status == "active":
            return MirrorResult(reminder_id=reminder_id, status="already_mirrored", reason="Already mirrored")

        # Eligibility check
        reason = _check_eligibility(self.store, reminder)
        if reason:
            return MirrorResult(reminder_id=reminder_id, status="ineligible", reason=reason)

        # Governance check
        gov_decision = governance.evaluate(
            household_id=reminder.household_id,
            action_type="calendar_mirror",
            actor_type="system",
            requires_tier=governance.AutonomyTier.TIER_1,
            reminder_id=reminder_id,
        )
        if not gov_decision.allowed:
            return MirrorResult(reminder_id=reminder_id, status="governance_blocked", reason=gov_decision.reason)

        # Resolve target member's calendar identity
        targets = self.store.get_targets(reminder_id)
        individual_targets = [t for t in targets if t.target_type == TargetType.INDIVIDUAL and t.member_id]
        if not individual_targets:
            return MirrorResult(reminder_id=reminder_id, status="ineligible", reason="No individual target")

        member = self.store.get_member(individual_targets[0].member_id)
        if member is None:
            return MirrorResult(reminder_id=reminder_id, status="ineligible", reason="Target member not found")

        identity = resolve_identity(self.store, member)
        if identity is None:
            return MirrorResult(reminder_id=reminder_id, status="ineligible", reason="No CalendarIdentity for member")

        # Get schedule for timing
        schedule = self.store.get_schedule(reminder_id)
        if not schedule or not schedule.next_fire_at:
            return MirrorResult(reminder_id=reminder_id, status="ineligible", reason="No schedule/fire time")

        # Create calendar event
        start = schedule.next_fire_at
        end = start + timedelta(minutes=15)
        body = f"{reminder.body}\n\n— Woodcreek Reminder (source: {reminder.source.value})"

        try:
            external_id = self.adapter.create_event(
                identity_id=identity.id,
                title=reminder.subject,
                start=start,
                end=end,
                body=body,
            )
        except Exception as e:
            logger.error(f"Calendar mirror failed for {reminder_id}: {e}")
            record = CalendarMirrorRecord(
                id=uuid4(), reminder_id=reminder_id,
                calendar_identity_id=identity.id, provider=identity.provider.value,
                external_event_id="", status="failed",
                created_at=datetime.now(UTC), failure_reason=str(e),
            )
            self._mirrors[reminder_id] = record
            return MirrorResult(reminder_id=reminder_id, status="failed", reason=str(e))

        if external_id:
            record = CalendarMirrorRecord(
                id=uuid4(), reminder_id=reminder_id,
                calendar_identity_id=identity.id, provider=identity.provider.value,
                external_event_id=external_id, status="active",
                created_at=datetime.now(UTC),
            )
            self._mirrors[reminder_id] = record
            logger.info(f"Reminder {str(reminder_id)[:8]} mirrored to calendar: {external_id}")
            return MirrorResult(
                reminder_id=reminder_id, status="mirrored",
                external_event_id=external_id, reason="Mirrored to calendar",
            )

        return MirrorResult(reminder_id=reminder_id, status="failed", reason="Adapter returned no event ID")

    def delete_mirror(self, reminder_id: UUID) -> bool:
        """Delete a mirrored calendar event (e.g., on reminder cancellation)."""
        record = self._mirrors.get(reminder_id)
        if not record or record.status != "active":
            return False

        try:
            success = self.adapter.delete_event(record.calendar_identity_id, record.external_event_id)
            if success:
                self._mirrors[reminder_id] = CalendarMirrorRecord(
                    id=record.id, reminder_id=record.reminder_id,
                    calendar_identity_id=record.calendar_identity_id,
                    provider=record.provider, external_event_id=record.external_event_id,
                    status="deleted", created_at=record.created_at,
                    deleted_at=datetime.now(UTC),
                )
                logger.info(f"Calendar mirror deleted for {str(reminder_id)[:8]}")
                return True
        except Exception as e:
            logger.error(f"Calendar mirror delete failed for {reminder_id}: {e}")
        return False

    def get_mirror_status(self, reminder_id: UUID) -> CalendarMirrorRecord | None:
        """Return the mirror record for a reminder, or None."""
        return self._mirrors.get(reminder_id)


def _check_eligibility(store: ReminderStore, reminder: Reminder) -> str | None:
    """Return None if eligible, or a reason string if ineligible."""
    if reminder.state not in {ReminderState.SCHEDULED, ReminderState.APPROVED}:
        return f"State {reminder.state.value} is not eligible (must be SCHEDULED or APPROVED)"

    if reminder.intent == NotificationIntent.DIGEST:
        return "DIGEST intent reminders are not mirrored to calendar"

    targets = store.get_targets(reminder.id)
    if not any(t.target_type == TargetType.INDIVIDUAL for t in targets):
        return "Only INDIVIDUAL-targeted reminders are mirrored (HOUSEHOLD fanout deferred)"

    schedule = store.get_schedule(reminder.id)
    if not schedule or not schedule.next_fire_at:
        return "No schedule or fire time — cannot create calendar event"

    return None
