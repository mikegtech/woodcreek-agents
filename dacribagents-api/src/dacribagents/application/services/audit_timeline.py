"""Unified audit timeline — queryable view across all reminder/governance actions.

Correlates reminder lifecycle, approval, delivery, acknowledgment, governance,
and mute/kill-switch events into a single timeline queryable by reminder_id,
household_id, or time window.

For MVP, the timeline is assembled in-memory from existing store data.
In production, the ``unified_timeline`` PostgreSQL table provides indexed queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services.governance import get_audit_log


@dataclass(frozen=True)
class TimelineEntry:
    """A single entry in the unified audit timeline."""

    timestamp: datetime
    event_type: str
    actor: str
    summary: str
    reminder_id: UUID | None = None
    household_id: UUID | None = None
    metadata: dict = field(default_factory=dict)


def get_reminder_timeline(
    store: ReminderStore,
    reminder_id: UUID,
) -> list[TimelineEntry]:
    """Build a timeline for a single reminder from all available sources."""
    entries: list[TimelineEntry] = []
    reminder = store.get_reminder(reminder_id)
    if reminder is None:
        return entries

    # Reminder creation
    entries.append(TimelineEntry(
        timestamp=reminder.created_at, event_type="reminder.created",
        actor="system", summary=f"Created: {reminder.subject}",
        reminder_id=reminder_id, household_id=reminder.household_id,
        metadata={"state": reminder.state.value, "source": reminder.source.value},
    ))

    # Approval records
    for ar in store.get_approval_records(reminder_id):
        entries.append(TimelineEntry(
            timestamp=ar.created_at, event_type=f"approval.{ar.action.value}",
            actor=str(ar.actor_id)[:8],
            summary=f"Approval {ar.action.value}" + (f": {ar.reason}" if ar.reason else ""),
            reminder_id=reminder_id, household_id=reminder.household_id,
        ))

    # Delivery records (scan executions)
    for eid, execution in getattr(store, "executions", {}).items():
        if execution.reminder_id != reminder_id:
            continue
        entries.append(TimelineEntry(
            timestamp=execution.fired_at, event_type="execution.fired",
            actor="scheduler", summary="Execution fired",
            reminder_id=reminder_id, household_id=reminder.household_id,
        ))
        for d in store.get_deliveries_for_execution(eid):
            entries.append(TimelineEntry(
                timestamp=d.created_at, event_type=f"delivery.{d.status.value}",
                actor="dispatcher", summary=f"{d.channel.value} → {d.status.value}",
                reminder_id=reminder_id, household_id=reminder.household_id,
                metadata={"channel": d.channel.value, "escalation_step": d.escalation_step},
            ))

    # Governance decisions for this reminder
    for entry in get_audit_log(reminder.household_id):
        if entry.reminder_id == reminder_id:
            entries.append(TimelineEntry(
                timestamp=entry.timestamp, event_type=f"governance.{entry.decision}",
                actor=entry.actor_type,
                summary=f"{entry.action_type}: {entry.reason}",
                reminder_id=reminder_id, household_id=reminder.household_id,
            ))

    entries.sort(key=lambda e: e.timestamp)
    return entries


def get_household_timeline(
    store: ReminderStore,
    household_id: UUID,
    limit: int = 50,
) -> list[TimelineEntry]:
    """Build a recent timeline for a household from governance audit log."""
    entries: list[TimelineEntry] = []

    for entry in get_audit_log(household_id, limit=limit):
        entries.append(TimelineEntry(
            timestamp=entry.timestamp, event_type=f"governance.{entry.decision}",
            actor=entry.actor_type,
            summary=f"{entry.action_type}: {entry.reason}",
            reminder_id=entry.reminder_id, household_id=entry.household_id,
        ))

    entries.sort(key=lambda e: e.timestamp)
    return entries[-limit:]
