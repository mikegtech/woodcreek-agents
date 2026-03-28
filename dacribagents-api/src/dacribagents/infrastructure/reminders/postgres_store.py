"""PostgreSQL-backed ReminderStore implementation.

Uses ``psycopg`` (sync) via the existing ``PostgresClientWrapper`` pattern.
All methods execute against the ``reminders``, ``reminder_targets``,
``reminder_schedules``, etc. tables defined in ``schema.py``.

For MVP, this is a thin SQL layer.  Queries are explicit and auditable.
"""
# ruff: noqa: D102

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import psycopg
from psycopg.types.json import Json

from dacribagents.domain.reminders.entities import (
    ApprovalRecord,
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
    ApprovalAction,
    DeliveryChannel,
    DeliveryStatus,
    MemberRole,
    NotificationIntent,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)


class PostgresReminderStore:
    """PostgreSQL-backed ReminderStore."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def _cur(self):
        return self._conn.cursor()

    # ── Household ───────────────────────────────────────────────────────

    def get_household(self, household_id: UUID) -> Household | None:
        with self._cur() as cur:
            cur.execute("SELECT id, name, created_at, metadata FROM households WHERE id = %s", (household_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return Household(id=row[0], name=row[1], created_at=row[2], metadata=row[3] or {})

    def get_household_members(self, household_id: UUID) -> list[HouseholdMember]:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, household_id, name, role, timezone, created_at, email, phone, slack_id, "
                "quiet_hours_start, quiet_hours_end, metadata FROM household_members WHERE household_id = %s",
                (household_id,),
            )
            return [_row_to_member(r) for r in cur.fetchall()]

    def get_member(self, member_id: UUID) -> HouseholdMember | None:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, household_id, name, role, timezone, created_at, email, phone, slack_id, "
                "quiet_hours_start, quiet_hours_end, metadata FROM household_members WHERE id = %s",
                (member_id,),
            )
            row = cur.fetchone()
            return _row_to_member(row) if row else None

    # ── Reminders ───────────────────────────────────────────────────────

    def create_reminder(self, reminder: Reminder) -> Reminder:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO reminders (id, household_id, subject, body, urgency, intent, source, source_agent, "
                "source_event_id, dedupe_key, state, requires_approval, created_by, created_at, updated_at, metadata) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (reminder.id, reminder.household_id, reminder.subject, reminder.body,
                 reminder.urgency.value, reminder.intent.value, reminder.source.value,
                 reminder.source_agent, reminder.source_event_id, reminder.dedupe_key,
                 reminder.state.value, reminder.requires_approval, reminder.created_by,
                 reminder.created_at, reminder.updated_at, Json({})),
            )
            self._conn.commit()
        return reminder

    def get_reminder(self, reminder_id: UUID) -> Reminder | None:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, household_id, subject, body, urgency, intent, source, source_agent, "
                "source_event_id, dedupe_key, state, requires_approval, created_by, created_at, "
                "updated_at, metadata FROM reminders WHERE id = %s",
                (reminder_id,),
            )
            row = cur.fetchone()
            return _row_to_reminder(row) if row else None

    def update_reminder(self, reminder: Reminder) -> Reminder:
        with self._cur() as cur:
            cur.execute(
                "UPDATE reminders SET subject=%s, body=%s, urgency=%s, updated_at=%s WHERE id=%s",
                (reminder.subject, reminder.body, reminder.urgency.value, reminder.updated_at, reminder.id),
            )
            self._conn.commit()
        return reminder

    def update_reminder_state(self, reminder_id: UUID, new_state: ReminderState) -> Reminder:
        now = datetime.now(UTC)
        with self._cur() as cur:
            cur.execute(
                "UPDATE reminders SET state=%s, updated_at=%s WHERE id=%s",
                (new_state.value, now, reminder_id),
            )
            self._conn.commit()
        return self.get_reminder(reminder_id)

    def list_reminders(self, household_id: UUID, *, states=None, member_id=None, limit=50, offset=0):
        with self._cur() as cur:
            query = "SELECT id, household_id, subject, body, urgency, intent, source, source_agent, " \
                    "source_event_id, dedupe_key, state, requires_approval, created_by, created_at, " \
                    "updated_at, metadata FROM reminders WHERE household_id = %s"
            params: list = [household_id]
            if states:
                placeholders = ",".join(["%s"] * len(states))
                query += f" AND state IN ({placeholders})"
                params.extend(s.value for s in states)
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cur.execute(query, params)
            rows = cur.fetchall()

            # Count
            count_q = "SELECT COUNT(*) FROM reminders WHERE household_id = %s"
            count_p: list = [household_id]
            if states:
                placeholders = ",".join(["%s"] * len(states))
                count_q += f" AND state IN ({placeholders})"
                count_p.extend(s.value for s in states)
            cur.execute(count_q, count_p)
            total = cur.fetchone()[0]

        return [_row_to_reminder(r) for r in rows], total

    def find_by_dedupe_key(self, household_id: UUID, dedupe_key: str) -> Reminder | None:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, household_id, subject, body, urgency, intent, source, source_agent, "
                "source_event_id, dedupe_key, state, requires_approval, created_by, created_at, "
                "updated_at, metadata FROM reminders "
                "WHERE household_id = %s AND dedupe_key = %s AND state NOT IN ('cancelled','failed','acknowledged','rejected') "
                "LIMIT 1",
                (household_id, dedupe_key),
            )
            row = cur.fetchone()
            return _row_to_reminder(row) if row else None

    # ── Targets ─────────────────────────────────────────────────────────

    def set_targets(self, reminder_id: UUID, targets: list[ReminderTarget]) -> list[ReminderTarget]:
        with self._cur() as cur:
            cur.execute("DELETE FROM reminder_targets WHERE reminder_id = %s", (reminder_id,))
            for t in targets:
                cur.execute(
                    "INSERT INTO reminder_targets (id, reminder_id, target_type, member_id, role) VALUES (%s,%s,%s,%s,%s)",
                    (t.id, t.reminder_id, t.target_type.value, t.member_id, t.role.value if t.role else None),
                )
            self._conn.commit()
        return targets

    def get_targets(self, reminder_id: UUID) -> list[ReminderTarget]:
        with self._cur() as cur:
            cur.execute("SELECT id, reminder_id, target_type, member_id, role FROM reminder_targets WHERE reminder_id = %s", (reminder_id,))
            return [
                ReminderTarget(id=r[0], reminder_id=r[1], target_type=TargetType(r[2]),
                               member_id=r[3], role=MemberRole(r[4]) if r[4] else None)
                for r in cur.fetchall()
            ]

    # ── Schedules ───────────────────────────────────────────────────────

    def set_schedule(self, schedule: ReminderSchedule) -> ReminderSchedule:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO reminder_schedules (id, reminder_id, schedule_type, timezone, fire_at, "
                "relative_to, relative_offset_minutes, cron_expression, next_fire_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (reminder_id) DO UPDATE SET schedule_type=EXCLUDED.schedule_type, "
                "fire_at=EXCLUDED.fire_at, cron_expression=EXCLUDED.cron_expression, next_fire_at=EXCLUDED.next_fire_at",
                (schedule.id, schedule.reminder_id, schedule.schedule_type.value, schedule.timezone,
                 schedule.fire_at, schedule.relative_to, schedule.relative_offset_minutes,
                 schedule.cron_expression, schedule.next_fire_at),
            )
            self._conn.commit()
        return schedule

    def get_schedule(self, reminder_id: UUID) -> ReminderSchedule | None:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, reminder_id, schedule_type, timezone, fire_at, relative_to, "
                "relative_offset_minutes, cron_expression, next_fire_at FROM reminder_schedules WHERE reminder_id = %s",
                (reminder_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return ReminderSchedule(
                id=row[0], reminder_id=row[1], schedule_type=ScheduleType(row[2]),
                timezone=row[3], fire_at=row[4], relative_to=row[5],
                relative_offset_minutes=row[6], cron_expression=row[7], next_fire_at=row[8],
            )

    def update_schedule(self, schedule: ReminderSchedule) -> ReminderSchedule:
        with self._cur() as cur:
            cur.execute(
                "UPDATE reminder_schedules SET next_fire_at=%s WHERE reminder_id=%s",
                (schedule.next_fire_at, schedule.reminder_id),
            )
            self._conn.commit()
        return schedule

    def list_due_reminders(self, now: datetime) -> list[tuple]:
        with self._cur() as cur:
            cur.execute(
                "SELECT r.id, r.household_id, r.subject, r.body, r.urgency, r.intent, r.source, "
                "r.source_agent, r.source_event_id, r.dedupe_key, r.state, r.requires_approval, "
                "r.created_by, r.created_at, r.updated_at, r.metadata, "
                "s.id, s.reminder_id, s.schedule_type, s.timezone, s.fire_at, s.relative_to, "
                "s.relative_offset_minutes, s.cron_expression, s.next_fire_at "
                "FROM reminders r JOIN reminder_schedules s ON r.id = s.reminder_id "
                "WHERE r.state = 'scheduled' AND s.next_fire_at <= %s",
                (now,),
            )
            results = []
            for row in cur.fetchall():
                reminder = _row_to_reminder(row[:16])
                schedule = ReminderSchedule(
                    id=row[16], reminder_id=row[17], schedule_type=ScheduleType(row[18]),
                    timezone=row[19], fire_at=row[20], relative_to=row[21],
                    relative_offset_minutes=row[22], cron_expression=row[23], next_fire_at=row[24],
                )
                results.append((reminder, schedule))
            return results

    # ── Executions & Deliveries ─────────────────────────────────────────

    def create_execution(self, execution: ReminderExecution) -> ReminderExecution:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO reminder_executions (id, reminder_id, schedule_id, fired_at, created_at) VALUES (%s,%s,%s,%s,%s)",
                (execution.id, execution.reminder_id, execution.schedule_id, execution.fired_at, execution.created_at),
            )
            self._conn.commit()
        return execution

    def create_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO reminder_deliveries (id, execution_id, member_id, channel, status, escalation_step, "
                "provider_message_id, sent_at, delivered_at, failed_at, failure_reason, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (delivery.id, delivery.execution_id, delivery.member_id, delivery.channel.value,
                 delivery.status.value, delivery.escalation_step, delivery.provider_message_id,
                 delivery.sent_at, delivery.delivered_at, delivery.failed_at, delivery.failure_reason,
                 delivery.created_at),
            )
            self._conn.commit()
        return delivery

    def update_delivery(self, delivery: ReminderDelivery) -> ReminderDelivery:
        with self._cur() as cur:
            cur.execute(
                "UPDATE reminder_deliveries SET status=%s, provider_message_id=%s, sent_at=%s, "
                "delivered_at=%s, failed_at=%s, failure_reason=%s WHERE id=%s",
                (delivery.status.value, delivery.provider_message_id, delivery.sent_at,
                 delivery.delivered_at, delivery.failed_at, delivery.failure_reason, delivery.id),
            )
            self._conn.commit()
        return delivery

    def get_deliveries_for_execution(self, execution_id: UUID) -> list[ReminderDelivery]:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, execution_id, member_id, channel, status, created_at, escalation_step, "
                "provider_message_id, sent_at, delivered_at, failed_at, failure_reason "
                "FROM reminder_deliveries WHERE execution_id = %s ORDER BY created_at",
                (execution_id,),
            )
            return [
                ReminderDelivery(
                    id=r[0], execution_id=r[1], member_id=r[2], channel=DeliveryChannel(r[3]),
                    status=DeliveryStatus(r[4]), created_at=r[5], escalation_step=r[6],
                    provider_message_id=r[7], sent_at=r[8], delivered_at=r[9],
                    failed_at=r[10], failure_reason=r[11],
                )
                for r in cur.fetchall()
            ]

    # ── Acknowledgements ────────────────────────────────────────────────

    def create_acknowledgement(self, ack: ReminderAcknowledgement) -> ReminderAcknowledgement:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO reminder_acknowledgements (id, delivery_id, member_id, method, acknowledged_at, note, snoozed_until) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (ack.id, ack.delivery_id, ack.member_id, ack.method.value, ack.acknowledged_at, ack.note, ack.snoozed_until),
            )
            self._conn.commit()
        return ack

    def get_acknowledgement(self, delivery_id: UUID) -> ReminderAcknowledgement | None:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, delivery_id, member_id, method, acknowledged_at, note, snoozed_until "
                "FROM reminder_acknowledgements WHERE delivery_id = %s",
                (delivery_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return ReminderAcknowledgement(
                id=row[0], delivery_id=row[1], member_id=row[2], method=AckMethod(row[3]),
                acknowledged_at=row[4], note=row[5], snoozed_until=row[6],
            )

    # ── Approval Records ───────────────────────────────────────────────

    def create_approval_record(self, record: ApprovalRecord) -> ApprovalRecord:
        with self._cur() as cur:
            cur.execute(
                "INSERT INTO reminder_approval_records (id, reminder_id, action, actor_id, reason, created_at, metadata) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (record.id, record.reminder_id, record.action.value, record.actor_id, record.reason, record.created_at, Json({})),
            )
            self._conn.commit()
        return record

    def get_approval_records(self, reminder_id: UUID) -> list[ApprovalRecord]:
        with self._cur() as cur:
            cur.execute(
                "SELECT id, reminder_id, action, actor_id, created_at, reason, metadata "
                "FROM reminder_approval_records WHERE reminder_id = %s ORDER BY created_at",
                (reminder_id,),
            )
            return [
                ApprovalRecord(
                    id=r[0], reminder_id=r[1], action=ApprovalAction(r[2]),
                    actor_id=r[3], created_at=r[4], reason=r[5], metadata=r[6] or {},
                )
                for r in cur.fetchall()
            ]

    # ── Preferences ─────────────────────────────────────────────────────

    def get_preference_rules(self, household_id: UUID, member_id: UUID | None = None) -> list[PreferenceRule]:
        with self._cur() as cur:
            query = "SELECT id, household_id, member_id, urgency, preferred_channel, fallback_channel, " \
                    "quiet_hours_override, active, created_at FROM preference_rules WHERE household_id = %s AND active = true"
            params: list = [household_id]
            if member_id:
                query += " AND (member_id = %s OR member_id IS NULL)"
                params.append(member_id)
            cur.execute(query, params)
            return [
                PreferenceRule(
                    id=r[0], household_id=r[1], member_id=r[2],
                    urgency=UrgencyLevel(r[3]) if r[3] else None,
                    preferred_channel=DeliveryChannel(r[4]),
                    fallback_channel=DeliveryChannel(r[5]) if r[5] else None,
                    quiet_hours_override=r[6], active=r[7], created_at=r[8],
                )
                for r in cur.fetchall()
            ]


# ── Row mappers ─────────────────────────────────────────────────────────────


def _row_to_reminder(row) -> Reminder:
    return Reminder(
        id=row[0], household_id=row[1], subject=row[2], body=row[3],
        urgency=UrgencyLevel(row[4]), source=ReminderSource(row[6]),
        state=ReminderState(row[10]), created_by=row[12],
        created_at=row[13], updated_at=row[14],
        intent=NotificationIntent(row[5]),
        source_agent=row[7], source_event_id=row[8], dedupe_key=row[9],
        requires_approval=row[11], metadata=row[15] or {},
    )


def _row_to_member(row) -> HouseholdMember:
    return HouseholdMember(
        id=row[0], household_id=row[1], name=row[2],
        role=MemberRole(row[3]), timezone=row[4], created_at=row[5],
        email=row[6], phone=row[7], slack_id=row[8],
        quiet_hours_start=row[9], quiet_hours_end=row[10],
        metadata=row[11] or {},
    )
