"""Escalation and retry service for unacknowledged reminders.

Retry: re-attempt delivery on the same channel after transient failure.
Escalation: re-deliver via fallback channel or to another member after ack timeout.

Backoff: ``base_delay * 2^attempt`` capped at ``max_delay``.
Limits: ``max_retries`` per delivery, ``max_escalation_steps`` per reminder.

This is a pure application service.  The control-loop driver calls
``check_escalations()`` on each tick.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from dacribagents.application.ports.delivery_channel import DeliveryChannelAdapter
from dacribagents.application.ports.event_publisher import DomainEvent, EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services import channel_policy, governance
from dacribagents.domain.reminders.entities import ReminderDelivery, ReminderExecution
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
    ReminderState,
)

# ── Configuration ───────────────────────────────────────────────────────────

MAX_RETRIES = 3
BASE_RETRY_DELAY = timedelta(minutes=2)
MAX_RETRY_DELAY = timedelta(minutes=30)

ACK_TIMEOUT_NORMAL = timedelta(hours=2)
ACK_TIMEOUT_URGENT = timedelta(minutes=30)

MAX_ESCALATION_STEPS = 2

# Failures that are worth retrying (transient)
_RETRIABLE_REASONS = frozenset({"request timeout", "connection refused", "rate limit", "503", "502", "429"})


def compute_backoff(attempt: int) -> timedelta:
    """Exponential backoff: ``base * 2^attempt``, capped at max."""
    delay = BASE_RETRY_DELAY * (2**attempt)
    return min(delay, MAX_RETRY_DELAY)


def is_retriable(failure_reason: str | None) -> bool:
    """Return True if the failure reason suggests a transient issue."""
    if not failure_reason:
        return False
    reason_lower = failure_reason.lower()
    return any(kw in reason_lower for kw in _RETRIABLE_REASONS)


def check_retries(
    store: ReminderStore,
    adapters: dict[DeliveryChannel, DeliveryChannelAdapter],
    household_id: UUID,
    now: datetime | None = None,
) -> list[ReminderDelivery]:
    """Find failed deliveries eligible for retry and re-attempt them."""
    now = now or datetime.now(UTC)
    retried: list[ReminderDelivery] = []

    reminders, _ = store.list_reminders(household_id, states=[ReminderState.FAILED])
    for reminder in reminders:
        for exec_id, execution in getattr(store, "executions", {}).items():
            if execution.reminder_id != reminder.id:
                continue
            deliveries = store.get_deliveries_for_execution(exec_id)
            for d in deliveries:
                if d.status != DeliveryStatus.FAILED:
                    continue
                if not is_retriable(d.failure_reason):
                    continue
                if d.escalation_step >= MAX_RETRIES:
                    continue
                # Check backoff
                retry_after = (d.failed_at or d.created_at) + compute_backoff(d.escalation_step)
                if now < retry_after:
                    continue

                # Retry on same channel
                adapter = adapters.get(d.channel)
                if adapter is None:
                    continue

                member = store.get_member(d.member_id)
                if member is None:
                    continue

                address = channel_policy._address_for(d.channel, member.phone, member.email, member.slack_id)
                if not address:
                    continue

                new_delivery = _retry_delivery(store, adapter, reminder, execution, d, member.id, address, now)
                retried.append(new_delivery)

                if new_delivery.status in {DeliveryStatus.SENT, DeliveryStatus.DELIVERED}:
                    store.update_reminder_state(reminder.id, ReminderState.DELIVERED)

    return retried


def check_escalations(  # noqa: PLR0913
    store: ReminderStore,
    adapters: dict[DeliveryChannel, DeliveryChannelAdapter],
    household_id: UUID,
    now: datetime | None = None,
    events: EventPublisher | None = None,
) -> list[ReminderDelivery]:
    """Escalate unacknowledged DELIVERED reminders via fallback channel or member."""
    now = now or datetime.now(UTC)
    publisher = events or NoOpEventPublisher()
    escalated: list[ReminderDelivery] = []

    reminders, _ = store.list_reminders(household_id, states=[ReminderState.DELIVERED])
    for reminder in reminders:
        # Governance gate: check if autonomous escalation is allowed
        gov_decision = governance.evaluate(
            household_id=household_id,
            action_type="auto_escalate",
            actor_type="system",
            requires_tier=governance.AutonomyTier.TIER_3,
            reminder_id=reminder.id,
        )
        if not gov_decision.allowed:
            continue

        # Determine ack timeout based on urgency
        from dacribagents.domain.reminders.enums import UrgencyLevel  # noqa: PLC0415

        timeout = ACK_TIMEOUT_URGENT if reminder.urgency in {UrgencyLevel.CRITICAL, UrgencyLevel.URGENT} else ACK_TIMEOUT_NORMAL

        # Check if delivered long enough ago
        if now - reminder.updated_at < timeout:
            continue

        # Check existing escalation count
        all_deliveries = _get_all_deliveries(store, reminder.id)
        max_step = max((d.escalation_step for d in all_deliveries), default=0)
        if max_step >= MAX_ESCALATION_STEPS:
            continue

        # Try fallback channel first
        last = all_deliveries[-1] if all_deliveries else None
        targets = store.get_targets(reminder.id)
        member_ids = _resolve_targets(store, reminder.household_id, targets)

        for mid in member_ids:
            member = store.get_member(mid)
            if not member:
                continue

            selection = channel_policy.select_channel(
                member_id=member.id,
                phone=member.phone,
                email=member.email,
                slack_id=member.slack_id,
                urgency=reminder.urgency,
                intent=reminder.intent,
                preference_rules=store.get_preference_rules(reminder.household_id, member.id),
            )

            # Use fallback channel if available and different from last attempt
            channel = selection.fallback_channel or selection.primary_channel
            if last and channel == last.channel and selection.fallback_channel:
                channel = selection.primary_channel  # flip to primary if fallback was used last

            address = channel_policy._address_for(channel, member.phone, member.email, member.slack_id)
            if not address or selection.suppressed:
                continue

            adapter = adapters.get(channel)
            if not adapter:
                continue

            execution = _ensure_execution(store, reminder)
            new_delivery = _escalate_delivery(store, adapter, reminder, execution, mid, channel, address, max_step + 1, now)
            escalated.append(new_delivery)

            if new_delivery.status in {DeliveryStatus.SENT, DeliveryStatus.DELIVERED}:
                publisher.publish(DomainEvent(
                    event_type="reminder.escalation_triggered",
                    reminder_id=reminder.id,
                    household_id=reminder.household_id,
                    timestamp=now,
                    payload={"channel": channel.value, "escalation_step": max_step + 1},
                ))
            break  # one escalation per reminder per tick

    return escalated


# ── Internal helpers ────────────────────────────────────────────────────────


def _retry_delivery(store, adapter, reminder, execution, failed, member_id, address, now):  # noqa: PLR0913
    new_delivery = ReminderDelivery(
        id=uuid4(),
        execution_id=execution.id,
        member_id=member_id,
        channel=failed.channel,
        status=DeliveryStatus.QUEUED,
        created_at=now,
        escalation_step=failed.escalation_step + 1,
    )
    new_delivery = store.create_delivery(new_delivery)

    try:
        result = adapter.send(
            recipient_id=member_id, recipient_address=address,
            subject=reminder.subject, body=reminder.body,
            reminder_id=reminder.id, urgency=reminder.urgency.value,
        )
    except Exception as e:
        return store.update_delivery(replace(
            new_delivery, status=DeliveryStatus.FAILED, failed_at=now, failure_reason=str(e),
        ))

    if result.status in {DeliveryStatus.SENT, DeliveryStatus.DELIVERED}:
        return store.update_delivery(replace(
            new_delivery, status=result.status, provider_message_id=result.provider_message_id,
            sent_at=now, delivered_at=now if result.status == DeliveryStatus.DELIVERED else None,
        ))
    return store.update_delivery(replace(
        new_delivery, status=DeliveryStatus.FAILED, failed_at=now, failure_reason=result.failure_reason,
    ))


def _escalate_delivery(store, adapter, reminder, execution, member_id, channel, address, step, now):  # noqa: PLR0913
    new_delivery = ReminderDelivery(
        id=uuid4(),
        execution_id=execution.id,
        member_id=member_id,
        channel=channel,
        status=DeliveryStatus.QUEUED,
        created_at=now,
        escalation_step=step,
    )
    new_delivery = store.create_delivery(new_delivery)

    try:
        result = adapter.send(
            recipient_id=member_id, recipient_address=address,
            subject=reminder.subject, body=reminder.body,
            reminder_id=reminder.id, urgency=reminder.urgency.value,
        )
    except Exception as e:
        return store.update_delivery(replace(
            new_delivery, status=DeliveryStatus.FAILED, failed_at=now, failure_reason=str(e),
        ))

    if result.status in {DeliveryStatus.SENT, DeliveryStatus.DELIVERED}:
        return store.update_delivery(replace(
            new_delivery, status=result.status, provider_message_id=result.provider_message_id,
            sent_at=now, delivered_at=now if result.status == DeliveryStatus.DELIVERED else None,
        ))
    return store.update_delivery(replace(
        new_delivery, status=DeliveryStatus.FAILED, failed_at=now, failure_reason=result.failure_reason,
    ))


def _get_all_deliveries(store, reminder_id):
    deliveries = []
    for eid, ex in getattr(store, "executions", {}).items():
        if ex.reminder_id == reminder_id:
            deliveries.extend(store.get_deliveries_for_execution(eid))
    return deliveries


def _resolve_targets(store, household_id, targets):
    from dacribagents.domain.reminders.enums import TargetType  # noqa: PLC0415

    ids = []
    for t in targets:
        if t.target_type == TargetType.INDIVIDUAL and t.member_id:
            ids.append(t.member_id)
        elif t.target_type == TargetType.HOUSEHOLD:
            ids.extend(m.id for m in store.get_household_members(household_id))
        elif t.target_type == TargetType.ROLE and t.role:
            ids.extend(m.id for m in store.get_household_members(household_id) if m.role == t.role)
    return list(dict.fromkeys(ids))


def _ensure_execution(store, reminder):
    schedule = store.get_schedule(reminder.id)
    now = datetime.now(UTC)
    execution = ReminderExecution(
        id=uuid4(), reminder_id=reminder.id,
        schedule_id=schedule.id if schedule else uuid4(),
        fired_at=now, created_at=now,
    )
    return store.create_execution(execution)
