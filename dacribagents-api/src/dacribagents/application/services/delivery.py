"""Delivery dispatcher — moves PENDING_DELIVERY reminders through outbound channels.

Orchestration flow:
1. Find reminders in PENDING_DELIVERY state.
2. Resolve target members from ReminderTarget records.
3. Evaluate channel policy per member (urgency, intent, contact info).
4. Dispatch via registered DeliveryChannelAdapters.
5. Record ReminderDelivery with provider result.
6. Transition reminder to DELIVERED or FAILED.
7. Publish domain events via EventPublisher seam (Kafka in production).

This is an application service — it depends on ports, not infrastructure.
LangGraph can orchestrate this as a durable workflow in Phase 4+.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from loguru import logger

from dacribagents.application.ports.delivery_channel import DeliveryChannelAdapter, DeliveryResult
from dacribagents.application.ports.event_publisher import DomainEvent, EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services import channel_policy, governance
from dacribagents.domain.reminders.entities import (
    Reminder,
    ReminderDelivery,
    ReminderExecution,
)
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
    ReminderState,
    TargetType,
)


class DeliveryDispatcher:
    """Dispatches PENDING_DELIVERY reminders through registered channel adapters."""

    def __init__(
        self,
        store: ReminderStore,
        adapters: dict[DeliveryChannel, DeliveryChannelAdapter],
        events: EventPublisher | None = None,
    ) -> None:
        self.store = store
        self.adapters = adapters
        self.events: EventPublisher = events or NoOpEventPublisher()

    def dispatch_pending(self, household_id: UUID) -> list[ReminderDelivery]:
        """Find all PENDING_DELIVERY reminders and dispatch them."""
        reminders, _ = self.store.list_reminders(household_id, states=[ReminderState.PENDING_DELIVERY])
        results: list[ReminderDelivery] = []
        for reminder in reminders:
            results.extend(self._dispatch_reminder(reminder))
        return results

    def dispatch_one(self, reminder_id: UUID) -> list[ReminderDelivery]:
        """Dispatch a specific reminder by ID."""
        reminder = self.store.get_reminder(reminder_id)
        if reminder is None or reminder.state != ReminderState.PENDING_DELIVERY:
            return []
        return self._dispatch_reminder(reminder)

    def _dispatch_reminder(self, reminder: Reminder) -> list[ReminderDelivery]:
        """Resolve targets, select channels, send, record results."""
        # Governance gate: check if autonomous dispatch is allowed
        gov_decision = governance.evaluate(
            household_id=reminder.household_id,
            action_type="auto_send",
            actor_type="system",
            requires_tier=governance.AutonomyTier.TIER_2 if governance.is_tier2_auto_eligible(reminder.source)
            else governance.AutonomyTier.TIER_1,
            reminder_id=reminder.id,
        )
        if not gov_decision.allowed and gov_decision.requires_approval:
            logger.info(f"Governance blocked dispatch for {reminder.id}: {gov_decision.reason}")
            return []

        targets = self.store.get_targets(reminder.id)
        member_ids = self._resolve_member_ids(reminder.household_id, targets)

        if not member_ids:
            logger.warning(f"No resolvable members for reminder {reminder.id}")
            return []

        execution = self._create_execution(reminder)
        deliveries: list[ReminderDelivery] = []
        any_success = False

        for mid in member_ids:
            member = self.store.get_member(mid)
            if member is None:
                continue

            # Member mute check: skip non-critical if muted
            from dacribagents.domain.reminders.enums import UrgencyLevel  # noqa: PLC0415

            if governance.is_member_muted(mid) and reminder.urgency not in {UrgencyLevel.CRITICAL, UrgencyLevel.URGENT}:
                logger.info(f"Delivery skipped for muted member {member.name}: {governance.get_mute_reason(mid)}")
                continue

            rules = self.store.get_preference_rules(reminder.household_id, member.id)

            now_time = datetime.now(UTC).time()
            selection = channel_policy.select_channel(
                member_id=member.id,
                phone=member.phone,
                email=member.email,
                slack_id=member.slack_id,
                urgency=reminder.urgency,
                intent=reminder.intent,
                preference_rules=rules,
                quiet_hours_start=member.quiet_hours_start,
                quiet_hours_end=member.quiet_hours_end,
                current_time=now_time,
            )

            if selection.suppressed:
                logger.warning(f"Delivery suppressed for {member.name}: {selection.suppression_reason}")
                continue

            delivery = self._send(
                reminder=reminder,
                execution=execution,
                member_id=member.id,
                channel=selection.primary_channel,
                address=selection.primary_address,
            )
            deliveries.append(delivery)
            if delivery.status in {DeliveryStatus.DELIVERED, DeliveryStatus.SENT}:
                any_success = True

        # State transition
        if any_success:
            self.store.update_reminder_state(reminder.id, ReminderState.DELIVERED)
            self.events.publish(DomainEvent(
                event_type="reminder.delivered",
                reminder_id=reminder.id,
                household_id=reminder.household_id,
                timestamp=datetime.now(UTC),
                payload={"channels": [d.channel.value for d in deliveries]},
            ))
        elif deliveries:
            self.store.update_reminder_state(reminder.id, ReminderState.FAILED)
            self.events.publish(DomainEvent(
                event_type="reminder.delivery_failed",
                reminder_id=reminder.id,
                household_id=reminder.household_id,
                timestamp=datetime.now(UTC),
            ))

        return deliveries

    def _resolve_member_ids(self, household_id: UUID, targets: list) -> list[UUID]:
        member_ids: list[UUID] = []
        for t in targets:
            if t.target_type == TargetType.INDIVIDUAL and t.member_id:
                member_ids.append(t.member_id)
            elif t.target_type == TargetType.HOUSEHOLD:
                member_ids.extend(m.id for m in self.store.get_household_members(household_id))
            elif t.target_type == TargetType.ROLE and t.role:
                member_ids.extend(
                    m.id for m in self.store.get_household_members(household_id) if m.role == t.role
                )
        return list(dict.fromkeys(member_ids))  # dedupe, preserve order

    def _create_execution(self, reminder: Reminder) -> ReminderExecution:
        schedule = self.store.get_schedule(reminder.id)
        now = datetime.now(UTC)
        execution = ReminderExecution(
            id=uuid4(),
            reminder_id=reminder.id,
            schedule_id=schedule.id if schedule else uuid4(),
            fired_at=now,
            created_at=now,
        )
        return self.store.create_execution(execution)

    def _send(  # noqa: PLR0913
        self,
        *,
        reminder: Reminder,
        execution: ReminderExecution,
        member_id: UUID,
        channel: DeliveryChannel,
        address: str,
    ) -> ReminderDelivery:
        now = datetime.now(UTC)
        delivery = ReminderDelivery(
            id=uuid4(),
            execution_id=execution.id,
            member_id=member_id,
            channel=channel,
            status=DeliveryStatus.QUEUED,
            created_at=now,
        )
        delivery = self.store.create_delivery(delivery)

        adapter = self.adapters.get(channel)
        if adapter is None:
            logger.error(f"No adapter for channel {channel.value}")
            return self.store.update_delivery(replace(
                delivery, status=DeliveryStatus.FAILED, failed_at=now,
                failure_reason=f"No adapter for {channel.value}",
            ))

        try:
            result: DeliveryResult = adapter.send(
                recipient_id=member_id,
                recipient_address=address,
                subject=reminder.subject,
                body=reminder.body,
                reminder_id=reminder.id,
                urgency=reminder.urgency.value,
            )
        except Exception as e:
            logger.exception(f"Adapter {channel.value} raised: {e}")
            return self.store.update_delivery(replace(
                delivery, status=DeliveryStatus.FAILED,
                failed_at=datetime.now(UTC), failure_reason=str(e),
            ))

        if result.status in {DeliveryStatus.DELIVERED, DeliveryStatus.SENT}:
            return self.store.update_delivery(replace(
                delivery,
                status=result.status,
                provider_message_id=result.provider_message_id,
                sent_at=now,
                delivered_at=now if result.status == DeliveryStatus.DELIVERED else None,
            ))

        return self.store.update_delivery(replace(
            delivery, status=DeliveryStatus.FAILED,
            provider_message_id=result.provider_message_id,
            failed_at=now, failure_reason=result.failure_reason,
        ))
