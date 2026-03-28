"""Digest aggregation and delivery service.

Collects eligible low-priority reminders into a daily digest email
instead of delivering them individually.

Eligibility rules (deterministic):
- ``intent == DIGEST``  → always eligible
- ``urgency == LOW`` and ``intent != ALERT``  → eligible
- ``urgency in {CRITICAL, URGENT}``  → never eligible
- Reminders already delivered or in terminal states → excluded

Digest window: reminders in PENDING_DELIVERY state that match eligibility.
The digest service claims eligible reminders, generates a formatted email,
delivers via the email adapter, and records delivery per-reminder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from loguru import logger

from dacribagents.application.ports.delivery_channel import DeliveryChannelAdapter, DeliveryResult
from dacribagents.application.ports.event_publisher import DomainEvent, EventPublisher, NoOpEventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.domain.reminders.entities import (
    Reminder,
    ReminderDelivery,
    ReminderExecution,
)
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    DeliveryStatus,
    NotificationIntent,
    ReminderState,
    UrgencyLevel,
)

# ── Digest batch record ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DigestBatch:
    """Record of a generated digest."""

    id: UUID
    household_id: UUID
    generated_at: datetime
    reminder_ids: tuple[UUID, ...]
    recipient_emails: tuple[str, ...]
    delivered: bool = False
    delivery_error: str | None = None


# ── Eligibility ─────────────────────────────────────────────────────────────


def is_digest_eligible(reminder: Reminder) -> bool:
    """Return True if this reminder should be aggregated into a digest."""
    if reminder.intent == NotificationIntent.DIGEST:
        return True
    if reminder.urgency == UrgencyLevel.LOW and reminder.intent != NotificationIntent.ALERT:
        return True
    return False


def collect_eligible(
    store: ReminderStore,
    household_id: UUID,
) -> list[Reminder]:
    """Return PENDING_DELIVERY reminders eligible for digest aggregation."""
    reminders, _ = store.list_reminders(household_id, states=[ReminderState.PENDING_DELIVERY])
    return [r for r in reminders if is_digest_eligible(r)]


# ── Digest generation and delivery ──────────────────────────────────────────


def generate_and_deliver(  # noqa: PLR0913
    store: ReminderStore,
    household_id: UUID,
    email_adapter: DeliveryChannelAdapter,
    *,
    events: EventPublisher | None = None,
    now: datetime | None = None,
) -> DigestBatch | None:
    """Collect eligible reminders, format a digest, and deliver via email.

    Returns the DigestBatch if a digest was generated, or None if no
    eligible reminders were found.
    """
    now = now or datetime.now(UTC)
    publisher = events or NoOpEventPublisher()

    eligible = collect_eligible(store, household_id)
    if not eligible:
        logger.debug("Digest: no eligible reminders")
        return None

    # Resolve recipient emails from household members
    members = store.get_household_members(household_id)
    emails = [m.email for m in members if m.email]
    if not emails:
        logger.warning("Digest: no email addresses in household")
        return None

    batch_id = uuid4()
    reminder_ids = tuple(r.id for r in eligible)

    # Format and deliver
    subject = f"Woodcreek Daily Digest — {now:%b %d, %Y}"
    body_text = _format_digest_text(eligible, now)
    body_html = _format_digest_html(eligible, now)

    delivered = False
    error = None

    for email_addr in emails:
        result: DeliveryResult = email_adapter.send(
            recipient_id=batch_id,
            recipient_address=email_addr,
            subject=subject,
            body=body_text,
            reminder_id=batch_id,
            urgency="low",
            metadata={"html_body": body_html},
        )

        if result.status in {DeliveryStatus.DELIVERED, DeliveryStatus.SENT}:
            delivered = True
        else:
            error = result.failure_reason

    # Mark each reminder as DELIVERED and create delivery records
    for r in eligible:
        execution = ReminderExecution(
            id=uuid4(), reminder_id=r.id,
            schedule_id=uuid4(), fired_at=now, created_at=now,
        )
        store.create_execution(execution)
        delivery = ReminderDelivery(
            id=uuid4(), execution_id=execution.id,
            member_id=batch_id, channel=DeliveryChannel.EMAIL,
            status=DeliveryStatus.DELIVERED if delivered else DeliveryStatus.FAILED,
            created_at=now,
            provider_message_id=f"digest-{batch_id}",
            sent_at=now if delivered else None,
            delivered_at=now if delivered else None,
            failure_reason=error if not delivered else None,
        )
        store.create_delivery(delivery)
        if delivered:
            store.update_reminder_state(r.id, ReminderState.DELIVERED)

    batch = DigestBatch(
        id=batch_id,
        household_id=household_id,
        generated_at=now,
        reminder_ids=reminder_ids,
        recipient_emails=tuple(emails),
        delivered=delivered,
        delivery_error=error,
    )

    publisher.publish(DomainEvent(
        event_type="digest.delivered" if delivered else "digest.delivery_failed",
        reminder_id=batch_id,
        household_id=household_id,
        timestamp=now,
        payload={"count": len(eligible), "recipients": len(emails)},
    ))

    logger.info(f"Digest generated: {len(eligible)} reminders, delivered={delivered}")
    return batch


# ── Formatting ──────────────────────────────────────────────────────────────

_URGENCY_LABEL = {UrgencyLevel.LOW: "low", UrgencyLevel.NORMAL: "normal"}


def _format_digest_text(reminders: list[Reminder], now: datetime) -> str:
    lines = [f"Woodcreek Daily Digest — {now:%B %d, %Y}", "=" * 40, ""]
    for i, r in enumerate(reminders, 1):
        lines.append(f"{i}. {r.subject}")
        if r.body:
            lines.append(f"   {r.body}")
        lines.append(f"   Source: {r.source.value} | Urgency: {r.urgency.value}")
        lines.append("")
    lines.append(f"Total: {len(reminders)} items")
    lines.append("---")
    lines.append("Woodcreek Household Reminders")
    return "\n".join(lines)


def _format_digest_html(reminders: list[Reminder], now: datetime) -> str:
    items = ""
    for r in reminders:
        body_html = f"<p style='color:#555;margin:0 0 4px 16px;'>{r.body}</p>" if r.body else ""
        items += f"""
        <div style="margin-bottom:16px;padding:8px 0;border-bottom:1px solid #eee;">
            <strong>{r.subject}</strong>
            {body_html}
            <p style="color:#999;font-size:12px;margin:4px 0 0 16px;">
                {r.source.value} &middot; {r.urgency.value}
            </p>
        </div>"""

    return f"""<html><body style="font-family:sans-serif;max-width:600px;margin:0 auto;">
<h2 style="color:#333;">Woodcreek Daily Digest</h2>
<p style="color:#666;">{now:%B %d, %Y} &middot; {len(reminders)} items</p>
{items}
<hr style="border:1px solid #eee;">
<p style="color:#888;font-size:12px;">Woodcreek Household Reminders</p>
</body></html>"""
