"""SMS inbound reply handler — parses bounded commands and routes to workflows.

Supported commands:
- ``OK`` / ``DONE`` / ``ACK``  → acknowledge reminder
- ``SNOOZE 30``               → snooze for N minutes (default 30)
- ``CANCEL``                  → cancel reminder

Correlation: sender phone → household member → most recent SMS delivery
→ reminder_id.

Unknown replies are logged but do not corrupt state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from loguru import logger

from dacribagents.application.ports.event_publisher import EventPublisher
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services.acknowledgment import acknowledge_delivery
from dacribagents.application.use_cases.reminder_workflows import cancel_reminder, snooze_reminder
from dacribagents.domain.reminders.enums import AckMethod, DeliveryChannel, DeliveryStatus, ReminderState
from dacribagents.domain.reminders.lifecycle import InvalidTransitionError, can_transition
from dacribagents.domain.reminders.models import SnoozeReminderRequest


@dataclass(frozen=True)
class SmsReplyResult:
    """Outcome of processing an inbound SMS reply."""

    status: str  # "acknowledged", "snoozed", "cancelled", "unrecognized", "no_match", "error"
    reminder_id: UUID | None = None
    action: str | None = None
    reply_text: str = ""
    reason: str | None = None


# ── Command parsing ─────────────────────────────────────────────────────────

_ACK_COMMANDS = frozenset({"ok", "done", "ack", "got it", "yes", "acknowledged"})
_CANCEL_COMMANDS = frozenset({"cancel", "stop"})
_SNOOZE_RE = re.compile(r"^snooze\s*(\d+)?$", re.IGNORECASE)


def parse_sms_command(text: str) -> tuple[str, int]:
    """Parse an SMS reply into (action, snooze_minutes).

    Returns:
        ("ack", 0), ("snooze", N), ("cancel", 0), or ("unknown", 0).

    """
    cleaned = text.strip().lower()

    if cleaned in _ACK_COMMANDS:
        return ("ack", 0)

    if cleaned in _CANCEL_COMMANDS:
        return ("cancel", 0)

    snooze_match = _SNOOZE_RE.match(cleaned)
    if snooze_match:
        minutes = int(snooze_match.group(1) or "30")
        return ("snooze", max(1, min(minutes, 1440)))  # clamp 1min–24hr

    return ("unknown", 0)


# ── Reply processing ────────────────────────────────────────────────────────


def handle_sms_reply(  # noqa: PLR0913, PLR0911
    store: ReminderStore,
    *,
    from_number: str,
    text: str,
    provider_message_id: str | None = None,
    events: EventPublisher | None = None,
) -> SmsReplyResult:
    """Process an inbound SMS reply and route to the appropriate workflow."""
    action, snooze_minutes = parse_sms_command(text)

    if action == "unknown":
        logger.info(f"Unrecognized SMS reply from {from_number}: {text!r}")
        return SmsReplyResult(status="unrecognized", reply_text=text, reason=f"Unrecognized command: {text!r}")

    # Correlate: phone → member → most recent SMS delivery → reminder
    match = _correlate_reply(store, from_number)
    if match is None:
        logger.warning(f"No delivery match for SMS from {from_number}")
        return SmsReplyResult(status="no_match", reply_text=text, reason="No recent SMS delivery found for this number")

    member_id, reminder_id, delivery_id = match

    try:
        if action == "ack":
            acknowledge_delivery(
                store,
                reminder_id=reminder_id,
                member_id=member_id,
                method=AckMethod.SMS_REPLY,
                delivery_id=delivery_id,
                note=f"SMS reply: {text}",
                events=events,
            )
            return SmsReplyResult(status="acknowledged", reminder_id=reminder_id, action="ack", reply_text=text)

        if action == "snooze":
            snooze_until = datetime.now(UTC) + timedelta(minutes=snooze_minutes)
            snooze_reminder(
                store, reminder_id,
                SnoozeReminderRequest(
                    member_id=member_id,
                    method=AckMethod.SMS_REPLY,
                    snooze_until=snooze_until,
                    note=f"SMS reply: {text}",
                ),
            )
            return SmsReplyResult(status="snoozed", reminder_id=reminder_id, action="snooze", reply_text=text)

        if action == "cancel":
            cancel_reminder(store, reminder_id)
            return SmsReplyResult(status="cancelled", reminder_id=reminder_id, action="cancel", reply_text=text)

    except InvalidTransitionError as e:
        logger.warning(f"SMS reply action {action} invalid for reminder {reminder_id}: {e}")
        return SmsReplyResult(
            status="error", reminder_id=reminder_id, action=action,
            reply_text=text, reason=f"Invalid transition: {e}",
        )

    return SmsReplyResult(status="error", reply_text=text, reason="Unexpected state")


# ── Correlation ─────────────────────────────────────────────────────────────


def _correlate_reply(
    store: ReminderStore,
    phone: str,
) -> tuple[UUID, UUID, UUID] | None:
    """Find (member_id, reminder_id, delivery_id) for an inbound SMS.

    Strategy: find the member by phone number, then scan for their most
    recent SMS delivery in a DELIVERED or SENT state.
    """
    phone_normalized = _normalize_phone(phone)

    # Find member by phone
    member = None
    for m in getattr(store, "members", {}).values():
        if m.phone and _normalize_phone(m.phone) == phone_normalized:
            member = m
            break
    if member is None:
        return None

    # Find most recent SMS delivery for this member
    best_delivery = None
    best_reminder_id = None

    for exec_id, execution in getattr(store, "executions", {}).items():
        deliveries = store.get_deliveries_for_execution(exec_id)
        for d in deliveries:
            if d.member_id != member.id:
                continue
            if d.channel != DeliveryChannel.SMS:
                continue
            if d.status not in {DeliveryStatus.SENT, DeliveryStatus.DELIVERED}:
                continue
            # Check if the parent reminder is in a state that can be acted on
            reminder = store.get_reminder(execution.reminder_id)
            if reminder is None:
                continue
            if not can_transition(reminder.state, ReminderState.ACKNOWLEDGED):
                continue
            if best_delivery is None or d.created_at > best_delivery.created_at:
                best_delivery = d
                best_reminder_id = execution.reminder_id

    if best_delivery and best_reminder_id:
        return (member.id, best_reminder_id, best_delivery.id)
    return None


def _normalize_phone(phone: str) -> str:
    """Strip non-digits for comparison."""
    return "".join(c for c in phone if c.isdigit())
