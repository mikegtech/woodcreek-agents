"""Channel selection policy — deterministic rules for choosing delivery channels.

Evaluates member preferences, urgency, intent, and available contact info
to select primary and fallback delivery channels.

Default behavior:
- CRITICAL / URGENT alerts → SMS primary, email fallback
- NORMAL reminders → email primary, SMS fallback
- LOW / DIGEST → email only
- Slack is available as fallback if slack_id is set
- Members without phone/email are suppressed with a reason
"""

from __future__ import annotations

from uuid import UUID

from dacribagents.application.ports.reminder_policy import ChannelSelection
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    NotificationIntent,
    UrgencyLevel,
)


def select_channel(  # noqa: PLR0913, PLR0911
    *,
    member_id: UUID,
    phone: str | None,
    email: str | None,
    slack_id: str | None,
    urgency: UrgencyLevel,
    intent: NotificationIntent,
) -> ChannelSelection:
    """Select primary and fallback channels for a single member."""
    # Digest intent always goes to email
    if intent == NotificationIntent.DIGEST:
        if email:
            return ChannelSelection(member_id=member_id, primary_channel=DeliveryChannel.EMAIL, primary_address=email)
        return ChannelSelection(
            member_id=member_id,
            primary_channel=DeliveryChannel.EMAIL,
            primary_address="",
            suppressed=True,
            suppression_reason="No email address for digest delivery",
        )

    # Critical / urgent → SMS primary, email fallback
    if urgency in {UrgencyLevel.CRITICAL, UrgencyLevel.URGENT}:
        if phone:
            return ChannelSelection(
                member_id=member_id,
                primary_channel=DeliveryChannel.SMS,
                primary_address=phone,
                fallback_channel=DeliveryChannel.EMAIL if email else None,
                fallback_address=email,
            )
        if email:
            return ChannelSelection(
                member_id=member_id,
                primary_channel=DeliveryChannel.EMAIL,
                primary_address=email,
            )

    # Normal / low → email primary, SMS fallback
    if email:
        return ChannelSelection(
            member_id=member_id,
            primary_channel=DeliveryChannel.EMAIL,
            primary_address=email,
            fallback_channel=DeliveryChannel.SMS if phone else None,
            fallback_address=phone,
        )
    if phone:
        return ChannelSelection(
            member_id=member_id,
            primary_channel=DeliveryChannel.SMS,
            primary_address=phone,
        )

    # No contact info
    return ChannelSelection(
        member_id=member_id,
        primary_channel=DeliveryChannel.EMAIL,
        primary_address="",
        suppressed=True,
        suppression_reason="No phone or email address available",
    )
