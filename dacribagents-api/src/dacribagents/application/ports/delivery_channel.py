"""Port for provider-neutral delivery channel adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from dacribagents.domain.reminders.enums import DeliveryChannel, DeliveryStatus


@dataclass(frozen=True)
class DeliveryResult:
    """Outcome of a single delivery attempt."""

    status: DeliveryStatus
    provider_message_id: str | None = None
    failure_reason: str | None = None


class DeliveryChannelAdapter(Protocol):
    """Adapter for sending reminders through a specific channel.

    Implementations: Telnyx SMS, woodcreek.me email, Slack, APNs push, in-app.
    Each adapter is registered by its DeliveryChannel enum value.
    """

    @property
    def channel(self) -> DeliveryChannel:
        """Which channel this adapter serves."""
        ...

    def send(  # noqa: PLR0913
        self,
        *,
        recipient_id: UUID,
        recipient_address: str,
        subject: str,
        body: str,
        reminder_id: UUID,
        urgency: str,
        metadata: dict[str, str] | None = None,
    ) -> DeliveryResult:
        """Attempt to deliver a reminder message.

        Args:
            recipient_id: HouseholdMember ID.
            recipient_address: Channel-specific address (phone, email, Slack ID, device token).
            subject: Reminder subject line.
            body: Reminder body text.
            reminder_id: For correlation/threading.
            urgency: Urgency level as string.
            metadata: Optional provider-specific metadata.

        Returns:
            DeliveryResult with status and optional provider message ID.

        """
        ...
