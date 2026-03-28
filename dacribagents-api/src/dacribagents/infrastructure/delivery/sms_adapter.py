"""Telnyx SMS delivery adapter — implements DeliveryChannelAdapter for SMS.

Wraps the existing ``TelnyxProvider`` from ``infrastructure/sms/providers/telnyx``.
"""

from __future__ import annotations

from uuid import UUID

from loguru import logger

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.domain.reminders.enums import DeliveryChannel, DeliveryStatus


class TelnyxSmsAdapter:
    """DeliveryChannelAdapter implementation backed by TelnyxProvider."""

    def __init__(self, provider: object | None = None) -> None:
        self._provider = provider

    @property
    def channel(self) -> DeliveryChannel:  # noqa: D102
        return DeliveryChannel.SMS

    def _get_provider(self):
        if self._provider is None:
            from dacribagents.infrastructure.sms.providers.telnyx.outbound import get_telnyx_provider  # noqa: PLC0415
            self._provider = get_telnyx_provider()
        return self._provider

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
        """Send an SMS reminder via Telnyx."""
        text = _format_sms(subject, body, urgency)
        provider = self._get_provider()

        try:
            # Use sync path — the delivery dispatcher is called from sync context.
            result = provider.send_sms_sync(to=recipient_address, text=text)
        except Exception as e:
            logger.error(f"Telnyx SMS failed for {recipient_address}: {e}")
            return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason=str(e))

        if result.success:
            logger.info(f"SMS delivered to {recipient_address}, id={result.message_id}")
            return DeliveryResult(
                status=DeliveryStatus.SENT,
                provider_message_id=result.message_id,
            )

        return DeliveryResult(
            status=DeliveryStatus.FAILED,
            failure_reason=result.error or "Unknown Telnyx error",
        )


def _format_sms(subject: str, body: str, urgency: str) -> str:
    """Format a reminder into SMS-safe plain text."""
    prefix = "[URGENT] " if urgency in {"critical", "urgent"} else ""
    text = f"{prefix}{subject}"
    if body:
        text += f"\n{body}"
    text += "\n\nReply OK to acknowledge."
    return text[:1600]
