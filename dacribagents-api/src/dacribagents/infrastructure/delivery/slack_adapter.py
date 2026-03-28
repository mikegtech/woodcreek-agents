"""Slack delivery adapter — implements DeliveryChannelAdapter for Slack.

Posts reminder/alert notifications to a configured Slack channel using
the existing ``SlackClient``.  This makes Slack a delivery channel in
addition to its operator/control-surface role.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from loguru import logger

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.domain.reminders.enums import DeliveryChannel, DeliveryStatus


class SlackDeliveryAdapter:
    """DeliveryChannelAdapter implementation backed by SlackClient."""

    def __init__(self, default_channel: str = "", bot_token: str | None = None) -> None:
        self._default_channel = default_channel
        self._bot_token = bot_token
        self._client = None

    def _get_client(self):
        if self._client is None:
            from dacribagents.infrastructure.slack.client import SlackClient  # noqa: PLC0415

            self._client = SlackClient(bot_token=self._bot_token)
        return self._client

    @property
    def channel(self) -> DeliveryChannel:  # noqa: D102
        return DeliveryChannel.SLACK

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
        """Post a reminder notification to Slack."""
        # recipient_address is slack_id or channel; fall back to default channel
        channel = recipient_address or self._default_channel
        if not channel:
            return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason="No Slack channel configured")

        text = _format_slack_notification(subject, body, urgency, str(reminder_id)[:8])

        try:
            client = self._get_client()
            # Run async post_message in sync context
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(client.post_message(channel=channel, text=text))
            finally:
                loop.close()

            if result.get("ok"):
                ts = result.get("ts", "")
                logger.info(f"Slack delivery to {channel}, ts={ts}")
                return DeliveryResult(status=DeliveryStatus.DELIVERED, provider_message_id=ts)

            error = result.get("error", "unknown")
            logger.error(f"Slack delivery failed: {error}")
            return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason=f"Slack API: {error}")

        except Exception as e:
            logger.error(f"Slack delivery exception: {e}")
            return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason=str(e))


def _format_slack_notification(subject: str, body: str, urgency: str, short_id: str) -> str:
    """Format a reminder as a Slack notification message."""
    icons = {"critical": ":rotating_light:", "urgent": ":warning:", "normal": ":bell:", "low": ":large_blue_circle:"}
    icon = icons.get(urgency, ":bell:")

    lines = [f"{icon} *{subject}*  `{short_id}`"]
    if body:
        lines.append(body)
    if urgency in {"critical", "urgent"}:
        lines.append(f"_Urgency: {urgency}_")
    return "\n".join(lines)
