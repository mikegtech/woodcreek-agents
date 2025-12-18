"""Telnyx SMS provider for sending outbound messages."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from loguru import logger


@dataclass
class SMSResult:
    """Result of sending an SMS."""

    success: bool
    message_id: str | None = None
    error: str | None = None
    carrier: str | None = None
    cost: str | None = None


class TelnyxProvider:
    """Telnyx SMS provider for sending outbound messages."""

    BASE_URL = "https://api.telnyx.com/v2"

    def __init__(
        self,
        api_key: str | None = None,
        default_from: str | None = None,
        messaging_profile_id: str | None = None,
    ):
        self.api_key = api_key or os.getenv("TELNYX_API_KEY")
        self.default_from = default_from or os.getenv("TELNYX_FROM_NUMBER")
        self.messaging_profile_id = messaging_profile_id or os.getenv("TELNYX_MESSAGING_PROFILE_ID")

        if not self.api_key:
            raise ValueError("TELNYX_API_KEY is required")

    async def send_sms(
        self,
        to: str,
        text: str,
        from_: str | None = None,
        messaging_profile_id: str | None = None,
    ) -> SMSResult:
        """Send an SMS message via Telnyx API."""
        from_number = from_ or self.default_from
        profile_id = messaging_profile_id or self.messaging_profile_id

        if not from_number:
            return SMSResult(success=False, error="No 'from' number specified")

        # Truncate if too long (SMS limit ~1600 chars for 10 segments)
        if len(text) > 1600:
            text = text[:1597] + "..."
            logger.warning(f"SMS truncated to 1600 chars for {to}")

        payload = {
            "from": from_number,
            "to": to,
            "text": text,
        }

        if profile_id:
            payload["messaging_profile_id"] = profile_id

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/messages",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    msg_data = data.get("data", {})
                    message_id = msg_data.get("id")
                    cost = msg_data.get("cost", {}).get("amount")

                    logger.info(f"SMS sent to {to}, message_id={message_id}")
                    return SMSResult(
                        success=True,
                        message_id=message_id,
                        cost=cost,
                    )
                else:
                    error_text = response.text
                    logger.error(f"Telnyx API error {response.status_code}: {error_text}")
                    return SMSResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {error_text[:200]}",
                    )

        except httpx.TimeoutException:
            logger.error(f"Telnyx API timeout sending to {to}")
            return SMSResult(success=False, error="Request timeout")
        except Exception as e:
            logger.error(f"Telnyx API exception: {e}")
            return SMSResult(success=False, error=str(e))

    async def send_reply(
        self,
        to: str,
        text: str,
        from_: str | None = None,
    ) -> SMSResult:
        """Convenience method for sending a reply."""
        return await self.send_sms(to=to, text=text, from_=from_)

    def send_sms_sync(
        self,
        to: str,
        text: str,
        from_: str | None = None,
        messaging_profile_id: str | None = None,
    ) -> SMSResult:
        """Synchronous version of send_sms for non-async contexts."""
        from_number = from_ or self.default_from
        profile_id = messaging_profile_id or self.messaging_profile_id

        if not from_number:
            return SMSResult(success=False, error="No 'from' number specified")

        if len(text) > 1600:
            text = text[:1597] + "..."
            logger.warning(f"SMS truncated to 1600 chars for {to}")

        payload = {
            "from": from_number,
            "to": to,
            "text": text,
        }

        if profile_id:
            payload["messaging_profile_id"] = profile_id

        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.BASE_URL}/messages",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    msg_data = data.get("data", {})
                    message_id = msg_data.get("id")
                    cost = msg_data.get("cost", {}).get("amount")

                    logger.info(f"SMS sent to {to}, message_id={message_id}")
                    return SMSResult(
                        success=True,
                        message_id=message_id,
                        cost=cost,
                    )
                else:
                    error_text = response.text
                    logger.error(f"Telnyx API error {response.status_code}: {error_text}")
                    return SMSResult(
                        success=False,
                        error=f"HTTP {response.status_code}: {error_text[:200]}",
                    )

        except httpx.TimeoutException:
            logger.error(f"Telnyx API timeout sending to {to}")
            return SMSResult(success=False, error="Request timeout")
        except Exception as e:
            logger.error(f"Telnyx API exception: {e}")
            return SMSResult(success=False, error=str(e))


# Singleton instance
_provider: TelnyxProvider | None = None


def get_telnyx_provider() -> TelnyxProvider:
    """Get or create Telnyx provider singleton."""
    global _provider
    if _provider is None:
        _provider = TelnyxProvider()
    return _provider