"""Slack API client for sending messages.

Uses httpx directly (no slack-sdk dependency) to stay aligned with the
existing Telnyx provider pattern.
"""

from __future__ import annotations

import httpx
from loguru import logger

from dacribagents.infrastructure.settings import get_settings

_BASE_URL = "https://slack.com/api"


class SlackClient:
    """Thin wrapper around the Slack Web API for chat.postMessage."""

    def __init__(self, bot_token: str | None = None) -> None:
        settings = get_settings()
        self.bot_token = bot_token or (
            settings.slack_bot_token.get_secret_value() if settings.slack_bot_token else ""
        )

    async def post_message(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> dict:
        """Post a message to a Slack channel, optionally in a thread."""
        payload: dict = {"channel": channel, "text": text}
        if thread_ts:
            payload["thread_ts"] = thread_ts

        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(
                f"{_BASE_URL}/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        data = resp.json()
        if not data.get("ok"):
            logger.error(f"Slack API error: {data.get('error')}")
        return data


_slack_client: SlackClient | None = None


def get_slack_client() -> SlackClient:
    """Singleton accessor."""
    global _slack_client  # noqa: PLW0603
    if _slack_client is None:
        _slack_client = SlackClient()
    return _slack_client
