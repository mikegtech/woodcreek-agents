"""Slack Events API endpoint.

Handles:
- ``url_verification`` challenge during Slack app setup
- ``event_callback`` with ``app_mention`` events for @woodcreek queries
- Request signature validation via HMAC-SHA256

Follows the same pattern as ``sms_ingest.py``: validate, acknowledge fast,
process in background.
"""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from loguru import logger

from dacribagents.infrastructure.settings import get_settings

router = APIRouter()


# ── Signature verification ──────────────────────────────────────────────────


def _verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str,
) -> bool:
    """Verify Slack request signature (HMAC-SHA256)."""
    if abs(time.time() - int(timestamp)) > 300:
        return False
    sig_basestring = f"v0:{timestamp}:{body.decode()}"
    expected = "v0=" + hmac.new(
        signing_secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── Background processing ──────────────────────────────────────────────────


async def _process_app_mention(
    channel: str,
    text: str,
    user: str,
    thread_ts: str | None,
    event_ts: str,
) -> None:
    """Background task: route an @woodcreek mention through the command handler."""
    # Lazy import to avoid circular deps and allow testing with stubs.
    from dacribagents.infrastructure.slack._store import get_household_store  # noqa: PLC0415
    from dacribagents.infrastructure.slack.client import get_slack_client  # noqa: PLC0415
    from dacribagents.infrastructure.slack.handler import SlackCommandHandler  # noqa: PLC0415

    try:
        store, calendar, household_id = get_household_store()
        handler = SlackCommandHandler(store=store, calendar=calendar, household_id=household_id)
        response = handler.handle(text)

        client = get_slack_client()
        await client.post_message(
            channel=channel,
            text=response.text,
            thread_ts=thread_ts or event_ts,
        )
        logger.info(f"Slack response sent to {channel} for user {user}")
    except Exception as e:
        logger.exception(f"Failed to process Slack mention from {user}: {e}")


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.post("/internal/slack/events")
async def slack_events(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    """Receive Slack Events API callbacks.

    Handles url_verification and app_mention events.
    """
    body = await request.body()
    settings = get_settings()

    # Signature validation (skip in dev if no signing secret configured)
    signing_secret = settings.slack_signing_secret.get_secret_value() if settings.slack_signing_secret else ""
    if signing_secret:
        ts = request.headers.get("x-slack-request-timestamp", "")
        sig = request.headers.get("x-slack-signature", "")
        if not _verify_slack_signature(body, ts, sig, signing_secret):
            raise HTTPException(status_code=401, detail="Invalid Slack signature")

    payload = await request.json()

    # URL verification challenge (Slack app setup)
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    # Event callback
    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        event_type = event.get("type")

        if event_type == "app_mention":
            background_tasks.add_task(
                _process_app_mention,
                channel=event.get("channel", ""),
                text=event.get("text", ""),
                user=event.get("user", ""),
                thread_ts=event.get("thread_ts"),
                event_ts=event.get("ts", ""),
            )
            return {"status": "accepted"}

        logger.debug(f"Unhandled Slack event type: {event_type}")

    return {"status": "ok"}


# ── Health ──────────────────────────────────────────────────────────────────


@router.get("/internal/slack/health")
async def slack_health() -> dict:
    """Health check for Slack subsystem."""
    settings = get_settings()
    return {
        "status": "healthy",
        "bot_token_configured": bool(settings.slack_bot_token),
        "signing_secret_configured": bool(settings.slack_signing_secret),
    }
