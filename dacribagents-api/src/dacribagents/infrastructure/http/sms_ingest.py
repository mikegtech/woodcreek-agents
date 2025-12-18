"""SMS ingest endpoint for receiving events from Cloudflare Queue consumer."""

from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from loguru import logger
from pydantic import BaseModel


router = APIRouter()


# ============================================================================
# Request Models
# ============================================================================


class InboundSmsEvent(BaseModel):
    """A single inbound SMS event from Telnyx via Cloudflare."""

    provider: Literal["telnyx"]
    event_type: Literal["message.received"]
    telnyx_message_id: str
    from_number: str
    to_number: str
    text: str
    received_at: str | None = None
    messaging_profile_id: str | None = None
    organization_id: str | None = None


class BatchPayload(BaseModel):
    """Batch of SMS events from Cloudflare Queue consumer."""

    source: Literal["cloudflare-queue"]
    env: str
    received_at: str
    events: list[InboundSmsEvent]


# ============================================================================
# Background Task
# ============================================================================


async def _process_sms_event(event: InboundSmsEvent) -> None:
    """Background task to process a single SMS event."""
    try:
        from dacribagents.application.use_cases.process_sms import process_sms_event

        result = await process_sms_event(event.model_dump())
        logger.info(
            f"SMS processed: {event.from_number} -> {result.get('status')}",
            extra={"result": result},
        )
    except Exception as e:
        logger.exception(f"Failed to process SMS from {event.from_number}: {e}")


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/internal/sms/events")
async def ingest_sms_events(
    payload: BatchPayload,
    background_tasks: BackgroundTasks,
    x_sms_ingest_secret: str = Header(default="", alias="x-sms-ingest-secret"),
) -> dict:
    """
    Receive SMS events from Cloudflare Queue consumer.

    This endpoint:
    1. Validates the shared secret
    2. Queues each event for background processing
    3. Returns immediately (Cloudflare consumer expects fast 2xx)

    Background processing:
    - Stores inbound message in SQLite
    - Generates agent response (TODO: currently echoes)
    - Sends reply via Telnyx
    - Stores outbound message in SQLite
    """
    # Validate secret
    expected = os.getenv("SMS_INGEST_SECRET", "")
    if not expected or x_sms_ingest_secret != expected:
        logger.warning("SMS ingest unauthorized attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Log receipt
    logger.info(
        f"SMS batch received: {len(payload.events)} events from {payload.source}",
        extra={
            "source": payload.source,
            "env": payload.env,
            "received_at": payload.received_at,
            "event_count": len(payload.events),
        },
    )

    # Queue each event for background processing
    for event in payload.events:
        background_tasks.add_task(_process_sms_event, event)
        logger.debug(f"Queued SMS for processing: {event.telnyx_message_id}")

    # Return immediately - Cloudflare consumer expects fast 2xx
    return {
        "status": "accepted",
        "count": len(payload.events),
        "env": payload.env,
    }


# ============================================================================
# Health check for SMS subsystem
# ============================================================================


@router.get("/internal/sms/health")
async def sms_health() -> dict:
    """Health check for SMS subsystem."""
    from dacribagents.infrastructure.sqlite.client import get_sqlite_client

    try:
        # Check SQLite
        sqlite = get_sqlite_client()
        # Simple query to verify connection
        with sqlite._connection() as conn:
            conn.execute("SELECT 1")

        return {
            "status": "healthy",
            "sqlite": "connected",
            "telnyx": "configured" if os.getenv("TELNYX_API_KEY") else "missing_api_key",
        }
    except Exception as e:
        logger.error(f"SMS health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }