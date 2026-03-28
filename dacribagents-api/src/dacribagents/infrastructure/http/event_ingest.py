"""HTTP event intake endpoint for dev/testing without Kafka.

Accepts the same normalized event payload that the Kafka consumer
processes, routes it through the event-intake application service.

In production, events arrive via Kafka.  This endpoint allows
testing and manual event injection.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter()


class InboundEventPayload(BaseModel):
    """Normalized upstream event payload."""

    event_type: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    household_id: UUID
    timestamp: datetime
    subject: str = Field(min_length=1, max_length=500)
    body: str = ""
    severity: str = "normal"
    source_service: str = ""
    metadata: dict = Field(default_factory=dict)


@router.post("/internal/events/ingest")
async def ingest_event_http(
    payload: InboundEventPayload,
    background_tasks: BackgroundTasks,
) -> dict:
    """Ingest an upstream event and create a reminder if appropriate."""
    background_tasks.add_task(_process_event, payload)
    return {"status": "accepted", "event_id": payload.event_id}


async def _process_event(payload: InboundEventPayload) -> None:
    from dacribagents.application.services.event_intake import UpstreamEvent, process_upstream_event  # noqa: PLC0415
    from dacribagents.infrastructure.slack._store import get_household_store  # noqa: PLC0415

    store, _, _ = get_household_store()

    event = UpstreamEvent(
        event_type=payload.event_type,
        event_id=payload.event_id,
        household_id=payload.household_id,
        timestamp=payload.timestamp,
        subject=payload.subject,
        body=payload.body,
        severity=payload.severity,
        source_service=payload.source_service,
        metadata=payload.metadata,
    )

    result = process_upstream_event(store, event)
    logger.info(f"Event intake: {payload.event_type} → {result.status}")
