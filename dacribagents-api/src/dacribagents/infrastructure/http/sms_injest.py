from __future__ import annotations

import os
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional

router = APIRouter()

class InboundSmsEvent(BaseModel):
    provider: Literal["telnyx"]
    event_type: Literal["message.received"]
    telnyx_message_id: str
    from_number: str
    to_number: str
    text: str
    received_at: Optional[str] = None
    messaging_profile_id: Optional[str] = None
    organization_id: Optional[str] = None

class BatchPayload(BaseModel):
    source: Literal["cloudflare-queue"]
    env: str
    received_at: str
    events: List[InboundSmsEvent]

@router.post("/internal/sms/events")
def ingest_sms_events(
    payload: BatchPayload,
    x_sms_ingest_secret: str = Header(default="", convert_underscores=False),
):
    expected = os.getenv("SMS_INGEST_SECRET", "")
    if not expected or x_sms_ingest_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # For now: log and return. You will wire to SQLite + Milvus + LangGraph later.
    # Keep it fast; batch size is small.
    return {
        "status": "accepted",
        "count": len(payload.events),
        "env": payload.env,
    }
