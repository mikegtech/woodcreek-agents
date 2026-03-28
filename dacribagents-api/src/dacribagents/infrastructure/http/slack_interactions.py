"""Slack interactive components endpoint (block_actions, shortcuts, etc.).

Handles button clicks from Block Kit messages (approve, reject, snooze,
cancel) and routes them through the same application workflows as text
commands.
"""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException
from loguru import logger

from dacribagents.application.use_cases import reminder_workflows as wf
from dacribagents.domain.reminders.lifecycle import InvalidTransitionError
from dacribagents.domain.reminders.models import (
    ApproveReminderRequest,
    RejectReminderRequest,
    SnoozeReminderRequest,
)
from dacribagents.infrastructure.slack._store import get_household_store

router = APIRouter()

# Default actor for Slack interactions (resolved from Slack user in production)
_DEFAULT_ACTOR = UUID("00000000-0000-0000-0000-000000000001")


@router.post("/internal/slack/interactions")
async def slack_interactions(payload: str = Form(...)) -> dict:
    """Handle Slack interactive component callbacks.

    Slack sends ``application/x-www-form-urlencoded`` with a JSON ``payload``
    field for block_actions, shortcuts, and other interactive components.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as err:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from err

    action_type = data.get("type")
    if action_type != "block_actions":
        return {"status": "ignored", "type": action_type}

    actions = data.get("actions", [])
    if not actions:
        return {"status": "no_actions"}

    action = actions[0]
    action_id = action.get("action_id", "")
    reminder_id_str = action.get("value", "")

    try:
        reminder_id = UUID(reminder_id_str)
    except ValueError:
        return {"status": "invalid_reminder_id"}

    store, _, _ = get_household_store()
    actor_id = _DEFAULT_ACTOR

    try:
        result = _dispatch_action(store, action_id, reminder_id, actor_id)
    except InvalidTransitionError as e:
        logger.warning(f"Invalid transition for {action_id} on {reminder_id}: {e}")
        return {"status": "invalid_transition", "error": str(e)}
    except wf.ReminderNotFoundError:
        return {"status": "not_found"}

    logger.info(f"Slack interaction: {action_id} on {reminder_id} → {result.state.value}")
    return {
        "status": "ok",
        "action": action_id,
        "reminder_id": str(reminder_id),
        "new_state": result.state.value,
    }


def _dispatch_action(store, action_id: str, reminder_id: UUID, actor_id: UUID):  # noqa: PLR0911
    """Route a Slack button action to the appropriate workflow."""
    if action_id == "approve_reminder":
        return wf.approve_reminder(
            store, reminder_id,
            ApproveReminderRequest(actor_id=actor_id, reason="Approved via Slack button"),
        )

    if action_id == "reject_reminder":
        return wf.reject_reminder(
            store, reminder_id,
            RejectReminderRequest(actor_id=actor_id, reason="Rejected via Slack button"),
        )

    if action_id == "cancel_reminder":
        return wf.cancel_reminder(store, reminder_id)

    if action_id == "snooze_reminder":
        from datetime import UTC, datetime, timedelta  # noqa: PLC0415

        return wf.snooze_reminder(
            store, reminder_id,
            SnoozeReminderRequest(
                member_id=actor_id,
                method="slack_button",
                snooze_until=datetime.now(UTC) + timedelta(minutes=30),
            ),
        )

    raise HTTPException(status_code=400, detail=f"Unknown action: {action_id}")
