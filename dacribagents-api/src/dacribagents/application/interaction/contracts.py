"""Typed interaction contracts for Slack/operator intent classification.

These are the structured outputs from the LLM intent-classification step.
Each contract carries enough typed data to route directly into the
deterministic reminder/governance/calendar workflows without further
string parsing.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Classified intent categories."""

    CREATE_REMINDER = "create_reminder"
    SNOOZE_REMINDER = "snooze_reminder"
    CANCEL_REMINDER = "cancel_reminder"
    ACKNOWLEDGE_REMINDER = "acknowledge_reminder"
    APPROVE_REMINDER = "approve_reminder"
    REJECT_REMINDER = "reject_reminder"
    REMINDER_QUERY = "reminder_query"
    GOVERNANCE_QUERY = "governance_query"
    CALENDAR_QUERY = "calendar_query"
    SUGGESTION_QUERY = "suggestion_query"
    UNKNOWN = "unknown"


class TargetScope(str, Enum):
    """Who the reminder targets."""

    HOUSEHOLD = "household"
    INDIVIDUAL = "individual"


class ParsedIntent(BaseModel):
    """Structured output from LLM intent classification + entity extraction.

    One LLM call produces this entire contract.  Downstream routing and
    validation are deterministic Python — no further LLM calls needed.
    """

    intent: IntentType = IntentType.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Reminder creation / update entities
    subject: str = ""
    target_scope: TargetScope = TargetScope.HOUSEHOLD
    target_member_name: str = ""
    schedule_text: str = ""  # raw time expression ("at 5pm", "tomorrow morning")
    urgency: str = "normal"

    # Action entities (snooze, cancel, approve, reject)
    reminder_short_id: str = ""
    snooze_minutes: int = 30
    reject_reason: str = ""

    # Query entities
    query_subject: str = ""  # "pending", "failed", "recurring", "tomorrow", etc.

    # Clarification
    needs_clarification: bool = False
    clarification_question: str = ""

    # Original input
    original_text: str = ""
