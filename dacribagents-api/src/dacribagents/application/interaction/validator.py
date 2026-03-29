"""Deterministic validation for parsed intents.

No LLM calls — pure Python validation of the typed contract against
store data and business rules.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from uuid import UUID

from dacribagents.application.interaction.contracts import IntentType, ParsedIntent, TargetScope
from dacribagents.application.ports.reminder_store import ReminderStore


def validate(intent: ParsedIntent, store: ReminderStore, household_id: UUID) -> ParsedIntent:
    """Validate and enrich a parsed intent. Returns a possibly-modified intent."""
    if intent.intent == IntentType.CREATE_REMINDER:
        return _validate_create(intent, store, household_id)
    if intent.intent in {IntentType.SNOOZE_REMINDER, IntentType.CANCEL_REMINDER,
                         IntentType.APPROVE_REMINDER, IntentType.REJECT_REMINDER,
                         IntentType.ACKNOWLEDGE_REMINDER}:
        return _validate_action(intent, store, household_id)
    return intent


def _validate_create(intent: ParsedIntent, store: ReminderStore, household_id: UUID) -> ParsedIntent:
    """Validate reminder creation intent."""
    # Check subject
    if not intent.subject or len(intent.subject.strip()) < 2:
        return intent.model_copy(update={
            "needs_clarification": True,
            "clarification_question": "What should the reminder be about?",
        })

    # Resolve member if individual targeting
    if intent.target_scope == TargetScope.INDIVIDUAL and intent.target_member_name:
        members = store.get_household_members(household_id)
        name_lower = intent.target_member_name.lower()
        match = next((m for m in members if m.name.lower() == name_lower), None)
        if match is None:
            available = ", ".join(m.name for m in members)
            return intent.model_copy(update={
                "needs_clarification": True,
                "clarification_question": f"I don't recognize '{intent.target_member_name}'. Available members: {available}",
            })

    return intent


def _validate_action(intent: ParsedIntent, store: ReminderStore, household_id: UUID) -> ParsedIntent:
    """Validate action intents (snooze, cancel, approve, reject)."""
    if not intent.reminder_short_id:
        return intent.model_copy(update={
            "needs_clarification": True,
            "clarification_question": "Which reminder? Provide the short ID (first 8 characters).",
        })

    # Try to find the reminder
    short_id = intent.reminder_short_id.lower().strip()
    reminders, _ = store.list_reminders(household_id)
    match = next((r for r in reminders if str(r.id).lower().startswith(short_id)), None)
    if match is None:
        return intent.model_copy(update={
            "needs_clarification": True,
            "clarification_question": f"No reminder found matching `{short_id}`. Try `@Woodcreek active reminders` to see IDs.",
        })

    return intent


def parse_schedule_to_datetime(schedule_text: str) -> datetime | None:  # noqa: PLR0911
    """Parse a natural-language schedule expression into a datetime.

    Handles: "at 5pm", "at 5", "at 3:30pm", "tomorrow morning",
    "tomorrow at 5pm", "in 30 minutes".
    """
    if not schedule_text:
        return None

    text = schedule_text.lower().strip()
    now = datetime.now(tz=timezone(timedelta(hours=-5)))  # CDT approximation

    # "in N minutes/hours"
    in_match = re.match(r"in\s+(\d+)\s*(min|minute|minutes|hour|hours|hr|hrs)", text)
    if in_match:
        val = int(in_match.group(1))
        if "hour" in in_match.group(2) or "hr" in in_match.group(2):
            val *= 60
        return now + timedelta(minutes=val)

    # Determine base date
    base_date = now.date()
    if "tomorrow" in text:
        base_date = (now + timedelta(days=1)).date()

    # "morning", "afternoon", "evening" defaults
    if "morning" in text and not re.search(r"\d", text):
        return datetime.combine(base_date, datetime.min.time().replace(hour=8), tzinfo=now.tzinfo)
    if "afternoon" in text and not re.search(r"\d", text):
        return datetime.combine(base_date, datetime.min.time().replace(hour=14), tzinfo=now.tzinfo)
    if "evening" in text and not re.search(r"\d", text):
        return datetime.combine(base_date, datetime.min.time().replace(hour=18), tzinfo=now.tzinfo)

    # "at H:MMam/pm" or "at H" or "at Hpm"
    time_match = re.search(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        ampm = time_match.group(3)

        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        elif ampm is None and hour < 8:
            # Bare number < 8 likely means PM (e.g., "at 5" = 5pm)
            hour += 12

        target = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute), tzinfo=now.tzinfo)
        # If the time already passed today, bump to tomorrow
        if target <= now and "tomorrow" not in text:
            target += timedelta(days=1)
        return target

    return None
