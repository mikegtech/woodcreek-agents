"""Intent classifier — single LLM call that produces a typed ParsedIntent.

Uses the repo's ``create_llm()`` factory with ``temperature=0`` for
deterministic classification.  Returns a ``ParsedIntent`` contract
with intent type, extracted entities, and confidence.

The classification prompt is kept tight and bounded — it does NOT
perform reminder business logic.
"""

from __future__ import annotations

import json

from loguru import logger

from dacribagents.application.interaction.contracts import IntentType, ParsedIntent, TargetScope

_SYSTEM_PROMPT = """You are a household reminder assistant intent classifier.

Given a user message, extract the intent and entities into a JSON object.
Respond with ONLY valid JSON matching this schema:

{
  "intent": one of: "create_reminder", "snooze_reminder", "cancel_reminder",
    "acknowledge_reminder", "approve_reminder", "reject_reminder",
    "reminder_query", "governance_query", "calendar_query",
    "suggestion_query", "unknown",
  "confidence": 0.0-1.0,
  "subject": "the reminder subject/task (clean, without command words)",
  "target_scope": "household" or "individual",
  "target_member_name": "member name if individual, empty otherwise",
  "schedule_text": "time expression if present, empty otherwise",
  "urgency": "normal", "urgent", "critical", or "low",
  "reminder_short_id": "short id if referencing existing reminder",
  "snooze_minutes": 30,
  "reject_reason": "reason if rejecting",
  "query_subject": "what the query is about",
  "needs_clarification": false,
  "clarification_question": ""
}

Rules:
- "remind Mike to pick up groceries at 5" → create_reminder, subject="pick up groceries", target=individual, member="Mike", schedule="at 5"
- "remind the family about trash night" → create_reminder, subject="trash night", target=household
- "tell Mike about dentist at 3pm" → create_reminder, subject="dentist", target=individual, member="Mike", schedule="at 3pm"
- "what reminders are pending?" → reminder_query, query_subject="pending"
- "snooze abc123 for 30 minutes" → snooze_reminder, short_id="abc123", snooze_minutes=30
- "approve abc123" → approve_reminder, short_id="abc123"
- "reject abc123 too late" → reject_reminder, short_id="abc123", reject_reason="too late"
- "cancel abc123" → cancel_reminder, short_id="abc123"
- "governance status" → governance_query, query_subject="status"
- "what does tomorrow look like?" → calendar_query, query_subject="tomorrow"
- "who is free tomorrow afternoon?" → calendar_query, query_subject="free tomorrow afternoon"
- "kill switch on" → governance_query, query_subject="kill_switch_on"
- "kill switch off" → governance_query, query_subject="kill_switch_off"
- "mute Mike" → governance_query, query_subject="mute", target_member_name="Mike"
- "unmute Mike" → governance_query, query_subject="unmute", target_member_name="Mike"
- "suggestions" → suggestion_query
- Extract the subject cleanly — strip command words like "remind", "tell", "create a reminder", "about", "to".
- If the time is ambiguous, set needs_clarification=true and ask about the time.
- If the target member is unclear, default to household.
"""


def classify_intent(text: str, llm: object | None = None) -> ParsedIntent:
    """Classify user text into a typed ParsedIntent via LLM.

    Falls back to a safe UNKNOWN intent on any error.
    """
    if llm is None:
        llm = _get_llm()

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=text)]
        response = llm.invoke(messages)
        content = response.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        return ParsedIntent(
            intent=IntentType(data.get("intent", "unknown")),
            confidence=float(data.get("confidence", 0.8)),
            subject=data.get("subject", ""),
            target_scope=TargetScope(data.get("target_scope", "household")),
            target_member_name=data.get("target_member_name", ""),
            schedule_text=data.get("schedule_text", ""),
            urgency=data.get("urgency", "normal"),
            reminder_short_id=data.get("reminder_short_id", ""),
            snooze_minutes=int(data.get("snooze_minutes", 30)),
            reject_reason=data.get("reject_reason", ""),
            query_subject=data.get("query_subject", ""),
            needs_clarification=bool(data.get("needs_clarification", False)),
            clarification_question=data.get("clarification_question", ""),
            original_text=text,
        )

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return ParsedIntent(intent=IntentType.UNKNOWN, original_text=text, confidence=0.0)


def _get_llm():
    """Get the reminder_classifier LLM from the factory (Mistral 7B, fast)."""
    from dacribagents.infrastructure.llm.factory import get_llm_factory  # noqa: PLC0415

    factory = get_llm_factory()
    return factory.get_llm_for_agent("reminder_classifier")
