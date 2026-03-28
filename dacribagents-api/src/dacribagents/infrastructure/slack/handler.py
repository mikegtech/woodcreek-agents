"""Slack command handler — parses @woodcreek mentions and routes to queries.

This is a keyword-based router for Phase 1.  It handles the known reminder
intelligence commands directly and returns structured Slack responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from dacribagents.application.ports.calendar_adapter import DateRange
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.use_cases import reminder_queries as rq
from dacribagents.domain.reminders.enums import (
    NotificationIntent,
    ReminderSource,
    TargetType,
    UrgencyLevel,
)
from dacribagents.infrastructure.calendar.mock_adapter import MockCalendarAdapter
from dacribagents.infrastructure.slack import formatters as fmt

# Timezone for the Woodcreek household
_TZ = timezone(timedelta(hours=-5))  # CDT approximation; use zoneinfo in production


@dataclass(frozen=True)
class SlackResponse:
    """Result of handling a Slack command."""

    text: str
    ephemeral: bool = False


class SlackCommandHandler:
    """Routes Slack @woodcreek mentions to reminder and calendar queries."""

    def __init__(
        self,
        store: ReminderStore,
        calendar: MockCalendarAdapter | None = None,
        household_id: UUID | None = None,
    ) -> None:
        self.store = store
        self.calendar = calendar or MockCalendarAdapter()
        self.household_id = household_id or UUID("00000000-0000-0000-0000-000000000001")

    def handle(self, raw_text: str) -> SlackResponse:  # noqa: PLR0911
        """Parse a Slack message and return a formatted response."""
        text = _strip_mention(raw_text).strip().lower()

        if _matches(text, ["pending", "waiting"]) and _matches(text, ["approval", "approve"]):
            return self._handle_approval_list()

        if _matches(text, ["alert", "alerts"]):
            return self._handle_alerts()

        if _matches(text, ["pending", "active", "open"]) and _matches(text, ["reminder"]):
            return self._handle_active_reminders()

        if _matches(text, ["tomorrow", "today", "schedule", "look like"]):
            return self._handle_schedule_query(text)

        if _matches(text, ["conflict", "conflicts", "busy"]):
            return self._handle_conflicts(text)

        if _matches(text, ["draft"]) and _matches(text, ["reminder"]):
            return self._handle_draft_preview(raw_text)

        if _matches(text, ["explain", "why"]) and _matches(text, ["approval", "reminder"]):
            return self._handle_explain(text)

        if _matches(text, ["reminder"]):
            return self._handle_active_reminders()

        return SlackResponse(
            text=(
                "I can help with reminders, alerts, calendar, and household scheduling.\n"
                "Try: `pending approvals`, `active alerts`, `what does tomorrow look like?`, "
                "or `draft a reminder for the family about ...`"
            ),
        )

    # ── Command handlers ────────────────────────────────────────────────

    def _handle_approval_list(self) -> SlackResponse:
        reminders = rq.list_pending_approval(self.store, self.household_id)
        return SlackResponse(text=fmt.format_approval_list(reminders))

    def _handle_alerts(self) -> SlackResponse:
        alerts = rq.list_active_alerts(self.store, self.household_id)
        return SlackResponse(text=fmt.format_reminder_list(alerts, "Active Alerts"))

    def _handle_active_reminders(self) -> SlackResponse:
        reminders = rq.list_active(self.store, self.household_id)
        return SlackResponse(text=fmt.format_reminder_list(reminders, "Active Reminders"))

    def _handle_schedule_query(self, text: str) -> SlackResponse:
        date_range = _parse_date_range(text)
        events = self.calendar.list_all_events(date_range)
        summary = rq.build_schedule_summary(self.store, self.household_id, date_range, events)
        return SlackResponse(text=fmt.format_schedule_summary(summary))

    def _handle_conflicts(self, text: str) -> SlackResponse:
        date_range = _parse_date_range(text)
        events = self.calendar.list_all_events(date_range)
        label = _date_range_label(date_range)
        return SlackResponse(text=fmt.format_calendar_conflicts(events, label))

    def _handle_draft_preview(self, raw_text: str) -> SlackResponse:
        parsed = _parse_draft_intent(raw_text)
        preview = rq.preview_draft(
            subject=parsed["subject"],
            body=parsed.get("body", ""),
            intent=NotificationIntent.REMINDER,
            urgency=UrgencyLevel.NORMAL,
            source=ReminderSource.USER,
            target_type=TargetType.HOUSEHOLD if parsed.get("household") else TargetType.INDIVIDUAL,
            schedule_description=parsed.get("schedule"),
        )
        return SlackResponse(text=fmt.format_draft_preview(preview))

    def _handle_explain(self, text: str) -> SlackResponse:
        # In Phase 1, explain the most recent approval-pending reminder.
        pending = rq.list_pending_approval(self.store, self.household_id)
        if not pending:
            return SlackResponse(text="No reminders currently need approval.")
        expl = rq.explain_reminder(self.store, pending[0].id)
        if expl is None:
            return SlackResponse(text="Could not find reminder details.")
        return SlackResponse(text=fmt.format_explanation(expl))


# ── Parsing helpers ─────────────────────────────────────────────────────────

_MENTION_RE = re.compile(r"<@\w+>\s*")


def _strip_mention(text: str) -> str:
    """Remove Slack user mention prefix (``<@U123> ...``)."""
    return _MENTION_RE.sub("", text)


def _matches(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword is found in *text*."""
    return any(kw in text for kw in keywords)


def _parse_date_range(text: str) -> DateRange:
    """Extract a date range from natural language (simple heuristic)."""
    now = datetime.now(tz=_TZ)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if "tomorrow" in text:
        start = today_start + timedelta(days=1)
        return DateRange(start=start, end=start + timedelta(days=1))
    if "today" in text:
        return DateRange(start=today_start, end=today_start + timedelta(days=1))

    # Look for day names
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in text:
            days_ahead = (i - today_start.weekday()) % 7 or 7
            target = today_start + timedelta(days=days_ahead)
            # If "night" mentioned, narrow to evening
            if "night" in text or "evening" in text:
                return DateRange(
                    start=target.replace(hour=17),
                    end=target.replace(hour=23, minute=59),
                )
            return DateRange(start=target, end=target + timedelta(days=1))

    # Default: next 24 hours
    return DateRange(start=now, end=now + timedelta(hours=24))


def _date_range_label(dr: DateRange) -> str:
    if dr.end - dr.start <= timedelta(days=1):
        return dr.start.strftime("%A %b %d")
    return f"{dr.start:%b %d} – {dr.end:%b %d}"


def _parse_draft_intent(raw_text: str) -> dict:
    """Extract subject, target, and schedule hints from a draft command."""
    text = _strip_mention(raw_text).strip()

    # Remove "draft a reminder" prefix variations
    for prefix in [
        "draft a reminder for the family about ",
        "draft a reminder for the household about ",
        "draft a reminder about ",
        "draft reminder about ",
        "draft reminder for ",
        "draft a reminder ",
        "draft reminder ",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break

    household = any(kw in raw_text.lower() for kw in ["family", "household", "everyone"])

    # Look for "at <time>" schedule hint
    schedule = None
    at_match = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", text, re.IGNORECASE)
    if at_match:
        schedule = f"One-shot at {at_match.group(1)}"

    # The remaining text (minus schedule hint) is the subject
    subject = text
    if at_match:
        subject = text[: at_match.start()].strip().rstrip(",")
    subject = subject.strip() or "Untitled reminder"

    return {"subject": subject, "household": household, "schedule": schedule}
