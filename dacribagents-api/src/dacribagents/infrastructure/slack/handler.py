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
from dacribagents.application.use_cases import reminder_workflows as wf
from dacribagents.domain.reminders.enums import (
    NotificationIntent,
    ReminderSource,
    ReminderState,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.lifecycle import InvalidTransitionError
from dacribagents.domain.reminders.models import (
    ApproveReminderRequest,
    CreateReminderRequest,
    RejectReminderRequest,
    ReminderTargetInput,
    SubmitForApprovalRequest,
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
        actor_id: UUID | None = None,
    ) -> None:
        self.store = store
        self.calendar = calendar or MockCalendarAdapter()
        self.household_id = household_id or UUID("00000000-0000-0000-0000-000000000001")
        self.actor_id = actor_id or self.household_id

    def handle(self, raw_text: str) -> SlackResponse:  # noqa: PLR0911
        """Parse a Slack message and return a formatted response."""
        text = _strip_mention(raw_text).strip().lower()

        # ── Write commands (order matters: match specific before generic) ──
        if text.startswith("approve "):
            return self._handle_approve(text)

        if text.startswith("reject "):
            return self._handle_reject(raw_text)

        if _matches(text, ["create"]) and _matches(text, ["reminder"]):
            return self._handle_create(raw_text)

        # ── Read commands ───────────────────────────────────────────────
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

        if _matches(text, ["explain", "why"]) and _matches(text, ["approval", "reminder", "blocked"]):
            return self._handle_explain(text)

        if _matches(text, ["reminder"]):
            return self._handle_active_reminders()

        return SlackResponse(
            text=(
                "I can help with reminders, alerts, calendar, and household scheduling.\n"
                "Try: `pending approvals`, `active alerts`, `what does tomorrow look like?`, "
                "`create a reminder for the family about ...`, `approve <id>`, `reject <id> <reason>`"
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
        pending = rq.list_pending_approval(self.store, self.household_id)
        if not pending:
            return SlackResponse(text="No reminders currently need approval.")
        expl = rq.explain_reminder(self.store, pending[0].id)
        if expl is None:
            return SlackResponse(text="Could not find reminder details.")
        return SlackResponse(text=fmt.format_explanation(expl))

    # ── Write command handlers ──────────────────────────────────────────

    def _handle_create(self, raw_text: str) -> SlackResponse:
        """Create a reminder draft and auto-submit for approval if needed."""
        parsed = _parse_draft_intent(raw_text)
        target_type = TargetType.HOUSEHOLD if parsed.get("household") else TargetType.INDIVIDUAL

        try:
            reminder = wf.create_reminder(
                self.store,
                self.household_id,
                self.actor_id,
                CreateReminderRequest(
                    subject=parsed["subject"],
                    source=ReminderSource.USER,
                    targets=[ReminderTargetInput(
                        target_type=target_type,
                        member_id=self.actor_id if target_type == TargetType.INDIVIDUAL else None,
                    )],
                ),
            )

            lines = [fmt.format_created_reminder(reminder)]

            # Auto-submit for approval if needed (user-created don't need it)
            if reminder.requires_approval and reminder.state == ReminderState.DRAFT:
                reminder = wf.submit_for_approval(
                    self.store, reminder.id,
                    SubmitForApprovalRequest(actor_id=self.actor_id),
                )
                lines.append(":clipboard: Submitted for approval.")

            return SlackResponse(text="\n".join(lines))
        except Exception as e:
            return SlackResponse(text=f":x: Failed to create reminder: {e}")

    def _handle_approve(self, text: str) -> SlackResponse:
        """Approve a pending-approval reminder by short ID."""
        short_id = text.removeprefix("approve ").strip().removeprefix("reminder ").strip()
        reminder = _find_by_short_id(self.store, self.household_id, short_id)
        if reminder is None:
            return SlackResponse(text=f":x: No reminder found matching `{short_id}`.")

        try:
            result = wf.approve_reminder(
                self.store, reminder.id,
                ApproveReminderRequest(actor_id=self.actor_id),
            )
            return SlackResponse(text=fmt.format_approval_action(result, "approved"))
        except InvalidTransitionError:
            return SlackResponse(
                text=f":x: Cannot approve *{reminder.subject}* — current state is `{reminder.state.value}`."
            )

    def _handle_reject(self, raw_text: str) -> SlackResponse:
        """Reject a pending-approval reminder by short ID with reason."""
        text = _strip_mention(raw_text).strip()
        # Parse: "reject <id> <reason>" or "reject <id> because <reason>"
        parts = text.removeprefix("reject ").strip().removeprefix("reminder ").strip()
        tokens = parts.split(None, 1)
        short_id = tokens[0] if tokens else ""
        reason = tokens[1] if len(tokens) > 1 else ""
        reason = reason.removeprefix("because ").strip() or "Rejected via Slack"

        reminder = _find_by_short_id(self.store, self.household_id, short_id)
        if reminder is None:
            return SlackResponse(text=f":x: No reminder found matching `{short_id}`.")

        try:
            result = wf.reject_reminder(
                self.store, reminder.id,
                RejectReminderRequest(actor_id=self.actor_id, reason=reason),
            )
            return SlackResponse(text=fmt.format_approval_action(result, "rejected", reason))
        except InvalidTransitionError:
            return SlackResponse(
                text=f":x: Cannot reject *{reminder.subject}* — current state is `{reminder.state.value}`."
            )


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
        "create a reminder for the family about ",
        "create a reminder for the household about ",
        "create a reminder about ",
        "create reminder about ",
        "create reminder for ",
        "create a reminder ",
        "create reminder ",
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


def _find_by_short_id(
    store: ReminderStore,
    household_id: UUID,
    short_id: str,
) -> object | None:
    """Find a reminder by the first 8 chars of its UUID."""
    short_id = short_id.lower().strip()
    all_reminders, _ = store.list_reminders(household_id)
    for r in all_reminders:
        if str(r.id).lower().startswith(short_id):
            return r
    return None
