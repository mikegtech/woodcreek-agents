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
from dacribagents.application.services import anomaly_review, audit_timeline
from dacribagents.application.services import calendar_queries as cq
from dacribagents.application.services import governance as gov
from dacribagents.application.services import suggestions as sug
from dacribagents.application.use_cases import reminder_queries as rq
from dacribagents.application.use_cases import reminder_workflows as wf
from dacribagents.domain.reminders.enums import (
    AckMethod,
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
    SnoozeReminderRequest,
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

    def handle(self, raw_text: str) -> SlackResponse:  # noqa: PLR0911, PLR0912
        """Parse a Slack message and return a formatted response."""
        text = _strip_mention(raw_text).strip().lower()

        # ── Write commands (order matters: match specific before generic) ──
        if text.startswith("approve "):
            return self._handle_approve(text)

        if text.startswith("reject "):
            return self._handle_reject(raw_text)

        if text.startswith("kill switch on") or text.startswith("kill-switch on"):
            return self._handle_kill_switch_on()

        if text.startswith("kill switch off") or text.startswith("kill-switch off"):
            return self._handle_kill_switch_off()

        if _matches(text, ["governance", "autonomy"]) and _matches(text, ["status", "summary", "tier"]):
            return self._handle_governance_status()

        if _matches(text, ["governance review", "autonomy review"]):
            return self._handle_governance_review()

        if text.startswith("mute "):
            return self._handle_mute(text)

        if text.startswith("unmute "):
            return self._handle_unmute(text)

        if _matches(text, ["muted", "muted members"]):
            return self._handle_list_muted()

        if _matches(text, ["timeline"]) and _matches(text, ["reminder"]):
            return self._handle_timeline(text)

        if _matches(text, ["anomal", "flag"]):
            return self._handle_anomalies()

        if _matches(text, ["suggest", "suggestion"]):
            return self._handle_suggestions()

        if text.startswith("cancel "):
            return self._handle_cancel(text)

        if text.startswith("snooze "):
            return self._handle_snooze(text)

        # "create a reminder ..." — explicit create command
        if text.startswith("create ") and "reminder" in text:
            return self._handle_create(raw_text)

        # "remind Mike about ..." — per-member targeting
        if re.match(r"remind\s+\w+\s+about\s+", text):
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

        if _matches(text, ["free"]) and _matches(text, ["minute", "min", "hour"]):
            return self._handle_free_window(text)

        if _matches(text, ["overlap"]) and _matches(text, ["reminder", "event"]):
            return self._handle_overlap(text)

        if _matches(text, ["draft"]) and _matches(text, ["reminder"]):
            return self._handle_draft_preview(raw_text)

        if _matches(text, ["scheduled", "next", "upcoming"]) and _matches(text, ["reminder"]):
            return self._handle_scheduled_next()

        if _matches(text, ["recurring"]):
            return self._handle_recurring()

        if _matches(text, ["pending delivery", "pending_delivery", "ready to send"]):
            return self._handle_pending_delivery()

        if _matches(text, ["delivered today", "sent today", "what was delivered", "what was sent"]):
            return self._handle_delivered()

        if _matches(text, ["failed", "delivery failed"]):
            return self._handle_failed()

        if _matches(text, ["acknowledged", "acked"]):
            return self._handle_acknowledged()

        if _matches(text, ["digest"]) and _matches(text, ["pending", "eligible", "next"]):
            return self._handle_digest_pending()

        if _matches(text, ["event originated", "event created", "from events", "upstream"]):
            return self._handle_event_originated()

        if _matches(text, ["mirror", "mirrored", "calendar mirror"]):
            return self._handle_mirror_status()

        if _matches(text, ["deferred", "quiet hours", "suppressed"]):
            return self._handle_deferred()

        if _matches(text, ["explain", "why"]) and _matches(text, ["approval", "reminder", "blocked", "not sent"]):
            return self._handle_explain(text)

        if _matches(text, ["reminder"]):
            return self._handle_active_reminders()

        return SlackResponse(
            text=(
                "I can help with reminders, alerts, calendar, and household scheduling.\n"
                "Try: `pending approvals`, `active alerts`, `scheduled reminders`, `recurring`,\n"
                "`create a reminder for the family about ...`, `remind Mike about ...`,\n"
                "`approve <id>`, `reject <id> <reason>`, `cancel <id>`, `snooze <id>`"
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
        report = cq.compute_conflicts(self.calendar, self.store, self.household_id, date_range)
        if report.member_conflicts:
            return SlackResponse(text=fmt.format_conflict_report(report, _date_range_label(date_range)))
        # Fallback to simple event listing if no identity-based conflicts found
        events = self.calendar.list_all_events(date_range)
        label = _date_range_label(date_range)
        return SlackResponse(text=fmt.format_calendar_conflicts(events, label))

    def _handle_free_window(self, text: str) -> SlackResponse:
        """Find a free window for a member."""
        date_range = _parse_date_range(text)
        # Parse duration: "free for 30 minutes" or "free for 1 hour"
        duration = 30
        dur_match = re.search(r"(\d+)\s*(?:minute|min|hour|hr)", text)
        if dur_match:
            val = int(dur_match.group(1))
            if "hour" in text or "hr" in text:
                val *= 60
            duration = val

        events = self.calendar.list_all_events(date_range)
        summary = cq.compute_free_busy(events, date_range, "Household")
        slot = cq.find_free_window(summary, duration)
        if slot:
            return SlackResponse(
                text=f":white_check_mark: Free window found: *{slot.start:%H:%M}–{slot.end:%H:%M}* ({duration}min)"
            )
        return SlackResponse(text=f":x: No free {duration}-minute window found in {_date_range_label(date_range)}")

    def _handle_overlap(self, text: str) -> SlackResponse:
        """Show reminders that overlap calendar events."""
        date_range = _parse_date_range(text)
        events = self.calendar.list_all_events(date_range)
        overlaps = cq.find_reminder_overlaps(self.store, self.household_id, events)
        return SlackResponse(text=fmt.format_reminder_overlaps(overlaps))

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

    def _handle_event_originated(self) -> SlackResponse:
        reminders = rq.list_event_originated(self.store, self.household_id)
        return SlackResponse(text=fmt.format_reminder_list(reminders, "Event-Originated Reminders"))

    def _handle_mirror_status(self) -> SlackResponse:
        active = rq.list_active(self.store, self.household_id)
        if not active:
            return SlackResponse(text="*Calendar Mirrors*\nNo active reminders.")
        lines = ["*Calendar Mirror Status*\n"]
        lines.append("Calendar mirroring is available for INDIVIDUAL-targeted, SCHEDULED reminders.")
        lines.append("Use `explain reminder` for detailed mirror status per reminder.")
        return SlackResponse(text="\n".join(lines))

    def _handle_deferred(self) -> SlackResponse:
        """Show reminders that were deferred (quiet hours, suppressed)."""
        # In the current model, deferred reminders stay in PENDING_DELIVERY
        # with suppressed deliveries. Show pending + any with quiet hours notes.
        reminders = rq.list_pending_delivery(self.store, self.household_id)
        if not reminders:
            return SlackResponse(text="*Deferred Reminders*\nNone currently deferred.")
        return SlackResponse(
            text=fmt.format_reminder_list(reminders, "Reminders Pending (possibly deferred due to quiet hours)")
        )

    def _handle_explain(self, text: str) -> SlackResponse:
        pending = rq.list_pending_approval(self.store, self.household_id)
        if not pending:
            return SlackResponse(text="No reminders currently need approval.")
        expl = rq.explain_reminder(self.store, pending[0].id)
        if expl is None:
            return SlackResponse(text="Could not find reminder details.")
        return SlackResponse(text=fmt.format_explanation(expl))

    # ── Write command handlers ──────────────────────────────────────────

    def _handle_kill_switch_on(self) -> SlackResponse:
        gov.activate_kill_switch("Slack operator")
        return SlackResponse(text=":octagonal_sign: *Kill switch ACTIVATED.* All autonomous actions now require approval.")

    def _handle_kill_switch_off(self) -> SlackResponse:
        gov.deactivate_kill_switch("Slack operator")
        return SlackResponse(text=":white_check_mark: *Kill switch deactivated.* Autonomous actions resumed per tier policy.")

    def _handle_governance_status(self) -> SlackResponse:
        summary = gov.get_governance_summary(self.household_id)
        return SlackResponse(text=fmt.format_governance_summary(summary))

    def _handle_governance_review(self) -> SlackResponse:
        review = gov.generate_review_summary(self.household_id)
        return SlackResponse(text=fmt.format_governance_review(review))

    def _handle_mute(self, text: str) -> SlackResponse:
        name = text.removeprefix("mute ").strip()
        member = _resolve_member(self.store, self.household_id, name)
        if member is None:
            return SlackResponse(text=f":x: No member found matching `{name}`.")
        gov.mute_member(member.id, f"Muted by Slack operator for {name}")
        return SlackResponse(text=f":mute: *{member.name}* muted for non-critical reminders.")

    def _handle_unmute(self, text: str) -> SlackResponse:
        name = text.removeprefix("unmute ").strip()
        member = _resolve_member(self.store, self.household_id, name)
        if member is None:
            return SlackResponse(text=f":x: No member found matching `{name}`.")
        gov.unmute_member(member.id)
        return SlackResponse(text=f":loud_sound: *{member.name}* unmuted — normal delivery resumed.")

    def _handle_timeline(self, text: str) -> SlackResponse:
        """Show unified timeline for the most recent pending-approval reminder."""
        pending = rq.list_pending_approval(self.store, self.household_id)
        if pending:
            entries = audit_timeline.get_reminder_timeline(self.store, pending[0].id)
        else:
            entries = audit_timeline.get_household_timeline(self.store, self.household_id, limit=15)
        if not entries:
            return SlackResponse(text="*Timeline*\nNo entries found.")
        lines = ["*Audit Timeline*\n"]
        for e in entries[-15:]:
            lines.append(f"`{e.timestamp:%H:%M}` *{e.event_type}*  {e.summary}")
        return SlackResponse(text="\n".join(lines))

    def _handle_suggestions(self) -> SlackResponse:
        date_range = _parse_date_range("tomorrow")
        events = self.calendar.list_all_events(date_range)
        drafts = sug.generate_suggestions(self.store, self.household_id, events)
        if not drafts:
            return SlackResponse(text="*Suggestions*\nNo suggestions at this time.")
        lines = [f"*Suggested Reminders* ({len(drafts)} items)\n"]
        for d in drafts:
            lines.append(f":bulb: *{d.subject}*")
            lines.append(f"  _{d.reason}_")
            lines.append(f"  Sources: {', '.join(d.context_sources)}")
        return SlackResponse(text="\n".join(lines))

    def _handle_anomalies(self) -> SlackResponse:
        flags = anomaly_review.check_anomalies(self.household_id)
        if not flags:
            return SlackResponse(text="*Anomaly Review*\nNo anomalies detected — all clear.")
        lines = ["*Anomaly Review*\n"]
        for f in flags:
            icon = ":rotating_light:" if f.severity == "critical" else ":warning:"
            lines.append(f"{icon} `{f.rule}` — {f.description}")
        return SlackResponse(text="\n".join(lines))

    def _handle_list_muted(self) -> SlackResponse:
        muted = gov.list_muted_members()
        if not muted:
            return SlackResponse(text="*Muted Members*\nNo members currently muted.")
        lines = ["*Muted Members*\n"]
        members = self.store.get_household_members(self.household_id)
        name_map = {m.id: m.name for m in members}
        for mid, reason in muted.items():
            name = name_map.get(mid, str(mid)[:8])
            lines.append(f":mute: *{name}*  —  {reason}")
        return SlackResponse(text="\n".join(lines))

    def _handle_scheduled_next(self) -> SlackResponse:
        pairs = rq.list_scheduled_next(self.store, self.household_id)
        return SlackResponse(text=fmt.format_scheduled_list(pairs))

    def _handle_recurring(self) -> SlackResponse:
        pairs = rq.list_recurring(self.store, self.household_id)
        return SlackResponse(text=fmt.format_recurring_list(pairs))

    def _handle_pending_delivery(self) -> SlackResponse:
        reminders = rq.list_pending_delivery(self.store, self.household_id)
        return SlackResponse(text=fmt.format_reminder_list(reminders, "Pending Delivery"))

    def _handle_delivered(self) -> SlackResponse:
        reminders = rq.list_delivered(self.store, self.household_id)
        return SlackResponse(text=fmt.format_reminder_list(reminders, "Delivered Reminders"))

    def _handle_failed(self) -> SlackResponse:
        reminders = rq.list_failed_deliveries(self.store, self.household_id)
        return SlackResponse(text=fmt.format_reminder_list(reminders, "Failed Deliveries"))

    def _handle_acknowledged(self) -> SlackResponse:
        reminders = rq.list_acknowledged(self.store, self.household_id)
        return SlackResponse(text=fmt.format_reminder_list(reminders, "Acknowledged Reminders"))

    def _handle_digest_pending(self) -> SlackResponse:
        reminders = rq.list_digest_eligible(self.store, self.household_id)
        if not reminders:
            return SlackResponse(text="*Digest Queue*\nNo reminders pending for digest.")
        return SlackResponse(text=fmt.format_reminder_list(reminders, "Pending Digest Items"))

    def _handle_cancel(self, text: str) -> SlackResponse:
        short_id = text.removeprefix("cancel ").strip().removeprefix("reminder ").strip()
        reminder = _find_by_short_id(self.store, self.household_id, short_id)
        if reminder is None:
            return SlackResponse(text=f":x: No reminder found matching `{short_id}`.")
        try:
            result = wf.cancel_reminder(self.store, reminder.id)
            return SlackResponse(text=fmt.format_approval_action(result, "cancelled"))
        except InvalidTransitionError:
            return SlackResponse(
                text=f":x: Cannot cancel *{reminder.subject}* — current state is `{reminder.state.value}`."
            )

    def _handle_snooze(self, text: str) -> SlackResponse:
        """Snooze a delivered reminder. Usage: snooze <id> [minutes]."""
        parts = text.removeprefix("snooze ").strip().split(None, 1)
        short_id = parts[0] if parts else ""
        minutes = 30  # default
        if len(parts) > 1:
            try:
                minutes = int(parts[1].removesuffix("m").removesuffix("min").removesuffix("minutes").strip())
            except ValueError:
                pass

        reminder = _find_by_short_id(self.store, self.household_id, short_id)
        if reminder is None:
            return SlackResponse(text=f":x: No reminder found matching `{short_id}`.")
        try:
            snooze_until = datetime.now(tz=_TZ) + timedelta(minutes=minutes)
            result = wf.snooze_reminder(
                self.store, reminder.id,
                SnoozeReminderRequest(
                    member_id=self.actor_id,
                    method=AckMethod.SLACK_BUTTON,
                    snooze_until=snooze_until,
                ),
            )
            return SlackResponse(text=f":zzz: *{result.subject}* snoozed for {minutes} minutes.")
        except InvalidTransitionError:
            return SlackResponse(
                text=f":x: Cannot snooze *{reminder.subject}* — current state is `{reminder.state.value}`."
            )

    def _handle_create(self, raw_text: str) -> SlackResponse:
        """Create a reminder draft and auto-submit for approval if needed."""
        parsed = _parse_draft_intent(raw_text)

        # Per-member targeting: resolve "remind Mike about..." to member_id
        member_name = parsed.get("member_name")
        if member_name:
            member = _resolve_member(self.store, self.household_id, member_name)
            if member:
                target = ReminderTargetInput(target_type=TargetType.INDIVIDUAL, member_id=member.id)
            else:
                target = ReminderTargetInput(target_type=TargetType.HOUSEHOLD)
        elif parsed.get("household"):
            target = ReminderTargetInput(target_type=TargetType.HOUSEHOLD)
        else:
            target = ReminderTargetInput(target_type=TargetType.HOUSEHOLD)

        try:
            reminder = wf.create_reminder(
                self.store,
                self.household_id,
                self.actor_id,
                CreateReminderRequest(
                    subject=parsed["subject"],
                    source=ReminderSource.USER,
                    targets=[target],
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

    # Per-member targeting: "remind Mike about ..."
    member_name = None
    member_match = re.search(
        r"(?:remind|create a reminder for)\s+([A-Z][a-z]+)\s+about\s+",
        _strip_mention(raw_text),
    )
    if member_match and member_match.group(1).lower() not in {"the", "a", "my"}:
        name = member_match.group(1)
        if name.lower() not in {"family", "household", "everyone"}:
            member_name = name

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

    return {"subject": subject, "household": household, "schedule": schedule, "member_name": member_name}


def _resolve_member(
    store: ReminderStore,
    household_id: UUID,
    name: str,
) -> object | None:
    """Resolve a member name to a HouseholdMember (case-insensitive)."""
    members = store.get_household_members(household_id)
    name_lower = name.lower()
    for m in members:
        if m.name.lower() == name_lower:
            return m
    return None


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
