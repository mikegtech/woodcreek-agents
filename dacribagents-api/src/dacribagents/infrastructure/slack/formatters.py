"""Format reminder domain objects for Slack mrkdwn responses."""

from __future__ import annotations

from dacribagents.application.ports.calendar_adapter import CalendarEvent
from dacribagents.application.services.calendar_queries import ConflictReport, ReminderOverlap
from dacribagents.application.use_cases.reminder_queries import (
    DraftPreview,
    ReminderExplanation,
    ScheduleSummary,
)
from dacribagents.domain.reminders.entities import Reminder, ReminderSchedule
from dacribagents.domain.reminders.enums import NotificationIntent, UrgencyLevel

# ── Urgency / intent labels ─────────────────────────────────────────────────

_URGENCY_ICON: dict[UrgencyLevel, str] = {
    UrgencyLevel.CRITICAL: ":rotating_light:",
    UrgencyLevel.URGENT: ":warning:",
    UrgencyLevel.NORMAL: ":bell:",
    UrgencyLevel.LOW: ":large_blue_circle:",
}

_INTENT_LABEL: dict[NotificationIntent, str] = {
    NotificationIntent.REMINDER: "Reminder",
    NotificationIntent.ALERT: "Alert",
    NotificationIntent.DIGEST: "Digest",
}


# ── Public formatters ───────────────────────────────────────────────────────


def format_reminder_list(reminders: list[Reminder], title: str) -> str:
    """Format a list of reminders as a Slack mrkdwn block."""
    if not reminders:
        return f"*{title}*\nNo items found."

    lines = [f"*{title}* ({len(reminders)} items)\n"]
    for r in reminders:
        icon = _URGENCY_ICON.get(r.urgency, ":bell:")
        intent = _INTENT_LABEL.get(r.intent, r.intent.value)
        lines.append(
            f"{icon} *{r.subject}*  |  _{intent}_  |  `{r.state.value}`  |  source: {r.source.value}"
        )
    return "\n".join(lines)


def format_approval_list(reminders: list[Reminder]) -> str:
    """Format reminders waiting for approval."""
    if not reminders:
        return "*Reminders waiting for approval*\nAll clear — nothing pending."

    lines = ["*Reminders waiting for approval* :clipboard:\n"]
    for r in reminders:
        icon = _URGENCY_ICON.get(r.urgency, ":bell:")
        source = r.source_agent or r.source.value
        short_id = str(r.id)[:8]
        lines.append(
            f"{icon} `{short_id}` *{r.subject}*  |  from: _{source}_"
            f"  |  `{r.state.value}`  |  created: {r.created_at:%Y-%m-%d %H:%M}"
        )
    lines.append("\nUse `approve <id>` or `reject <id> <reason>` to act.")
    return "\n".join(lines)


def format_created_reminder(reminder: Reminder) -> str:
    """Format a newly created reminder confirmation."""
    icon = _URGENCY_ICON.get(reminder.urgency, ":bell:")
    short_id = str(reminder.id)[:8]
    return (
        f"{icon} *Reminder created:* `{short_id}`\n"
        f"*Subject:* {reminder.subject}\n"
        f"*State:* `{reminder.state.value}`  |  *Approval required:* {'yes' if reminder.requires_approval else 'no'}"
    )


def format_approval_action(reminder: Reminder, action: str, reason: str | None = None) -> str:
    """Format an approval/rejection confirmation."""
    icon = ":white_check_mark:" if action == "approved" else ":no_entry_sign:"
    short_id = str(reminder.id)[:8]
    lines = [f"{icon} Reminder `{short_id}` *{reminder.subject}* has been *{action}*."]
    lines.append(f"*New state:* `{reminder.state.value}`")
    if reason:
        lines.append(f"*Reason:* {reason}")
    return "\n".join(lines)


def format_explanation(expl: ReminderExplanation) -> str:
    """Format a reminder explanation."""
    r = expl.reminder
    lines = [
        f"*{r.subject}*",
        f"> {r.body}" if r.body else "",
        "",
        f"*State:* `{r.state.value}`  |  *Urgency:* {r.urgency.value}  |  *Intent:* {r.intent.value}",
        f"*Source:* {expl.source_description}",
        f"*Lifecycle:* {expl.lifecycle_summary}",
    ]
    if expl.approval_reason:
        lines.append(f"*Approval required:* {expl.approval_reason}")
    if expl.schedule:
        s = expl.schedule
        if s.fire_at:
            lines.append(f"*Scheduled:* {s.fire_at:%Y-%m-%d %H:%M %Z}  ({s.schedule_type.value})")
        elif s.cron_expression:
            lines.append(f"*Recurring:* `{s.cron_expression}`")
    if expl.targets:
        target_strs = [t.target_type.value for t in expl.targets]
        lines.append(f"*Targets:* {', '.join(target_strs)}")
    return "\n".join(line for line in lines if line is not None)


def format_draft_preview(preview: DraftPreview) -> str:
    """Format a draft preview for Slack."""
    icon = _URGENCY_ICON.get(preview.urgency, ":bell:")
    intent = _INTENT_LABEL.get(preview.intent, preview.intent.value)
    channels = ", ".join(c.value for c in preview.likely_channels)

    lines = [
        f"{icon} *Draft Preview*\n",
        f"*Subject:* {preview.subject}",
    ]
    if preview.body:
        lines.append(f"*Body:* {preview.body}")
    lines.extend([
        f"*Intent:* {intent}  |  *Urgency:* {preview.urgency.value}",
        f"*Targets:* {preview.target_description}",
    ])
    if preview.schedule_description:
        lines.append(f"*Schedule:* {preview.schedule_description}")
    lines.append(f"*Likely channels:* {channels}")
    if preview.requires_approval:
        lines.append(f":lock: *Approval required:* {preview.approval_reason}")
    else:
        lines.append(":white_check_mark: No approval required — would schedule immediately.")
    if preview.notes:
        lines.append("\n_Notes:_")
        for note in preview.notes:
            lines.append(f"  • {note}")
    return "\n".join(lines)


def format_scheduled_list(pairs: list[tuple[Reminder, ReminderSchedule | None]]) -> str:
    """Format upcoming scheduled reminders with next fire time."""
    if not pairs:
        return "*Upcoming Scheduled Reminders*\nNone scheduled."

    lines = [f"*Upcoming Scheduled Reminders* ({len(pairs)} items)\n"]
    for r, sched in pairs:
        icon = _URGENCY_ICON.get(r.urgency, ":bell:")
        short_id = str(r.id)[:8]
        fire_str = _format_fire_time(sched)
        stype = sched.schedule_type.value if sched else "?"
        lines.append(f"{icon} `{short_id}` *{r.subject}*  |  {fire_str}  |  _{stype}_")
    return "\n".join(lines)


def format_recurring_list(pairs: list[tuple[Reminder, ReminderSchedule]]) -> str:
    """Format active recurring reminders."""
    if not pairs:
        return "*Recurring Reminders*\nNone active."

    lines = [f"*Recurring Reminders* ({len(pairs)} items)\n"]
    for r, sched in pairs:
        icon = _URGENCY_ICON.get(r.urgency, ":bell:")
        short_id = str(r.id)[:8]
        fire_str = _format_fire_time(sched)
        lines.append(f"{icon} `{short_id}` *{r.subject}*  |  `{sched.cron_expression}`  |  next: {fire_str}")
    return "\n".join(lines)


def format_approval_buttons(reminder: Reminder) -> list[dict]:
    """Build Slack Block Kit action buttons for a pending-approval reminder."""
    short_id = str(reminder.id)[:8]
    return [
        {
            "type": "actions",
            "block_id": f"approval_{short_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "style": "primary",
                    "action_id": "approve_reminder",
                    "value": str(reminder.id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "style": "danger",
                    "action_id": "reject_reminder",
                    "value": str(reminder.id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Cancel"},
                    "action_id": "cancel_reminder",
                    "value": str(reminder.id),
                },
            ],
        },
    ]


def format_management_buttons(reminder: Reminder) -> list[dict]:
    """Build Slack Block Kit action buttons for managing a delivered reminder."""
    return [
        {
            "type": "actions",
            "block_id": f"manage_{str(reminder.id)[:8]}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Snooze 30m"},
                    "action_id": "snooze_reminder",
                    "value": str(reminder.id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Cancel"},
                    "style": "danger",
                    "action_id": "cancel_reminder",
                    "value": str(reminder.id),
                },
            ],
        },
    ]


def _format_fire_time(sched: ReminderSchedule | None) -> str:
    if not sched or not sched.next_fire_at:
        return "no fire time"
    return sched.next_fire_at.strftime("%Y-%m-%d %H:%M")


def format_schedule_summary(summary: ScheduleSummary) -> str:
    """Format a combined reminder + calendar summary."""
    start = summary.date_range.start
    end = summary.date_range.end

    lines = [f"*Schedule: {start:%a %b %d} — {end:%a %b %d}*\n"]

    if summary.calendar_events:
        lines.append(f"*Calendar* ({len(summary.calendar_events)} events)")
        for e in sorted(summary.calendar_events, key=lambda x: x.start):
            time_str = "all day" if e.all_day else f"{e.start:%H:%M}–{e.end:%H:%M}"
            lines.append(f"  :calendar: {time_str}  *{e.title}*")
        lines.append("")

    if summary.reminders:
        lines.append(f"*Reminders* ({len(summary.reminders)} active)")
        for r in summary.reminders:
            icon = _URGENCY_ICON.get(r.urgency, ":bell:")
            lines.append(f"  {icon} *{r.subject}*  `{r.state.value}`")
    else:
        lines.append("*Reminders:* None active.")

    if not summary.calendar_events and not summary.reminders:
        lines.append("Nothing scheduled — all clear!")

    return "\n".join(lines)


def format_calendar_conflicts(
    events: list[CalendarEvent],
    window_label: str,
) -> str:
    """Format calendar conflicts for a time window."""
    if not events:
        return f"*Conflicts for {window_label}:* None found — all clear!"

    lines = [f"*Conflicts for {window_label}* ({len(events)} events)\n"]
    for e in sorted(events, key=lambda x: x.start):
        time_str = "all day" if e.all_day else f"{e.start:%H:%M}–{e.end:%H:%M}"
        lines.append(f"  :calendar: {time_str}  *{e.title}*")
    return "\n".join(lines)


def format_conflict_report(report: ConflictReport, label: str) -> str:
    """Format identity-based conflict report per member."""
    if not report.member_conflicts:
        return f"*Conflicts for {label}:* None found — all clear!"

    lines = [f"*Conflicts for {label}*\n"]
    for name, slots in report.member_conflicts.items():
        lines.append(f"*{name}:*")
        for s in sorted(slots, key=lambda x: x.start):
            lines.append(f"  :calendar: {s.start:%H:%M}–{s.end:%H:%M}  {s.title}")
    return "\n".join(lines)


def format_reminder_overlaps(overlaps: list[ReminderOverlap]) -> str:
    """Format reminders that overlap with calendar events."""
    if not overlaps:
        return "*Reminder/Calendar Overlaps*\nNo overlaps found — all clear!"

    lines = [f"*Reminder/Calendar Overlaps* ({len(overlaps)} reminders)\n"]
    for o in overlaps:
        icon = _URGENCY_ICON.get(o.reminder.urgency, ":bell:")
        events_str = ", ".join(e.title for e in o.overlapping_events)
        lines.append(f"{icon} *{o.reminder.subject}* overlaps with: _{events_str}_")
    return "\n".join(lines)
