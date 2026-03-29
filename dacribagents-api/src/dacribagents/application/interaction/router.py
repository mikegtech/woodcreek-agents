"""Deterministic router — maps validated ParsedIntent to application workflows.

No LLM calls.  Takes a typed contract and calls the appropriate
existing service function.  Returns a structured result for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from loguru import logger

from dacribagents.application.interaction.contracts import IntentType, ParsedIntent, TargetScope
from dacribagents.application.interaction.validator import parse_schedule_to_datetime
from dacribagents.application.ports.reminder_store import ReminderStore
from dacribagents.application.services import governance as gov
from dacribagents.application.use_cases import reminder_queries as rq
from dacribagents.application.use_cases import reminder_workflows as wf
from dacribagents.domain.reminders.enums import (
    AckMethod,
    ReminderSource,
    ReminderState,
    ScheduleType,
    TargetType,
    UrgencyLevel,
)
from dacribagents.domain.reminders.models import (
    ApproveReminderRequest,
    CreateReminderRequest,
    RejectReminderRequest,
    ReminderScheduleInput,
    ReminderTargetInput,
    SnoozeReminderRequest,
    SubmitForApprovalRequest,
)


@dataclass
class RouteResult:
    """Outcome of routing a validated intent to a service."""

    success: bool
    message: str
    reminder_id: str | None = None


def route(  # noqa: PLR0911
    intent: ParsedIntent,
    store: ReminderStore,
    household_id: UUID,
    actor_id: UUID,
) -> RouteResult:
    """Route a validated ParsedIntent to the appropriate workflow."""
    if intent.needs_clarification:
        return RouteResult(success=False, message=intent.clarification_question)

    try:
        if intent.intent == IntentType.CREATE_REMINDER:
            return _route_create(intent, store, household_id, actor_id)
        if intent.intent == IntentType.SNOOZE_REMINDER:
            return _route_snooze(intent, store, actor_id)
        if intent.intent == IntentType.CANCEL_REMINDER:
            return _route_cancel(intent, store, household_id)
        if intent.intent == IntentType.APPROVE_REMINDER:
            return _route_approve(intent, store, household_id, actor_id)
        if intent.intent == IntentType.REJECT_REMINDER:
            return _route_reject(intent, store, household_id, actor_id)
        if intent.intent == IntentType.ACKNOWLEDGE_REMINDER:
            return _route_ack(intent, store, actor_id)
        if intent.intent == IntentType.REMINDER_QUERY:
            return _route_reminder_query(intent, store, household_id)
        if intent.intent == IntentType.GOVERNANCE_QUERY:
            return _route_governance_query(intent, store, household_id, actor_id)
        if intent.intent == IntentType.CALENDAR_QUERY:
            return _route_calendar_query(intent, store, household_id)
        if intent.intent == IntentType.SUGGESTION_QUERY:
            return _route_suggestion_query(intent, store, household_id)
        return RouteResult(success=False, message="I'm not sure what you're asking. Try `@Woodcreek active reminders` or `remind Mike about ...`")
    except Exception as e:
        logger.error(f"Route failed for {intent.intent.value}: {e}")
        return RouteResult(success=False, message=f"Something went wrong: {e}")


# ── Create ──────────────────────────────────────────────────────────────────


def _route_create(intent: ParsedIntent, store: ReminderStore, household_id: UUID, actor_id: UUID) -> RouteResult:
    # Build target
    if intent.target_scope == TargetScope.INDIVIDUAL and intent.target_member_name:
        members = store.get_household_members(household_id)
        member = next((m for m in members if m.name.lower() == intent.target_member_name.lower()), None)
        if member:
            target = ReminderTargetInput(target_type=TargetType.INDIVIDUAL, member_id=member.id)
        else:
            target = ReminderTargetInput(target_type=TargetType.HOUSEHOLD)
    else:
        target = ReminderTargetInput(target_type=TargetType.HOUSEHOLD)

    # Build schedule if time provided
    schedule = None
    if intent.schedule_text:
        fire_at = parse_schedule_to_datetime(intent.schedule_text)
        if fire_at:
            schedule = ReminderScheduleInput(schedule_type=ScheduleType.ONE_SHOT, fire_at=fire_at)

    # Map urgency
    urgency_map = {"critical": UrgencyLevel.CRITICAL, "urgent": UrgencyLevel.URGENT, "low": UrgencyLevel.LOW}
    urgency = urgency_map.get(intent.urgency, UrgencyLevel.NORMAL)

    request = CreateReminderRequest(
        subject=intent.subject,
        urgency=urgency,
        source=ReminderSource.USER,
        targets=[target],
        schedule=schedule,
    )

    reminder = wf.create_reminder(store, household_id, actor_id, request)

    # Auto-submit for approval if needed
    if reminder.requires_approval and reminder.state == ReminderState.DRAFT:
        wf.submit_for_approval(store, reminder.id, SubmitForApprovalRequest(actor_id=actor_id))

    short_id = str(reminder.id)[:8]
    target_desc = intent.target_member_name or "the household"
    schedule_desc = f" at {intent.schedule_text}" if intent.schedule_text else ""
    state_desc = reminder.state.value

    return RouteResult(
        success=True,
        message=f":bell: *Reminder created:* `{short_id}`\n*Subject:* {intent.subject}\n*For:* {target_desc}{schedule_desc}\n*State:* `{state_desc}`",
        reminder_id=short_id,
    )


# ── Actions ─────────────────────────────────────────────────────────────────


def _find_reminder(store, household_id, short_id):
    reminders, _ = store.list_reminders(household_id)
    return next((r for r in reminders if str(r.id).lower().startswith(short_id.lower())), None)


def _route_snooze(intent: ParsedIntent, store: ReminderStore, actor_id: UUID) -> RouteResult:
    # Find by short ID — need household_id, get from reminder
    # For now, search across all reminders in store
    reminder = store.get_reminder(UUID(intent.reminder_short_id)) if len(intent.reminder_short_id) > 8 else None
    if reminder is None:
        return RouteResult(success=False, message=f"Reminder `{intent.reminder_short_id}` not found.")

    snooze_until = datetime.now(UTC) + timedelta(minutes=intent.snooze_minutes)
    wf.snooze_reminder(store, reminder.id, SnoozeReminderRequest(
        member_id=actor_id, method=AckMethod.SLACK_BUTTON, snooze_until=snooze_until,
    ))
    return RouteResult(success=True, message=f":zzz: Snoozed `{intent.reminder_short_id}` for {intent.snooze_minutes} minutes.")


def _route_cancel(intent: ParsedIntent, store: ReminderStore, household_id: UUID) -> RouteResult:
    reminder = _find_reminder(store, household_id, intent.reminder_short_id)
    if reminder is None:
        return RouteResult(success=False, message=f"Reminder `{intent.reminder_short_id}` not found.")
    wf.cancel_reminder(store, reminder.id)
    return RouteResult(success=True, message=f":no_entry_sign: Cancelled `{intent.reminder_short_id}` — *{reminder.subject}*")


def _route_approve(intent: ParsedIntent, store: ReminderStore, household_id: UUID, actor_id: UUID) -> RouteResult:
    reminder = _find_reminder(store, household_id, intent.reminder_short_id)
    if reminder is None:
        return RouteResult(success=False, message=f"Reminder `{intent.reminder_short_id}` not found.")
    wf.approve_reminder(store, reminder.id, ApproveReminderRequest(actor_id=actor_id))
    return RouteResult(success=True, message=f":white_check_mark: Approved `{intent.reminder_short_id}` — *{reminder.subject}*")


def _route_reject(intent: ParsedIntent, store: ReminderStore, household_id: UUID, actor_id: UUID) -> RouteResult:
    reminder = _find_reminder(store, household_id, intent.reminder_short_id)
    if reminder is None:
        return RouteResult(success=False, message=f"Reminder `{intent.reminder_short_id}` not found.")
    reason = intent.reject_reason or "Rejected via Slack"
    wf.reject_reminder(store, reminder.id, RejectReminderRequest(actor_id=actor_id, reason=reason))
    return RouteResult(success=True, message=f":no_entry_sign: Rejected `{intent.reminder_short_id}` — {reason}")


def _route_ack(intent: ParsedIntent, store: ReminderStore, actor_id: UUID) -> RouteResult:
    return RouteResult(success=False, message="Acknowledgment via Slack coming soon. Reply OK to SMS instead.")


# ── Queries ─────────────────────────────────────────────────────────────────


def _route_reminder_query(intent: ParsedIntent, store: ReminderStore, household_id: UUID) -> RouteResult:  # noqa: PLR0911
    q = intent.query_subject.lower()

    if "pending" in q and "approv" in q:
        items = rq.list_pending_approval(store, household_id)
        return _format_reminder_list(items, "Pending Approval")
    if "active" in q or "pending" in q:
        items = rq.list_active(store, household_id)
        return _format_reminder_list(items, "Active Reminders")
    if "schedul" in q or "upcoming" in q or "next" in q:
        pairs = rq.list_scheduled_next(store, household_id)
        lines = [f"*Scheduled Reminders* ({len(pairs)} items)"]
        for r, s in pairs:
            fire = s.next_fire_at.strftime("%Y-%m-%d %H:%M") if s and s.next_fire_at else "?"
            lines.append(f":bell: `{str(r.id)[:8]}` *{r.subject}* — {fire}")
        return RouteResult(success=True, message="\n".join(lines) if pairs else "No scheduled reminders.")
    if "recurring" in q:
        pairs = rq.list_recurring(store, household_id)
        lines = [f"*Recurring Reminders* ({len(pairs)} items)"]
        for r, s in pairs:
            lines.append(f":bell: `{str(r.id)[:8]}` *{r.subject}* — `{s.cron_expression}`")
        return RouteResult(success=True, message="\n".join(lines) if pairs else "No recurring reminders.")
    if "deliver" in q:
        items = rq.list_delivered(store, household_id)
        return _format_reminder_list(items, "Delivered")
    if "fail" in q:
        items = rq.list_failed_deliveries(store, household_id)
        return _format_reminder_list(items, "Failed Deliveries")
    if "alert" in q:
        items = rq.list_active_alerts(store, household_id)
        return _format_reminder_list(items, "Active Alerts")
    if "digest" in q:
        items = rq.list_digest_eligible(store, household_id)
        return _format_reminder_list(items, "Digest Queue")
    if "ack" in q:
        items = rq.list_acknowledged(store, household_id)
        return _format_reminder_list(items, "Acknowledged")

    # Default: show active
    items = rq.list_active(store, household_id)
    return _format_reminder_list(items, "Active Reminders")


def _route_governance_query(intent: ParsedIntent, store: ReminderStore, household_id: UUID, actor_id: UUID) -> RouteResult:  # noqa: PLR0911
    q = intent.query_subject.lower()

    if "kill_switch_on" in q or "kill switch on" in q:
        gov.activate_kill_switch("Slack operator")
        return RouteResult(success=True, message=":octagonal_sign: *Kill switch ACTIVATED.* All autonomous actions now require approval.")
    if "kill_switch_off" in q or "kill switch off" in q:
        gov.deactivate_kill_switch("Slack operator")
        return RouteResult(success=True, message=":white_check_mark: *Kill switch deactivated.* Autonomous actions resumed per tier policy.")
    if "mute" in q and intent.target_member_name:
        members = store.get_household_members(household_id)
        member = next((m for m in members if m.name.lower() == intent.target_member_name.lower()), None)
        if member:
            gov.mute_member(member.id, "Muted via Slack by operator")
            return RouteResult(success=True, message=f":mute: *{member.name}* muted for non-critical reminders.")
        return RouteResult(success=False, message=f"Member '{intent.target_member_name}' not found.")
    if "unmute" in q and intent.target_member_name:
        members = store.get_household_members(household_id)
        member = next((m for m in members if m.name.lower() == intent.target_member_name.lower()), None)
        if member:
            gov.unmute_member(member.id)
            return RouteResult(success=True, message=f":loud_sound: *{member.name}* unmuted.")
        return RouteResult(success=False, message=f"Member '{intent.target_member_name}' not found.")
    if "review" in q:
        review = gov.generate_review_summary(household_id)
        lines = [
            "*Governance Review*",
            f"Tier: {review['tier']} | Allowed: {review['allowed']} | Blocked: {review['blocked']}",
            f"Budget: {review['daily_budget']} | Muted: {review['muted_members']}",
        ]
        return RouteResult(success=True, message="\n".join(lines))

    # Default: governance status
    summary = gov.get_governance_summary(household_id)
    ks = ":octagonal_sign: ACTIVE" if summary["kill_switch_active"] else ":white_check_mark: Off"
    lines = [
        "*Governance Status*",
        f"*Tier:* {summary['tier']} ({summary['tier_name']})",
        f"*Kill switch:* {ks}",
        f"*Budget:* {summary['daily_count']}/{summary['daily_budget']} ({summary['budget_remaining']} remaining)",
    ]
    return RouteResult(success=True, message="\n".join(lines))


def _route_calendar_query(intent: ParsedIntent, store: ReminderStore, household_id: UUID) -> RouteResult:
    return RouteResult(success=True, message="Calendar queries require the calendar adapter to be configured. Try `@Woodcreek active reminders` for now.")


def _route_suggestion_query(intent: ParsedIntent, store: ReminderStore, household_id: UUID) -> RouteResult:
    from dacribagents.application.services.suggestions import generate_suggestions  # noqa: PLC0415

    suggestions = generate_suggestions(store, household_id, [])
    if not suggestions:
        return RouteResult(success=True, message="*Suggestions*\nNo suggestions at this time.")
    lines = [f"*Suggested Reminders* ({len(suggestions)} items)"]
    for s in suggestions:
        lines.append(f":bulb: *{s.subject}* — _{s.reason}_")
    return RouteResult(success=True, message="\n".join(lines))


# ── Helpers ─────────────────────────────────────────────────────────────────


def _format_reminder_list(items: list, title: str) -> RouteResult:
    if not items:
        return RouteResult(success=True, message=f"*{title}*\nNo items found.")
    lines = [f"*{title}* ({len(items)} items)"]
    for r in items[:15]:
        lines.append(f":bell: `{str(r.id)[:8]}` *{r.subject}* | `{r.state.value}` | {r.source.value}")
    return RouteResult(success=True, message="\n".join(lines))
