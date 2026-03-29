"""Channel selection policy — deterministic rules for choosing delivery channels.

Resolution order:
1. Member-specific PreferenceRule for the given urgency (most specific)
2. Member-specific PreferenceRule with urgency=None (member default)
3. Household-wide PreferenceRule for the given urgency
4. Household-wide PreferenceRule with urgency=None (household default)
5. Hard-coded defaults based on urgency/intent

Quiet hours: suppress non-critical delivery if current time falls within
the member's quiet window.  CRITICAL urgency always bypasses quiet hours.
PreferenceRules with ``quiet_hours_override=True`` also bypass.
"""

from __future__ import annotations

from datetime import time
from uuid import UUID

from dacribagents.application.ports.reminder_policy import ChannelSelection
from dacribagents.domain.reminders.entities import PreferenceRule
from dacribagents.domain.reminders.enums import (
    DeliveryChannel,
    NotificationIntent,
    UrgencyLevel,
)


def select_channel(  # noqa: PLR0913, PLR0911
    *,
    member_id: UUID,
    phone: str | None,
    email: str | None,
    slack_id: str | None,
    urgency: UrgencyLevel,
    intent: NotificationIntent,
    preference_rules: list[PreferenceRule] | None = None,
    quiet_hours_start: time | None = None,
    quiet_hours_end: time | None = None,
    current_time: time | None = None,
) -> ChannelSelection:
    """Select primary and fallback channels for a single member."""
    # ── Quiet hours check ───────────────────────────────────────────
    rule = _resolve_rule(preference_rules or [], member_id, urgency)
    bypass_quiet = urgency == UrgencyLevel.CRITICAL or (rule and rule.quiet_hours_override)

    if not bypass_quiet and _in_quiet_hours(quiet_hours_start, quiet_hours_end, current_time):
        return ChannelSelection(
            member_id=member_id,
            primary_channel=DeliveryChannel.EMAIL,
            primary_address=email or "",
            suppressed=True,
            suppression_reason="Deferred: quiet hours active (non-critical)",
        )

    # ── PreferenceRule-driven selection ──────────────────────────────
    if rule:
        primary = rule.preferred_channel
        fallback = rule.fallback_channel
        primary_addr = _address_for(primary, phone, email, slack_id)
        fallback_addr = _address_for(fallback, phone, email, slack_id) if fallback else None

        if primary_addr:
            return ChannelSelection(
                member_id=member_id,
                primary_channel=primary,
                primary_address=primary_addr,
                fallback_channel=fallback if fallback_addr else None,
                fallback_address=fallback_addr,
            )
        # Preferred channel not available — fall through to defaults

    # ── Default rules (no matching preference) ──────────────────────
    return _default_selection(member_id, phone, email, urgency, intent)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _resolve_rule(
    rules: list[PreferenceRule],
    member_id: UUID,
    urgency: UrgencyLevel,
) -> PreferenceRule | None:
    """Find the best matching active rule (most specific wins)."""
    candidates = [r for r in rules if r.active]

    # 1. Member + urgency match
    for r in candidates:
        if r.member_id == member_id and r.urgency == urgency:
            return r
    # 2. Member default (urgency=None)
    for r in candidates:
        if r.member_id == member_id and r.urgency is None:
            return r
    # 3. Household + urgency (member_id=None)
    for r in candidates:
        if r.member_id is None and r.urgency == urgency:
            return r
    # 4. Household default
    for r in candidates:
        if r.member_id is None and r.urgency is None:
            return r
    return None


def _in_quiet_hours(
    start: time | None,
    end: time | None,
    now: time | None,
) -> bool:
    """Return True if *now* falls within the quiet window."""
    if not start or not end or not now:
        return False
    if start <= end:
        return start <= now < end
    # Wraps midnight: e.g., 22:00 → 07:00
    return now >= start or now < end


def _address_for(
    channel: DeliveryChannel | None,
    phone: str | None,
    email: str | None,
    slack_id: str | None,
) -> str | None:
    if channel == DeliveryChannel.SMS:
        return phone
    if channel == DeliveryChannel.EMAIL:
        return email
    if channel == DeliveryChannel.SLACK:
        return slack_id
    return None


def _default_selection(  # noqa: PLR0911
    member_id: UUID,
    phone: str | None,
    email: str | None,
    urgency: UrgencyLevel,
    intent: NotificationIntent,
) -> ChannelSelection:
    """Hard-coded defaults when no PreferenceRule matches."""
    if intent == NotificationIntent.DIGEST:
        if email:
            return ChannelSelection(member_id=member_id, primary_channel=DeliveryChannel.EMAIL, primary_address=email)
        return ChannelSelection(
            member_id=member_id, primary_channel=DeliveryChannel.EMAIL, primary_address="",
            suppressed=True, suppression_reason="No email address for digest delivery",
        )

    if urgency in {UrgencyLevel.CRITICAL, UrgencyLevel.URGENT}:
        if phone:
            return ChannelSelection(
                member_id=member_id, primary_channel=DeliveryChannel.SMS, primary_address=phone,
                fallback_channel=DeliveryChannel.EMAIL if email else None, fallback_address=email,
            )
        if email:
            return ChannelSelection(member_id=member_id, primary_channel=DeliveryChannel.EMAIL, primary_address=email)

    # Normal / low urgency: email primary, SMS fallback
    if email:
        return ChannelSelection(
            member_id=member_id, primary_channel=DeliveryChannel.EMAIL, primary_address=email,
            fallback_channel=DeliveryChannel.SMS if phone else None, fallback_address=phone,
        )
    if phone:
        return ChannelSelection(member_id=member_id, primary_channel=DeliveryChannel.SMS, primary_address=phone)

    return ChannelSelection(
        member_id=member_id, primary_channel=DeliveryChannel.EMAIL, primary_address="",
        suppressed=True, suppression_reason="No phone or email address available",
    )
