"""Governance service — autonomy tiers, kill switch, budgets, and audit.

Controls what the system may do autonomously vs. what requires human
approval.  All governance decisions are deterministic, inspectable, and
auditable.

Autonomy Tiers:
- **Tier 0** — human approval required for all writes.
- **Tier 1** — delivery allowed after approval; no autonomous creation.
- **Tier 2** — pre-approved reminder types may auto-send.
- **Tier 3** — bounded autonomous escalation / follow-on behavior.

Kill switch: instantly disables all autonomous actions (Tier >= 2)
without affecting manual/operator workflows.

Budget: caps autonomous actions per household per day.  Exceeded budget
forces escalation to human approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from uuid import UUID, uuid4

from loguru import logger

# ── Autonomy tiers ──────────────────────────────────────────────────────────


class AutonomyTier(IntEnum):
    """Governance tiers from most restrictive to most autonomous."""

    TIER_0 = 0  # Human approves all writes
    TIER_1 = 1  # Delivery on approval; no autonomous creation
    TIER_2 = 2  # Pre-approved types auto-send
    TIER_3 = 3  # Bounded autonomous escalation


# ── Governance decision ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class GovernanceDecision:
    """Result of a governance policy evaluation."""

    allowed: bool
    requires_approval: bool
    tier: AutonomyTier
    reason: str
    blocked_by_kill_switch: bool = False
    blocked_by_budget: bool = False


# ── Audit record ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GovernanceAuditEntry:
    """Audit record for an autonomous or governance-mediated action."""

    id: UUID
    timestamp: datetime
    household_id: UUID
    action_type: str
    actor_type: str  # "human", "system", "autonomous_policy"
    tier: AutonomyTier
    decision: str  # "allowed", "blocked", "approval_required"
    reason: str
    reminder_id: UUID | None = None
    metadata: dict = field(default_factory=dict)


# ── Governance state (in-memory for MVP) ────────────────────────────────────


class GovernanceState:
    """Mutable governance state — kill switch and budget tracking."""

    def __init__(self) -> None:
        self.kill_switch_active: bool = False
        self.kill_switch_activated_at: datetime | None = None
        self.kill_switch_activated_by: str | None = None
        self.household_tiers: dict[UUID, AutonomyTier] = {}
        self.action_counts: dict[str, int] = {}  # "household_id:date" → count
        self.audit_log: list[GovernanceAuditEntry] = []
        self.daily_budget: int = 50  # max autonomous actions per household per day
        self.muted_members: dict[UUID, str] = {}  # member_id → reason

    def get_tier(self, household_id: UUID) -> AutonomyTier:
        """Return the autonomy tier for a household (default: Tier 0)."""
        return self.household_tiers.get(household_id, AutonomyTier.TIER_0)

    def set_tier(self, household_id: UUID, tier: AutonomyTier) -> None:
        """Set the autonomy tier for a household."""
        self.household_tiers[household_id] = tier


_state: GovernanceState | None = None


def get_governance_state() -> GovernanceState:
    """Singleton accessor for governance state."""
    global _state  # noqa: PLW0603
    if _state is None:
        _state = GovernanceState()
    return _state


# ── Kill switch ─────────────────────────────────────────────────────────────


def activate_kill_switch(activated_by: str = "operator") -> None:
    """Instantly disable all autonomous actions (Tier >= 2)."""
    state = get_governance_state()
    state.kill_switch_active = True
    state.kill_switch_activated_at = datetime.now(UTC)
    state.kill_switch_activated_by = activated_by
    logger.warning(f"KILL SWITCH ACTIVATED by {activated_by}")

    state.audit_log.append(GovernanceAuditEntry(
        id=uuid4(), timestamp=datetime.now(UTC),
        household_id=UUID(int=0), action_type="kill_switch.activated",
        actor_type="human", tier=AutonomyTier.TIER_0,
        decision="blocked", reason=f"Kill switch activated by {activated_by}",
    ))


def deactivate_kill_switch(deactivated_by: str = "operator") -> None:
    """Re-enable autonomous actions."""
    state = get_governance_state()
    state.kill_switch_active = False
    state.kill_switch_activated_at = None
    state.kill_switch_activated_by = None
    logger.info(f"Kill switch deactivated by {deactivated_by}")

    state.audit_log.append(GovernanceAuditEntry(
        id=uuid4(), timestamp=datetime.now(UTC),
        household_id=UUID(int=0), action_type="kill_switch.deactivated",
        actor_type="human", tier=AutonomyTier.TIER_0,
        decision="allowed", reason=f"Kill switch deactivated by {deactivated_by}",
    ))


def is_kill_switch_active() -> bool:
    """Check if the kill switch is currently active."""
    return get_governance_state().kill_switch_active


# ── Budget / rate limiting ──────────────────────────────────────────────────


def _budget_key(household_id: UUID, date: datetime | None = None) -> str:
    d = (date or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"{household_id}:{d}"


def get_daily_count(household_id: UUID, date: datetime | None = None) -> int:
    """Return the number of autonomous actions today for a household."""
    state = get_governance_state()
    return state.action_counts.get(_budget_key(household_id, date), 0)


def increment_count(household_id: UUID, date: datetime | None = None) -> int:
    """Increment and return the daily autonomous action count."""
    state = get_governance_state()
    key = _budget_key(household_id, date)
    state.action_counts[key] = state.action_counts.get(key, 0) + 1
    return state.action_counts[key]


def is_budget_exceeded(household_id: UUID, date: datetime | None = None) -> bool:
    """Check if the household has exceeded its daily autonomy budget."""
    state = get_governance_state()
    return get_daily_count(household_id, date) >= state.daily_budget


# ── Governance decision ─────────────────────────────────────────────────────


def evaluate(  # noqa: PLR0913
    *,
    household_id: UUID,
    action_type: str,
    actor_type: str = "system",
    requires_tier: AutonomyTier = AutonomyTier.TIER_2,
    reminder_id: UUID | None = None,
) -> GovernanceDecision:
    """Evaluate whether an action is allowed under current governance rules.

    Args:
        household_id: The household scope.
        action_type: What is being attempted (e.g., "auto_send", "auto_escalate").
        actor_type: Who/what is performing the action.
        requires_tier: Minimum tier needed for autonomous execution.
        reminder_id: Optional reminder for audit context.

    Returns:
        GovernanceDecision with allow/block/approval-required ruling.

    """
    state = get_governance_state()
    now = datetime.now(UTC)
    tier = state.get_tier(household_id)

    # Kill switch blocks all autonomous actions
    if state.kill_switch_active and requires_tier >= AutonomyTier.TIER_2:
        decision = GovernanceDecision(
            allowed=False, requires_approval=True, tier=tier,
            reason="Kill switch is active — all autonomous actions require approval",
            blocked_by_kill_switch=True,
        )
        _record_audit(state, household_id, action_type, actor_type, tier, "blocked", decision.reason, reminder_id, now)
        return decision

    # Tier check
    if tier < requires_tier:
        decision = GovernanceDecision(
            allowed=False, requires_approval=True, tier=tier,
            reason=f"Household tier {tier.value} is below required tier {requires_tier.value} for {action_type}",
        )
        _record_audit(state, household_id, action_type, actor_type, tier, "approval_required", decision.reason, reminder_id, now)
        return decision

    # Budget check
    if is_budget_exceeded(household_id):
        decision = GovernanceDecision(
            allowed=False, requires_approval=True, tier=tier,
            reason=f"Daily autonomy budget exceeded ({state.daily_budget} actions/day)",
            blocked_by_budget=True,
        )
        _record_audit(state, household_id, action_type, actor_type, tier, "blocked", decision.reason, reminder_id, now)
        return decision

    # Allowed
    increment_count(household_id)
    decision = GovernanceDecision(
        allowed=True, requires_approval=False, tier=tier,
        reason=f"Allowed: tier {tier.value} >= required {requires_tier.value}",
    )
    _record_audit(state, household_id, action_type, actor_type, tier, "allowed", decision.reason, reminder_id, now)
    return decision


# ── Queries ─────────────────────────────────────────────────────────────────


def get_audit_log(
    household_id: UUID | None = None,
    limit: int = 50,
) -> list[GovernanceAuditEntry]:
    """Return recent governance audit entries."""
    state = get_governance_state()
    entries = state.audit_log
    if household_id:
        entries = [e for e in entries if e.household_id == household_id or e.household_id == UUID(int=0)]
    return entries[-limit:]


def get_governance_summary(household_id: UUID) -> dict:
    """Generate a governance status summary for operator visibility."""
    state = get_governance_state()
    tier = state.get_tier(household_id)
    count = get_daily_count(household_id)

    recent = get_audit_log(household_id, limit=20)
    allowed = sum(1 for e in recent if e.decision == "allowed")
    blocked = sum(1 for e in recent if e.decision == "blocked")
    approval_req = sum(1 for e in recent if e.decision == "approval_required")

    return {
        "household_id": str(household_id),
        "tier": tier.value,
        "tier_name": tier.name,
        "kill_switch_active": state.kill_switch_active,
        "kill_switch_activated_by": state.kill_switch_activated_by,
        "daily_count": count,
        "daily_budget": state.daily_budget,
        "budget_remaining": max(0, state.daily_budget - count),
        "recent_allowed": allowed,
        "recent_blocked": blocked,
        "recent_approval_required": approval_req,
    }


# ── Internal ────────────────────────────────────────────────────────────────


# ── Member mute / opt-out ───────────────────────────────────────────────────


def mute_member(member_id: UUID, reason: str = "Operator muted") -> None:
    """Mute non-critical reminder delivery for a member."""
    state = get_governance_state()
    state.muted_members[member_id] = reason
    logger.info(f"Member {member_id} muted: {reason}")
    state.audit_log.append(GovernanceAuditEntry(
        id=uuid4(), timestamp=datetime.now(UTC),
        household_id=UUID(int=0), action_type="member.muted",
        actor_type="human", tier=AutonomyTier.TIER_0,
        decision="allowed", reason=reason,
        metadata={"member_id": str(member_id)},
    ))


def unmute_member(member_id: UUID) -> None:
    """Unmute a member — resume normal delivery."""
    state = get_governance_state()
    state.muted_members.pop(member_id, None)
    logger.info(f"Member {member_id} unmuted")


def is_member_muted(member_id: UUID) -> bool:
    """Check if a member has muted non-critical reminders."""
    return member_id in get_governance_state().muted_members


def get_mute_reason(member_id: UUID) -> str | None:
    """Return the mute reason for a member, or None if not muted."""
    return get_governance_state().muted_members.get(member_id)


def list_muted_members() -> dict[UUID, str]:
    """Return all muted members and their reasons."""
    return dict(get_governance_state().muted_members)


# ── Tier 2/3 auto-approve helpers ───────────────────────────────────────────

# Sources that are pre-approved for auto-send at Tier 2
from dacribagents.domain.reminders.enums import ReminderSource  # noqa: E402

TIER2_AUTO_SEND_SOURCES: frozenset[ReminderSource] = frozenset({
    ReminderSource.HOUSEHOLD_ROUTINE,
})


def is_tier2_auto_eligible(source: ReminderSource) -> bool:
    """Return True if this source qualifies for Tier 2 auto-send."""
    return source in TIER2_AUTO_SEND_SOURCES


# ── Governance review summary ───────────────────────────────────────────────


def generate_review_summary(household_id: UUID) -> dict:
    """Generate a governance review summary for the household.

    Intended for monthly operator review (delivered via digest or Slack).
    """
    state = get_governance_state()
    entries = [e for e in state.audit_log if e.household_id == household_id or e.household_id == UUID(int=0)]

    allowed = [e for e in entries if e.decision == "allowed"]
    blocked = [e for e in entries if e.decision == "blocked"]
    approval_req = [e for e in entries if e.decision == "approval_required"]

    action_types = {}
    for e in entries:
        action_types[e.action_type] = action_types.get(e.action_type, 0) + 1

    kill_switch_events = [e for e in entries if "kill_switch" in e.action_type]
    mute_events = [e for e in entries if "muted" in e.action_type]

    return {
        "household_id": str(household_id),
        "tier": state.get_tier(household_id).value,
        "total_entries": len(entries),
        "allowed": len(allowed),
        "blocked": len(blocked),
        "approval_required": len(approval_req),
        "action_type_counts": action_types,
        "kill_switch_events": len(kill_switch_events),
        "mute_events": len(mute_events),
        "muted_members": len(state.muted_members),
        "daily_budget": state.daily_budget,
    }


# ── Internal ────────────────────────────────────────────────────────────────


def _record_audit(state, household_id, action_type, actor_type, tier, decision, reason, reminder_id, now):  # noqa: PLR0913
    state.audit_log.append(GovernanceAuditEntry(
        id=uuid4(), timestamp=now,
        household_id=household_id, action_type=action_type,
        actor_type=actor_type, tier=tier,
        decision=decision, reason=reason, reminder_id=reminder_id,
    ))
