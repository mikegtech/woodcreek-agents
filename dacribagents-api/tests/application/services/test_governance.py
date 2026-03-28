"""Tests for governance: autonomy tiers, kill switch, budgets, audit."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from dacribagents.application.services.governance import (
    AutonomyTier,
    GovernanceState,
    activate_kill_switch,
    deactivate_kill_switch,
    evaluate,
    get_audit_log,
    get_daily_count,
    get_governance_state,
    get_governance_summary,
    is_budget_exceeded,
    is_kill_switch_active,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset governance state before each test."""
    import dacribagents.application.services.governance as mod

    mod._state = GovernanceState()
    yield
    mod._state = None


@pytest.fixture()
def household_id():
    return uuid4()


# ── Tier model ──────────────────────────────────────────────────────────────


def test_default_tier_is_zero(household_id):
    state = get_governance_state()
    assert state.get_tier(household_id) == AutonomyTier.TIER_0


def test_set_and_get_tier(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)
    assert state.get_tier(household_id) == AutonomyTier.TIER_2


def test_tier_ordering():
    assert AutonomyTier.TIER_0 < AutonomyTier.TIER_1
    assert AutonomyTier.TIER_1 < AutonomyTier.TIER_2
    assert AutonomyTier.TIER_2 < AutonomyTier.TIER_3


# ── Kill switch ─────────────────────────────────────────────────────────────


def test_kill_switch_default_off():
    assert is_kill_switch_active() is False


def test_kill_switch_activate():
    activate_kill_switch("test")
    assert is_kill_switch_active() is True
    state = get_governance_state()
    assert state.kill_switch_activated_by == "test"


def test_kill_switch_deactivate():
    activate_kill_switch("test")
    deactivate_kill_switch("test")
    assert is_kill_switch_active() is False


def test_kill_switch_blocks_autonomous(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_3)
    activate_kill_switch("test")

    decision = evaluate(
        household_id=household_id,
        action_type="auto_send",
        requires_tier=AutonomyTier.TIER_2,
    )
    assert decision.allowed is False
    assert decision.blocked_by_kill_switch is True
    assert decision.requires_approval is True


def test_kill_switch_allows_tier0_actions(household_id):
    """Tier 0 actions (human-driven) are not blocked by kill switch."""
    activate_kill_switch("test")

    decision = evaluate(
        household_id=household_id,
        action_type="manual_create",
        requires_tier=AutonomyTier.TIER_0,
    )
    # Tier 0 doesn't require autonomous permission, so kill switch doesn't block
    assert decision.blocked_by_kill_switch is False


def test_kill_switch_audit_trail():
    activate_kill_switch("operator")
    log = get_audit_log()
    assert any(e.action_type == "kill_switch.activated" for e in log)


# ── Budget / rate limiting ──────────────────────────────────────────────────


def test_budget_default_not_exceeded(household_id):
    assert is_budget_exceeded(household_id) is False
    assert get_daily_count(household_id) == 0


def test_budget_increments(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    # Each allowed autonomous action increments the count
    evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)
    assert get_daily_count(household_id) == 1


def test_budget_exceeded_blocks(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_3)
    state.daily_budget = 3

    # Exhaust budget
    for _ in range(3):
        evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)

    assert is_budget_exceeded(household_id) is True

    decision = evaluate(
        household_id=household_id,
        action_type="auto_send",
        requires_tier=AutonomyTier.TIER_2,
    )
    assert decision.allowed is False
    assert decision.blocked_by_budget is True


# ── Governance evaluation ───────────────────────────────────────────────────


def test_tier_too_low_requires_approval(household_id):
    # Default tier 0, action requires tier 2
    decision = evaluate(
        household_id=household_id,
        action_type="auto_send",
        requires_tier=AutonomyTier.TIER_2,
    )
    assert decision.allowed is False
    assert decision.requires_approval is True
    assert "tier" in decision.reason.lower()


def test_tier_sufficient_allows(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    decision = evaluate(
        household_id=household_id,
        action_type="auto_send",
        requires_tier=AutonomyTier.TIER_2,
    )
    assert decision.allowed is True
    assert decision.requires_approval is False


def test_tier3_allows_escalation(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_3)

    decision = evaluate(
        household_id=household_id,
        action_type="auto_escalate",
        requires_tier=AutonomyTier.TIER_3,
    )
    assert decision.allowed is True


def test_human_actions_always_allowed(household_id):
    """Tier 0 actions (manual, operator-driven) are always allowed."""
    decision = evaluate(
        household_id=household_id,
        action_type="manual_create",
        actor_type="human",
        requires_tier=AutonomyTier.TIER_0,
    )
    assert decision.allowed is True


# ── Audit trail ─────────────────────────────────────────────────────────────


def test_audit_records_decisions(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)
    evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_3)

    log = get_audit_log(household_id)
    assert len(log) == 2
    assert log[0].decision == "allowed"
    assert log[1].decision == "approval_required"


def test_audit_includes_tier_and_reason(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_1)

    evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)

    log = get_audit_log(household_id)
    entry = log[-1]
    assert entry.tier == AutonomyTier.TIER_1
    assert entry.action_type == "auto_send"
    assert "tier" in entry.reason.lower()


# ── Governance summary ──────────────────────────────────────────────────────


def test_governance_summary(household_id):
    state = get_governance_state()
    state.set_tier(household_id, AutonomyTier.TIER_2)

    evaluate(household_id=household_id, action_type="auto_send", requires_tier=AutonomyTier.TIER_2)
    evaluate(household_id=household_id, action_type="auto_escalate", requires_tier=AutonomyTier.TIER_3)

    summary = get_governance_summary(household_id)
    assert summary["tier"] == 2
    assert summary["tier_name"] == "TIER_2"
    assert summary["daily_count"] == 1
    assert summary["recent_allowed"] == 1
    assert summary["recent_approval_required"] == 1
    assert summary["kill_switch_active"] is False
