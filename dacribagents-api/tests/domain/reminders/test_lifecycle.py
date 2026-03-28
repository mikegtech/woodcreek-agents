"""Tests for reminder lifecycle state machine."""

from __future__ import annotations

import pytest

from dacribagents.domain.reminders.enums import ReminderState
from dacribagents.domain.reminders.lifecycle import (
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    InvalidTransitionError,
    can_transition,
    is_terminal,
    reachable_states,
    validate_transition,
)


# ── Every state is accounted for in the transition map ──────────────────────


def test_all_states_have_transition_entry():
    for state in ReminderState:
        assert state in VALID_TRANSITIONS, f"{state} missing from VALID_TRANSITIONS"


# ── Happy-path transitions ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("current", "target"),
    [
        # No-approval path
        (ReminderState.DRAFT, ReminderState.SCHEDULED),
        (ReminderState.DRAFT, ReminderState.CANCELLED),
        # Approval path
        (ReminderState.DRAFT, ReminderState.PENDING_APPROVAL),
        (ReminderState.PENDING_APPROVAL, ReminderState.APPROVED),
        (ReminderState.PENDING_APPROVAL, ReminderState.REJECTED),
        (ReminderState.PENDING_APPROVAL, ReminderState.CANCELLED),
        (ReminderState.APPROVED, ReminderState.SCHEDULED),
        (ReminderState.APPROVED, ReminderState.CANCELLED),
        # Delivery path
        (ReminderState.SCHEDULED, ReminderState.PENDING_DELIVERY),
        (ReminderState.SCHEDULED, ReminderState.CANCELLED),
        (ReminderState.PENDING_DELIVERY, ReminderState.DELIVERED),
        (ReminderState.PENDING_DELIVERY, ReminderState.FAILED),
        (ReminderState.PENDING_DELIVERY, ReminderState.CANCELLED),
        (ReminderState.DELIVERED, ReminderState.ACKNOWLEDGED),
        (ReminderState.DELIVERED, ReminderState.SNOOZED),
        (ReminderState.DELIVERED, ReminderState.CANCELLED),
        (ReminderState.DELIVERED, ReminderState.FAILED),
        (ReminderState.SNOOZED, ReminderState.PENDING_DELIVERY),
        (ReminderState.SNOOZED, ReminderState.CANCELLED),
    ],
)
def test_valid_transition(current: ReminderState, target: ReminderState):
    assert can_transition(current, target) is True
    validate_transition(current, target)  # should not raise


# ── Invalid transitions ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("current", "target"),
    [
        # Can't go backwards
        (ReminderState.SCHEDULED, ReminderState.DRAFT),
        (ReminderState.DELIVERED, ReminderState.SCHEDULED),
        (ReminderState.ACKNOWLEDGED, ReminderState.DELIVERED),
        (ReminderState.APPROVED, ReminderState.PENDING_APPROVAL),
        # Can't skip states
        (ReminderState.DRAFT, ReminderState.DELIVERED),
        (ReminderState.DRAFT, ReminderState.ACKNOWLEDGED),
        (ReminderState.SCHEDULED, ReminderState.ACKNOWLEDGED),
        (ReminderState.PENDING_APPROVAL, ReminderState.SCHEDULED),  # must go through APPROVED
        # Terminal states can't transition
        (ReminderState.ACKNOWLEDGED, ReminderState.CANCELLED),
        (ReminderState.CANCELLED, ReminderState.DRAFT),
        (ReminderState.FAILED, ReminderState.PENDING_DELIVERY),
        (ReminderState.REJECTED, ReminderState.DRAFT),
        (ReminderState.REJECTED, ReminderState.APPROVED),
        # Self-transitions not allowed
        (ReminderState.DRAFT, ReminderState.DRAFT),
        (ReminderState.DELIVERED, ReminderState.DELIVERED),
    ],
)
def test_invalid_transition(current: ReminderState, target: ReminderState):
    assert can_transition(current, target) is False
    with pytest.raises(InvalidTransitionError) as exc_info:
        validate_transition(current, target)
    assert exc_info.value.current == current
    assert exc_info.value.target == target


# ── Terminal states ─────────────────────────────────────────────────────────


def test_terminal_states_are_correct():
    assert TERMINAL_STATES == frozenset({
        ReminderState.REJECTED,
        ReminderState.ACKNOWLEDGED,
        ReminderState.CANCELLED,
        ReminderState.FAILED,
    })


@pytest.mark.parametrize("state", list(TERMINAL_STATES))
def test_terminal_state_has_no_outbound(state: ReminderState):
    assert is_terminal(state) is True
    assert reachable_states(state) == frozenset()


@pytest.mark.parametrize(
    "state",
    [s for s in ReminderState if s not in TERMINAL_STATES],
)
def test_non_terminal_state_has_outbound(state: ReminderState):
    assert is_terminal(state) is False
    assert len(reachable_states(state)) > 0


# ── Cancellation is reachable from all non-terminal states ──────────────────


@pytest.mark.parametrize(
    "state",
    [s for s in ReminderState if s not in TERMINAL_STATES],
)
def test_cancel_reachable_from_all_non_terminal(state: ReminderState):
    assert can_transition(state, ReminderState.CANCELLED)


# ── Approval path ───────────────────────────────────────────────────────────


def test_approval_path():
    """DRAFT -> PENDING_APPROVAL -> APPROVED -> SCHEDULED is valid."""
    validate_transition(ReminderState.DRAFT, ReminderState.PENDING_APPROVAL)
    validate_transition(ReminderState.PENDING_APPROVAL, ReminderState.APPROVED)
    validate_transition(ReminderState.APPROVED, ReminderState.SCHEDULED)


def test_rejection_is_terminal():
    validate_transition(ReminderState.PENDING_APPROVAL, ReminderState.REJECTED)
    assert is_terminal(ReminderState.REJECTED)


def test_cannot_skip_approval_to_scheduled():
    """PENDING_APPROVAL -> SCHEDULED is invalid — must go through APPROVED."""
    assert not can_transition(ReminderState.PENDING_APPROVAL, ReminderState.SCHEDULED)


# ── Snooze cycle ────────────────────────────────────────────────────────────


def test_snooze_cycle():
    """DELIVERED -> SNOOZED -> PENDING_DELIVERY -> DELIVERED is valid."""
    validate_transition(ReminderState.DELIVERED, ReminderState.SNOOZED)
    validate_transition(ReminderState.SNOOZED, ReminderState.PENDING_DELIVERY)
    validate_transition(ReminderState.PENDING_DELIVERY, ReminderState.DELIVERED)


# ── InvalidTransitionError message ──────────────────────────────────────────


def test_error_message_includes_states():
    err = InvalidTransitionError(ReminderState.DRAFT, ReminderState.ACKNOWLEDGED)
    assert "draft" in str(err)
    assert "acknowledged" in str(err)
