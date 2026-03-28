"""Reminder lifecycle state machine with transition rules and guardrails."""

from __future__ import annotations

from dacribagents.domain.reminders.enums import ReminderState


class InvalidTransitionError(Exception):
    """Raised when a reminder state transition violates lifecycle rules."""

    def __init__(self, current: ReminderState, target: ReminderState) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Cannot transition from {current.value!r} to {target.value!r}")


# ── Valid transition map ────────────────────────────────────────────────────
#
# Approval path:    DRAFT → PENDING_APPROVAL → APPROVED → SCHEDULED → ...
# No-approval path: DRAFT → SCHEDULED → ...

VALID_TRANSITIONS: dict[ReminderState, frozenset[ReminderState]] = {
    ReminderState.DRAFT: frozenset({
        ReminderState.PENDING_APPROVAL,
        ReminderState.SCHEDULED,
        ReminderState.CANCELLED,
    }),
    ReminderState.PENDING_APPROVAL: frozenset({
        ReminderState.APPROVED,
        ReminderState.REJECTED,
        ReminderState.CANCELLED,
    }),
    ReminderState.APPROVED: frozenset({
        ReminderState.SCHEDULED,
        ReminderState.CANCELLED,
    }),
    ReminderState.SCHEDULED: frozenset({
        ReminderState.PENDING_DELIVERY,
        ReminderState.CANCELLED,
    }),
    ReminderState.PENDING_DELIVERY: frozenset({
        ReminderState.DELIVERED,
        ReminderState.FAILED,
        ReminderState.CANCELLED,
    }),
    ReminderState.DELIVERED: frozenset({
        ReminderState.ACKNOWLEDGED,
        ReminderState.SNOOZED,
        ReminderState.CANCELLED,
        ReminderState.FAILED,
    }),
    ReminderState.SNOOZED: frozenset({
        ReminderState.PENDING_DELIVERY,
        ReminderState.CANCELLED,
    }),
    # Terminal states — no outbound transitions.
    ReminderState.REJECTED: frozenset(),
    ReminderState.ACKNOWLEDGED: frozenset(),
    ReminderState.CANCELLED: frozenset(),
    ReminderState.FAILED: frozenset(),
}

TERMINAL_STATES: frozenset[ReminderState] = frozenset({
    ReminderState.REJECTED,
    ReminderState.ACKNOWLEDGED,
    ReminderState.CANCELLED,
    ReminderState.FAILED,
})


def validate_transition(current: ReminderState, target: ReminderState) -> None:
    """Raise InvalidTransitionError if *current* -> *target* is not allowed."""
    allowed = VALID_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidTransitionError(current, target)


def can_transition(current: ReminderState, target: ReminderState) -> bool:
    """Return True if *current* -> *target* is a valid lifecycle transition."""
    return target in VALID_TRANSITIONS.get(current, frozenset())


def is_terminal(state: ReminderState) -> bool:
    """Return True if *state* is a terminal (no further transitions)."""
    return state in TERMINAL_STATES


def reachable_states(current: ReminderState) -> frozenset[ReminderState]:
    """Return the set of states directly reachable from *current*."""
    return VALID_TRANSITIONS.get(current, frozenset())
