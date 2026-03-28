"""Deterministic approval policy for reminders.

Evaluates whether a reminder requires human approval before scheduling.
This is a pure function of the reminder's attributes — no LLM, no external
calls.  The policy can be extended with additional rules in future phases.

Default behavior:
- USER and HOUSEHOLD_ROUTINE sources → no approval required
- All other sources (AGENT, TELEMETRY, HOA, etc.) → approval required
- Explicit ``requires_approval=True`` on the request → always requires approval
"""

from __future__ import annotations

from dataclasses import dataclass

from dacribagents.domain.reminders.enums import ReminderSource, UrgencyLevel


@dataclass(frozen=True)
class ApprovalDecision:
    """Result of the approval policy evaluation."""

    required: bool
    reason: str | None


# Sources that are considered safe — created by the operator directly.
_NO_APPROVAL_SOURCES: frozenset[ReminderSource] = frozenset({
    ReminderSource.USER,
    ReminderSource.HOUSEHOLD_ROUTINE,
})


def evaluate(
    source: ReminderSource,
    urgency: UrgencyLevel = UrgencyLevel.NORMAL,
    requires_approval_override: bool = False,
    source_agent: str | None = None,
) -> ApprovalDecision:
    """Determine whether a reminder needs approval.

    Args:
        source: Where the reminder originated.
        urgency: Urgency level (reserved for future policy expansion).
        requires_approval_override: If True, approval is always required.
        source_agent: Agent name (for the explanation string).

    Returns:
        ApprovalDecision with ``required`` flag and ``reason``.

    """
    if requires_approval_override:
        return ApprovalDecision(required=True, reason="Explicitly flagged for approval.")

    if source in _NO_APPROVAL_SOURCES:
        return ApprovalDecision(required=False, reason=None)

    agent_desc = f" ({source_agent})" if source_agent else ""
    return ApprovalDecision(
        required=True,
        reason=f"Reminders from '{source.value}'{agent_desc} sources require human approval before delivery.",
    )
