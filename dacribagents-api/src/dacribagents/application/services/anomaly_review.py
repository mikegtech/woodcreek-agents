"""Bounded anomaly review — deterministic flags for unusual patterns.

Checks rule-based thresholds and flags households or behaviors for
operator review.  This is NOT an ML system — thresholds are explicit
and inspectable.

Anomaly rules:
1. Volume spike: daily reminder count > 3x recent average
2. Repeated escalation: >3 escalations to same member in 24h
3. Kill-switch churn: >2 kill-switch toggles in 24h
4. Delivery failure spike: >30% failure rate in a day
5. Budget exhaustion: daily budget hit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from dacribagents.application.services.governance import get_audit_log, get_daily_count, get_governance_state


@dataclass(frozen=True)
class AnomalyFlag:
    """A flagged anomaly for operator review."""

    rule: str
    severity: str  # "warning", "critical"
    description: str
    household_id: UUID
    metadata: dict = field(default_factory=dict)


def check_anomalies(household_id: UUID) -> list[AnomalyFlag]:
    """Run all anomaly checks for a household. Returns flags for review."""
    flags: list[AnomalyFlag] = []
    state = get_governance_state()
    log = get_audit_log(household_id, limit=200)

    # Rule 1: Budget exhaustion
    count = get_daily_count(household_id)
    if count >= state.daily_budget:
        flags.append(AnomalyFlag(
            rule="budget_exhaustion",
            severity="warning",
            description=f"Daily autonomy budget exhausted: {count}/{state.daily_budget}",
            household_id=household_id,
        ))

    # Rule 2: Kill-switch churn (>2 toggles in 24h)
    now = datetime.now(UTC)
    ks_events = [e for e in log if "kill_switch" in e.action_type and now - e.timestamp < timedelta(hours=24)]
    if len(ks_events) > 2:
        flags.append(AnomalyFlag(
            rule="kill_switch_churn",
            severity="warning",
            description=f"Kill switch toggled {len(ks_events)} times in 24h",
            household_id=household_id,
        ))

    # Rule 3: Blocked action spike (>10 blocked in recent log)
    blocked = [e for e in log if e.decision == "blocked" and now - e.timestamp < timedelta(hours=24)]
    if len(blocked) > 10:
        flags.append(AnomalyFlag(
            rule="blocked_spike",
            severity="warning",
            description=f"{len(blocked)} actions blocked by governance in 24h",
            household_id=household_id,
        ))

    # Rule 4: Escalation concentration (>3 escalations in 24h)
    escalations = [e for e in log if e.action_type == "auto_escalate" and e.decision == "allowed" and now - e.timestamp < timedelta(hours=24)]
    if len(escalations) > 3:
        flags.append(AnomalyFlag(
            rule="escalation_concentration",
            severity="critical",
            description=f"{len(escalations)} autonomous escalations in 24h",
            household_id=household_id,
        ))

    return flags
