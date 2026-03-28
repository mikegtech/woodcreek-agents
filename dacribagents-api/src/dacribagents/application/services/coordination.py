"""Cross-agent reminder coordination — enrichment, merge, suppression.

Handles multiple agent-originated reminder candidates for the same
household/time window.  Deterministic rules resolve overlaps:

1. **Enrich**: One agent's candidate adds context to another's.
2. **Merge**: Overlapping candidates collapse into one better draft.
3. **Suppress**: A higher-priority candidate suppresses a lower-priority duplicate.
4. **Keep separate**: Candidates are independent — no merge needed.

All outputs enter the existing draft/approval/governance lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from dacribagents.domain.reminders.enums import (
    NotificationIntent,
    ReminderSource,
    UrgencyLevel,
)


@dataclass
class ReminderCandidate:
    """A proposed reminder from an agent before coordination."""

    id: UUID = field(default_factory=uuid4)
    household_id: UUID = field(default_factory=lambda: UUID(int=0))
    source: ReminderSource = ReminderSource.AGENT
    source_agent: str = ""
    subject: str = ""
    body: str = ""
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    intent: NotificationIntent = NotificationIntent.REMINDER
    event_id: str | None = None
    dedupe_key: str | None = None
    enrichments: list[Enrichment] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Enrichment:
    """Additional context added by a coordinating agent."""

    agent: str
    context: str
    urgency_hint: UrgencyLevel | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class CoordinationResult:
    """Outcome of coordinating multiple candidates."""

    action: str  # "emit", "merge", "suppress", "enrich"
    candidates: list[ReminderCandidate]
    merged: ReminderCandidate | None = None
    reason: str = ""


# ── Enrichment ──────────────────────────────────────────────────────────────


def enrich_candidate(
    candidate: ReminderCandidate,
    enrichment: Enrichment,
) -> ReminderCandidate:
    """Add enrichment context to a candidate without creating a new one."""
    candidate.enrichments.append(enrichment)
    if enrichment.context:
        candidate.body = f"{candidate.body}\n\n[{enrichment.agent}] {enrichment.context}".strip()
    if enrichment.urgency_hint and enrichment.urgency_hint.value < candidate.urgency.value:
        # Promote urgency if the enriching agent suggests higher
        candidate.urgency = enrichment.urgency_hint
    return candidate


# ── Coordination ────────────────────────────────────────────────────────────

_URGENCY_RANK = {UrgencyLevel.CRITICAL: 0, UrgencyLevel.URGENT: 1, UrgencyLevel.NORMAL: 2, UrgencyLevel.LOW: 3}


def coordinate(candidates: list[ReminderCandidate]) -> list[CoordinationResult]:
    """Coordinate a batch of candidates from different agents.

    Rules (deterministic):
    1. Group by dedupe_key — candidates with same key merge.
    2. Within a merge group, keep the highest-urgency candidate as primary,
       append other subjects/body as enrichments.
    3. Candidates without dedupe_key are independent — emitted as-is.
    """
    if not candidates:
        return []

    # Group by dedupe_key
    groups: dict[str | None, list[ReminderCandidate]] = {}
    for c in candidates:
        groups.setdefault(c.dedupe_key, []).append(c)

    results: list[CoordinationResult] = []

    for key, group in groups.items():
        if key is None or len(group) == 1:
            # Independent candidates — emit each
            for c in group:
                results.append(CoordinationResult(action="emit", candidates=[c], reason="Independent candidate"))
        elif len(group) > 1:
            # Merge: keep highest urgency, append others as enrichments
            group.sort(key=lambda c: _URGENCY_RANK.get(c.urgency, 99))
            primary = group[0]
            for secondary in group[1:]:
                enrich_candidate(primary, Enrichment(
                    agent=secondary.source_agent,
                    context=f"Also reported by {secondary.source_agent}: {secondary.subject}",
                    urgency_hint=secondary.urgency,
                    source_ref=secondary.event_id,
                ))
            results.append(CoordinationResult(
                action="merge",
                candidates=group,
                merged=primary,
                reason=f"Merged {len(group)} candidates with dedupe_key={key!r}",
            ))

    return results


def suppress_if_exists(
    candidate: ReminderCandidate,
    existing_subjects: list[str],
) -> CoordinationResult:
    """Suppress a candidate if an identical subject already exists."""
    if candidate.subject in existing_subjects:
        return CoordinationResult(
            action="suppress", candidates=[candidate],
            reason=f"Suppressed: duplicate subject '{candidate.subject}'",
        )
    return CoordinationResult(action="emit", candidates=[candidate], reason="No conflict")
