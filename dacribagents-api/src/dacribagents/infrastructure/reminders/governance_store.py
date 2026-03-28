"""PostgreSQL-backed governance state persistence.

Reads/writes governance tier, kill-switch, budget, and mute state
from the ``governance_*`` tables defined in ``schema.py``.

This replaces the in-memory ``GovernanceState`` singleton for
production deployments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import psycopg

from dacribagents.application.services.governance import AutonomyTier, GovernanceState


class PostgresGovernanceStore:
    """Reads/writes governance state from PostgreSQL."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def load_state(self, state: GovernanceState, household_id: UUID) -> None:
        """Load governance state for a household from the database."""
        with self._conn.cursor() as cur:
            # Tier and kill switch
            cur.execute(
                "SELECT tier, kill_switch, kill_switch_by, kill_switch_at, daily_budget "
                "FROM governance_state WHERE household_id = %s",
                (household_id,),
            )
            row = cur.fetchone()
            if row:
                state.household_tiers[household_id] = AutonomyTier(row[0])
                state.kill_switch_active = row[1]
                state.kill_switch_activated_by = row[2]
                state.kill_switch_activated_at = row[3]
                state.daily_budget = row[4]

            # Muted members
            cur.execute(
                "SELECT member_id, reason FROM governance_muted_members WHERE household_id = %s",
                (household_id,),
            )
            for r in cur.fetchall():
                state.muted_members[r[0]] = r[1]

            # Budget counter for today
            cur.execute(
                "SELECT count FROM governance_budget_counters WHERE household_id = %s AND date = CURRENT_DATE",
                (household_id,),
            )
            row = cur.fetchone()
            if row:
                key = f"{household_id}:{datetime.now(UTC).strftime('%Y-%m-%d')}"
                state.action_counts[key] = row[0]

    def save_tier(self, household_id: UUID, tier: AutonomyTier) -> None:
        """Persist the autonomy tier for a household."""
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO governance_state (household_id, tier, updated_at) VALUES (%s, %s, %s) "
                "ON CONFLICT (household_id) DO UPDATE SET tier = EXCLUDED.tier, updated_at = EXCLUDED.updated_at",
                (household_id, tier.value, datetime.now(UTC)),
            )
            self._conn.commit()

    def save_kill_switch(self, active: bool, activated_by: str | None) -> None:
        """Persist kill switch state (global — updates all households or uses a sentinel)."""
        now = datetime.now(UTC)
        sentinel = UUID("00000000-0000-0000-0000-000000000001")
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO governance_state (household_id, kill_switch, kill_switch_by, kill_switch_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (household_id) DO UPDATE SET kill_switch=EXCLUDED.kill_switch, "
                "kill_switch_by=EXCLUDED.kill_switch_by, kill_switch_at=EXCLUDED.kill_switch_at, updated_at=EXCLUDED.updated_at",
                (sentinel, active, activated_by, now if active else None, now),
            )
            self._conn.commit()

    def save_mute(self, member_id: UUID, household_id: UUID, reason: str) -> None:
        """Persist member mute state."""
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO governance_muted_members (member_id, household_id, reason, muted_at) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (member_id) DO UPDATE SET reason=EXCLUDED.reason",
                (member_id, household_id, reason, datetime.now(UTC)),
            )
            self._conn.commit()

    def remove_mute(self, member_id: UUID) -> None:
        """Remove member mute state."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM governance_muted_members WHERE member_id = %s", (member_id,))
            self._conn.commit()

    def increment_budget(self, household_id: UUID) -> int:
        """Increment and return the daily budget counter."""
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO governance_budget_counters (household_id, date, count) VALUES (%s, CURRENT_DATE, 1) "
                "ON CONFLICT (household_id, date) DO UPDATE SET count = governance_budget_counters.count + 1 "
                "RETURNING count",
                (household_id,),
            )
            count = cur.fetchone()[0]
            self._conn.commit()
            return count
