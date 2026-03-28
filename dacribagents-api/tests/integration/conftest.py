"""Shared fixtures for integration tests.

Integration tests use a real PostgreSQL instance.
Set TEST_POSTGRES_DSN to point to a test database.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from dacribagents.application.services.governance import GovernanceState

_TEST_DSN = os.getenv("TEST_POSTGRES_DSN", "")


def _skip_if_no_postgres():
    if not _TEST_DSN:
        pytest.skip("TEST_POSTGRES_DSN not set — skipping integration tests")


@pytest.fixture()
def pg_conn():
    """Provide a real psycopg connection with clean reminder tables."""
    _skip_if_no_postgres()
    import psycopg  # noqa: PLC0415

    conn = psycopg.connect(_TEST_DSN)
    # Apply schema
    from dacribagents.infrastructure.reminders.schema import setup_reminder_schema  # noqa: PLC0415

    setup_reminder_schema(conn)
    # Clean tables for test isolation
    with conn.cursor() as cur:
        for table in [
            "reminder_acknowledgements", "reminder_deliveries", "reminder_executions",
            "reminder_approval_records", "reminder_targets", "reminder_schedules",
            "reminder_audit_log", "unified_timeline", "governance_audit_log",
            "governance_budget_counters", "governance_muted_members", "governance_state",
            "reminder_context_sources", "preference_rules", "reminders",
            "household_members", "households",
        ]:
            cur.execute(f"DELETE FROM {table}")  # noqa: S608
        conn.commit()
    yield conn
    conn.close()


@pytest.fixture()
def pg_store(pg_conn):
    """Provide a PostgresReminderStore backed by real Postgres."""
    from dacribagents.infrastructure.reminders.postgres_store import PostgresReminderStore  # noqa: PLC0415

    return PostgresReminderStore(pg_conn)


@pytest.fixture()
def seeded_store(pg_store, pg_conn):
    """Store with a household and member pre-seeded."""
    household_id = uuid4()
    member_id = uuid4()
    with pg_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO households (id, name) VALUES (%s, %s)",
            (household_id, "Woodcreek Test"),
        )
        cur.execute(
            "INSERT INTO household_members (id, household_id, name, role, timezone, email, phone) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (member_id, household_id, "Mike", "admin", "America/Chicago", "mike@test.com", "+15125551234"),
        )
        pg_conn.commit()
    return pg_store, household_id, member_id


@pytest.fixture(autouse=True)
def _reset_governance():
    import dacribagents.application.services.governance as mod  # noqa: PLC0415

    mod._state = GovernanceState()
    yield
    mod._state = None
