"""PostgreSQL DDL for the household reminder domain.

Run ``setup_reminder_schema(conn)`` during application startup to create or
migrate the reminder tables.  All statements are idempotent (IF NOT EXISTS).
"""

from __future__ import annotations

import psycopg
from loguru import logger

# ── DDL ─────────────────────────────────────────────────────────────────────

REMINDER_SCHEMA_SQL = """
-- ==========================================================================
-- Household
-- ==========================================================================

CREATE TABLE IF NOT EXISTS households (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb
);

-- ==========================================================================
-- Household Members
-- ==========================================================================

CREATE TABLE IF NOT EXISTS household_members (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    household_id        UUID         NOT NULL REFERENCES households(id),
    name                VARCHAR(255) NOT NULL,
    role                VARCHAR(50)  NOT NULL,  -- admin | member | child
    timezone            VARCHAR(100) NOT NULL DEFAULT 'America/Chicago',
    email               VARCHAR(255),
    phone               VARCHAR(50),
    slack_id            VARCHAR(100),
    quiet_hours_start   TIME,
    quiet_hours_end     TIME,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata            JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_hm_household ON household_members(household_id);

-- ==========================================================================
-- Reminders
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminders (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    household_id        UUID         NOT NULL REFERENCES households(id),
    subject             VARCHAR(500) NOT NULL,
    body                TEXT         NOT NULL DEFAULT '',
    urgency             VARCHAR(50)  NOT NULL DEFAULT 'normal',
    intent              VARCHAR(50)  NOT NULL DEFAULT 'reminder',  -- reminder | alert | digest
    source              VARCHAR(50)  NOT NULL,  -- user | agent | calendar | email | maintenance | hoa | warranty | telemetry | household_routine | event_bus
    source_agent        VARCHAR(255),
    source_event_id     VARCHAR(500),            -- correlation ID from the upstream event source
    dedupe_key          VARCHAR(500),            -- prevents duplicate reminders from the same event
    state               VARCHAR(50)  NOT NULL DEFAULT 'draft',  -- draft | pending_approval | approved | rejected | scheduled | pending_delivery | delivered | acknowledged | snoozed | cancelled | failed
    requires_approval   BOOLEAN      NOT NULL DEFAULT false,
    created_by          UUID         NOT NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata            JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_rem_household      ON reminders(household_id);
CREATE INDEX IF NOT EXISTS idx_rem_state          ON reminders(state);
CREATE INDEX IF NOT EXISTS idx_rem_household_state ON reminders(household_id, state);
CREATE INDEX IF NOT EXISTS idx_rem_created_by     ON reminders(created_by);
CREATE INDEX IF NOT EXISTS idx_rem_source_event   ON reminders(source_event_id) WHERE source_event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_rem_dedupe  ON reminders(household_id, dedupe_key)
    WHERE dedupe_key IS NOT NULL AND state NOT IN ('cancelled', 'failed', 'acknowledged');

-- ==========================================================================
-- Reminder Targets
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_targets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id     UUID        NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
    target_type     VARCHAR(50) NOT NULL,  -- individual | household | role
    member_id       UUID        REFERENCES household_members(id),
    role            VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_rt_reminder ON reminder_targets(reminder_id);
CREATE INDEX IF NOT EXISTS idx_rt_member   ON reminder_targets(member_id);

-- ==========================================================================
-- Reminder Schedules
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_schedules (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id             UUID         NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
    schedule_type           VARCHAR(50)  NOT NULL,  -- one_shot | relative | recurring
    timezone                VARCHAR(100) NOT NULL DEFAULT 'America/Chicago',
    fire_at                 TIMESTAMPTZ,
    relative_to             VARCHAR(500),
    relative_offset_minutes INTEGER,
    cron_expression         VARCHAR(255),
    next_fire_at            TIMESTAMPTZ,
    UNIQUE (reminder_id)
);

CREATE INDEX IF NOT EXISTS idx_rs_next_fire ON reminder_schedules(next_fire_at)
    WHERE next_fire_at IS NOT NULL;

-- ==========================================================================
-- Reminder Executions
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_executions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id     UUID        NOT NULL REFERENCES reminders(id),
    schedule_id     UUID        NOT NULL REFERENCES reminder_schedules(id),
    fired_at        TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_re_reminder ON reminder_executions(reminder_id);

-- ==========================================================================
-- Reminder Deliveries
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_deliveries (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id        UUID        NOT NULL REFERENCES reminder_executions(id),
    member_id           UUID        NOT NULL REFERENCES household_members(id),
    channel             VARCHAR(50) NOT NULL,  -- push | sms | email | slack | in_app
    status              VARCHAR(50) NOT NULL DEFAULT 'queued',
    escalation_step     INTEGER     NOT NULL DEFAULT 0,
    provider_message_id VARCHAR(500),
    sent_at             TIMESTAMPTZ,
    delivered_at        TIMESTAMPTZ,
    failed_at           TIMESTAMPTZ,
    failure_reason      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rd_execution ON reminder_deliveries(execution_id);
CREATE INDEX IF NOT EXISTS idx_rd_member    ON reminder_deliveries(member_id);
CREATE INDEX IF NOT EXISTS idx_rd_status    ON reminder_deliveries(status);

-- ==========================================================================
-- Reminder Acknowledgements
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_acknowledgements (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    delivery_id     UUID        NOT NULL REFERENCES reminder_deliveries(id),
    member_id       UUID        NOT NULL REFERENCES household_members(id),
    method          VARCHAR(50) NOT NULL,  -- slack_reaction | sms_reply | push_action | email_link | dashboard
    acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    note            TEXT,
    snoozed_until   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ra_delivery ON reminder_acknowledgements(delivery_id);

-- ==========================================================================
-- Approval Records
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_approval_records (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id     UUID         NOT NULL REFERENCES reminders(id),
    action          VARCHAR(50)  NOT NULL,  -- submitted | approved | rejected
    actor_id        UUID         NOT NULL,
    reason          TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_rar_reminder ON reminder_approval_records(reminder_id);
CREATE INDEX IF NOT EXISTS idx_rar_actor    ON reminder_approval_records(actor_id);

-- ==========================================================================
-- Preference Rules
-- ==========================================================================

CREATE TABLE IF NOT EXISTS preference_rules (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    household_id        UUID        NOT NULL REFERENCES households(id),
    member_id           UUID        REFERENCES household_members(id),
    urgency             VARCHAR(50),
    preferred_channel   VARCHAR(50) NOT NULL,
    fallback_channel    VARCHAR(50),
    quiet_hours_override BOOLEAN    NOT NULL DEFAULT false,
    active              BOOLEAN     NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pr_household ON preference_rules(household_id);
CREATE INDEX IF NOT EXISTS idx_pr_member    ON preference_rules(member_id);

-- ==========================================================================
-- Reminder Context Sources
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_context_sources (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    household_id    UUID         NOT NULL REFERENCES households(id),
    source_type     VARCHAR(100) NOT NULL,
    source_ref      VARCHAR(500) NOT NULL,
    title           VARCHAR(500) NOT NULL,
    event_date      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_rcs_household ON reminder_context_sources(household_id);

-- ==========================================================================
-- Calendar Providers
-- ==========================================================================

CREATE TABLE IF NOT EXISTS calendar_providers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_type   VARCHAR(50)  NOT NULL UNIQUE,  -- workmail_ews | ical_feed | google | caldav | manual
    display_name    VARCHAR(255) NOT NULL,
    active          BOOLEAN      NOT NULL DEFAULT true,
    config          JSONB        NOT NULL DEFAULT '{}'::jsonb
);

-- ==========================================================================
-- Calendar Identities
-- ==========================================================================

CREATE TABLE IF NOT EXISTS calendar_identities (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    household_id        UUID         NOT NULL REFERENCES households(id),
    member_id           UUID         REFERENCES household_members(id),
    provider            VARCHAR(50)  NOT NULL,
    provider_account_id VARCHAR(500) NOT NULL,
    display_name        VARCHAR(255) NOT NULL,
    active              BOOLEAN      NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata            JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ci_household ON calendar_identities(household_id);
CREATE INDEX IF NOT EXISTS idx_ci_member    ON calendar_identities(member_id);

-- ==========================================================================
-- Calendar Access Policies
-- ==========================================================================

CREATE TABLE IF NOT EXISTS calendar_access_policies (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calendar_identity_id    UUID        NOT NULL REFERENCES calendar_identities(id) ON DELETE CASCADE,
    access_level            VARCHAR(50) NOT NULL DEFAULT 'read_only',
    visible_calendars       TEXT[]      NOT NULL DEFAULT '{}',
    sync_frequency_minutes  INTEGER     NOT NULL DEFAULT 15,
    write_back_enabled      BOOLEAN     NOT NULL DEFAULT false,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (calendar_identity_id)
);

-- ==========================================================================
-- Member Devices (future push / APNs)
-- ==========================================================================

CREATE TABLE IF NOT EXISTS member_devices (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    member_id       UUID         NOT NULL REFERENCES household_members(id),
    platform        VARCHAR(50)  NOT NULL,  -- ios
    device_token    VARCHAR(500) NOT NULL,
    display_name    VARCHAR(255),
    active          BOOLEAN      NOT NULL DEFAULT true,
    registered_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (member_id, device_token)
);

CREATE INDEX IF NOT EXISTS idx_md_member ON member_devices(member_id);

-- ==========================================================================
-- Audit Trail
-- ==========================================================================

CREATE TABLE IF NOT EXISTS reminder_audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reminder_id     UUID         NOT NULL,
    actor           VARCHAR(255) NOT NULL,
    action          VARCHAR(100) NOT NULL,
    target_member_id UUID,
    channel         VARCHAR(50),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ral_reminder  ON reminder_audit_log(reminder_id);
CREATE INDEX IF NOT EXISTS idx_ral_actor     ON reminder_audit_log(actor);
CREATE INDEX IF NOT EXISTS idx_ral_created   ON reminder_audit_log(created_at);

-- ==========================================================================
-- Governance State
-- ==========================================================================

CREATE TABLE IF NOT EXISTS governance_state (
    household_id    UUID PRIMARY KEY,
    tier            INTEGER      NOT NULL DEFAULT 0,
    kill_switch     BOOLEAN      NOT NULL DEFAULT false,
    kill_switch_by  VARCHAR(255),
    kill_switch_at  TIMESTAMPTZ,
    daily_budget    INTEGER      NOT NULL DEFAULT 50,
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS governance_muted_members (
    member_id       UUID PRIMARY KEY,
    household_id    UUID         NOT NULL,
    reason          TEXT         NOT NULL DEFAULT '',
    muted_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gmm_household ON governance_muted_members(household_id);

CREATE TABLE IF NOT EXISTS governance_budget_counters (
    household_id    UUID         NOT NULL,
    date            DATE         NOT NULL DEFAULT CURRENT_DATE,
    count           INTEGER      NOT NULL DEFAULT 0,
    PRIMARY KEY (household_id, date)
);

CREATE TABLE IF NOT EXISTS governance_audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    household_id    UUID         NOT NULL,
    action_type     VARCHAR(100) NOT NULL,
    actor_type      VARCHAR(50)  NOT NULL,
    tier            INTEGER      NOT NULL,
    decision        VARCHAR(50)  NOT NULL,
    reason          TEXT         NOT NULL DEFAULT '',
    reminder_id     UUID,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_gal_household ON governance_audit_log(household_id);
CREATE INDEX IF NOT EXISTS idx_gal_created   ON governance_audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_gal_decision  ON governance_audit_log(decision);

-- ==========================================================================
-- Unified Audit Timeline
-- ==========================================================================

CREATE TABLE IF NOT EXISTS unified_timeline (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    household_id    UUID         NOT NULL,
    reminder_id     UUID,
    event_type      VARCHAR(100) NOT NULL,
    actor           VARCHAR(255) NOT NULL DEFAULT 'system',
    summary         TEXT         NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ut_household  ON unified_timeline(household_id);
CREATE INDEX IF NOT EXISTS idx_ut_reminder   ON unified_timeline(reminder_id) WHERE reminder_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ut_created    ON unified_timeline(created_at);
CREATE INDEX IF NOT EXISTS idx_ut_type       ON unified_timeline(event_type);
"""


def setup_reminder_schema(conn: psycopg.Connection) -> None:
    """Create all reminder-domain tables and indexes (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(REMINDER_SCHEMA_SQL)
    conn.commit()
    logger.info("Reminder domain schema setup complete")
