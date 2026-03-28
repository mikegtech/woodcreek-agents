"""Baseline reminder/governance schema.

Revision ID: 001
Revises: None
Create Date: 2026-03-28

This baseline migration matches the idempotent DDL in
``infrastructure/reminders/schema.py``.  For existing deployments
where the schema was applied via ``setup_reminder_schema()``,
stamp this revision without running: ``alembic stamp 001``
"""

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The full schema DDL — matches schema.py exactly
    op.execute("""
    CREATE TABLE IF NOT EXISTS households (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name VARCHAR(255) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS household_members (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        household_id UUID NOT NULL REFERENCES households(id),
        name VARCHAR(255) NOT NULL,
        role VARCHAR(50) NOT NULL,
        timezone VARCHAR(100) NOT NULL DEFAULT 'America/Chicago',
        email VARCHAR(255),
        phone VARCHAR(50),
        slack_id VARCHAR(100),
        quiet_hours_start TIME,
        quiet_hours_end TIME,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    CREATE INDEX IF NOT EXISTS idx_hm_household ON household_members(household_id);

    CREATE TABLE IF NOT EXISTS reminders (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        household_id UUID NOT NULL REFERENCES households(id),
        subject VARCHAR(500) NOT NULL,
        body TEXT NOT NULL DEFAULT '',
        urgency VARCHAR(50) NOT NULL DEFAULT 'normal',
        intent VARCHAR(50) NOT NULL DEFAULT 'reminder',
        source VARCHAR(50) NOT NULL,
        source_agent VARCHAR(255),
        source_event_id VARCHAR(500),
        dedupe_key VARCHAR(500),
        state VARCHAR(50) NOT NULL DEFAULT 'draft',
        requires_approval BOOLEAN NOT NULL DEFAULT false,
        created_by UUID NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    CREATE INDEX IF NOT EXISTS idx_rem_household ON reminders(household_id);
    CREATE INDEX IF NOT EXISTS idx_rem_state ON reminders(state);
    CREATE INDEX IF NOT EXISTS idx_rem_household_state ON reminders(household_id, state);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_rem_dedupe ON reminders(household_id, dedupe_key)
        WHERE dedupe_key IS NOT NULL AND state NOT IN ('cancelled', 'failed', 'acknowledged');

    CREATE TABLE IF NOT EXISTS reminder_targets (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        reminder_id UUID NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
        target_type VARCHAR(50) NOT NULL,
        member_id UUID REFERENCES household_members(id),
        role VARCHAR(50)
    );
    CREATE INDEX IF NOT EXISTS idx_rt_reminder ON reminder_targets(reminder_id);

    CREATE TABLE IF NOT EXISTS reminder_schedules (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        reminder_id UUID NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
        schedule_type VARCHAR(50) NOT NULL,
        timezone VARCHAR(100) NOT NULL DEFAULT 'America/Chicago',
        fire_at TIMESTAMPTZ,
        relative_to VARCHAR(500),
        relative_offset_minutes INTEGER,
        cron_expression VARCHAR(255),
        next_fire_at TIMESTAMPTZ,
        UNIQUE (reminder_id)
    );
    CREATE INDEX IF NOT EXISTS idx_rs_next_fire ON reminder_schedules(next_fire_at)
        WHERE next_fire_at IS NOT NULL;

    CREATE TABLE IF NOT EXISTS reminder_executions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        reminder_id UUID NOT NULL REFERENCES reminders(id),
        schedule_id UUID NOT NULL,
        fired_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS reminder_deliveries (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        execution_id UUID NOT NULL REFERENCES reminder_executions(id),
        member_id UUID NOT NULL,
        channel VARCHAR(50) NOT NULL,
        status VARCHAR(50) NOT NULL DEFAULT 'queued',
        escalation_step INTEGER NOT NULL DEFAULT 0,
        provider_message_id VARCHAR(500),
        sent_at TIMESTAMPTZ,
        delivered_at TIMESTAMPTZ,
        failed_at TIMESTAMPTZ,
        failure_reason TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS reminder_acknowledgements (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        delivery_id UUID NOT NULL,
        member_id UUID NOT NULL,
        method VARCHAR(50) NOT NULL,
        acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        note TEXT,
        snoozed_until TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS reminder_approval_records (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        reminder_id UUID NOT NULL REFERENCES reminders(id),
        action VARCHAR(50) NOT NULL,
        actor_id UUID NOT NULL,
        reason TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS preference_rules (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        household_id UUID NOT NULL REFERENCES households(id),
        member_id UUID,
        urgency VARCHAR(50),
        preferred_channel VARCHAR(50) NOT NULL,
        fallback_channel VARCHAR(50),
        quiet_hours_override BOOLEAN NOT NULL DEFAULT false,
        active BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS governance_state (
        household_id UUID PRIMARY KEY,
        tier INTEGER NOT NULL DEFAULT 0,
        kill_switch BOOLEAN NOT NULL DEFAULT false,
        kill_switch_by VARCHAR(255),
        kill_switch_at TIMESTAMPTZ,
        daily_budget INTEGER NOT NULL DEFAULT 50,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS governance_muted_members (
        member_id UUID PRIMARY KEY,
        household_id UUID NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        muted_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS governance_budget_counters (
        household_id UUID NOT NULL,
        date DATE NOT NULL DEFAULT CURRENT_DATE,
        count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (household_id, date)
    );

    CREATE TABLE IF NOT EXISTS governance_audit_log (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        household_id UUID NOT NULL,
        action_type VARCHAR(100) NOT NULL,
        actor_type VARCHAR(50) NOT NULL,
        tier INTEGER NOT NULL,
        decision VARCHAR(50) NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        reminder_id UUID,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS unified_timeline (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        household_id UUID NOT NULL,
        reminder_id UUID,
        event_type VARCHAR(100) NOT NULL,
        actor VARCHAR(255) NOT NULL DEFAULT 'system',
        summary TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );

    CREATE TABLE IF NOT EXISTS reminder_audit_log (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        reminder_id UUID NOT NULL,
        actor VARCHAR(255) NOT NULL,
        action VARCHAR(100) NOT NULL,
        target_member_id UUID,
        channel VARCHAR(50),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    """)


def downgrade() -> None:
    op.execute("""
    DROP TABLE IF EXISTS unified_timeline CASCADE;
    DROP TABLE IF EXISTS governance_audit_log CASCADE;
    DROP TABLE IF EXISTS governance_budget_counters CASCADE;
    DROP TABLE IF EXISTS governance_muted_members CASCADE;
    DROP TABLE IF EXISTS governance_state CASCADE;
    DROP TABLE IF EXISTS reminder_audit_log CASCADE;
    DROP TABLE IF EXISTS reminder_approval_records CASCADE;
    DROP TABLE IF EXISTS reminder_acknowledgements CASCADE;
    DROP TABLE IF EXISTS reminder_deliveries CASCADE;
    DROP TABLE IF EXISTS reminder_executions CASCADE;
    DROP TABLE IF EXISTS reminder_schedules CASCADE;
    DROP TABLE IF EXISTS reminder_targets CASCADE;
    DROP TABLE IF EXISTS preference_rules CASCADE;
    DROP TABLE IF EXISTS reminders CASCADE;
    DROP TABLE IF EXISTS household_members CASCADE;
    DROP TABLE IF EXISTS households CASCADE;
    """)
