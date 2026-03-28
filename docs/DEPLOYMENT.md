# Reminder Platform — Deployment Runbook

## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL 16+ (provided by Docker Compose)
- Environment variables configured (see `.env.example`)
- Optional: Kafka cluster (Confluent Cloud), Slack app, Telnyx account, WorkMail

## 1. Configuration

Copy the environment template and fill in values:

```bash
cp dacribagents-api/.env.example .env
# Edit .env with your credentials
```

Required for production:
- `POSTGRES_PASSWORD`
- `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`
- `TELNYX_API_KEY`, `TELNYX_FROM_NUMBER`
- `SMTP_USER`, `SMTP_PASSWORD`

Optional (enables additional features):
- `KAFKA_BROKERS`, `KAFKA_SASL_USERNAME`, `KAFKA_SASL_PASSWORD` — Kafka event publishing
- `WORKMAIL_EWS_EMAIL`, `WORKMAIL_EWS_PASSWORD` — Calendar integration
- `SLACK_DELIVERY_CHANNEL` — Slack reminder delivery

## 2. Database Setup

### Option A: Idempotent DDL (existing deployments)

The reminder schema is applied automatically on startup via `setup_reminder_schema()`.
All DDL statements use `CREATE TABLE IF NOT EXISTS` — safe to run repeatedly.

### Option B: Alembic Migrations (versioned evolution)

```bash
cd dacribagents-api

# Set the database URL
export DATABASE_URL="postgresql://woodcreek:${POSTGRES_PASSWORD}@localhost:5433/woodcreek_agents"

# For NEW deployments — apply all migrations:
uv run alembic upgrade head

# For EXISTING deployments — stamp the current schema version without running DDL:
uv run alembic stamp 001

# Future schema changes:
uv run alembic revision -m "description of change"
uv run alembic upgrade head
```

## 3. Service Startup

```bash
# Start all services
docker compose up -d

# Verify
docker compose ps
docker compose logs agents-api --tail=20
```

The `agents-api` container:
- Applies reminder schema on startup
- Initializes `PostgresReminderStore` (production mode)
- Starts scheduler background task (60s interval)
- Configures Kafka publisher if `KAFKA_BROKERS` is set

## 4. Health Checks

```bash
# Subsystem readiness
curl http://localhost:8080/health/subsystems

# Expected response:
# {"status":"healthy","subsystems":{"postgresql":{"status":"healthy"},...}}

# Scheduler health
curl http://localhost:8080/internal/scheduler/health

# Slack health
curl http://localhost:8080/internal/slack/health
```

## 5. Verify Slack Integration

1. Install the Slack app in your workspace
2. Set `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET`
3. Configure Events API URL: `https://your-domain/internal/slack/events`
4. Configure Interactivity URL: `https://your-domain/internal/slack/interactions`
5. Test: `@woodcreek governance status`

## 6. Verify Delivery Channels

### SMS (Telnyx)
- Ensure `TELNYX_API_KEY` and `TELNYX_FROM_NUMBER` are set
- Webhook URL for inbound: `https://your-domain/internal/sms/events`

### Email (SMTP)
- Ensure `SMTP_USER` and `SMTP_PASSWORD` are set
- Uses WorkMail SMTP by default (port 465, SSL)

## 7. Scheduler Troubleshooting

If the scheduler is not advancing reminders:

1. Check logs: `docker compose logs agents-api | grep "Scheduler"`
2. Verify `ENVIRONMENT=production` (scheduler skips in development mode)
3. Manual tick: `curl -X POST http://localhost:8080/internal/scheduler/tick`
4. Check governance tier: `@woodcreek governance status` (Tier 0 blocks autonomous dispatch)

## 8. Kill Switch

If autonomous behavior is unsafe:

```
@woodcreek kill switch on
```

This immediately blocks all Tier 2+ autonomous actions (auto-send, auto-escalate).
Manual/operator-approved actions continue to work.

To restore:
```
@woodcreek kill switch off
```

## 9. Rollback

1. Activate kill switch: `@woodcreek kill switch on`
2. Stop the service: `docker compose stop agents-api`
3. To rollback schema: `uv run alembic downgrade -1`
4. Deploy previous version and restart

## 10. Integration Testing

```bash
cd dacribagents-api

# Unit tests (no external deps)
uv run pytest tests/ -m "not integration" --no-cov

# Integration tests (requires Postgres at localhost:5433)
docker compose up -d postgres
TEST_POSTGRES_DSN="postgresql://woodcreek:${POSTGRES_PASSWORD}@localhost:5433/woodcreek_agents" \
    uv run pytest tests/integration/ -m integration --no-cov -v
```
