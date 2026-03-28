---
name: Reminder Orchestration Initiative
description: Household reminder platform (ADR-006) — Phase 0 complete, Phase 1 Slack + read-only intelligence in progress
type: project
---

Household reminder orchestration is the major platform initiative (ADR-006, 2026-03-28).

**Phase 0** (complete): Domain entities, lifecycle state machine, ports, use cases, PostgreSQL schema, 117 tests.

**Phase 1** (in progress): Read-only context and reminder intelligence.
- Slack `@woodcreek` operator surface via `/internal/slack/events` FastAPI endpoint
- Reminder queries: pending approvals, active alerts, explain, draft preview, schedule summary
- Provider-neutral calendar: MockCalendarAdapter (ready for WorkMail/EWS swap)
- InMemoryReminderStore for dev (will be replaced by PostgreSQL-backed store)
- 161 total tests

**Channel model:** Slack = control surface, Telnyx SMS = urgent/fallback, Email = context/archive, iPhone/APNs = future primary.

**Calendar strategy:** Provider-neutral from day one. WorkMail/EWS first adapter (not yet built). Gmail OAuth deferred.

**Key files:**
- Domain: `dacribagents-api/src/dacribagents/domain/reminders/`
- Queries: `application/use_cases/reminder_queries.py`
- Slack: `infrastructure/slack/` (handler, formatters, client, _store)
- Calendar: `infrastructure/calendar/mock_adapter.py`
- Endpoint: `infrastructure/http/slack_ingest.py`
- Store: `infrastructure/reminders/in_memory_store.py`

**How to apply:** Route Slack operator queries through SlackCommandHandler. All notification delivery goes through reminder domain. CalendarIdentity separate from email identity. Autonomy gated behind Tier 0-3 progression.
