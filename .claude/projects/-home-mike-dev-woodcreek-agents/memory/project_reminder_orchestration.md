---
name: Reminder Orchestration Initiative
description: Household reminder orchestration platform (ADR-006) with Slack control surface, WorkMail calendar, iPhone/APNs end-state, Telnyx SMS fallback
type: project
---

Household reminder orchestration was introduced as a first-class platform capability in ADR-006 (2026-03-28, revised same day). Reminders are a shared domain — not per-agent notification logic.

**Channel model:**
- Slack = conversational control surface, approval UX, `@woodcreek` interactions
- Telnyx SMS = urgent delivery, fallback, escalation
- Email (woodcreek.me) = rich context, digests, archival
- iPhone app / APNs push = long-term primary household notification (future)

**Calendar strategy:** Provider-neutral interface from day one. AWS WorkMail / EWS is the first adapter (service-owned credentials, no token churn). Gmail OAuth explicitly deferred.

**Why:** Agents (HOA, warranties, maintenance, security) all generate time-sensitive actions. Without a unified layer, delivery is fragmented with no ack/escalation/audit.

**How to apply:** Route all agent notifications through the reminder domain model. CalendarIdentity is separate from email identity. MemberDevice is anticipated from Phase 0 for future push. Autonomy is gated behind Tier 0-3 progression. Home Warranty ADR bumped to ADR-007.
