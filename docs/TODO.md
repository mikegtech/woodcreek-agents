# Woodcreek Agents — Roadmap

## Domain Configuration
- **API Domain:** woodcreek.ai (agents, dashboard, traefik)
- **Email Domain:** woodcreek.me (RAG content source, context/archive delivery)
- **Calendar Identity:** AWS WorkMail (Woodcreek-managed accounts, EWS calendar adapter)
- **Control Surface:** Slack (conversational command/control, approval UX)
- **SMS:** Telnyx (urgent delivery, fallback, escalation)
- **Push (future):** iPhone app / APNs (long-term primary household notification)

---

## Foundation (Completed / In Progress)

### Agent Implementation
- [x] General Assistant Agent (Groq/local vLLM)
- [ ] Supervisor Agent (intent routing)
- [ ] HOA Compliance Agent (RAG)
- [ ] Home Maintenance Agent (tool calling)
- [ ] Security & Cameras Agent (event-driven)

### Infrastructure
- [x] Milvus (GPU-accelerated vectors)
- [x] PostgreSQL (LangGraph checkpoints)
- [x] Docker containerization
- [x] Traefik routing (woodcreek.ai)
- [x] Cloudflare tunnel

### Email Integration
- [x] IMAP/SMTP client for woodcreek.me
- [x] Email ingestion pipeline → Milvus
- [x] Email send capability (notifications, reports)
- [x] Email RAG (search HOA communications, receipts, etc.)

### SMS/Messaging (Telnyx)
- [x] Telnyx account setup
- [x] Inbound SMS webhook
- [ ] Outbound SMS (alerts, reminders)
- [ ] Two-way conversation threading
- [ ] SMS → Agent routing

### RAG Pipeline
- [ ] Document ingestion (HOA docs, CC&Rs, manuals)
- [ ] Email ingestion (woodcreek.me inbox)
- [ ] Chunking strategy
- [ ] Embedding pipeline (sentence-transformers)
- [ ] Retrieval integration with agents

---

## Household Reminder Orchestration Platform

Reminders are a first-class platform capability — not an afterthought bolted onto individual agents. The system serves both household-wide and individual use cases with a clear channel model:

- **Slack** is the conversational control surface: create reminders, approve agent drafts, query schedules, interact with `@woodcreek`.
- **Telnyx SMS** is the urgent/fallback/escalation delivery channel with external reach.
- **Email (woodcreek.me)** is the rich-context, digest, and archival channel.
- **iPhone app / APNs push** is the intended long-term primary notification and acknowledgment surface for the household.

Reminders originate from many sources: calendar events, email-extracted dates, agent observations (HOA deadlines, warranty expirations, maintenance windows), household routines, and direct user requests via Slack or SMS.

Calendar integrations use a **provider-neutral abstraction from day one**, with **AWS WorkMail / EWS as the first adapter** for Woodcreek-managed identities. Gmail OAuth is explicitly deferred — delegated-auth lifecycle friction makes it a poor MVP baseline.

See [ADR-006](ADRs/ADR-006.md) for domain model, lifecycle, delivery architecture, and governance decisions.

### Phase 0 — Reminder Domain & Architecture
Define the core domain model and persistence layer before any runtime behavior.

- [ ] Define domain entities: `Household`, `HouseholdMember`, `Reminder`, `ReminderTarget`, `ReminderSchedule`, `ReminderDelivery`, `ReminderExecution`, `ReminderAcknowledgement`, `PreferenceRule`, `ReminderContextSource`
- [ ] Define calendar identity entities: `CalendarIdentity`, `CalendarProvider`, `CalendarAccessPolicy`
- [ ] Anticipate device registration: `MemberDevice` for future APNs/push support
- [ ] Design reminder lifecycle state machine (draft → scheduled → pending_delivery → delivered → acknowledged / snoozed / escalated → cancelled / failed)
- [ ] Schema design in PostgreSQL (reminder tables, household/member tables, delivery log, audit trail)
- [ ] Define Pydantic/TypeScript models for inter-agent reminder protocol
- [ ] Establish `packages/shared/reminders/` as the shared reminder domain library
- [ ] Design provider-neutral `CalendarAdapter` interface
- [ ] Write ADR-006 decision record *(this document)*

### Phase 1 — Read-Only Context and Reminder Intelligence
Give agents visibility into existing schedules and household context without writing or sending anything.

- [ ] Implement `CalendarAdapter` interface with AWS WorkMail / EWS as first adapter
- [ ] Build `ReminderContextSource` adapters: WorkMail calendar events, email-derived dates, HOA calendar, manual entries
- [ ] Reminder suggestion engine: agents propose reminders based on context (e.g., HOA compliance deadlines, warranty expirations, maintenance windows)
- [ ] Context enrichment: attach relevant household member info, delivery preferences, and prior reminder history to suggestions
- [ ] iCal feed adapter (read-only — works with any published calendar)
- [ ] Email-derived date extraction from existing RAG pipeline

### Phase 2 — Reminder Authoring with Human Approval
Enable agents and users to create reminders, but require explicit human approval before anything is sent.

- [ ] Reminder draft/create API: agents and Slack can create reminder drafts
- [ ] Slack approval flow: agent-authored reminders surface in Slack for household member approval
- [ ] Manual reminder creation via Slack (`@woodcreek remind the family about soccer at 5`), SMS, or CLI
- [ ] Scheduling engine: cron-like recurring reminders, one-shot reminders, relative-time reminders ("3 days before X")
- [ ] Reminder editing, snooze, and cancellation with full audit trail
- [ ] Household-level and individual targeting from the start (e.g., "remind the household" vs. "remind Mike")

### Phase 3 — Multi-Channel Delivery and Family Context
Wire reminders to real outbound channels and support household/group dynamics.

- [ ] SMS delivery via Telnyx (outbound SMS — urgent, fallback, escalation)
- [ ] Email delivery via woodcreek.me SMTP (rich context, digests, archival)
- [ ] Slack notification delivery (operator/admin workflows, interactive approval buttons)
- [ ] Delivery preferences per `HouseholdMember`: preferred channel, quiet hours, urgency overrides
- [ ] `PreferenceRule` engine: "Mike prefers SMS for urgent, email for informational"; "No reminders after 10pm unless critical"
- [ ] Delivery receipts and failure handling (bounce, undeliverable, retry logic)
- [ ] Acknowledgement model: members ack via Slack reaction, SMS reply, email link, or future push action
- [ ] Snooze flow: "snooze 30 min" via Slack or SMS reply
- [ ] Escalation: if no ack within configurable window, re-deliver via alternate channel or escalate to another household member

### Phase 4 — Event-Driven Household Orchestration
Move from scheduled reminders to reactive, event-driven triggers across the agent ecosystem.

- [ ] Event bus integration: agents emit events (e.g., `warranty.expiring`, `hoa.deadline.approaching`, `camera.offline`)
- [ ] Reminder triggers: event → reminder creation rules (configurable per household)
- [ ] Cross-agent reminder coordination: maintenance agent triggers reminder, compliance agent enriches it, delivery system sends it
- [ ] Calendar write-back: approved reminders optionally sync to WorkMail calendar (via CalendarAdapter)
- [ ] Batch/digest mode: group low-priority reminders into daily or weekly household digests (delivered via email)
- [ ] LangGraph workflow integration: reminder lifecycle as a durable LangGraph graph with checkpointing
- [ ] Slack-native household schedule queries: `@woodcreek what does tomorrow look like?`, `@woodcreek who has conflicts Thursday night?`

### Phase 5 — Guardrailed Autonomy and Governance
Allow agents limited autonomous action only after audit, approval, and guardrail infrastructure is proven.

- [ ] Full audit trail review: every reminder action (create, approve, send, ack, snooze, escalate) logged with actor, timestamp, and context
- [ ] Governance dashboard: household admins review all autonomous actions, set budgets/limits, revoke agent permissions
- [ ] Limited autonomy tiers:
  - Tier 1: Agent can draft reminders (human approves via Slack)
  - Tier 2: Agent can send pre-approved reminder types without human approval (e.g., recurring maintenance reminders)
  - Tier 3: Agent can escalate and re-route based on acknowledgement status
- [ ] Rate limiting and anomaly detection: cap reminders per member per day, flag unusual patterns
- [ ] Opt-out and override: any household member can mute, snooze, or opt out of non-critical reminders
- [ ] Kill switch: household admin can disable all autonomous actions instantly via Slack or dashboard
- [ ] Periodic governance review: monthly summary of autonomous actions for household admin

---

## iPhone App & Push Notification Evolution

The long-term primary household UX is a native iPhone app with APNs push notifications. Everyone in the household is on iPhone, making native push the natural acknowledgment and notification surface.

- [ ] `MemberDevice` entity and APNs device token registration
- [ ] Push notification delivery channel in delivery abstraction
- [ ] Reminder ack/snooze via push notification actions
- [ ] Household activity feed / reminder inbox in-app
- [ ] Member-to-device mapping for multi-device households

SMS and email remain as fallback, escalation, and archival channels. Slack remains the operator/admin control surface. The architecture anticipates this from Phase 0 (device registration entity) so the transition is additive, not a rewrite.

---

## Future Considerations
- [ ] Web dashboard (Next.js) — household activity view, governance metrics
- [ ] Voice interface (Whisper + TTS)
- [ ] Smart home device control
- [ ] Gmail OAuth calendar adapter (optional, deferred — see ADR-006 calendar strategy)
- [ ] Apple Calendar (CalDAV) adapter
- [ ] Slack/Discord as additional household channels beyond operator use
