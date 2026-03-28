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

### Phase 0 — Reminder Domain & Architecture ✓
Define the core domain model and persistence layer before any runtime behavior.

- [x] Define domain entities: `Household`, `HouseholdMember`, `Reminder`, `ReminderTarget`, `ReminderSchedule`, `ReminderDelivery`, `ReminderExecution`, `ReminderAcknowledgement`, `PreferenceRule`, `ReminderContextSource` — `domain/reminders/entities.py`
- [x] Define calendar identity entities: `CalendarIdentity`, `CalendarProvider`, `CalendarAccessPolicy` — `domain/reminders/entities.py`
- [x] Anticipate device registration: `MemberDevice` for future APNs/push support — `domain/reminders/entities.py`
- [x] Design reminder lifecycle state machine (draft → scheduled → pending_delivery → delivered → acknowledged / snoozed → cancelled / failed) — `domain/reminders/lifecycle.py`
- [x] Schema design in PostgreSQL (reminder tables, household/member tables, delivery log, audit trail) — `infrastructure/reminders/schema.py`
- [x] Define Pydantic models for reminder API contracts — `domain/reminders/models.py`
- [x] Establish `domain/reminders/` as the shared reminder domain package
- [x] Design provider-neutral `CalendarAdapter` interface — `application/ports/calendar_adapter.py`
- [x] Design provider-neutral `DeliveryChannelAdapter` interface — `application/ports/delivery_channel.py`
- [x] Design `ReminderPolicy` interface — `application/ports/reminder_policy.py`
- [x] Design `ReminderStore` persistence interface — `application/ports/reminder_store.py`
- [x] Implement reminder workflow use cases (create, update, schedule, cancel, ack, snooze, list) — `application/use_cases/reminder_workflows.py`
- [x] Write ADR-006 decision record
- [x] Tests: 101 passing (lifecycle transitions, entity validation, API model validation, workflow use cases)

### Phase 1 — Read-Only Context and Reminder Intelligence (in progress)
Give agents and the operator visibility into existing schedules and household context without writing or sending anything. Slack is the first operator-facing interaction surface.

#### Implemented
- [x] Slack operator interaction surface — `@woodcreek` app mention handler via `/internal/slack/events` endpoint
  - Pending approvals, active alerts, active reminders queries
  - Schedule/calendar queries ("what does tomorrow look like?")
  - Conflict detection ("who has conflicts Thursday night?")
  - Draft preview ("draft a reminder for the family about soccer at 5")
  - Approval explanation ("explain why this reminder needs approval")
  - Keyword-based command router with structured Slack mrkdwn responses
- [x] Read-only reminder intelligence queries — `application/use_cases/reminder_queries.py`
  - `list_pending_approval`, `list_active_alerts`, `list_active`, `list_by_states`
  - `explain_reminder` with source/approval/lifecycle descriptions
  - `preview_draft` non-destructive draft preview with channel estimates and notes
  - `build_schedule_summary` combined reminder + calendar view
- [x] Provider-neutral calendar read boundary — `MockCalendarAdapter` implementing `CalendarAdapter` protocol
  - `list_events`, `get_event`, `list_all_events` (household view)
  - Ready for WorkMail/EWS adapter swap without interface changes
- [x] `InMemoryReminderStore` — dev/test store implementation in `infrastructure/reminders/in_memory_store.py`
- [x] Slack mrkdwn formatters for all query results — `infrastructure/slack/formatters.py`
- [x] Slack client (httpx-based, no slack-sdk dependency) — `infrastructure/slack/client.py`
- [x] Settings: `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` configuration
- [x] Tests: 44 new tests (161 total) covering handler routing, formatters, queries, calendar mock

#### Remaining
- [ ] AWS WorkMail / EWS calendar adapter (replaces MockCalendarAdapter for production)
- [ ] iCal feed adapter (read-only)
- [ ] Build `ReminderContextSource` adapters: email-derived dates, HOA calendar, manual entries
- [ ] Reminder suggestion engine: agents propose reminders based on context
- [ ] Context enrichment: attach household member info, delivery preferences, and prior history
- [ ] Email-derived date extraction from existing RAG pipeline
- [ ] Supervisor/agent routing integration for unrecognized Slack queries

### Phase 2 — Reminder Authoring with Human Approval (in progress)
Enable agents and users to create reminders, but require explicit human approval before anything is sent.

#### Phase 2A — APIs and Human-in-the-Loop Approval Workflows ✓
- [x] Approval lifecycle states: `pending_approval`, `approved`, `rejected` added to state machine
- [x] Approval path: DRAFT → PENDING_APPROVAL → APPROVED → SCHEDULED (or REJECTED, terminal)
- [x] `ApprovalRecord` entity for audit trail (who submitted/approved/rejected, when, why)
- [x] `ApprovalPolicy` service — deterministic evaluation of whether a reminder needs approval
- [x] Write workflows: `submit_for_approval`, `approve_reminder`, `reject_reminder`, `get_approval_history`
- [x] Schedule gating: `schedule_reminder` raises `ApprovalRequiredError` if approval is needed but not granted
- [x] Slack write commands: `create a reminder ...`, `approve <id>`, `reject <id> <reason>`
- [x] Approval list shows short IDs for actionable approve/reject commands
- [x] `reminder_approval_records` table in PostgreSQL schema
- [x] Tests: 205 total (44 new) covering approval workflows, policy, Slack commands, lifecycle transitions

#### Phase 2B — Scheduling Engine and Slack Interactive Control ✓
- [x] Scheduling engine with `tick()` function — finds due reminders, transitions one-shot to `pending_delivery`, creates recurring executions
- [x] Cron-based recurring schedule computation via `croniter` — `compute_next_fire()` for standard cron expressions
- [x] One-shot reminders: SCHEDULED → PENDING_DELIVERY on fire time
- [x] Recurring reminders: stay SCHEDULED, create `ReminderExecution` per occurrence, update `next_fire_at`
- [x] Duplicate-fire prevention: only SCHEDULED reminders with `next_fire_at <= now` are eligible
- [x] Per-member targeting: `remind Mike about ...` resolves member by name via store
- [x] Slack interactive buttons (Block Kit): approve/reject/cancel/snooze actions with `block_actions` endpoint
- [x] Slack snooze command: `snooze <id> [minutes]`
- [x] Slack cancel command: `cancel <id>`
- [x] Scheduling queries: `scheduled reminders`, `recurring`, `pending delivery`
- [x] `ReminderStore` extensions: `update_schedule`, `list_due_reminders`
- [x] `SlackClient.update_message` for post-action message updates
- [x] `/internal/slack/interactions` endpoint for Block Kit button callbacks
- [x] Tests: 220 total (15 new) covering scheduler tick, cron computation, recurrence, per-member targeting, Slack commands

#### Phase 2 — Remaining
- [ ] Relative-time reminders ("3 days before X")
- [ ] Manual reminder creation via SMS or CLI
- [ ] Scheduling timer/poller (drives `tick()` on interval — FastAPI background task or external cron)

### Phase 3 — Multi-Channel Delivery and Family Context (in progress)
Wire reminders to real outbound channels and support household/group dynamics.

#### Phase 2C / 3A — Outbound Delivery via Telnyx and Email ✓
- [x] `DeliveryDispatcher` application service: finds PENDING_DELIVERY reminders, resolves targets to members, dispatches via adapters, records outcomes, transitions state
- [x] `TelnyxSmsAdapter`: wraps existing `TelnyxProvider`, formats SMS-safe messages with urgency prefix and ack instructions
- [x] `SmtpEmailAdapter`: AWS WorkMail SMTP_SSL delivery with HTML+text multipart, reminder metadata headers
- [x] `channel_policy.select_channel()`: deterministic channel selection by urgency/intent/contact info (critical→SMS, normal→email, digest→email only)
- [x] `acknowledge_delivery()` service: channel-neutral ack intake seam, transitions DELIVERED→ACKNOWLEDGED
- [x] `EventPublisher` port + `NoOpEventPublisher`: Kafka-aligned seam for domain events (reminder.delivered, reminder.delivery_failed, reminder.acknowledged) — wired to Kafka in Phase 4
- [x] Delivery status queries: `list_delivered`, `list_failed_deliveries`, `get_delivery_details`
- [x] Slack delivery visibility: `delivered today`, `failed deliveries` commands
- [x] SMTP settings: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_ADDRESS`
- [x] Tests: 236 total (16 new) covering dispatch, channel selection, ack, failure handling, event publishing

#### Phase 3B — Escalation, Retry, PreferenceRule Engine, and Scheduling Driver ✓
- [x] `PreferenceRule` integration: `select_channel()` consults stored rules with 4-level precedence (member+urgency → member default → household+urgency → household default)
- [x] Quiet hours enforcement: non-critical reminders deferred during member quiet window; CRITICAL and `quiet_hours_override` rules bypass
- [x] Delivery retry with bounded exponential backoff: `2^attempt * base_delay`, capped at 30min, max 3 retries, retriable vs non-retriable failure detection
- [x] Escalation: re-deliver via fallback channel after ack timeout (2hr normal, 30min urgent), escalate to alternate member, max 2 escalation steps, publishes `reminder.escalation_triggered` event
- [x] Control-loop driver: `run_cycle()` executes schedule → dispatch → retry → escalate in one tick; `POST /internal/scheduler/tick` endpoint
- [x] Slack visibility: `deferred`/`quiet hours` query, updated help text
- [x] Fixed `InMemoryReminderStore.update_delivery` (was no-op, now persists status changes)
- [x] Tests: 256 total (20 new) covering preference resolution, quiet hours, backoff, retry/escalation, control loop

#### Phase 3 — Remaining
- [ ] Slack notification delivery adapter
- [ ] Batch/digest mode: group low-priority reminders into daily/weekly household digests

### Phase 4 — Event-Driven Household Orchestration (in progress)
Move from scheduled reminders to reactive, event-driven triggers across the agent ecosystem.

#### Phase 4A — Kafka Event Intake and Event-Driven Reminder Creation ✓
- [x] Event intake service: deterministic mapping of upstream events → reminders with 4 supported event types (`warranty.expiring`, `hoa.deadline.approaching`, `camera.offline`, `maintenance.due`)
- [x] `UpstreamEvent` model with severity-to-urgency mapping, source metadata preservation
- [x] `ReminderEventConsumer`: Kafka consumer using `confluent_kafka.Consumer`, aligned with existing SASL_SSL pattern
- [x] `KafkaEventPublisher`: Kafka-backed `EventPublisher` implementation using `confluent_kafka.Producer`
- [x] Durable dedupe: `dedupe_key = "{event_type}:{event_id}"`, checked against non-terminal reminders, replay-safe
- [x] Domain events published: `reminder.created_from_event`, `reminder.duplicate_ignored`
- [x] HTTP event intake endpoint: `POST /internal/events/ingest` for dev/testing without Kafka
- [x] Slack visibility: `event originated` / `from events` / `upstream` query command
- [x] `list_event_originated()` query: filters reminders with `source_event_id != None`
- [x] Settings: `kafka_topic_events = "woodcreek.events.v1"`
- [x] Tests: 271 total (15 new) covering event mapping, severity override, dedupe/replay, cancel+re-create, event publishing, source metadata

#### Phase 4B — SMS Inbound Acknowledgment and Slack Delivery Adapter ✓
- [x] `SlackDeliveryAdapter`: posts reminder/alert notifications to Slack channels, implements `DeliveryChannelAdapter`
- [x] SMS inbound reply handler: bounded command parsing (OK/DONE/ACK → acknowledge, SNOOZE N → snooze, CANCEL → cancel)
- [x] Reminder-execution correlation: sender phone → member lookup → most recent SMS delivery → reminder_id
- [x] SMS ingest wired: Telnyx webhook → `_process_sms_event` → `handle_sms_reply()` first, fallback to general processing
- [x] Slack visibility: `acknowledged` reminders query
- [x] Settings: `SLACK_DELIVERY_CHANNEL` for reminder notification destination
- [x] Tests: 297 total (26 new) covering SMS command parsing, ack/snooze/cancel via reply, correlation, Slack adapter

#### Phase 4 — Remaining
- [ ] Cross-agent reminder coordination: maintenance agent triggers reminder, compliance agent enriches it
- [ ] Calendar write-back: approved reminders optionally sync to WorkMail calendar (via CalendarAdapter)
- [ ] Batch/digest mode: group low-priority reminders into daily or weekly household digests
- [ ] LangGraph workflow integration: reminder lifecycle as a durable LangGraph graph with checkpointing

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
