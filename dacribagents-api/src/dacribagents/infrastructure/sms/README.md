# SMS Integration

## Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   Telnyx    │────▶│  Cloudflare Worker   │────▶│  Cloudflare      │
│   Webhook   │     │  (webhook.ts)        │     │  Queue           │
└─────────────┘     │  - Signature verify  │     │  (sms-events)    │
                    │  - Fast ACK          │     └────────┬─────────┘
                    └──────────────────────┘              │
                                                          │
                    ┌─────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Cloudflare Worker   │────▶│  Woodcreek API   │────▶│  Telnyx API  │
│  (consumer.ts)       │     │  /internal/sms   │     │  (send SMS)  │
│  - Batch delivery    │     │  - Background    │     └──────────────┘
└──────────────────────┘     │    processing    │
                             │  - Agent call    │
                             │  - Store msgs    │
                             └────────┬─────────┘
                                      │
                             ┌────────┴─────────┐
                             │                  │
                             ▼                  ▼
                    ┌──────────────┐   ┌──────────────┐
                    │   SQLite     │   │   Milvus     │
                    │ (short-term) │   │ (long-term)  │
                    │  last 20 msg │   │  archival    │
                    └──────────────┘   └──────────────┘
```

## Components

### 1. Cloudflare Workers (`telnyx-sms-gateway/`)

| File | Purpose |
|------|---------|
| `webhook.ts` | Receives Telnyx webhooks, verifies signatures, queues for processing |
| `consumer.ts` | Consumes queue, forwards batches to Woodcreek API |
| `telnyx_verify.ts` | Ed25519 signature verification |
| `types.ts` | TypeScript type definitions |

### 2. Woodcreek API (`dacribagents-api/`)

| File | Purpose |
|------|---------|
| `infrastructure/http/sms_ingest.py` | FastAPI endpoint for receiving SMS batches |
| `infrastructure/sqlite/client.py` | SQLite client for conversation storage |
| `infrastructure/sms/telnyx_provider.py` | Telnyx API client for sending SMS |
| `application/use_cases/process_sms.py` | SMS processing orchestration |

## Setup

### 1. Environment Variables

Add to `.env`:

```bash
# Telnyx SMS
TELNYX_API_KEY=your-api-key
TELNYX_FROM_NUMBER=+12142866568
TELNYX_MESSAGING_PROFILE_ID=4001982f-b4d7-4f63-8451-e9c6636da24b

# SMS Ingest Security (generate a random secret)
SMS_INGEST_SECRET=$(openssl rand -hex 32)

# SQLite
SQLITE_DB_PATH=/app/data/conversations.db
```

### 2. Docker Compose

Add to `docker-compose.yml`:

```yaml
services:
  agents-api:
    volumes:
      - sqlite_data:/app/data
    environment:
      - TELNYX_API_KEY=${TELNYX_API_KEY}
      - TELNYX_FROM_NUMBER=${TELNYX_FROM_NUMBER}
      - TELNYX_MESSAGING_PROFILE_ID=${TELNYX_MESSAGING_PROFILE_ID}
      - SMS_INGEST_SECRET=${SMS_INGEST_SECRET}
      - SQLITE_DB_PATH=/app/data/conversations.db

volumes:
  sqlite_data:
```

### 3. Cloudflare Workers

```bash
cd telnyx-sms-gateway

# Create queue
npx wrangler queues create sms-events

# Deploy workers
npm run deploy

# Set secrets
npx wrangler secret put TELNYX_PUBLIC_KEY_BASE64 --name telnyx-sms-webhook
npx wrangler secret put AGENT_INGEST_URL --name telnyx-sms-consumer
# Value: https://api.woodcreek.ai/internal/sms/events

npx wrangler secret put AGENT_INGEST_SECRET --name telnyx-sms-consumer
# Value: same as SMS_INGEST_SECRET
```

### 4. Telnyx Dashboard

Update webhook URL to:
```
https://telnyx-sms-webhook.<your-subdomain>.workers.dev/
```

## Message Flow

1. **Inbound SMS** → Telnyx sends webhook to Cloudflare Worker
2. **Signature Verify** → Worker verifies Ed25519 signature
3. **Queue** → Message added to Cloudflare Queue
4. **Fast ACK** → 200 returned to Telnyx immediately
5. **Consumer** → Queue consumer batches and forwards to Woodcreek API
6. **Process** → API stores inbound, calls agent, sends reply
7. **Reply** → Telnyx API sends SMS back to user
8. **Store** → Both messages saved to SQLite

## SQLite Schema

```sql
-- Conversations table
CREATE TABLE conversations (
  conversation_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  state_json TEXT
);

-- Messages table
CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  direction TEXT NOT NULL CHECK(direction IN ('inbound','outbound')),
  provider TEXT NOT NULL,
  provider_message_id TEXT,
  ts TEXT NOT NULL,
  from_number TEXT NOT NULL,
  to_number TEXT NOT NULL,
  text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'stored',
  
  UNIQUE(provider, provider_message_id),
  FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);
```

## Testing

```bash
# Test the ingest endpoint locally
curl -X POST http://localhost:8080/internal/sms/events \
  -H "Content-Type: application/json" \
  -H "x-sms-ingest-secret: your-secret" \
  -d '{
    "source": "cloudflare-queue",
    "env": "dev",
    "received_at": "2025-12-17T12:00:00Z",
    "events": [{
      "provider": "telnyx",
      "event_type": "message.received",
      "telnyx_message_id": "test-123",
      "from_number": "+12145551234",
      "to_number": "+12142866568",
      "text": "Hello from test!"
    }]
  }'

# Check SMS health
curl http://localhost:8080/internal/sms/health
```

## TODO

- [ ] Replace echo response with actual agent call
- [ ] Implement Milvus archival for old messages
- [ ] Add rate limiting
- [ ] Add message deduplication check
- [ ] Multi-agent routing based on keywords