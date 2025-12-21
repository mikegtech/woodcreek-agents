# Woodcreek Agents - Roadmap

## Domain Configuration
- **API Domain:** woodcreek.ai (agents, dashboard, traefik)
- **Email Domain:** woodcreek.me (RAG content source)

---

## Phase 2: Agent Implementation
- [x] General Assistant Agent (Groq/local vLLM)
- [ ] Supervisor Agent (intent routing)
- [ ] HOA Compliance Agent (RAG)
- [ ] Home Maintenance Agent (tool calling)
- [ ] Security & Cameras Agent (event-driven)

## Phase 3: Infrastructure
- [x] Milvus (GPU-accelerated vectors)
- [x] PostgreSQL (LangGraph checkpoints)
- [x] Docker containerization
- [x] Traefik routing (woodcreek.ai)
- [x] Cloudflare tunnel

## Phase 4: Email Integration
- [x] IMAP/SMTP client for woodcreek.me
- [x] Email ingestion pipeline → Milvus
- [x] Email send capability (notifications, reports)
- [x] Email RAG (search HOA communications, receipts, etc.)

## Phase 5: SMS/Messaging (Telnyx)
- [x] Telnyx account setup
- [x] Inbound SMS webhook
- [ ] Outbound SMS (alerts, reminders)
- [ ] Two-way conversation threading
- [ ] SMS → Agent routing

## Phase 6: RAG Pipeline
- [ ] Document ingestion (HOA docs, CC&Rs, manuals)
- [ ] Email ingestion (woodcreek.me inbox)
- [ ] Chunking strategy
- [ ] Embedding pipeline (sentence-transformers)
- [ ] Retrieval integration with agents

## Future Considerations
- [ ] Web dashboard (Next.js)
- [ ] Voice interface (Whisper + TTS)
- [ ] Calendar integrations
- [ ] Smart home device control