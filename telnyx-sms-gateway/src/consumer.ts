import { InboundSmsEvent } from "./types";

export interface Env {
    ENV: string;
    AGENT_INGEST_URL: string;
    AGENT_INGEST_SECRET: string;
}

type BatchPayload = {
    source: "cloudflare-queue";
    env: string;
    received_at: string;
    events: InboundSmsEvent[];
};

export default {
    async queue(batch: MessageBatch<InboundSmsEvent>, env: Env): Promise<void> {
        const events = batch.messages.map(m => m.body);

        const payload: BatchPayload = {
            source: "cloudflare-queue",
            env: env.ENV,
            received_at: new Date().toISOString(),
            events,
        };

        const res = await fetch(env.AGENT_INGEST_URL, {
            method: "POST",
            headers: {
                "content-type": "application/json",
                "x-sms-ingest-secret": env.AGENT_INGEST_SECRET,
            },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            // Let Cloudflare retry by throwing; messages will be retried / DLQ depending on queue config.
            const txt = await res.text().catch(() => "");
            throw new Error(`Agent ingest failed: ${res.status} ${txt}`);
        }

        // ACK messages only after successful forward
        batch.ackAll();
    },
};
