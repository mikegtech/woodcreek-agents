import { InboundSmsEvent, TelnyxWebhookEnvelope } from "./types";
import { verifyTelnyxWebhook } from "./telnyx_verify";

export interface Env {
    ENV: string;
    TELNYX_PUBLIC_KEY_BASE64: string;

    SMS_EVENTS_QUEUE: Queue<InboundSmsEvent>;
}

function getHeader(req: Request, name: string): string {
    return req.headers.get(name) ?? req.headers.get(name.toLowerCase()) ?? "";
}

export default {
    async fetch(req: Request, env: Env): Promise<Response> {
        if (req.method !== "POST") return new Response("Method Not Allowed", { status: 405 });

        const url = new URL(req.url);

        // ACK-only failover endpoint (no signature required)
        if (url.pathname.endsWith("/fallback")) {
        const body = await req.text().catch(() => "");
        console.warn("TELNYX FAILOVER HIT", { path: url.pathname, bodyPreview: body.slice(0, 200) });

        return new Response(JSON.stringify({ status: "acknowledged", mode: "failover" }), {
            status: 200,
            headers: { "content-type": "application/json" },
        });
        }

        // IMPORTANT: read raw body once and keep it exactly for signature verification
        const rawBody = await req.text();

        const signature = getHeader(req, "telnyx-signature-ed25519");
        const timestamp = getHeader(req, "telnyx-timestamp");

        const ok = await verifyTelnyxWebhook({
            rawBody,
            signatureBase64: signature,
            timestamp,
            publicKeyBase64: env.TELNYX_PUBLIC_KEY_BASE64,
            toleranceSeconds: 300,
        });

        if (!ok) {
            return new Response(JSON.stringify({ error: "Unauthorized" }), {
                status: 401,
                headers: { "content-type": "application/json" },
            });
        }

        let parsed: TelnyxWebhookEnvelope;
        try {
            parsed = JSON.parse(rawBody);
        } catch {
            return new Response(JSON.stringify({ error: "Invalid JSON" }), {
                status: 400,
                headers: { "content-type": "application/json" },
            });
        }

        const eventType = parsed?.data?.event_type ?? "";
        if (eventType !== "message.received") {
            // ACK other event types quickly (delivered, failed, etc.)
            return new Response(JSON.stringify({ status: "ignored", event_type: eventType }), {
                status: 200,
                headers: { "content-type": "application/json" },
            });
        }

        const payload = parsed?.data?.payload ?? {};
        const fromNum = payload?.from?.phone_number ?? "";
        const toNum = (payload?.to?.[0]?.phone_number) ?? "";
        const telnyxId = payload?.id ?? "";
        const text = payload?.text ?? "";

        if (!fromNum || !toNum || !telnyxId) {
            return new Response(JSON.stringify({ error: "Missing required fields" }), {
                status: 400,
                headers: { "content-type": "application/json" },
            });
        }

        const msg: InboundSmsEvent = {
            provider: "telnyx",
            event_type: "message.received",
            telnyx_message_id: telnyxId,
            from_number: fromNum,
            to_number: toNum,
            text,
            received_at: payload?.received_at,
            messaging_profile_id: payload?.messaging_profile_id,
            organization_id: payload?.organization_id,
        };

        // enqueue for async processing (fast)
        await env.SMS_EVENTS_QUEUE.send(msg);

        // immediate ACK to Telnyx
        return new Response(JSON.stringify({ status: "queued", id: telnyxId }), {
            status: 200,
            headers: { "content-type": "application/json" },
        });
    },
};
