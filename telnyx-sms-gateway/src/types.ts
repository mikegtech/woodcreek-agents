export type TelnyxWebhookEnvelope = {
    data?: {
        event_type?: string;
        payload?: any;
    };
};

export type InboundSmsEvent = {
    provider: "telnyx";
    event_type: "message.received";
    telnyx_message_id: string;
    from_number: string;
    to_number: string;
    text: string;
    received_at?: string;
    messaging_profile_id?: string;
    organization_id?: string;
};
