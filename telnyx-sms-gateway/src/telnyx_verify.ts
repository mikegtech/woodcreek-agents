function b64ToBytes(b64: string): Uint8Array {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
}

async function importEd25519PublicKey(publicKeyBase64: string): Promise<CryptoKey> {
    const raw = b64ToBytes(publicKeyBase64); // 32 bytes
    return crypto.subtle.importKey(
        "raw",
        raw,
        { name: "Ed25519" } as any,
        false,
        ["verify"]
    );
}

export async function verifyTelnyxWebhook(params: {
    rawBody: string;
    signatureBase64: string;
    timestamp: string;
    publicKeyBase64: string;
    toleranceSeconds?: number;
}): Promise<boolean> {
    const { rawBody, signatureBase64, timestamp, publicKeyBase64 } = params;
    const tolerance = params.toleranceSeconds ?? 300;

    if (!signatureBase64 || !timestamp) return false;

    // Basic replay protection: timestamp freshness
    const tsNum = Number(timestamp);
    if (!Number.isFinite(tsNum)) return false;
    const now = Math.floor(Date.now() / 1000);
    if (tsNum < now - tolerance) return false;

    const key = await importEd25519PublicKey(publicKeyBase64);

    const signedPayload = new TextEncoder().encode(`${timestamp}|${rawBody}`);
    const sig = b64ToBytes(signatureBase64);

    const ok = await crypto.subtle.verify(
        { name: "Ed25519" } as any,
        key,
        sig,
        signedPayload
    );

    return ok;
}
