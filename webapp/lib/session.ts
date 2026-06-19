// HMAC-SHA256 session token utilities — runs in both Node.js and Edge runtime.
//
// Token format: `{expiry_unix_ms}.{hmac_hex}`
// HMAC covers only the expiry string, signed with DASHBOARD_SESSION_SECRET.
// crypto.subtle.verify provides constant-time comparison by spec.

const SESSION_DURATION_MS = 8 * 60 * 60 * 1000; // 8 hours

async function importKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

function toHex(buf: ArrayBuffer): string {
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function fromHex(hex: string): Uint8Array<ArrayBuffer> {
  const pairs = hex.match(/.{2}/g);
  if (!pairs) return new Uint8Array(new ArrayBuffer(0));
  const buf = new ArrayBuffer(pairs.length);
  const arr = new Uint8Array(buf);
  pairs.forEach((b, i) => {
    arr[i] = parseInt(b, 16);
  });
  return arr;
}

export async function signSessionToken(secret: string): Promise<string> {
  const expiry = String(Date.now() + SESSION_DURATION_MS);
  const key = await importKey(secret);
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(expiry));
  return `${expiry}.${toHex(sig)}`;
}

export async function verifySessionToken(
  token: string | undefined,
  secret: string,
): Promise<boolean> {
  if (!token || !secret) return false;
  const dot = token.indexOf(".");
  if (dot === -1) return false;
  const expiry = token.slice(0, dot);
  const sigHex = token.slice(dot + 1);
  const expiryMs = parseInt(expiry, 10);
  if (Number.isNaN(expiryMs) || Date.now() > expiryMs) return false;
  const key = await importKey(secret);
  const sigBytes = fromHex(sigHex);
  try {
    // subtle.verify is constant-time per the Web Crypto spec
    return await crypto.subtle.verify("HMAC", key, sigBytes, new TextEncoder().encode(expiry));
  } catch {
    return false;
  }
}
