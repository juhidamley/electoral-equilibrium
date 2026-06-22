// HMAC-SHA256 session token utilities — runs in both Node.js and Edge runtime.
//
// WHAT THIS DOES (beginner crypto): when a user logs in, we give them a "session
// token" proving they're authenticated. We must make sure they can't FORGE one.
// An HMAC ("hash-based message authentication code") is a tamper-proof stamp:
// HMAC(secret, message) produces a signature that you can only reproduce if you
// know the `secret`. The server keeps DASHBOARD_SESSION_SECRET private, so:
//   • signSessionToken()   stamps an expiry time → token "{expiry}.{signature}"
//   • verifySessionToken() re-stamps the expiry with the same secret and checks
//     it matches. If a user edits the expiry to extend their session, the
//     signature won't match and verification fails.
// This is the TypeScript twin of the Python _verify_session_token in
// shock_endpoint.py — both build/check tokens identically so a token signed by
// one is accepted by the other.
//
// Token format: `{expiry_unix_ms}.{hmac_hex}`
// HMAC covers only the expiry string, signed with DASHBOARD_SESSION_SECRET.
// crypto.subtle.verify provides constant-time comparison by spec (see the note
// in shock_endpoint.py on why constant-time matters — it defeats timing attacks).
// crypto.subtle is the standard Web Crypto API, available in both the Node and
// Edge (middleware) runtimes, so this one file works everywhere.

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
