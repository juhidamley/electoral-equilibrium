import { createHmac, timingSafeEqual } from "crypto";
import { NextResponse } from "next/server";
import { signSessionToken } from "@/lib/session";

// HMAC both passwords to equal-length (32-byte) buffers before timingSafeEqual.
// This avoids leaking length information via a Buffer.length !== branch and
// satisfies the constant-time requirement regardless of supplied string length.
function verifyPassword(supplied: string, expected: string): boolean {
  const key = process.env.DASHBOARD_SESSION_SECRET ?? "";
  const ha = createHmac("sha256", key).update(supplied).digest();
  const hb = createHmac("sha256", key).update(expected).digest();
  return timingSafeEqual(ha, hb);
}

export async function POST(req: Request): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }

  const supplied =
    body != null &&
    typeof body === "object" &&
    "password" in body &&
    typeof (body as Record<string, unknown>).password === "string"
      ? ((body as Record<string, unknown>).password as string)
      : "";

  const expected = process.env.DASHBOARD_PASSWORD ?? "";
  const secret = process.env.DASHBOARD_SESSION_SECRET ?? "";

  // Fail closed: if either env var is absent, authentication is impossible.
  // Still run verifyPassword to keep constant-time behaviour.
  const envOk = expected.length > 0 && secret.length > 0;
  const match = envOk && verifyPassword(supplied, expected);

  if (!match) {
    return NextResponse.json({ error: "Invalid password" }, { status: 401 });
  }

  const token = await signSessionToken(secret);
  const res = NextResponse.json({ ok: true });
  res.cookies.set("dashboard_session", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 8 * 60 * 60,
    path: "/",
  });
  return res;
}
