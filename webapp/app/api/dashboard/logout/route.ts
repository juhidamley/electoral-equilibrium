// Logout route: POST /api/dashboard/logout. Logging out just means destroying
// the session cookie. We overwrite "dashboard_session" with an empty value and
// maxAge: 0, which tells the browser to delete it immediately. After this the
// middleware/backend will treat the user as unauthenticated. (The same secure
// cookie attributes as login are set so the browser matches and replaces the
// existing cookie.)

import { NextResponse } from "next/server";

export async function POST(): Promise<NextResponse> {
  const res = NextResponse.json({ ok: true });
  res.cookies.set("dashboard_session", "", {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 0, // 0 = expire now → browser deletes the cookie
    path: "/",
  });
  return res;
}
