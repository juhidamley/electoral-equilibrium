// ============================================================================
// Next.js MIDDLEWARE — runs on the server BEFORE a matched page is served.
// ============================================================================
// A file named middleware.ts at the project root is special in Next.js: the
// function below runs for every request whose path matches `config.matcher`
// (bottom of file), before the page itself. We use it as the dashboard's
// gatekeeper: check the session cookie and, if it's invalid, redirect to the
// login page instead of ever rendering the protected content.
//
// This is the FRONTEND guard (it stops unauthenticated users from loading the
// dashboard pages). The real data is still independently protected on the
// backend (FastAPI's _require_dashboard_auth) — defense in depth: even if someone
// bypassed this, the API would still refuse them.

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { verifySessionToken } from "@/lib/session";

// Protects all /dashboard/* sub-paths (e.g. /dashboard/coverage).
// The root /dashboard page runs its own server-side check via cookies()
// and renders the login form inline when unauthenticated, so it is
// intentionally excluded from this matcher to avoid redirect loops.
export async function middleware(req: NextRequest): Promise<NextResponse> {
  const token = req.cookies.get("dashboard_session")?.value;
  const secret = process.env.DASHBOARD_SESSION_SECRET ?? "";
  const valid = await verifySessionToken(token, secret);
  if (!valid) {
    return NextResponse.redirect(new URL("/dashboard", req.url));
  }
  return NextResponse.next();
}

// matcher = which paths trigger the middleware above. "/dashboard/:path+" means
// any path with at least one segment AFTER /dashboard (e.g. /dashboard/coverage),
// but NOT bare /dashboard itself — that page handles its own login inline, so
// excluding it here avoids an infinite redirect loop back to itself.
export const config = {
  matcher: ["/dashboard/:path+"],
};
