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

export const config = {
  matcher: ["/dashboard/:path+"],
};
