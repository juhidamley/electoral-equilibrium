// ============================================================================
// DashboardPage — the analyst dashboard at /dashboard. Assembles the 6 panels.
// ============================================================================
// This is a SERVER COMPONENT (note: no "use client" — it's async and runs on the
// server). Because it runs server-side, it can read the session cookie and verify
// it BEFORE any HTML is sent: an unauthenticated visitor gets the login form and
// never receives the dashboard markup at all. Authenticated visitors get the nav
// + a grid of the panel components (each of which then fetches its own data from
// the API on the client). The middleware also guards /dashboard/* sub-paths; this
// page additionally guards the bare /dashboard entry point itself.

import { cookies } from "next/headers";
import { verifySessionToken } from "@/lib/session";
import DashboardLoginForm from "@/components/DashboardLoginForm";
import DashboardNav from "@/components/DashboardNav";
import CoverageMatrix from "@/components/dashboard/CoverageMatrix";
import SentimentDistribution from "@/components/dashboard/SentimentDistribution";
import BioCoverage from "@/components/dashboard/BioCoverage";
import LossCurves from "@/components/dashboard/LossCurves";
import Convergence from "@/components/dashboard/Convergence";
import AuditTable from "@/components/dashboard/AuditTable";
import EstimateCount from "@/components/dashboard/EstimateCount";

// Server component — cookie verification happens on the server before any
// HTML is sent to the client. A devtools-injected cookie with a forged value
// will fail HMAC verification here and render the login form instead.
export default async function DashboardPage() {
  const cookieStore = await cookies();
  const token = cookieStore.get("dashboard_session")?.value;
  const secret = process.env.DASHBOARD_SESSION_SECRET ?? "";
  const authenticated = await verifySessionToken(token, secret);

  if (!authenticated) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-gray-50">
        <DashboardLoginForm />
      </main>
    );
  }

  return (
    <div className="flex min-h-screen bg-gray-50">
      <DashboardNav />
      <main className="flex-1 p-8">
        {/* Page header with live estimate count */}
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-gray-900">
            Electoral Equilibrium — Analyst Dashboard
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Pipeline internals ·{" "}
            <EstimateCount />
          </p>
        </div>

        {/* Row 1 — two-column: Coverage Matrix + Sentiment Distribution */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <CoverageMatrix />
          <SentimentDistribution />
        </div>

        {/* Row 2 — full-width: Bio Classification Coverage */}
        <div className="mt-6">
          <BioCoverage />
        </div>

        {/* Row 3 — full-width: Training Loss Curves */}
        <div className="mt-6">
          <LossCurves />
        </div>

        {/* Row 4 — full-width: MC Convergence Audit */}
        <div className="mt-6">
          <Convergence />
        </div>

        {/* Row 5 — full-width: Estimate Audit Log */}
        <div className="mt-6 pb-8">
          <AuditTable />
        </div>
      </main>
    </div>
  );
}
