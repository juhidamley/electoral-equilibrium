import { cookies } from "next/headers";
import { verifySessionToken } from "@/lib/session";
import DashboardLoginForm from "@/components/DashboardLoginForm";
import DashboardNav from "@/components/DashboardNav";
import CoverageMatrix from "@/components/dashboard/CoverageMatrix";
import SentimentDistribution from "@/components/dashboard/SentimentDistribution";
import BioCoverage from "@/components/dashboard/BioCoverage";

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
        <h1 className="text-2xl font-semibold text-gray-900">Analyst Dashboard</h1>
        <p className="mt-2 text-sm text-gray-500">
          Pipeline internals — select a panel from the sidebar.
        </p>
        {/* Row 1 — Data Coverage Matrix (full width) */}
        <div className="mt-8">
          <CoverageMatrix />
        </div>

        {/* Row 2 — Sentiment + Bio coverage (two-column grid) */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <SentimentDistribution />
          <BioCoverage />
        </div>
      </main>
    </div>
  );
}
