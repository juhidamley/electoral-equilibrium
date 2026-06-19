"use client";

import { useState } from "react";
import { Lock } from "lucide-react";

export default function DashboardLoginForm() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/dashboard/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        // Full page reload so the server component re-reads the new cookie.
        window.location.href = "/dashboard";
      } else {
        const body = await res.json().catch(() => ({})) as { error?: string };
        setError(body.error ?? "Authentication failed");
      }
    } catch {
      setError("Network error — please try again");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="w-full max-w-sm rounded-xl border border-gray-200 bg-white p-8 shadow-sm">
      <div className="mb-6 flex flex-col items-center gap-2">
        <Lock className="h-8 w-8 text-gray-400" />
        <h1 className="text-lg font-semibold text-gray-900">Analyst Dashboard</h1>
        <p className="text-center text-xs text-gray-500">
          Electoral Equilibrium · internal use only
        </p>
      </div>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div>
          <label htmlFor="dashboard-password" className="mb-1.5 block text-xs font-medium text-gray-600">
            Password
          </label>
          <input
            id="dashboard-password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="Enter dashboard password"
            required
          />
        </div>
        {error && (
          <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-600" role="alert">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={submitting || password.length === 0}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
