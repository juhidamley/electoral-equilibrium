"use client";

// DashboardNav — the sidebar/nav for the analyst dashboard: links between the
// panel groups and a logout button (which clears the session cookie via the
// /api/dashboard/logout route). Purely navigational; holds no estimate data.

import Link from "next/link";
import { BarChart2, FileText, Users, Activity, LogOut } from "lucide-react";

const PANELS = [
  { href: "/dashboard/coverage", label: "Data Coverage", icon: BarChart2 },
  { href: "/dashboard/sentiment", label: "Sentiment Dist.", icon: Activity },
  { href: "/dashboard/bio", label: "Bio Coverage", icon: Users },
  { href: "/dashboard/audit", label: "Audit Log", icon: FileText },
] as const;

export default function DashboardNav() {
  async function handleLogout() {
    await fetch("/api/dashboard/logout", { method: "POST" });
    window.location.href = "/dashboard";
  }

  return (
    <nav className="flex w-56 flex-none flex-col border-r border-gray-200 bg-white px-3 py-6">
      <p className="mb-4 px-3 text-xs font-semibold uppercase tracking-wider text-gray-400">
        Analyst Dashboard
      </p>
      <ul className="flex flex-1 flex-col gap-1">
        {PANELS.map(({ href, label, icon: Icon }) => (
          <li key={href}>
            <Link
              href={href}
              className="flex items-center gap-3 rounded-md px-3 py-2 text-sm text-gray-700 hover:bg-gray-100"
            >
              <Icon className="h-4 w-4 text-gray-400" />
              {label}
            </Link>
          </li>
        ))}
      </ul>
      <button
        onClick={handleLogout}
        className="flex items-center gap-3 rounded-md px-3 py-2 text-sm text-gray-500 hover:bg-gray-100"
      >
        <LogOut className="h-4 w-4" />
        Log out
      </button>
    </nav>
  );
}
