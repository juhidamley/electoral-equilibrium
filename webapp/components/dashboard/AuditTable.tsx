"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { BLOC_LABEL } from "@/lib/blocs";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

interface AuditRow {
  id: number;
  timestamp: string;
  event_text: string;
  intensity: number | null;
  deltas_json: string | null;
  feasible: boolean | null;
  target_met: boolean | null;
  win_prob: number | null;
  llm_ms: number | null;
  optimizer_ms: number | null;
  montecarlo_ms: number | null;
  backend: string | null;
  party: string | null;
}

type SortKey = "timestamp" | "intensity" | "win_prob" | "total_ms";
type SortDir = "asc" | "desc";

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatMs(ms: number | null): string {
  if (ms == null) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${ms}ms`;
}

function totalMs(row: AuditRow): number | null {
  const a = row.llm_ms ?? 0;
  const b = row.optimizer_ms ?? 0;
  const c = row.montecarlo_ms ?? 0;
  if (row.llm_ms == null && row.optimizer_ms == null && row.montecarlo_ms == null) return null;
  return a + b + c;
}

function fmtTs(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toISOString().slice(0, 16).replace("T", " ");
  } catch {
    return ts.slice(0, 16);
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function parseDeltas(json: string | null): { bloc: string; label: string; delta: number }[] {
  if (!json) return [];
  try {
    const obj = JSON.parse(json) as Record<string, number>;
    return Object.entries(obj).map(([bloc, delta]) => ({
      bloc,
      label: BLOC_LABEL[bloc] ?? bloc,
      delta,
    }));
  } catch {
    return [];
  }
}

// ── Party badge ────────────────────────────────────────────────────────────────

function PartyBadge({ party }: { party: string | null }) {
  if (!party) return <span className="text-gray-400">—</span>;
  const cls =
    party === "democrat"
      ? "bg-blue-50 text-blue-700 border-blue-200"
      : party === "republican"
        ? "bg-red-50 text-red-700 border-red-200"
        : "bg-gray-50 text-gray-600 border-gray-200";
  return (
    <span className={`rounded border px-1.5 py-0.5 text-xs font-medium ${cls}`}>{party}</span>
  );
}

// ── Sort header cell ───────────────────────────────────────────────────────────

function SortTh({
  label,
  sortKey,
  currentKey,
  currentDir,
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  currentKey: SortKey;
  currentDir: SortDir;
  onSort: (k: SortKey) => void;
}) {
  const active = currentKey === sortKey;
  return (
    <th
      className="cursor-pointer select-none whitespace-nowrap px-3 py-2 text-left text-xs font-medium text-gray-500 hover:text-gray-800"
      onClick={() => onSort(sortKey)}
    >
      {label}
      <span className="ml-1 text-gray-300">
        {active ? (currentDir === "asc" ? "↑" : "↓") : "↕"}
      </span>
    </th>
  );
}

// ── Mini delta bar chart ──────────────────────────────────────────────────────

function DeltaChart({ deltasJson }: { deltasJson: string | null }) {
  const deltas = parseDeltas(deltasJson);
  if (!deltas.length) return <p className="text-xs text-gray-400">No delta data.</p>;

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart
        layout="vertical"
        data={deltas}
        margin={{ top: 2, right: 36, bottom: 2, left: 4 }}
        barSize={14}
      >
        <XAxis
          type="number"
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => (v >= 0 ? `+${(v * 100).toFixed(0)}pp` : `${(v * 100).toFixed(0)}pp`)}
          tick={{ fontSize: 10 }}
        />
        <YAxis type="category" dataKey="label" width={112} tick={{ fontSize: 11 }} />
        <Tooltip
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter={(v: any) => {
            const n = Number(v);
            if (isNaN(n)) return String(v);
            return `${n >= 0 ? "+" : ""}${(n * 100).toFixed(1)}pp`;
          }}
          itemStyle={{ fontSize: 11 }}
        />
        <ReferenceLine x={0} stroke="#d1d5db" />
        <Bar dataKey="delta" isAnimationActive={false}>
          {deltas.map((entry, i) => (
            <Cell key={i} fill={entry.delta >= 0 ? "#3b82f6" : "#ef4444"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Expanded row ──────────────────────────────────────────────────────────────

function ExpandedRow({ row }: { row: AuditRow }) {
  return (
    <tr className="bg-gray-50">
      <td colSpan={8} className="px-4 py-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Full event text */}
          <div>
            <p className="mb-1 text-xs font-medium text-gray-500">Full event</p>
            <p className="rounded border border-gray-200 bg-white p-2 text-xs text-gray-800">
              {row.event_text}
            </p>

            {/* Latency breakdown */}
            <div className="mt-3">
              <p className="mb-1 text-xs font-medium text-gray-500">Per-stage latency</p>
              <div className="flex flex-wrap gap-4 text-xs text-gray-600">
                <span>
                  <span className="font-medium">LLM:</span> {formatMs(row.llm_ms)}
                </span>
                <span>
                  <span className="font-medium">Optimizer:</span> {formatMs(row.optimizer_ms)}
                </span>
                <span>
                  <span className="font-medium">Monte Carlo:</span> {formatMs(row.montecarlo_ms)}
                </span>
                <span>
                  <span className="font-medium">Total:</span> {formatMs(totalMs(row))}
                </span>
              </div>
            </div>
          </div>

          {/* Delta vector mini chart */}
          <div>
            <p className="mb-1 text-xs font-medium text-gray-500">
              Race-bloc delta vector (pp shift)
            </p>
            <DeltaChart deltasJson={row.deltas_json} />
          </div>
        </div>
      </td>
    </tr>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function AuditTable() {
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "no_data">("idle");
  const [rows, setRows] = useState<AuditRow[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("timestamp");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounce search input → triggers refetch
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setSearch(inputValue), 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [inputValue]);

  useEffect(() => {
    setStatus("loading");
    const params = new URLSearchParams({ limit: "200" });
    if (search) params.set("search", search);
    fetch(`${API_URL}/api/audit?${params}`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: AuditRow[] | null) => {
        if (!data) { setStatus("no_data"); return; }
        setRows(data);
        setStatus(data.length ? "ok" : "no_data");
      })
      .catch(() => setStatus("no_data"));
  }, [search]);

  const sortedRows = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      let va: number, vb: number;
      if (sortKey === "timestamp") {
        va = new Date(a.timestamp).getTime();
        vb = new Date(b.timestamp).getTime();
      } else if (sortKey === "intensity") {
        va = a.intensity ?? -Infinity;
        vb = b.intensity ?? -Infinity;
      } else if (sortKey === "win_prob") {
        va = a.win_prob ?? -Infinity;
        vb = b.win_prob ?? -Infinity;
      } else {
        // total_ms
        va = totalMs(a) ?? -Infinity;
        vb = totalMs(b) ?? -Infinity;
      }
      return sortDir === "asc" ? va - vb : vb - va;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  function toggleExpand(id: number) {
    setExpandedId((cur) => (cur === id ? null : id));
  }

  // ── Loading skeleton ─────────────────────────────────────────────────────

  const header = (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
      <div>
        <h2 className="text-sm font-semibold text-gray-900">Estimate Audit Log</h2>
        <p className="mt-0.5 text-xs text-gray-500">
          All pipeline estimates — click a row to expand delta vector and latency breakdown.
        </p>
      </div>
      <input
        type="text"
        placeholder="Search events…"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        className="rounded border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-300"
      />
    </div>
  );

  if (status === "idle" || status === "loading") {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        {header}
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-8 animate-pulse rounded bg-gray-50" />
          ))}
        </div>
      </div>
    );
  }

  if (status === "no_data" && !rows.length) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        {header}
        <p className="py-6 text-center text-sm text-gray-400">
          {search ? `No estimates match "${search}".` : "No estimates logged yet."}
        </p>
      </div>
    );
  }

  // ── Table ──────────────────────────────────────────────────────────────────

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6">
      {header}

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-100">
              <SortTh
                label="Timestamp"
                sortKey="timestamp"
                currentKey={sortKey}
                currentDir={sortDir}
                onSort={handleSort}
              />
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Event</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Party</th>
              <SortTh
                label="Intensity"
                sortKey="intensity"
                currentKey={sortKey}
                currentDir={sortDir}
                onSort={handleSort}
              />
              <SortTh
                label="P(win)"
                sortKey="win_prob"
                currentKey={sortKey}
                currentDir={sortDir}
                onSort={handleSort}
              />
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Feasible</th>
              <SortTh
                label="Latency"
                sortKey="total_ms"
                currentKey={sortKey}
                currentDir={sortDir}
                onSort={handleSort}
              />
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">
                <span className="sr-only">Expand</span>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {sortedRows.map((row) => (
              <React.Fragment key={row.id}>
                <tr
                  className="cursor-pointer transition-colors hover:bg-gray-50"
                  onClick={() => toggleExpand(row.id)}
                >
                  <td className="whitespace-nowrap px-3 py-2 font-mono text-gray-500">
                    {fmtTs(row.timestamp)}
                  </td>
                  <td className="max-w-[280px] px-3 py-2 text-gray-800" title={row.event_text}>
                    {truncate(row.event_text, 40)}
                  </td>
                  <td className="px-3 py-2">
                    <PartyBadge party={row.party} />
                  </td>
                  <td className="px-3 py-2 text-gray-700">
                    {row.intensity != null ? row.intensity.toFixed(1) : "—"}
                  </td>
                  <td className="px-3 py-2 font-medium text-gray-700">
                    {row.win_prob != null ? `${(row.win_prob * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="px-3 py-2">
                    {row.feasible == null ? (
                      <span className="text-gray-400">—</span>
                    ) : row.feasible ? (
                      <span className="text-green-600">✓</span>
                    ) : (
                      <span className="text-red-500">✗</span>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-gray-500">
                    {formatMs(totalMs(row))}
                  </td>
                  <td className="px-3 py-2 text-gray-300">
                    {expandedId === row.id ? "▲" : "▼"}
                  </td>
                </tr>
                {expandedId === row.id && <ExpandedRow row={row} />}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-gray-400">
        {sortedRows.length} row{sortedRows.length !== 1 ? "s" : ""}
        {search && ` matching "${search}"`}
        {rows.length >= 200 && " (showing first 200 — use search to filter)"}
      </p>
    </div>
  );
}
