"use client";

// BioCoverage — Panel 3 of the analyst dashboard.
//
// Stacked horizontal BarChart: one row per shock, three stacked segments:
//   keyword  — detected via race/religion/gender lexicons (bio_classifier.py, stage 1)
//   setfit   — classified by SetFit model on Pi (bio_classifier.py, stage 2)
//   fallback — language-prior assignment (excluded from µ and Σ_Δ; held-out validation only)
//
// IMPORTANT — fallback bios are annotated with a note because per the pipeline
// invariant they must NOT enter mean or covariance estimation (CLAUDE.md §Language
// fallback rule).  Showing them here makes the exclusion auditable, not actionable.
//
// Shocks are ordered by total bios descending so the most-covered events appear first.
// Shock IDs are human-readable-ized: underscores → spaces, truncated to 24 chars.

import React, { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── API types ─────────────────────────────────────────────────────────────────

interface ShockCounts {
  keyword: number;
  setfit: number;
  fallback: number;
  total: number;
}

interface BioCoverageResponse {
  status: "ok" | "no_data";
  shocks?: Record<string, ShockCounts>;
}

// ── Chart row ─────────────────────────────────────────────────────────────────

interface ShockRow {
  shock: string;    // raw ID for keys
  label: string;   // display label
  keyword: number;
  setfit: number;
  fallback: number;
  total: number;
}

// ── Segment styling ───────────────────────────────────────────────────────────

const SEG_FILL = {
  keyword: "#3b82f6",   // blue  — lexicon match (high-precision)
  setfit:  "#22c55e",   // green — SetFit model (Pi endpoint)
  fallback: "#f59e0b",  // amber — language prior (excluded from optimizer inputs)
} as const;

const SEG_LABEL = {
  keyword:  "Keyword lexicon",
  setfit:   "SetFit model",
  fallback: "Language prior (fallback)",
} as const;

// ── Helpers ───────────────────────────────────────────────────────────────────

function humanShockId(id: string): string {
  const label = id.replace(/_/g, " ");
  return label.length > 24 ? label.slice(0, 22) + "…" : label;
}

function buildRows(shocks: Record<string, ShockCounts>): ShockRow[] {
  return Object.entries(shocks)
    .map(([shock, c]) => ({
      shock,
      label: humanShockId(shock),
      keyword: c.keyword ?? 0,
      setfit: c.setfit ?? 0,
      fallback: c.fallback ?? 0,
      total: c.total ?? (c.keyword ?? 0) + (c.setfit ?? 0) + (c.fallback ?? 0),
    }))
    .sort((a, b) => b.total - a.total);
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

function BioTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name: string; value: number; fill: string }[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const total = payload.reduce((s, p) => s + (p.value ?? 0), 0);
  return (
    <div className="min-w-[180px] space-y-1 rounded border border-gray-200 bg-white p-2.5 text-xs shadow-md">
      <p className="font-semibold text-gray-800">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1 text-gray-600">
            <span
              className="inline-block h-2 w-2 flex-none rounded-sm"
              style={{ background: p.fill }}
            />
            {SEG_LABEL[p.name as keyof typeof SEG_LABEL] ?? p.name}
          </span>
          <span className="tabular-nums font-medium text-gray-800">{p.value}</span>
        </div>
      ))}
      <div className="mt-1 border-t border-gray-100 pt-1 flex justify-between text-gray-500 tabular-nums">
        <span>Total</span>
        <span>{total}</span>
      </div>
      {payload.find((p) => p.name === "fallback" && (p.value ?? 0) > 0) && (
        <p className="mt-1 text-[10px] text-amber-600">
          † fallback bios excluded from µ / Σ_Δ estimation
        </p>
      )}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function BioCoverage() {
  const [fetchStatus, setFetchStatus] = useState<"idle" | "loading" | "ok" | "no_data">("idle");
  const [apiData, setApiData] = useState<BioCoverageResponse | null>(null);

  useEffect(() => {
    setFetchStatus("loading");
    fetch(`${API_URL}/api/bio-coverage`, { credentials: "include" })
      .then((r) => {
        if (r.status === 401 || !r.ok) {
          setFetchStatus("no_data");
          return null;
        }
        return r.json() as Promise<BioCoverageResponse>;
      })
      .then((data) => {
        if (!data) return;
        setApiData(data);
        setFetchStatus(
          data.status === "no_data" || !data.shocks || Object.keys(data.shocks).length === 0
            ? "no_data"
            : "ok",
        );
      })
      .catch(() => setFetchStatus("no_data"));
  }, []);

  const rows = useMemo(
    () => (apiData?.shocks ? buildRows(apiData.shocks) : []),
    [apiData],
  );

  const totalBios = useMemo(() => rows.reduce((s, r) => s + r.total, 0), [rows]);
  const hasFallback = useMemo(() => rows.some((r) => r.fallback > 0), [rows]);

  // Grow chart height with number of shocks; minimum for empty-state header.
  const ROW_H = 28;
  const chartH = Math.max(160, rows.length * ROW_H + 56);

  return (
    <div className="flex flex-col rounded-lg border border-gray-200 bg-white p-6">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-sm font-semibold text-gray-900">Bio Inference Coverage</h2>
        <p className="mt-0.5 text-xs text-gray-500">
          Bios classified per shock by inference method.
          {fetchStatus === "ok" && rows.length > 0 && (
            <> {rows.length} shock(s) · {totalBios.toLocaleString()} bios total.</>
          )}
        </p>
        {hasFallback && (
          <p className="mt-1 text-[10px] text-amber-600 bg-amber-50 rounded px-1.5 py-0.5 inline-block">
            † Language-prior fallback bios are excluded from µ / Σ_Δ (held-out validation only).
          </p>
        )}
      </div>

      {/* Loading */}
      {fetchStatus === "loading" && (
        <div className="space-y-2 animate-pulse">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="h-3 w-24 rounded bg-gray-100" />
              <div className="h-5 flex-1 rounded bg-gray-50" />
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {(fetchStatus === "no_data" || fetchStatus === "idle") && (
        <div className="flex flex-1 items-center justify-center py-8 text-sm text-gray-400">
          No bio classification data yet.
          <span className="ml-1 text-xs text-gray-300">
            Collect social posts and run the bio classifier to populate this panel.
          </span>
        </div>
      )}

      {/* Chart */}
      {fetchStatus === "ok" && rows.length > 0 && (
        <ResponsiveContainer width="100%" height={chartH}>
          <BarChart
            layout="vertical"
            data={rows}
            margin={{ top: 4, right: 24, bottom: 8, left: 8 }}
            barSize={ROW_H - 8}
          >
            <XAxis
              type="number"
              tick={{ fontSize: 10 }}
              label={{
                value: "Bio count",
                position: "insideBottom",
                offset: -4,
                fontSize: 10,
                fill: "#9ca3af",
              }}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={136}
              tick={{ fontSize: 11 }}
            />
            <Tooltip content={<BioTooltip />} cursor={{ fill: "#f9fafb" }} />
            <Legend
              iconType="square"
              iconSize={10}
              formatter={(value) =>
                SEG_LABEL[value as keyof typeof SEG_LABEL] ?? value
              }
              wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
            />
            <Bar dataKey="keyword" stackId="a" fill={SEG_FILL.keyword} isAnimationActive={false} name="keyword" />
            <Bar dataKey="setfit"  stackId="a" fill={SEG_FILL.setfit}  isAnimationActive={false} name="setfit" />
            <Bar dataKey="fallback" stackId="a" fill={SEG_FILL.fallback} isAnimationActive={false} name="fallback" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
