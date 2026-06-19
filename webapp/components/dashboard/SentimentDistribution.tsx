"use client";

// SentimentDistribution — Panel 2 of the analyst dashboard.
//
// Shows the distribution of RoBERTa elasticity scores per demographic bloc
// across filtered shocks, as a horizontal box-plot chart.
//
// IMPORTANT — these are NOT vote shares or win probabilities.
// They are raw RoBERTa sentiment scores in [−1, +1] that measure how strongly
// each bloc's social-media/news discourse reacted to a shock.  They feed the
// LLM fine-tuning pipeline as intermediate features; they do not flow directly
// into the optimizer.  Labels are explicit to prevent conflation with loyalty
// scalars (µ) or win-probability values elsewhere in the app.
//
// Box-plot geometry (horizontal, per bloc row):
//   ─┤  whisker min  |──[ IQR p25─p75 ]──|  whisker max  ├─
//                              │ median
//
// The score axis is stored internally as [0, 1] (score normalized by (v+1)/2)
// so the PercentileStrip-style custom shape can derive pixel positions from
// `x` (position of domain-min) and `width` (pixels to dataKey value = max01).
// Axis tick labels convert back to the original [−1, +1] range.

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { BLOC_LABEL } from "@/lib/blocs";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Taxonomy categories ───────────────────────────────────────────────────────
// Sourced from configs/shock_taxonomy.json — stable config, hardcoded to avoid
// a separate round-trip.

const CATEGORIES = [
  "All",
  "Geopolitical",
  "Economic",
  "Moral/Scandal",
  "Religious",
  "Immigration",
  "Criminal Justice",
  "Health/Pandemic",
  "Environmental",
  "Electoral/Voting Rights",
] as const;

type Category = (typeof CATEGORIES)[number];

// ── API types ─────────────────────────────────────────────────────────────────

interface SentimentApiResponse {
  status: "ok" | "no_data";
  model?: string;
  shocks?: string[];
  blocs?: Record<string, Record<string, number>>;
  note?: string | null;
}

// ── Box-plot data row ─────────────────────────────────────────────────────────

interface BoxRow {
  bloc: string;
  label: string;
  n: number;
  // Normalized to [0, 1] via (score + 1) / 2 for Recharts shape arithmetic.
  min01: number;
  p25_01: number;
  p50_01: number;
  p75_01: number;
  max01: number;
}

// ── Canonical bloc display order ──────────────────────────────────────────────

const BLOC_ORDER = [
  "african_american", "asian", "latino", "other_race", "white",
  "evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel",
  "women", "men", "other_gender",
];

// ── Client-side percentile helper ─────────────────────────────────────────────

function pct(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = p * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function norm(v: number): number {
  return (v + 1) / 2; // [−1, 1] → [0, 1]
}

function buildBoxRows(blocs: Record<string, Record<string, number>>): BoxRow[] {
  return BLOC_ORDER.flatMap((bloc) => {
    const scoreMap = blocs[bloc];
    if (!scoreMap) return [];
    const vals = Object.values(scoreMap)
      .filter((v) => typeof v === "number" && isFinite(v))
      .sort((a, b) => a - b);
    if (vals.length === 0) return [];
    const mn = vals[0];
    const mx = vals[vals.length - 1];
    const q25 = pct(vals, 0.25);
    const q50 = pct(vals, 0.5);
    const q75 = pct(vals, 0.75);
    return [
      {
        bloc,
        label: BLOC_LABEL[bloc] ?? bloc,
        n: vals.length,
        min01: norm(mn),
        p25_01: norm(q25),
        p50_01: norm(q50),
        p75_01: norm(q75),
        max01: norm(mx),
      },
    ];
  });
}

// ── Custom box-plot shape ─────────────────────────────────────────────────────
// Defined at module level (stable reference, no hook calls, no captured state).
// dataKey="max01" → width = max01 * scale (pixels from x=domain_min to x=max01).
// x is the pixel position of domain value 0 (i.e. normalized score 0 = original −1).
// All other positions: x_k = x + v_k * scale  where scale = width / max01.

function BoxShape(props: {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: BoxRow;
}): React.ReactElement {
  const { x = 0, y = 0, width = 0, height = 0 } = props;
  const entry = props.payload;
  if (!entry || width <= 0 || entry.max01 === 0) return <g />;

  const scale = width / entry.max01;
  const xMin = x + entry.min01 * scale;
  const xP25 = x + entry.p25_01 * scale;
  const xP50 = x + entry.p50_01 * scale;
  const xP75 = x + entry.p75_01 * scale;
  const xMax = x + width;

  const ym = y + height / 2;
  const boxH = Math.max(6, Math.round(height * 0.62));
  const yt = ym - boxH / 2;
  const yb = ym + boxH / 2;

  return (
    <g>
      {/* Whisker spine */}
      <line x1={xMin} y1={ym} x2={xMax} y2={ym} stroke="#94a3b8" strokeWidth={1.5} />
      {/* Whisker end-caps */}
      <line x1={xMin} y1={yt} x2={xMin} y2={yb} stroke="#94a3b8" strokeWidth={1.5} />
      <line x1={xMax} y1={yt} x2={xMax} y2={yb} stroke="#94a3b8" strokeWidth={1.5} />
      {/* IQR box (p25–p75) */}
      <rect
        x={xP25}
        y={yt}
        width={Math.max(0, xP75 - xP25)}
        height={boxH}
        fill="#3b82f6"
        fillOpacity={0.15}
        stroke="#3b82f6"
        strokeWidth={1}
      />
      {/* Median tick — slightly taller than box */}
      <line x1={xP50} y1={yt - 1} x2={xP50} y2={yb + 1} stroke="#3b82f6" strokeWidth={2} />
    </g>
  );
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

function BoxTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { payload: BoxRow }[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const denorm = (v: number) => (v * 2 - 1).toFixed(2);
  return (
    <div className="max-w-[200px] space-y-1 rounded border border-gray-200 bg-white p-2.5 text-xs shadow-md">
      <p className="font-semibold text-gray-800">{label ?? d.label}</p>
      <p className="text-gray-400 text-[10px]">n = {d.n} shock(s)</p>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-gray-600 tabular-nums">
        <span>Min</span>      <span className="text-right">{denorm(d.min01)}</span>
        <span>p25</span>      <span className="text-right">{denorm(d.p25_01)}</span>
        <span>Median</span>   <span className="text-right font-medium text-blue-600">{denorm(d.p50_01)}</span>
        <span>p75</span>      <span className="text-right">{denorm(d.p75_01)}</span>
        <span>Max</span>      <span className="text-right">{denorm(d.max01)}</span>
      </div>
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SentimentDistribution() {
  const [category, setCategory] = useState<Category>("All");
  const [fetchStatus, setFetchStatus] = useState<"idle" | "loading" | "ok" | "no_data">("idle");
  const [apiData, setApiData] = useState<SentimentApiResponse | null>(null);

  const doFetch = useCallback((cat: Category) => {
    setFetchStatus("loading");
    const url =
      cat === "All"
        ? `${API_URL}/api/sentiment-dist`
        : `${API_URL}/api/sentiment-dist?category=${encodeURIComponent(cat)}`;
    fetch(url, { credentials: "include" })
      .then((r) => {
        if (r.status === 401 || !r.ok) {
          setFetchStatus("no_data");
          return null;
        }
        return r.json() as Promise<SentimentApiResponse>;
      })
      .then((data) => {
        if (!data) return;
        setApiData(data);
        setFetchStatus(
          data.status === "no_data" || !data.blocs || Object.keys(data.blocs).length === 0
            ? "no_data"
            : "ok",
        );
      })
      .catch(() => setFetchStatus("no_data"));
  }, []);

  useEffect(() => {
    doFetch("All");
  }, [doFetch]);

  const handleCategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value as Category;
    setCategory(val);
    doFetch(val);
  };

  const boxRows = useMemo(
    () => (apiData?.blocs ? buildBoxRows(apiData.blocs) : []),
    [apiData],
  );

  const nShocks = apiData?.shocks?.length ?? 0;
  const ROW_H = 26;
  const chartH = boxRows.length * ROW_H + 52; // + x-axis space

  return (
    <div className="flex flex-col rounded-lg border border-gray-200 bg-white p-6">
      {/* Header */}
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-gray-900">
            Elasticity Score Distribution by Bloc
          </h2>
          {/* Explicit unit note: scores ≠ vote shares */}
          <p className="mt-0.5 text-xs text-gray-500">
            RoBERTa sentiment scores (−1 to +1) — intermediate features, not vote shares.{" "}
            {fetchStatus === "ok" && nShocks > 0 && (
              <span className="text-gray-400">{nShocks} shock(s) in view.</span>
            )}
          </p>
          {apiData?.model && (
            <p className="mt-0.5 text-[10px] text-gray-300 font-mono">{apiData.model}</p>
          )}
        </div>
        {/* Category dropdown */}
        <select
          value={category}
          onChange={handleCategoryChange}
          className="h-8 rounded-md border border-gray-200 bg-white px-2 text-xs text-gray-700 focus:border-blue-400 focus:outline-none"
          disabled={fetchStatus === "loading"}
        >
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      {/* Loading */}
      {fetchStatus === "loading" && (
        <div className="flex-1 space-y-2 animate-pulse">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="h-3 w-28 rounded bg-gray-100" />
              <div className="h-5 flex-1 rounded bg-gray-50" />
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {(fetchStatus === "no_data" || fetchStatus === "idle") && (
        <div className="flex flex-1 items-center justify-center py-8 text-sm text-gray-400">
          No sentiment data{category !== "All" ? ` for "${category}"` : " available yet"}.
        </div>
      )}

      {/* Chart — no_data note from backend */}
      {fetchStatus === "ok" && apiData?.note && (
        <p className="mb-2 text-xs text-amber-600 bg-amber-50 rounded px-2 py-1">
          {apiData.note}
        </p>
      )}

      {fetchStatus === "ok" && boxRows.length > 0 && (
        <>
          <ResponsiveContainer width="100%" height={chartH}>
            <BarChart
              layout="vertical"
              data={boxRows}
              margin={{ top: 4, right: 16, bottom: 28, left: 8 }}
              barSize={ROW_H - 6}
            >
              {/* Zero reference — score = 0 maps to normalized 0.5 */}
              <ReferenceLine
                x={0.5}
                stroke="#d1d5db"
                strokeDasharray="3 2"
              />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickCount={5}
                tickFormatter={(v: number) => (v * 2 - 1).toFixed(1)}
                tick={{ fontSize: 10 }}
                label={{
                  value: "RoBERTa elasticity score (−1 to +1)",
                  position: "insideBottom",
                  offset: -16,
                  fontSize: 10,
                  fill: "#9ca3af",
                }}
              />
              <YAxis
                type="category"
                dataKey="label"
                width={132}
                tick={{ fontSize: 11 }}
              />
              <Tooltip content={<BoxTooltip />} cursor={false} />
              {/* dataKey="max01" — see BoxShape comment for the pixel arithmetic. */}
              <Bar
                dataKey="max01"
                shape={BoxShape as never}
                isAnimationActive={false}
                fill="transparent"
              />
            </BarChart>
          </ResponsiveContainer>

          {/* Legend */}
          <div className="mt-2 flex flex-wrap gap-4 text-[10px] text-gray-500">
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-5 bg-gray-300" />
              whisker (min–max)
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-3 w-5 rounded-sm border border-blue-400 bg-blue-50" />
              IQR (p25–p75)
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-3 w-0.5 bg-blue-500" />
              median
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-3 border-l border-dashed border-gray-300" />
              score = 0
            </span>
          </div>
        </>
      )}
    </div>
  );
}
