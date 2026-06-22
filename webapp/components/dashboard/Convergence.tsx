"use client";

// Convergence — Panel 5 of the analyst dashboard.
// Shows the Monte Carlo win-probability estimate at N=1k/5k/10k draws, with a
// shaded p5–p95 band, to demonstrate the estimate has CONVERGED (stabilized) by
// 10k draws — i.e. running more simulations wouldn't change the answer. Data from
// GET /api/convergence. A Pass/Fail badge flags whether it settled within ±0.005.

import React, { useEffect, useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Convergence tolerance: |p(N=10k) - p(N=5k)| <= TOL is a pass
const CONVERGENCE_TOL = 0.005;

// ── Types ─────────────────────────────────────────────────────────────────────

interface SeriesPoint {
  n: number;
  win_probability: number;
  p5: number;
  p95: number;
}

interface ApiResponse {
  status: "ok" | "no_data";
  shock?: string | null;
  party?: string | null;
  series?: SeriesPoint[];
  note?: string;
}

// Recharts data format: include both raw values and band arithmetic
interface ChartPoint {
  n: number;
  win: number;
  // lower band baseline and upper width for Recharts stacked Area band trick
  p5: number;
  bandWidth: number; // p95 - p5
}

// ── Custom tooltip ─────────────────────────────────────────────────────────

function ConvergenceTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name: string; value: number; color?: string }[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  const win = payload.find((p) => p.name === "win")?.value;
  const p5 = payload.find((p) => p.name === "p5")?.value;
  const bw = payload.find((p) => p.name === "bandWidth")?.value;
  const p95 = p5 != null && bw != null ? p5 + bw : undefined;
  return (
    <div className="rounded border border-gray-200 bg-white p-2.5 text-xs shadow-md">
      <p className="mb-1 font-semibold text-gray-700">N = {(label as number).toLocaleString()}</p>
      {win != null && <p className="text-blue-700">P(win): {(win * 100).toFixed(1)}%</p>}
      {p5 != null && p95 != null && (
        <p className="text-gray-500">
          90% CI: [{(p5 * 100).toFixed(1)}%, {(p95 * 100).toFixed(1)}%]
        </p>
      )}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Convergence() {
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "no_data">("idle");
  const [data, setData] = useState<ApiResponse | null>(null);

  useEffect(() => {
    setStatus("loading");
    fetch(`${API_URL}/api/convergence`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((d: ApiResponse | null) => {
        if (!d || d.status === "no_data" || !d.series?.length) {
          setStatus("no_data");
          return;
        }
        setData(d);
        setStatus("ok");
      })
      .catch(() => setStatus("no_data"));
  }, []);

  const chartData: ChartPoint[] = useMemo(
    () =>
      (data?.series ?? []).map((s) => ({
        n: s.n,
        win: s.win_probability,
        p5: s.p5,
        bandWidth: Math.max(0, s.p95 - s.p5),
      })),
    [data],
  );

  const final10k = data?.series?.find((s) => s.n === 10000) ?? null;
  const final5k = data?.series?.find((s) => s.n === 5000) ?? null;

  const converged =
    final10k !== null &&
    final5k !== null &&
    Math.abs(final10k.win_probability - final5k.win_probability) <= CONVERGENCE_TOL;

  // Degenerate: all p5, p95, and win_probability saturated at 1.0 — zero variance
  const isDegenerate = (data?.series ?? []).every(
    (s) => s.win_probability >= 0.9999 && s.p5 >= 0.9999 && s.p95 >= 0.9999,
  );

  const yMin = useMemo(() => {
    if (!data?.series?.length) return 0;
    return Math.max(0, Math.min(...data.series.map((s) => s.p5)) - 0.05);
  }, [data]);

  // ── Loading skeleton ───────────────────────────────────────────────────────

  if (status === "idle" || status === "loading") {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        <div className="mb-4 h-5 w-48 animate-pulse rounded bg-gray-100" />
        <div className="h-64 animate-pulse rounded-md bg-gray-50" />
      </div>
    );
  }

  // ── Empty state ────────────────────────────────────────────────────────────

  if (status === "no_data" || !data) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        <h2 className="mb-1 text-sm font-semibold text-gray-900">MC Convergence Audit</h2>
        <p className="text-sm text-gray-400">No equilibrium artifacts available yet.</p>
        <p className="mt-1 text-xs text-gray-300">
          Run the full pipeline (LLM → optimizer → MC) to generate a{" "}
          <code className="rounded bg-gray-50 px-1 font-mono">SimulationData</code> artifact.
        </p>
      </div>
    );
  }

  // ── Chart ──────────────────────────────────────────────────────────────────

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6">
      {/* Header row */}
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-gray-900">MC Convergence Audit</h2>
          <p className="mt-0.5 text-xs text-gray-500">
            Win probability at N = 1k / 5k / 10k with 90% CI band.{" "}
            {data.shock && (
              <span className="text-gray-400">
                Shock:{" "}
                <span className="italic">
                  {data.shock.length > 60 ? data.shock.slice(0, 60) + "…" : data.shock}
                </span>
              </span>
            )}
          </p>
        </div>

        {/* Pass / Fail badge */}
        <div className="flex items-center gap-2">
          {final10k && (
            <span
              className={`rounded px-2 py-0.5 text-xs font-semibold border ${
                converged
                  ? "bg-green-50 text-green-700 border-green-200"
                  : "bg-red-50 text-red-700 border-red-200"
              }`}
            >
              {converged ? "Converged ✓" : "Not converged ✗"}
            </span>
          )}
          {data.party && (
            <span
              className={`rounded px-2 py-0.5 text-xs border ${
                data.party === "democrat"
                  ? "bg-blue-50 text-blue-700 border-blue-200"
                  : "bg-red-50 text-red-700 border-red-200"
              }`}
            >
              {data.party}
            </span>
          )}
        </div>
      </div>

      {/* Degenerate covariance warning */}
      {isDegenerate && (
        <div className="mb-4 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          <span className="font-semibold">Degenerate covariance: </span>
          Win probability saturated at 100% with zero CI width. This means the MC found no
          losing simulations — the result is an artifact of near-zero covariance in the
          ILR space, not a trustworthy estimate. The "Converged" badge is misleading here.
        </div>
      )}

      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
          <XAxis
            dataKey="n"
            tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
            tick={{ fontSize: 11 }}
            label={{
              value: "simulations (N)",
              position: "insideBottom",
              offset: -4,
              fontSize: 11,
              fill: "#9ca3af",
            }}
          />
          <YAxis
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            tick={{ fontSize: 11 }}
            domain={[yMin, 1]}
          />
          <Tooltip content={<ConvergenceTooltip />} />

          {/* CI band: stack lower-baseline area then band-width area on top */}
          <Area
            type="monotone"
            dataKey="p5"
            stackId="band"
            stroke="none"
            fill="transparent"
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="bandWidth"
            stackId="band"
            stroke="none"
            fill="#bfdbfe"
            fillOpacity={0.6}
            isAnimationActive={false}
            name="90% CI"
          />

          {/* Final estimate reference line */}
          {final10k && (
            <ReferenceLine
              y={final10k.win_probability}
              stroke="#1e3a8a"
              strokeDasharray="6 3"
              label={{
                value: `final ${(final10k.win_probability * 100).toFixed(1)}%`,
                position: "right",
                fontSize: 10,
                fill: "#1e3a8a",
              }}
            />
          )}

          {/* ±0.005 tolerance band around final estimate */}
          {final10k && (
            <ReferenceArea
              y1={Math.max(0, final10k.win_probability - CONVERGENCE_TOL)}
              y2={Math.min(1, final10k.win_probability + CONVERGENCE_TOL)}
              fill="#dbeafe"
              fillOpacity={0.25}
              strokeOpacity={0}
            />
          )}

          {/* Win probability line */}
          <Line
            type="monotone"
            dataKey="win"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 4, fill: "#2563eb", strokeWidth: 0 }}
            isAnimationActive={false}
            name="P(win)"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Methodology note */}
      {data.note && (
        <p className="mt-3 text-xs text-gray-400 italic">{data.note}</p>
      )}

      {/* Convergence detail */}
      {final10k && final5k && (
        <p className="mt-1 text-xs text-gray-500">
          |P(N=10k) − P(N=5k)| ={" "}
          <span className={converged ? "text-green-600" : "text-red-600"}>
            {Math.abs(final10k.win_probability - final5k.win_probability).toFixed(4)}
          </span>{" "}
          (tol ≤ {CONVERGENCE_TOL})
        </p>
      )}
    </div>
  );
}
