"use client";

// Three layers rendered as a single horizontal bar per race bloc:
//   1. Opaque bar    — shifted (equilibrium.mu_shifted per-bloc loyalty μ̃_i ∈ [0,1])
//   2. Translucent   — rebalanced (equilibrium.weights w̃_i, fillOpacity=0.35)
//   3. Baseline tick — thin dashed gray line at μ_i (pre-shock), when available.
//
// All three layers arrive together on the "equilibrium" SSE event. The chart shows a
// skeleton until that event arrives, then renders both layers without animation.
//
// Both μ̃_i and w̃_i are in [0,1], so they share a single x-axis.

import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Customized,
} from "recharts";

import type { Party } from "@/lib/types";
import { BLOC_LABEL, RACE_BLOCS } from "@/lib/blocs";

// ── Props ─────────────────────────────────────────────────────────────────────

interface CoalitionChartProps {
  baseline: Record<string, number> | null;    // μ_i pre-shock (null until backend exposes it)
  shifted: Record<string, number> | null;     // μ̃_i per-bloc — equilibrium.mu_shifted
  rebalanced: Record<string, number> | null;  // w̃_i optimizer weights — equilibrium.weights
  feasible: boolean;
  targetMet: boolean | null;                  // equilibrium.target_met (authoritative)
  muEffShifted: number | null;               // equilibrium.mu_eff_shifted (λ-weighted scalar)
  target: number | null;                      // V_eq threshold (for gap arithmetic)
  party: Party;
  loading?: boolean;
}

// ── Colors ────────────────────────────────────────────────────────────────────

const PARTY_COLOR: Record<Party, string> = {
  democrat:   "#2563eb",
  republican: "#dc2626",
};

// ── Chart data shape ──────────────────────────────────────────────────────────

interface ChartEntry {
  bloc: string;
  label: string;
  baseline: number | null;
  shifted: number | null;
  weight: number | null;
  delta: number | null; // shifted - baseline (null when baseline not yet available)
}

// ── SVG defs — infeasible stripe pattern ──────────────────────────────────────
// Rendered via <Customized> so it lives inside the BarChart SVG and the
// fill="url(#stripe-infeasible)" reference resolves correctly.

const InfeasibleDefs = (_: unknown) => (
  <defs>
    <pattern
      id="stripe-infeasible"
      patternUnits="userSpaceOnUse"
      width="8"
      height="8"
      patternTransform="rotate(45)"
    >
      <rect width="4" height="8" fill="#dc2626" fillOpacity="0.5" />
    </pattern>
  </defs>
);

// ── Custom tooltip ────────────────────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { payload: ChartEntry }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  const entry = payload[0].payload;
  return (
    <div className="rounded-md border border-gray-200 bg-white p-3 text-xs shadow-lg space-y-1">
      <p className="font-semibold">{label}</p>
      {entry.shifted != null && (
        <p>
          Loyalty after shock:{" "}
          <strong>{Math.round(entry.shifted * 100)}%</strong>
          {entry.baseline != null && entry.delta != null && (
            <span className={entry.delta >= 0 ? " text-green-600" : " text-red-600"}>
              {" "}(was {Math.round(entry.baseline * 100)}%,{" "}
              {entry.delta >= 0 ? "+" : ""}
              {Math.round(entry.delta * 100)}pp)
            </span>
          )}
        </p>
      )}
      {entry.weight != null && (
        <p>
          Coalition weight: <strong>{Math.round(entry.weight * 100)}%</strong>
        </p>
      )}
    </div>
  );
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function CoalitionChart({
  baseline,
  shifted,
  rebalanced,
  feasible,
  targetMet,
  muEffShifted,
  target,
  party,
  loading,
}: CoalitionChartProps) {
  const partyColor = PARTY_COLOR[party];
  const hasRebalanced = rebalanced !== null;

  // Gap in pp — derived from backend scalar to avoid client-side λ/stratum recomputation.
  const gapPP =
    muEffShifted !== null && target !== null
      ? (muEffShifted - target) * 100
      : null;

  // ── Skeleton ───────────────────────────────────────────────────────────────
  if (shifted === null) {
    return (
      <div className="rounded-md border border-gray-100 bg-white p-4">
        <div className="space-y-3 animate-pulse">
          {RACE_BLOCS.map((b) => (
            <div key={b} className="flex items-center gap-3">
              <div className="h-3 w-28 rounded bg-gray-100" />
              <div className="h-6 flex-1 rounded bg-gray-100" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── Chart data ─────────────────────────────────────────────────────────────
  const chartData: ChartEntry[] = RACE_BLOCS.map((bloc) => {
    const s = shifted[bloc] ?? null;
    const b = baseline?.[bloc] ?? null;
    // weight is null for a bloc absent from rebalanced (omit translucent bar, not 0)
    const w = hasRebalanced ? (rebalanced![bloc] ?? null) : null;
    return {
      bloc,
      label: BLOC_LABEL[bloc] ?? bloc,
      baseline: b,
      shifted: s,
      weight: w,
      delta: s != null && b != null ? s - b : null,
    };
  }).sort((a, b) => Math.abs(b.delta ?? 0) - Math.abs(a.delta ?? 0));

  // ── Custom bar shape ───────────────────────────────────────────────────────
  // All three layers (opaque prediction, translucent rebalance, baseline tick)
  // are drawn in one custom shape so they share a coordinate system and the
  // translucent layer never causes the opaque bar to re-mount.
  const BarShape = useMemo(
    () =>
      (props: {
        x?: number; y?: number; width?: number; height?: number;
        shifted?: number | null; weight?: number | null;
        baseline?: number | null; delta?: number | null;
      }) => {
        const { x = 0, y = 0, width = 0, height = 0, shifted: sv, weight: wv, baseline: bv, delta: dv } = props;
        if (sv == null || width <= 0) return <g />;

        // pixels per unit on the shared [0,1] x-axis
        const scale = width / sv;

        // Baseline tick x-position (null when μ_i not yet in SSE payload)
        const baselineX = bv != null ? x + bv * scale : null;

        // Translucent weight bar width
        const weightW = hasRebalanced && wv != null ? Math.max(0, wv * scale) : 0;

        // Bar labels
        const deltaStr =
          dv != null
            ? `${dv >= 0 ? "+" : ""}${Math.round(dv * 100)}pp`
            : null;
        const weightStr =
          hasRebalanced && wv != null
            ? `${Math.round(wv * 100)}%`
            : null;

        return (
          <g>
            {/* 1. Opaque prediction bar (shifted μ̃_i) */}
            <rect
              x={x} y={y}
              width={Math.max(0, width)} height={height}
              fill={partyColor} fillOpacity={1}
            />

            {/* 2. Translucent rebalance overlay (weight w̃_i) */}
            {hasRebalanced && wv != null && (
              <rect
                x={x} y={y}
                width={weightW} height={height}
                fill={feasible ? partyColor : "url(#stripe-infeasible)"}
                fillOpacity={feasible ? 0.35 : 1}
              />
            )}

            {/* 3. Baseline reference tick (μ_i pre-shock) */}
            {baselineX != null && (
              <line
                x1={baselineX} y1={y}
                x2={baselineX} y2={y + height}
                stroke="#9ca3af" strokeWidth={2} strokeDasharray="3 2"
              />
            )}

            {/* Delta label on opaque bar (+8pp / -3pp) */}
            {deltaStr != null && width > 30 && (
              <text
                x={x + width - 4} y={y + height / 2 + 4}
                textAnchor="end" fontSize={10} fill="white"
              >
                {deltaStr}
              </text>
            )}

            {/* Weight label on translucent bar (14%) */}
            {weightStr != null && weightW > 30 && (
              <text
                x={x + weightW - 4} y={y + height / 2 + 4}
                textAnchor="end" fontSize={10} fill={partyColor} fillOpacity={0.9}
              >
                {weightStr}
              </text>
            )}
          </g>
        );
      },
    [partyColor, feasible, hasRebalanced],
  );

  return (
    <div className="rounded-md border border-gray-100 bg-white p-4">
      {/* Infeasibility banner */}
      {!feasible && hasRebalanced && (
        <div className="mb-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm font-medium text-red-700">
          No feasible coalition path under this shock.
        </div>
      )}

      {/* Section header — explains each layer before the user reads the chart */}
      <div className="mb-3 flex flex-wrap items-baseline gap-x-5 gap-y-1.5">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-5 rounded-sm" style={{ background: partyColor }} />
          <strong className="text-sm text-gray-800">What will happen</strong>
          <span className="text-xs text-gray-400">(loyalty shift)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-5 rounded-sm"
            style={{ background: partyColor, opacity: 0.35 }}
          />
          <span className="text-sm text-gray-500">Most likely rebalance</span>
          <span className="text-xs text-gray-400">(coalition weight)</span>
        </span>
        {baseline !== null && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-5 border-t-2 border-dashed border-gray-400" />
            <span className="text-xs text-gray-400">Pre-shock baseline</span>
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 2, right: 44, bottom: 2, left: 8 }}
          barSize={22}
        >
          {/* SVG pattern for infeasible bars */}
          <Customized component={InfeasibleDefs} />

          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={130}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Single <Bar> renders all three layers via BarShape */}
          <Bar
            dataKey="shifted"
            shape={BarShape as any}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>

      {/* Equilibrium summary — visible once equilibrium SSE event arrives */}
      {shifted !== null && (
        <div className="mt-3 text-sm">
          {targetMet !== null ? (
            <p>
              <span className="font-medium text-gray-700">Equilibrium status: </span>
              {targetMet ? (
                <span className="font-semibold text-green-700">
                  MET
                  {gapPP !== null && (
                    <span className="font-normal text-green-600">
                      {" "}(+{gapPP.toFixed(1)} pp above target)
                    </span>
                  )}
                </span>
              ) : (
                <span className="font-semibold text-red-700">
                  NOT MET
                  {gapPP !== null && (
                    <span className="font-normal text-red-600">
                      {" "}({gapPP.toFixed(1)} pp below target)
                    </span>
                  )}
                </span>
              )}
            </p>
          ) : (
            <p className="animate-pulse text-gray-400">Equilibrium: pending…</p>
          )}
        </div>
      )}
    </div>
  );
}
