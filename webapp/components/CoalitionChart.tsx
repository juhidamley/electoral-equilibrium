"use client";

// ============================================================================
// CoalitionChart — TWO side-by-side panels, one quantity each.
// ============================================================================
// BEGINNER ORIENTATION (React + recharts):
//   • This file exports a React COMPONENT — a function that takes `props` (its
//     inputs) and returns JSX (a description of what to draw). React re-runs the
//     function and updates the screen whenever the props change. Here the props
//     arrive from the parent page as SSE events stream in.
//   • "use client" (top line) marks this as a browser component (Next.js renders
//     some components on the server; charts need the browser, hence this).
//   • We draw with `recharts`. Each panel is a plain horizontal bar chart; the
//     only custom part is a `shape` (makeBarShape) that hand-draws one bar + its
//     label as raw SVG so we control sizing precisely.
//
// WHY TWO PANELS (this used to be one overlaid chart):
//   Panel A — "Predicted loyalty shift" (μ̃_i ∈ [0,1]): how loyal each bloc is to
//             the selected party AFTER the shock.
//   Panel B — "Optimizer-recommended coalition emphasis" (w̃_i ∈ [0,1]): the
//             optimizer's strategic weighting — how heavily to lean on each bloc.
//             This is NOT each bloc's share of the population or electorate.
//   The previous single-chart overlay drew w̃ as a translucent bar on top of μ̃,
//   which (a) had a rendering bug where the translucent layer didn't show and
//   (b) led users to read w̃ as population share. Two separate panels fix both:
//   each quantity gets its own clean [0,1] axis and its own honest header.
//
// Both quantities arrive together on the "equilibrium" SSE event; the panels show
// a skeleton until then, then render without animation.

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
// Shared by both panels — shows loyalty and (strategic) emphasis together. With
// the panels visually separated, surfacing both numbers in the tooltip is a
// convenience, not an overlay: the wording makes clear emphasis ≠ population share.

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
          Coalition emphasis (strategic weighting):{" "}
          <strong>{Math.round(entry.weight * 100)}%</strong>
        </p>
      )}
    </div>
  );
};

// ── Single-quantity bar shape factory ─────────────────────────────────────────
// Returns a recharts `shape` function that draws ONE bar + its value label.
// Defined at module level so the closures are stable; each panel wraps it in a
// useMemo (constant hook count) below.
//
// `valueKey` selects which ChartEntry field this bar reads (shifted | weight) —
// we read from payload, not the spread `value`, so a null field cleanly renders
// nothing instead of a zero-width artifact.

function makeBarShape(opts: {
  valueKey: "shifted" | "weight";
  color: string;
  fillOpacity: number;
  stripeWhenInfeasible?: boolean;
  feasible?: boolean;
  showBaseline?: boolean;
}) {
  return (props: Record<string, unknown>) => {
    const x = (props.x as number) ?? 0;
    const y = (props.y as number) ?? 0;
    const height = (props.height as number) ?? 0;
    const payload = props.payload as ChartEntry | undefined;
    const bg = props.background as { width?: number } | undefined;

    const sv = payload ? payload[opts.valueKey] : null;

    // Derive pixel scale from background.width (full plot span for domain [0,1]).
    // Falls back to recharts' own width/sv only when background is omitted —
    // avoids the width=0-during-initial-layout bug that blanked bars.
    const rawW = (props.width as number) ?? 0;
    const plotW: number | null =
      bg?.width != null && bg.width > 0
        ? bg.width
        : sv != null && sv > 0 && rawW > 0
          ? rawW / sv
          : null;
    if (sv == null || plotW == null) return <g />;

    const barW = Math.max(0, sv * plotW);
    const infeasible = opts.stripeWhenInfeasible && opts.feasible === false;
    const fill = infeasible ? "url(#stripe-infeasible)" : opts.color;
    const fillOpacity = infeasible ? 1 : opts.fillOpacity;

    const bv = payload?.baseline ?? null;
    const baselineX = opts.showBaseline && bv != null ? x + bv * plotW : null;

    const labelStr = `${Math.round(sv * 100)}%`;
    // Inside (white) when the bar is wide enough; otherwise just outside the end
    // (gray) so small bars — common for coalition emphasis — stay readable.
    const labelInside = barW > 34;

    return (
      <g>
        <rect x={x} y={y} width={barW} height={height} fill={fill} fillOpacity={fillOpacity} />

        {/* Baseline reference tick (μ_i pre-shock) — loyalty panel only */}
        {baselineX != null && (
          <line
            x1={baselineX} y1={y}
            x2={baselineX} y2={y + height}
            stroke="#9ca3af" strokeWidth={2} strokeDasharray="3 2"
          />
        )}

        {labelInside ? (
          <text
            x={x + barW - 4} y={y + height / 2 + 3}
            textAnchor="end" fontSize={10} fill="white"
          >
            {labelStr}
          </text>
        ) : (
          <text
            x={x + barW + 4} y={y + height / 2 + 3}
            textAnchor="start" fontSize={10} fill="#4b5563"
          >
            {labelStr}
          </text>
        )}
      </g>
    );
  };
}

// ── One panel (header + chart) ────────────────────────────────────────────────

function ChartPanel({
  title,
  subtitle,
  data,
  dataKey,
  shape,
}: {
  title: string;
  subtitle: string;
  data: ChartEntry[];
  dataKey: "shifted" | "weight";
  shape: (props: Record<string, unknown>) => React.ReactElement;
}) {
  return (
    <div className="min-w-0">
      <div className="mb-2">
        <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
        <p className="text-xs text-gray-400">{subtitle}</p>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 2, right: 40, bottom: 2, left: 8 }}
          barSize={22}
        >
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
            width={120}
            tick={{ fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey={dataKey} shape={shape as any} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

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

  // Gap in percentage points = how far the coalition's effective loyalty is above
  // (+) or below (−) the win threshold V_eq. We use the BACKEND's μ_eff scalar
  // (muEffShifted) and only subtract the target here. We deliberately do NOT
  // recompute μ_eff in the browser: the true formula needs the λ layer weights
  // and the religion/gender strata, which the frontend doesn't have — an earlier
  // version tried a race-only recompute and produced a wrong gap. Trust the
  // backend's authoritative number (and equilibrium.target_met) instead.
  const gapPP =
    muEffShifted !== null && target !== null
      ? (muEffShifted - target) * 100
      : null;

  // Two single-quantity shapes. useMemo keeps a constant hook count across the
  // null→data transition (Rules of Hooks), and both are declared BEFORE the
  // skeleton early return below.
  const LoyaltyShape = useMemo(
    () =>
      makeBarShape({
        valueKey: "shifted",
        color: partyColor,
        fillOpacity: 1,
        showBaseline: true,
      }),
    [partyColor],
  );
  const WeightShape = useMemo(
    () =>
      makeBarShape({
        valueKey: "weight",
        color: partyColor,
        fillOpacity: 0.55,
        stripeWhenInfeasible: true,
        feasible,
      }),
    [partyColor, feasible],
  );

  // ── Skeleton ───────────────────────────────────────────────────────────────
  if (shifted === null) {
    return (
      <div className="rounded-md border border-gray-100 bg-white p-4">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {[0, 1].map((panel) => (
            <div key={panel} className="space-y-3 animate-pulse">
              {RACE_BLOCS.map((b) => (
                <div key={b} className="flex items-center gap-3">
                  <div className="h-3 w-24 rounded bg-gray-100" />
                  <div className="h-5 flex-1 rounded bg-gray-100" />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── Chart data ─────────────────────────────────────────────────────────────
  // Sorted by post-shock loyalty descending so both panels share row order and
  // line up for visual comparison.
  const chartData: ChartEntry[] = RACE_BLOCS.map((bloc) => {
    const s = shifted[bloc] ?? null;
    const b = baseline?.[bloc] ?? null;
    const w = hasRebalanced ? (rebalanced![bloc] ?? null) : null;
    return {
      bloc,
      label: BLOC_LABEL[bloc] ?? bloc,
      baseline: b,
      shifted: s,
      weight: w,
      delta: s != null && b != null ? s - b : null,
    };
  }).sort((a, b) => (b.shifted ?? 0) - (a.shifted ?? 0));

  return (
    <div className="rounded-md border border-gray-100 bg-white p-4">
      {/* Infeasibility banner */}
      {!feasible && hasRebalanced && (
        <div className="mb-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm font-medium text-red-700">
          No feasible coalition path under this shock.
        </div>
      )}

      {/* Two distinct panels — loyalty (μ̃) and strategic emphasis (w̃) — never
          overlaid, so emphasis can't be misread as a share of the population. */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <ChartPanel
          title="Predicted loyalty shift"
          subtitle="Per-bloc support for the party after the shock (μ̃)"
          data={chartData}
          dataKey="shifted"
          shape={LoyaltyShape}
        />
        <ChartPanel
          title="Optimizer-recommended coalition emphasis"
          subtitle="Strategic weighting (w̃) — not population share"
          data={chartData}
          dataKey="weight"
          shape={WeightShape}
        />
      </div>

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
