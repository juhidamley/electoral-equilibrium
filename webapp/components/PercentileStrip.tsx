"use client";

// ============================================================================
// PercentileStrip — a mini box-plot per race bloc showing weight UNCERTAINTY.
// ============================================================================
// WHAT'S A BOX-PLOT (in case it's unfamiliar): a compact way to show a range of
// outcomes. The Monte Carlo produced thousands of possible coalition weights for
// each bloc; rather than one number, this shows the spread:
//   • the BOX spans the middle 50% of outcomes (25th–75th percentile),
//   • the line inside the box is the median (50th),
//   • the WHISKERS reach the 5th and 95th percentiles (the 90% range).
// A wide box/whiskers = lots of uncertainty about that bloc's weight; a narrow
// one = the optimizer is confident. Supplementary detail beneath the WinGauge.
//
// Compact horizontal box-plot strip — shows the p5/p25/p50/p75/p95 distribution
// of coalition weights per race bloc, as produced by the Logistic-Normal ILR
// Monte Carlo in simulation/montecarlo.py.
//
// Box-plot geometry (per row, x-axis = value in [0,1]):
//   ───┤  IQR box (p25–p75, blue fill)  ├───  whisker line p5–p95
//       └─────── median tick (p50) ──────┘
//
// Drawn with a custom BarShape so a single <Bar dataKey="p95"> gives us the
// full [0, p95] pixel budget; all other x-positions are derived by:
//   scale = width / p95   (pixels per unit, since width = p95 * chartScale)
//   x_k   = x + p_k * scale    (x = pixel position of value=0 on x-axis)
// Reading from props.payload (not spread data keys) for Recharts version safety.

import React from "react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from "recharts";

import { BLOC_LABEL, RACE_BLOCS } from "@/lib/blocs";

// ── Props ─────────────────────────────────────────────────────────────────────

interface PercentileStripProps {
  percentiles: Record<string, number[]> | null; // [p5, p25, p50, p75, p95] per bloc
}

// ── Internal types ────────────────────────────────────────────────────────────

interface BoxEntry {
  bloc: string;
  label: string;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

// ── Colors ────────────────────────────────────────────────────────────────────

const BOX_FILL = "#3b82f6";    // blue-500
const WHISKER = "#94a3b8";     // slate-400

// ── Custom bar shape ──────────────────────────────────────────────────────────
// Defined at module level (not inside the component) so it is stable across
// renders and never triggers a Rules-of-Hooks violation from early returns.
// BOX_FILL and WHISKER are module constants — no component state is captured.

function BoxShape(props: {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: BoxEntry;
}): React.ReactElement {
  const { x = 0, y = 0, width = 0, height = 0 } = props;
  const entry = props.payload;
  if (!entry || width <= 0 || !entry.p95) return <g />;

  const { p5, p25, p50, p75, p95 } = entry;
  const scale = width / p95; // px per unit on [0,1] axis

  const x5 = x + p5 * scale;
  const x25 = x + p25 * scale;
  const x50 = x + p50 * scale;
  const x75 = x + p75 * scale;
  const x95 = x + width; // = x + p95 * scale

  const ym = y + height / 2;                    // vertical midline of row
  const boxH = Math.max(6, Math.round(height * 0.65)); // IQR box height
  const yt = ym - boxH / 2;
  const yb = ym + boxH / 2;

  return (
    <g>
      {/* Whisker spine p5→p95 */}
      <line x1={x5} y1={ym} x2={x95} y2={ym} stroke={WHISKER} strokeWidth={1} />
      {/* Whisker end-caps */}
      <line x1={x5}  y1={yt} x2={x5}  y2={yb} stroke={WHISKER} strokeWidth={1} />
      <line x1={x95} y1={yt} x2={x95} y2={yb} stroke={WHISKER} strokeWidth={1} />
      {/* IQR box (p25–p75) */}
      <rect
        x={x25} y={yt}
        width={Math.max(0, x75 - x25)} height={boxH}
        fill={BOX_FILL} fillOpacity={0.18} stroke={BOX_FILL} strokeWidth={1}
      />
      {/* Median tick (p50) — slightly taller than box for visibility */}
      <line x1={x50} y1={yt - 1} x2={x50} y2={yb + 1} stroke={BOX_FILL} strokeWidth={2} />
    </g>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function PercentileStrip({ percentiles }: PercentileStripProps) {
  if (!percentiles) return null;

  const chartData: BoxEntry[] = RACE_BLOCS.flatMap((b) => {
    const row = percentiles[b];
    if (!row || row.length < 5) return [];
    const [p5, p25, p50, p75, p95] = row;
    return [{ bloc: b, label: BLOC_LABEL[b] ?? b, p5, p25, p50, p75, p95 }];
  });

  if (chartData.length === 0) return null;

  const rowH = 24; // px per bloc row
  const chartH = chartData.length * rowH + 28; // + x-axis height

  return (
    <div className="mt-4">
      <p className="mb-1.5 text-xs font-medium text-gray-400">
        Per-bloc coalition weight distribution (p5–p95)
      </p>
      <ResponsiveContainer width="100%" height={chartH}>
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 2, right: 6, bottom: 2, left: 4 }}
          barSize={rowH - 8}
        >
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 10 }}
            tickCount={6}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={118}
            tick={{ fontSize: 11 }}
          />
          {/* dataKey="p95" so width prop = pixels spanning [0, p95].
              All other percentile positions are derived from scale = width / p95. */}
          <Bar
            dataKey="p95"
            shape={BoxShape as any}
            isAnimationActive={false}
            fill="transparent"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
