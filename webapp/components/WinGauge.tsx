"use client";

// ============================================================================
// WinGauge — a speedometer-style dial showing the win probability.
// ============================================================================
// HOW TO READ IT: the needle sweeps left (0% = certain loss) → up (50% = toss-up)
// → right (100% = certain win) over a red/yellow/green arc. Below it: the 90%
// confidence interval. The number comes from the Monte Carlo simulation (the
// "simulation" SSE event).
//
// It's an SVG drawing computed by hand (the geometry math is below).
//
// Semicircular gauge — win probability from Logistic-Normal Monte Carlo.
//
// Arc geometry: center (100,100), radius 80, viewBox "0 0 200 110".
//   p=0 → left (180°), p=0.5 → top (90°), p=1 → right (0°).
// Three fixed colored bands: red 0–45%, yellow 45–55%, green 55–100%.
// Needle: triangle tip at radius 70, rotated about (100,100) by
//   (winProbability * 180 - 90)°. Verified anchor cases:
//   0% → -90° → points left ✓, 50% → 0° → points up ✓, 100% → +90° → right ✓

import React from "react";
import PercentileStrip from "@/components/PercentileStrip";

// ── Props ─────────────────────────────────────────────────────────────────────

interface WinGaugeProps {
  winProbability: number | null;
  percentiles: Record<string, number[]> | null;
  winProbabilityLow?: number;
  winProbabilityHigh?: number;
  loading?: boolean;
}

// ── Arc geometry ──────────────────────────────────────────────────────────────

const CX = 100;
const CY = 100;
const R = 80;
const NEEDLE_R = 70;
const SW = 16; // stroke width for arc bands

function arcPoint(p: number): [number, number] {
  const theta = Math.PI * (1 - p); // 180°→0° as p goes 0→1 (left→right via top)
  return [CX + R * Math.cos(theta), CY - R * Math.sin(theta)];
}

// Returns an SVG arc path from win-prob p1 to p2 (both in [0,1]),
// going clockwise through the upper semicircle (sweep=1).
// Spans ≥ 100% are split at the midpoint to avoid the degenerate
// diametrically-opposite-points case where SVG arcs are undefined.
function arcPath(p1: number, p2: number): string {
  const [x1, y1] = arcPoint(p1);
  const [x2, y2] = arcPoint(p2);
  if (p2 - p1 >= 1) {
    const [mx, my] = arcPoint(0.5); // top midpoint (100, 20)
    return (
      `M ${x1.toFixed(2)} ${y1.toFixed(2)} ` +
      `A ${R} ${R} 0 0 1 ${mx.toFixed(2)} ${my.toFixed(2)} ` +
      `A ${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`
    );
  }
  const largeArc = p2 - p1 > 0.5 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 ${largeArc} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function WinGauge({
  winProbability,
  percentiles,
  winProbabilityLow,
  winProbabilityHigh,
  loading,
}: WinGaugeProps) {
  const isLoading = loading || winProbability === null;

  const pct = winProbability !== null ? Math.round(winProbability * 100) : null;

  // Rotation: p=0 → -90°, p=0.5 → 0°, p=1 → +90°
  const needleAngle =
    winProbability !== null ? winProbability * 180 - 90 : -90;

  const hasCi =
    winProbabilityLow !== undefined && winProbabilityHigh !== undefined;

  // Degenerate CI: bootstrap produced [1,1] or [0,0] due to diagonal covariance
  // fallback — must not be presented as a real interval.
  const ciDegenerate = hasCi && winProbabilityLow === winProbabilityHigh;
  const lowPct = hasCi ? Math.round((winProbabilityLow as number) * 100) : null;
  const highPct = hasCi
    ? Math.round((winProbabilityHigh as number) * 100)
    : null;

  return (
    <div className="flex flex-col items-center rounded-md border border-gray-100 bg-white p-4">
      <svg
        viewBox="0 0 200 110"
        className="w-full max-w-[220px]"
        aria-hidden="true"
      >
        {/* Background grey track — always visible */}
        <path
          d={arcPath(0, 1)}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={SW}
          strokeLinecap="butt"
        />

        {/* Colored bands — appear when data is ready, painted over grey track */}
        {!isLoading && (
          <>
            {/* Red: 0%–45% (losing territory) */}
            <path
              d={arcPath(0, 0.45)}
              fill="none"
              stroke="#ef4444"
              strokeWidth={SW}
              strokeLinecap="butt"
            />
            {/* Yellow: 45%–55% (toss-up) */}
            <path
              d={arcPath(0.45, 0.55)}
              fill="none"
              stroke="#eab308"
              strokeWidth={SW}
              strokeLinecap="butt"
            />
            {/* Green: 55%–100% (favored) */}
            <path
              d={arcPath(0.55, 1)}
              fill="none"
              stroke="#22c55e"
              strokeWidth={SW}
              strokeLinecap="butt"
            />
          </>
        )}

        {/* Needle — rotated triangle, CSS transition animates value changes */}
        {!isLoading && (
          <g
            style={{
              transformOrigin: `${CX}px ${CY}px`,
              transform: `rotate(${needleAngle}deg)`,
              transition: "transform 0.8s ease-out",
            }}
          >
            <polygon
              points={`${CX - 3},${CY} ${CX + 3},${CY} ${CX},${CY - NEEDLE_R}`}
              fill="#1f2937"
            />
          </g>
        )}

        {/* Pivot circle — drawn last so it sits on top of the needle base */}
        <circle cx={CX} cy={CY} r={6} fill="#374151" />

        {/* Percentage readout — inside the gauge face, above the pivot */}
        {isLoading ? (
          <text
            x={CX}
            y={80}
            textAnchor="middle"
            fontSize="22"
            fontWeight="700"
            fill="#9ca3af"
            className="animate-pulse"
          >
            —
          </text>
        ) : (
          <text
            x={CX}
            y={80}
            textAnchor="middle"
            fontSize="22"
            fontWeight="700"
            fill="#111827"
          >
            {pct}%
          </text>
        )}
      </svg>

      {/* CI strip — below SVG in HTML so Tailwind classes apply cleanly */}
      <div className="mt-1 min-h-[1.25rem] text-center text-xs">
        {isLoading ? (
          <span className="animate-pulse text-gray-400">Computing…</span>
        ) : hasCi ? (
          ciDegenerate ? (
            <span className="italic text-gray-400">
              CI unavailable (insufficient covariance data)
            </span>
          ) : (
            <span className="text-gray-500">
              90% CI: {lowPct}%–{highPct}%
            </span>
          )
        ) : null}
      </div>

      {/* Percentile distribution strip — supplementary detail, shown only once
          simulation data is present; PercentileStrip handles null silently. */}
      <div className="w-full">
        <PercentileStrip percentiles={isLoading ? null : percentiles} />
      </div>
    </div>
  );
}
