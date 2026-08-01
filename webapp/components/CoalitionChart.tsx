"use client";

// ============================================================================
// CoalitionChart — three strata: Race / Religion / Gender
// ============================================================================
// STRUCTURE:
//   All three strata show Δ loyalty (change) on a symmetric axis (see
//   lib/deltaScale.ts for the domain — matches the backend's decode-table
//   clip ceiling) with green/red bars and a zero reference line.
//
//   Race    — two side-by-side panels:
//               LEFT:  Δ loyalty per race bloc (DeltaPanel) — from "deltas" event
//               RIGHT: equilibrium coalition weighting (w̃, [0,1]) — fixed given
//                      baseline loyalties; the "equilibrium" event re-sends it
//                      per shock but the value never changes (see
//                      docs/design/optimizer_framing.md)
//   Religion + Gender — Δ loyalty panel only; fixed strata (no optimizer weight).
//             Values come from the "deltas" SSE event.
//
// SSE fields consumed:
//   "deltas"      → deltas_race             (Δμ per race bloc)
//                 → deltas_religion         (Δμ per religion bloc)
//                 → deltas_gender           (Δμ per gender bloc)
//   "equilibrium" → equilibrium.weights     (race w̃, [0,1])

import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Customized,
} from "recharts";

import type { Party } from "@/lib/types";
import {
  BLOC_LABEL,
  RACE_BLOCS,
  RELIGION_BLOCS,
  GENDER_BLOCS,
} from "@/lib/blocs";
import { DELTA_AXIS_DOMAIN, DELTA_AXIS_MAX, DELTA_AXIS_TICKS, formatDeltaPP } from "@/lib/deltaScale";

// ── Props ─────────────────────────────────────────────────────────────────────

interface CoalitionChartProps {
  baseline: Record<string, number> | null;
  shifted: Record<string, number> | null;         // equilibrium.mu_shifted (race, [0,1]) — gates skeleton
  rebalanced: Record<string, number> | null;      // equilibrium.weights    (race, [0,1])
  deltasRace: Record<string, number> | null;      // deltas_race     (Δμ, from "deltas" event)
  deltasReligion: Record<string, number> | null;  // deltas_religion (Δμ, from "deltas" event)
  deltasGender: Record<string, number> | null;    // deltas_gender   (Δμ, from "deltas" event)
  feasible: boolean;
  targetMet: boolean | null;
  muEffShifted: number | null;
  target: number | null;
  party: Party;
  loading?: boolean;
}

// ── Colors ────────────────────────────────────────────────────────────────────

const PARTY_COLOR: Record<Party, string> = {
  democrat:   "#2563eb",
  republican: "#dc2626",
};

// ── Chart entry shapes ────────────────────────────────────────────────────────

interface LoyaltyEntry {
  bloc: string;
  label: string;
  baseline: number | null;
  shifted: number | null;
  weight: number | null;
  delta: number | null;
}

interface DeltaEntry {
  bloc: string;
  label: string;
  value: number;      // Δμ, can be negative
  absValue: number;   // |Δμ| — recharts Bar dataKey must be ≥ 0
  positive: boolean;
}

// ── Infeasible stripe defs ────────────────────────────────────────────────────

const InfeasibleDefs = (_: unknown) => (
  <defs>
    <pattern id="stripe-infeasible" patternUnits="userSpaceOnUse"
      width="8" height="8" patternTransform="rotate(45)">
      <rect width="4" height="8" fill="#dc2626" fillOpacity="0.5" />
    </pattern>
  </defs>
);

// ── Loyalty tooltip ───────────────────────────────────────────────────────────

const LoyaltyTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { payload: LoyaltyEntry }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  const e = payload[0].payload;
  return (
    <div className="rounded-md border border-gray-200 bg-white p-3 text-xs shadow-lg space-y-1">
      <p className="font-semibold">{label}</p>
      {e.shifted != null && (
        <p>
          Loyalty after shock: <strong>{Math.round(e.shifted * 100)}%</strong>
          {e.baseline != null && e.delta != null && (
            <span className={e.delta >= 0 ? " text-green-600" : " text-red-600"}>
              {" "}(was {Math.round(e.baseline * 100)}%, {formatDeltaPP(e.delta)})
            </span>
          )}
        </p>
      )}
      {e.weight != null && (
        <p>
          Equilibrium weighting: <strong>{Math.round(e.weight * 100)}%</strong>{" "}
          <span className="text-gray-400">(fixed, not shock-specific)</span>
        </p>
      )}
    </div>
  );
};

// ── Delta tooltip ─────────────────────────────────────────────────────────────

const DeltaTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { payload: DeltaEntry }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  const value = payload[0].payload.value;
  return (
    <div className="rounded-md border border-gray-200 bg-white p-3 text-xs shadow-lg space-y-1">
      <p className="font-semibold">{label}</p>
      <p>Loyalty shift:{" "}
        <strong className={value >= 0 ? "text-green-700" : "text-red-700"}>
          {formatDeltaPP(value)}
        </strong>
      </p>
    </div>
  );
};

// ── Loyalty bar shape factory (race [0,1] axis) ───────────────────────────────

function makeLoyaltyShape(opts: {
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
    const payload = props.payload as LoyaltyEntry | undefined;
    const bg = props.background as { width?: number } | undefined;
    const sv = payload ? payload[opts.valueKey] : null;
    const rawW = (props.width as number) ?? 0;
    const plotW: number | null =
      bg?.width != null && bg.width > 0 ? bg.width
      : sv != null && sv > 0 && rawW > 0 ? rawW / sv
      : null;
    if (sv == null || plotW == null) return <g />;

    const barW = Math.max(0, sv * plotW);
    const infeasible = opts.stripeWhenInfeasible && opts.feasible === false;
    const fill = infeasible ? "url(#stripe-infeasible)" : opts.color;
    const fillOpacity = infeasible ? 1 : opts.fillOpacity;
    const bv = payload?.baseline ?? null;
    const baselineX = opts.showBaseline && bv != null ? x + bv * plotW : null;
    const labelStr = `${Math.round(sv * 100)}%`;
    const labelInside = barW > 34;

    return (
      <g>
        <rect x={x} y={y} width={barW} height={height} fill={fill} fillOpacity={fillOpacity} />
        {baselineX != null && (
          <line x1={baselineX} y1={y} x2={baselineX} y2={y + height}
            stroke="#9ca3af" strokeWidth={2} strokeDasharray="3 2" />
        )}
        {labelInside
          ? <text x={x + barW - 4} y={y + height / 2 + 3} textAnchor="end" fontSize={10} fill="white">{labelStr}</text>
          : <text x={x + barW + 4} y={y + height / 2 + 3} textAnchor="start" fontSize={10} fill="#4b5563">{labelStr}</text>
        }
      </g>
    );
  };
}

// ── Delta bar shape (religion/gender symmetric axis) ─────────────────────────
// recharts positions x at the left edge of the [0, DELTA_AXIS_MAX] half-domain
// (after domain transform). We override placement by deriving the zero-pixel
// from background.x + background.width/2 (the axis is symmetric
// [-DELTA_AXIS_MAX, DELTA_AXIS_MAX] — see lib/deltaScale.ts).
//
// DELTA_AXIS_MAX (not a locally-hardcoded half-domain) is used here
// deliberately: this shape's pixel math and DeltaPanel's <XAxis domain> below
// must always agree on the same half-domain, or bars render at the wrong
// width relative to the visible axis/ticks. Importing one shared constant
// instead of two independent copies is exactly how that drifted out of sync
// the first time (see DECISIONS.md "Step 2.1" for the same lesson on the
// Python side).

function DeltaShape(props: Record<string, unknown>) {
  const y = (props.y as number) ?? 0;
  const height = (props.height as number) ?? 0;
  const payload = props.payload as DeltaEntry | undefined;
  const bg = props.background as { x?: number; width?: number } | undefined;

  if (!payload || !bg?.width) return <g />;

  const plotW = bg.width;
  const bgX = (bg.x as number) ?? 0;
  // zero is at the midpoint of the background rect
  const zeroX = bgX + plotW / 2;
  const barPx = Math.abs(payload.value) / DELTA_AXIS_MAX * (plotW / 2);
  const barX = payload.positive ? zeroX : zeroX - barPx;
  const fill = payload.positive ? "#16a34a" : "#dc2626";
  const labelPp = formatDeltaPP(payload.value);
  const labelInside = barPx > 30;

  return (
    <g>
      <rect x={barX} y={y} width={Math.max(0, barPx)} height={height} fill={fill} fillOpacity={0.75} />
      {labelInside ? (
        <text
          x={payload.positive ? barX + barPx - 3 : barX + 3}
          y={y + height / 2 + 3}
          textAnchor={payload.positive ? "end" : "start"}
          fontSize={10} fill="white"
        >{labelPp}</text>
      ) : barPx > 4 ? (
        <text
          x={payload.positive ? barX + barPx + 3 : barX - 3}
          y={y + height / 2 + 3}
          textAnchor={payload.positive ? "start" : "end"}
          fontSize={10} fill="#4b5563"
        >{labelPp}</text>
      ) : null}
    </g>
  );
}

// ── Sub-panels ────────────────────────────────────────────────────────────────

function LoyaltyPanel({ title, subtitle, data, dataKey, shape }: {
  title: string;
  subtitle: string;
  data: LoyaltyEntry[];
  dataKey: "shifted" | "weight";
  shape: (props: Record<string, unknown>) => React.ReactElement;
}) {
  return (
    <div className="min-w-0">
      <div className="mb-2">
        <h4 className="text-sm font-semibold text-gray-800">{title}</h4>
        <p className="text-xs text-gray-400">{subtitle}</p>
      </div>
      <ResponsiveContainer width="100%" height={170}>
        <BarChart layout="vertical" data={data}
          margin={{ top: 2, right: 40, bottom: 2, left: 8 }} barSize={18}>
          <Customized component={InfeasibleDefs} />
          <XAxis type="number" domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="label" width={118} tick={{ fontSize: 10 }} />
          <Tooltip content={<LoyaltyTooltip />} />
          <Bar dataKey={dataKey} shape={shape as any} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function DeltaPanel({ title, subtitle, data }: {
  title: string;
  subtitle: string;
  data: DeltaEntry[];
}) {
  const height = Math.max(100, data.length * 30 + 20);
  return (
    <div className="min-w-0">
      <div className="mb-2">
        <h4 className="text-sm font-semibold text-gray-800">{title}</h4>
        <p className="text-xs text-gray-400">{subtitle}</p>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart layout="vertical" data={data}
          margin={{ top: 2, right: 48, bottom: 2, left: 8 }} barSize={18}>
          <XAxis type="number" domain={DELTA_AXIS_DOMAIN}
            tickFormatter={formatDeltaPP}
            ticks={DELTA_AXIS_TICKS}
            tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="label" width={118} tick={{ fontSize: 10 }} />
          <Tooltip content={<DeltaTooltip />} />
          <ReferenceLine x={0} stroke="#9ca3af" strokeWidth={1} />
          <Bar dataKey="absValue" shape={DeltaShape as any} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Section divider ───────────────────────────────────────────────────────────

function StratumHeader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 mt-5 mb-3 first:mt-0">
      <span className="text-xs font-bold uppercase tracking-wider text-gray-400">{label}</span>
      <div className="flex-1 h-px bg-gray-100" />
    </div>
  );
}

function SkeletonRows({ n }: { n: number }) {
  return (
    <div className="space-y-2 animate-pulse">
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="h-3 w-24 rounded bg-gray-100" />
          <div className="h-5 flex-1 rounded bg-gray-100" />
        </div>
      ))}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function CoalitionChart({
  baseline,
  shifted,
  rebalanced,
  deltasRace,
  deltasReligion,
  deltasGender,
  feasible,
  targetMet,
  muEffShifted,
  target,
  party,
  loading,
}: CoalitionChartProps) {
  const partyColor = PARTY_COLOR[party];

  const gapPP =
    muEffShifted !== null && target !== null ? (muEffShifted - target) * 100 : null;

  // Constant hook count — must be before any early return
  const WeightShape = useMemo(
    () => makeLoyaltyShape({ valueKey: "weight", color: partyColor, fillOpacity: 0.55,
        stripeWhenInfeasible: true, feasible }),
    [partyColor, feasible],
  );

  // ── Skeleton — show until equilibrium (race data) arrives ─────────────────
  if (shifted === null) {
    return (
      <div className="rounded-md border border-gray-100 bg-white p-4 space-y-2">
        <StratumHeader label="Race" />
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <SkeletonRows n={5} />
          <SkeletonRows n={5} />
        </div>
        <StratumHeader label="Religion" />
        <SkeletonRows n={7} />
        <StratumHeader label="Gender" />
        <SkeletonRows n={3} />
      </div>
    );
  }

  // ── Race chart data ────────────────────────────────────────────────────────
  const raceData: LoyaltyEntry[] = RACE_BLOCS.map((bloc) => {
    const s = shifted[bloc] ?? null;
    const b = baseline?.[bloc] ?? null;
    const w = rebalanced ? (rebalanced[bloc] ?? null) : null;
    return {
      bloc,
      label: BLOC_LABEL[bloc] ?? bloc,
      baseline: b,
      shifted: s,
      weight: w,
      delta: s != null && b != null ? s - b : null,
    };
  }).sort((a, b_) => (b_.weight ?? 0) - (a.weight ?? 0));

  // ── Delta data for all three strata ───────────────────────────────────────
  function toDeltaEntries(
    blocs: readonly string[],
    deltas: Record<string, number> | null,
  ): DeltaEntry[] {
    if (!deltas) return [];
    return blocs.map((bloc) => {
      const v = deltas[bloc] ?? 0;
      return { bloc, label: BLOC_LABEL[bloc] ?? bloc, value: v, absValue: Math.abs(v), positive: v >= 0 };
    });
  }

  const raceDeltaData  = toDeltaEntries(RACE_BLOCS,     deltasRace);
  const religionData   = toDeltaEntries(RELIGION_BLOCS, deltasReligion);
  const genderData     = toDeltaEntries(GENDER_BLOCS,   deltasGender);

  return (
    <div className="rounded-md border border-gray-100 bg-white p-4">
      {!feasible && rebalanced !== null && (
        <div className="mb-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm font-medium text-red-700">
          No feasible coalition path under this shock.
        </div>
      )}

      {/* ── RACE: Δ loyalty + emphasis (w̃) ── */}
      <StratumHeader label="Race" />
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {raceDeltaData.length > 0 ? (
          <DeltaPanel
            title="Predicted loyalty shift"
            subtitle="Δ loyalty toward the party after the shock (Δμ)"
            data={raceDeltaData}
          />
        ) : (
          <SkeletonRows n={5} />
        )}
        <LoyaltyPanel
          title="Equilibrium coalition (baseline)"
          subtitle="Strategic weighting (w̃), fixed by baseline loyalties — not population share"
          data={raceData}
          dataKey="weight"
          shape={WeightShape}
        />
      </div>

      {/* ── RELIGION: delta axis only (fixed stratum — no optimizer weight) ── */}
      <StratumHeader label="Religion" />
      {religionData.length > 0 ? (
        <DeltaPanel
          title="Predicted loyalty shift"
          subtitle="Δ loyalty toward the party after the shock — religion is a fixed stratum, no coalition reweighting"
          data={religionData}
        />
      ) : (
        <SkeletonRows n={7} />
      )}

      {/* ── GENDER: delta axis only ── */}
      <StratumHeader label="Gender" />
      {genderData.length > 0 ? (
        <DeltaPanel
          title="Predicted loyalty shift"
          subtitle="Δ loyalty toward the party after the shock — gender is a fixed stratum, no coalition reweighting"
          data={genderData}
        />
      ) : (
        <SkeletonRows n={3} />
      )}

      {/* Equilibrium summary */}
      {shifted !== null && (
        <div className="mt-4 text-sm">
          {targetMet !== null ? (
            <p>
              <span className="font-medium text-gray-700">Equilibrium status: </span>
              {targetMet ? (
                <span className="font-semibold text-green-700">
                  MET{gapPP !== null && (
                    <span className="font-normal text-green-600"> (+{gapPP.toFixed(1)} pp above target)</span>
                  )}
                </span>
              ) : (
                <span className="font-semibold text-red-700">
                  NOT MET{gapPP !== null && (
                    <span className="font-normal text-red-600"> ({gapPP.toFixed(1)} pp below target)</span>
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
