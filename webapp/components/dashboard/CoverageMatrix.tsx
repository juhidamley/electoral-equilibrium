"use client";

// CoverageMatrix — bloc × cycle data-quality heatmap for the analyst dashboard.
//
// Visual form of the 60-cell data-coverage flag logged by build_voter_panel():
// white (missing) cells make gaps immediately visible and double as Figure X in
// the paper's data section. The panel parquets produced by the pipeline carry a
// "source" field (e.g. "ANES+CES+GSS", "NEP", "imputed_pew_lgbtq") that the
// /api/coverage endpoint converts into the four discrete quality labels shown here.
//
// Color map is intentionally discrete, NOT a continuous gradient: imputed and
// missing are categorically different kinds of cells (one has a documented
// estimate, the other has nothing), so a blue→white ramp would falsely imply
// they differ only in degree.

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

import { BLOC_LABEL } from "@/lib/blocs";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

type Quality = "multi-source" | "single-source" | "imputed" | "missing";

interface CoverageCell {
  bloc: string;
  cycle: number;
  quality: Quality;
  vote_share: number | null;
  sources: string[];
}

interface ApiResponse {
  status: "ok" | "no_data";
  cells?: CoverageCell[];
}

// Scatter data point: original CoverageCell fields + recharts position indices.
interface ScatterPoint extends CoverageCell {
  x: number; // cycle index (0, 1, 2, …)
  y: number; // bloc index (0 = top)
}

// ── Color map — discrete, NOT a gradient (see file-top note) ─────────────────

const QUALITY_FILL: Record<Quality, string> = {
  "multi-source": "#1e3a8a", // dark blue  — two or more survey sources
  "single-source": "#60a5fa", // medium blue — exactly one source
  imputed: "#9ca3af", // gray        — documented estimate (e.g. Pew carry-forward)
  missing: "#ffffff", // white       — undocumented gap (the coverage flag)
};

const QUALITY_STROKE: Record<Quality, string> = {
  "multi-source": "#1e3a8a",
  "single-source": "#60a5fa",
  imputed: "#9ca3af",
  missing: "#e5e7eb", // thin border so the cell outline is visible on white
};

const QUALITY_TEXT: Record<Quality, string> = {
  "multi-source": "Multi-source",
  "single-source": "Single-source",
  imputed: "Imputed",
  missing: "Missing",
};

const QUALITY_ORDER: Quality[] = ["multi-source", "single-source", "imputed", "missing"];

// ── Canonical bloc ordering: race → religion → gender ────────────────────────

const BLOC_ORDER = [
  "african_american",
  "asian",
  "latino",
  "other_race",
  "white",
  "evangelical",
  "catholic",
  "protestant",
  "secular",
  "jewish",
  "muslim",
  "other_rel",
  "women",
  "men",
  "other_gender",
];

// ── Chart constants ───────────────────────────────────────────────────────────

const CHART_MARGIN = { top: 10, right: 24, bottom: 40, left: 148 };
const CELL_H = 24; // px per grid row

// ── Custom square cell shape ──────────────────────────────────────────────────
// Returns a memoizable render function. cellW / cellH are computed from the
// container's pixel dimensions so cells tile the plot area exactly.
// We subtract 1px on each dimension to leave a 1-pixel gap between cells.

function makeCellShape(cellW: number, cellH: number) {
  return function CellRect(props: {
    cx?: number;
    cy?: number;
    quality?: Quality;
    [key: string]: unknown;
  }) {
    const { cx, cy, quality = "missing" } = props;
    if (cx == null || cy == null) return null;
    return (
      <rect
        x={cx - cellW / 2}
        y={cy - cellH / 2}
        width={Math.max(0, cellW - 1)}
        height={Math.max(0, cellH - 1)}
        fill={QUALITY_FILL[quality]}
        stroke={QUALITY_STROKE[quality]}
        strokeWidth={1}
      />
    );
  };
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

interface TooltipProps {
  active?: boolean;
  payload?: { payload: ScatterPoint }[];
}

function CellTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const q = d.quality;
  // Use gray for the "missing" label so it stays readable (white text ≠ visible)
  const labelColor = q === "missing" ? "#6b7280" : QUALITY_FILL[q];
  return (
    <div className="max-w-[200px] space-y-1 rounded border border-gray-200 bg-white p-2.5 text-xs shadow-md">
      <p className="font-semibold text-gray-800">{BLOC_LABEL[d.bloc] ?? d.bloc}</p>
      <p className="text-gray-500">Cycle: {d.cycle}</p>
      <p>
        Quality:{" "}
        <span className="font-medium" style={{ color: labelColor }}>
          {QUALITY_TEXT[q]}
        </span>
      </p>
      <p>
        Vote share:{" "}
        <span className="font-medium">
          {d.vote_share !== null ? `${(d.vote_share * 100).toFixed(1)}%` : "—"}
        </span>
      </p>
      {d.sources.length > 0 && (
        <p className="text-gray-500">
          Sources: <span className="text-gray-700">{d.sources.join(", ")}</span>
        </p>
      )}
    </div>
  );
}

// ── Legend ────────────────────────────────────────────────────────────────────

function MatrixLegend() {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-gray-600">
      {QUALITY_ORDER.map((q) => (
        <span key={q} className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 flex-none rounded-sm"
            style={{
              background: QUALITY_FILL[q],
              border: `1px solid ${QUALITY_STROKE[q]}`,
            }}
          />
          {QUALITY_TEXT[q]}
        </span>
      ))}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function CoverageMatrix() {
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "no_data">("idle");
  const [cells, setCells] = useState<CoverageCell[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(640);

  // ── Fetch on mount ─────────────────────────────────────────────────────────
  useEffect(() => {
    setStatus("loading");
    fetch(`${API_URL}/api/coverage`, { credentials: "include" })
      .then((r) => {
        // 401 → session expired; treat as no_data rather than crashing the panel
        if (r.status === 401 || !r.ok) {
          setStatus("no_data");
          return null;
        }
        return r.json() as Promise<ApiResponse>;
      })
      .then((data) => {
        if (!data) return;
        if (data.status === "no_data" || !data.cells?.length) {
          setStatus("no_data");
        } else {
          setCells(data.cells);
          setStatus("ok");
        }
      })
      .catch(() => setStatus("no_data"));
  }, []);

  // ── Track container pixel width for cell sizing ────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // ── Derived geometry ───────────────────────────────────────────────────────

  const presentBlocs = useMemo(
    () => BLOC_ORDER.filter((b) => cells.some((c) => c.bloc === b)),
    [cells],
  );

  const cycles = useMemo(
    () => Array.from(new Set(cells.map((c) => c.cycle))).sort((a, b) => a - b),
    [cells],
  );

  // Integer-index maps so cells are uniformly spaced regardless of year gaps.
  const blocIndexMap = useMemo(
    () => Object.fromEntries(presentBlocs.map((b, i) => [b, i])),
    [presentBlocs],
  );

  const cycleIndexMap = useMemo(
    () => Object.fromEntries(cycles.map((c, i) => [c, i])),
    [cycles],
  );

  const scatterData: ScatterPoint[] = useMemo(
    () =>
      cells.map((cell) => ({
        ...cell,
        x: cycleIndexMap[cell.cycle] ?? 0,
        y: blocIndexMap[cell.bloc] ?? 0,
      })),
    [cells, cycleIndexMap, blocIndexMap],
  );

  // Domain extended by ±0.5 so cells tile flush with the axis edges.
  // With domain [-0.5, n-0.5] and innerWidth pixels, each unit = innerWidth/n px = cellW.
  const nCycles = cycles.length;
  const nBlocs = presentBlocs.length;
  const innerWidth = containerWidth - CHART_MARGIN.left - CHART_MARGIN.right;
  const cellW = nCycles > 0 ? innerWidth / nCycles : 60;
  const chartHeight = nBlocs * CELL_H + CHART_MARGIN.top + CHART_MARGIN.bottom;

  // Memoize the shape function so Recharts doesn't remount cells on every render.
  const cellShape = useMemo(() => makeCellShape(cellW, CELL_H), [cellW]);

  // ── Loading skeleton ───────────────────────────────────────────────────────

  if (status === "idle" || status === "loading") {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        <div className="mb-4 h-5 w-48 animate-pulse rounded bg-gray-100" />
        <div className="h-64 animate-pulse rounded-md bg-gray-50" />
      </div>
    );
  }

  // ── Empty / no_data state ──────────────────────────────────────────────────

  if (status === "no_data") {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        <h2 className="mb-1 text-sm font-semibold text-gray-900">Data Coverage Matrix</h2>
        <p className="text-sm text-gray-400">No coverage data available yet.</p>
        <p className="mt-1 text-xs text-gray-300">
          Run{" "}
          <code className="rounded bg-gray-50 px-1 font-mono">
            just build-voter-panel
          </code>{" "}
          to populate the panel parquets.
        </p>
      </div>
    );
  }

  // ── Chart ──────────────────────────────────────────────────────────────────

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6">
      {/* Header */}
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-900">Data Coverage Matrix</h2>
          <p className="mt-0.5 text-xs text-gray-500">
            Voter panel quality by stratum and election cycle.{" "}
            <span className="text-gray-400">
              White cells are undocumented gaps (the 60-cell flag from{" "}
              <code className="font-mono">build_voter_panel</code>).
            </span>
          </p>
        </div>
        <MatrixLegend />
      </div>

      {/* Heatmap */}
      <div ref={containerRef}>
        <ScatterChart
          width={containerWidth}
          height={chartHeight}
          margin={CHART_MARGIN}
        >
          <XAxis
            type="number"
            dataKey="x"
            name="Cycle"
            domain={nCycles > 0 ? [-0.5, nCycles - 0.5] : [0, 1]}
            ticks={cycles.map((_, i) => i)}
            tickFormatter={(i: number) => String(cycles[i] ?? "")}
            tick={{ fontSize: 11 }}
            label={{
              value: "Election cycle",
              position: "insideBottom",
              offset: -20,
              fontSize: 11,
              fill: "#6b7280",
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Bloc"
            domain={nBlocs > 0 ? [-0.5, nBlocs - 0.5] : [0, 1]}
            reversed
            ticks={presentBlocs.map((_, i) => i)}
            tickFormatter={(i: number) => {
              const b = presentBlocs[Math.round(i)];
              return BLOC_LABEL[b] ?? b ?? "";
            }}
            tick={{ fontSize: 11 }}
            width={CHART_MARGIN.left - 8}
          />
          {/* cursor={false} suppresses the default crosshair overlay on cells */}
          <Tooltip content={<CellTooltip />} cursor={false} />
          <Scatter data={scatterData} shape={cellShape as never} />
        </ScatterChart>
      </div>

      {/* Coverage summary */}
      {cells.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-gray-500">
          {QUALITY_ORDER.map((q) => {
            const n = cells.filter((c) => c.quality === q).length;
            if (n === 0) return null;
            return (
              <span key={q}>
                <span
                  className="mr-1 font-medium"
                  style={{
                    color: q === "missing" ? "#6b7280" : QUALITY_FILL[q],
                  }}
                >
                  {n}
                </span>
                {QUALITY_TEXT[q].toLowerCase()}
              </span>
            );
          })}
          <span className="text-gray-300">
            — {cells.length} cells total ({nBlocs} blocs × {nCycles} cycles)
          </span>
        </div>
      )}
    </div>
  );
}
