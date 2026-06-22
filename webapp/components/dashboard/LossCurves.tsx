"use client";

// LossCurves — Panel 4 of the analyst dashboard.
// Plots the model's training progress: train loss (solid) and validation loss
// (dashed) per epoch, with a dropdown to pick which training run to view. Data
// comes from GET /api/training-logs (parsed from the HPC SLURM logs). This is
// how you check the fine-tuning actually learned — loss should fall over epochs.

import React, { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// 2020 baseline eval MAE from models/mistral-r16/trainer_state.json
const BASELINE_MAE = 0.0362;

// ── Types ─────────────────────────────────────────────────────────────────────

interface LogEntry {
  step: number;
  epoch: number | null;
  train_loss: number | null;
  val_loss: number | null;
  lr: number | null;
}

interface RunInfo {
  run_id: string;
  complete: boolean;
  val_available: boolean;
  val_attempted: boolean;
  log_history: LogEntry[];
  summary: Record<string, number | null> | null;
}

interface ApiResponse {
  status: "ok" | "no_data";
  runs?: RunInfo[];
  note?: string;
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

function LossTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name: string; value: number | null; color: string }[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded border border-gray-200 bg-white p-2.5 text-xs shadow-md">
      <p className="mb-1 font-semibold text-gray-700">Step {label}</p>
      {payload.map((p) =>
        p.value != null ? (
          <p key={p.name} style={{ color: p.color }}>
            {p.name}: {p.value.toFixed(4)}
          </p>
        ) : null,
      )}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function LossCurves() {
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "no_data">("idle");
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    setStatus("loading");
    fetch(`${API_URL}/api/training-logs`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: ApiResponse | null) => {
        if (!data || data.status === "no_data" || !data.runs?.length) {
          setStatus("no_data");
          return;
        }
        setRuns(data.runs);
        setSelectedId(data.runs[0].run_id);
        setStatus("ok");
      })
      .catch(() => setStatus("no_data"));
  }, []);

  const selectedRun = useMemo(
    () => runs.find((r) => r.run_id === selectedId) ?? null,
    [runs, selectedId],
  );

  const chartData = useMemo(
    () =>
      selectedRun?.log_history.map((entry) => ({
        step: entry.step,
        "Train loss": entry.train_loss,
        "Val loss": entry.val_loss,
      })) ?? [],
    [selectedRun],
  );

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

  if (status === "no_data" || !selectedRun) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6">
        <h2 className="mb-1 text-sm font-semibold text-gray-900">Training Loss Curves</h2>
        <p className="text-sm text-gray-400">No HPC training logs synced yet.</p>
        <p className="mt-1 text-xs text-gray-300">
          Sync{" "}
          <code className="rounded bg-gray-50 px-1 font-mono">rawdata/hpc_logs/</code> from
          the A100 node via Syncthing to see per-step loss curves.
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
          <h2 className="text-sm font-semibold text-gray-900">Training Loss Curves</h2>
          <p className="mt-0.5 text-xs text-gray-500">
            Per-step cross-entropy loss from HPC SLURM logs.{" "}
            <span className="text-gray-400">
              Dashed line = 2020 baseline MAE ({BASELINE_MAE}).
            </span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          {/* Val n/a badge */}
          {selectedRun.val_attempted && !selectedRun.val_available && (
            <span className="rounded bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700 border border-amber-200">
              val n/a
            </span>
          )}

          {/* Incomplete badge */}
          {!selectedRun.complete && (
            <span className="rounded bg-red-50 px-2 py-0.5 text-xs font-medium text-red-600 border border-red-200">
              crashed / partial
            </span>
          )}

          {/* Run selector */}
          {runs.length > 1 && (
            <select
              value={selectedId ?? ""}
              onChange={(e) => setSelectedId(e.target.value)}
              className="rounded border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-300"
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id}
                  {!r.complete ? " (partial)" : ""}
                </option>
              ))}
            </select>
          )}
          {runs.length === 1 && (
            <span className="text-xs text-gray-500 font-mono">{selectedRun.run_id}</span>
          )}
        </div>
      </div>

      {chartData.length === 0 ? (
        <p className="py-8 text-center text-sm text-gray-400">No log entries parsed.</p>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 11 }}
              label={{ value: "step", position: "insideBottom", offset: -2, fontSize: 11, fill: "#9ca3af" }}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => v.toFixed(3)}
            />
            <Tooltip content={<LossTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11 }}
              formatter={(value: string) =>
                value === "Val loss" && selectedRun.val_attempted && !selectedRun.val_available
                  ? `${value} (n/a)`
                  : value
              }
            />

            {/* Baseline MAE reference */}
            <ReferenceLine
              y={BASELINE_MAE}
              stroke="#6b7280"
              strokeDasharray="6 3"
              label={{
                value: `baseline ${BASELINE_MAE}`,
                position: "right",
                fontSize: 10,
                fill: "#6b7280",
              }}
            />

            <Line
              type="monotone"
              dataKey="Train loss"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="Val loss"
              stroke="#9ca3af"
              strokeWidth={1.5}
              strokeDasharray="5 3"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {/* Summary row */}
      {selectedRun.summary && (
        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-gray-500">
          {Object.entries(selectedRun.summary).map(([k, v]) =>
            v != null ? (
              <span key={k}>
                <span className="font-medium text-gray-700">{k}:</span>{" "}
                {typeof v === "number" ? v.toFixed(4) : v}
              </span>
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}
