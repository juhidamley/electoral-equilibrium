"use client";

// ============================================================================
// EmpiricalSupportPanel — renders the real-reaction grounding of a prediction.
// ============================================================================
// Two things, both purely informational (base prediction is unaffected):
//   1. EMPIRICAL SUPPORT — for events with real archive coverage, a per-bloc
//      table comparing the base predicted delta against the real scored social
//      sentiment, with an agreement flag (aligned / diverged / no_data).
//   2. REFINEMENT — for SENTIMENT-ALIGNED events only, the optional refined
//      prediction (base bin → refined bin per bloc). Divergent/mobilizing events
//      show a "base only" note and no refined prediction; uncovered events render
//      nothing (empiricalSupport is null).
// The framing notes from the backend are shown verbatim so the honesty travels
// with the data: empirical support is an OUT-OF-SAMPLE COMPARISON (not accuracy),
// refinement is a SOFT inference-time correction (not a retrained driver).

import React from "react";

import type { EmpiricalSupport, Refinement } from "@/lib/types";
import { BLOC_LABEL } from "@/lib/blocs";

const AGREE_STYLE: Record<string, string> = {
  aligned: "bg-green-50 text-green-700 border-green-200",
  diverged: "bg-amber-50 text-amber-700 border-amber-200",
  no_data: "bg-gray-50 text-gray-400 border-gray-200",
};

const REGIME_BADGE: Record<string, { label: string; cls: string }> = {
  aligned: { label: "Sentiment-aligned", cls: "bg-blue-50 text-blue-700 border-blue-200" },
  divergent: { label: "Valence-divergent", cls: "bg-purple-50 text-purple-700 border-purple-200" },
  uncovered: { label: "Uncovered", cls: "bg-gray-50 text-gray-500 border-gray-200" },
};

function pct(x: number | null): string {
  return x == null ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}pp`;
}
function sentiment(x: number | null): string {
  return x == null ? "—" : x.toFixed(3);
}

export default function EmpiricalSupportPanel({
  empiricalSupport,
  refinement,
}: {
  empiricalSupport: EmpiricalSupport | null;
  refinement: Refinement | null;
}) {
  // Uncovered / no coverage → render nothing (base prediction stands alone).
  if (!empiricalSupport) return null;

  const es = empiricalSupport;
  const regime = REGIME_BADGE[es.regime] ?? REGIME_BADGE.uncovered;
  const refinedBins = refinement?.applied ? refinement.refined_prediction?.bins ?? {} : null;
  // Only annotate covered blocs (those with a real sentiment reading).
  const rows = es.blocs.filter((b) => b.real_social_sentiment != null);

  return (
    <div className="rounded-md border border-gray-200 bg-white p-4">
      <div className="mb-2 flex items-center gap-2">
        <h3 className="text-sm font-semibold text-gray-800">
          Empirical grounding — real reactions
        </h3>
        <span className={`rounded border px-1.5 py-0.5 text-[11px] font-medium ${regime.cls}`}>
          {regime.label}
        </span>
      </div>

      <p className="mb-3 text-xs text-gray-500">
        Base prediction vs real archived social sentiment for{" "}
        <span className="font-mono">{es.shock_id}</span> —{" "}
        <span className="text-green-700">{es.n_aligned} aligned</span> ·{" "}
        <span className="text-amber-700">{es.n_diverged} diverged</span> ·{" "}
        <span className="text-gray-400">{es.n_no_data} no-data</span>
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-200 text-left text-gray-500">
              <th className="py-1 pr-3 font-medium">Bloc</th>
              <th className="py-1 pr-3 font-medium">Predicted</th>
              {refinedBins && <th className="py-1 pr-3 font-medium">Refined</th>}
              <th className="py-1 pr-3 font-medium">Real sentiment</th>
              <th className="py-1 pr-3 font-medium">n posts</th>
              <th className="py-1 font-medium">Agreement</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((b) => (
              <tr key={b.bloc} className="border-b border-gray-100">
                <td className="py-1 pr-3 text-gray-700">{BLOC_LABEL[b.bloc] ?? b.bloc}</td>
                <td className="py-1 pr-3">
                  <span className="text-gray-800">{b.predicted_bin ?? "—"}</span>{" "}
                  <span className="text-gray-400">{pct(b.predicted_delta)}</span>
                </td>
                {refinedBins && (
                  <td className="py-1 pr-3 text-blue-700">
                    {refinedBins[b.bloc] && refinedBins[b.bloc] !== b.predicted_bin ? (
                      <span className="font-medium">{refinedBins[b.bloc]}</span>
                    ) : (
                      <span className="text-gray-300">{refinedBins[b.bloc] ?? "—"}</span>
                    )}
                  </td>
                )}
                <td className="py-1 pr-3 text-gray-600">{sentiment(b.real_social_sentiment)}</td>
                <td className="py-1 pr-3 text-gray-400">{b.n_posts}</td>
                <td className="py-1">
                  <span
                    className={`rounded border px-1.5 py-0.5 text-[10px] ${
                      AGREE_STYLE[b.agreement] ?? AGREE_STYLE.no_data
                    }`}
                  >
                    {b.agreement}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Refinement status — refined table above only when applied; here the note. */}
      {refinement && (
        <div className="mt-3 rounded border border-gray-100 bg-gray-50 p-2 text-xs">
          {refinement.applied ? (
            <p className="text-blue-700">
              <span className="font-medium">Refined prediction shown</span> (blue column):
              real sentiment injected at inference on this aligned event.
            </p>
          ) : (
            <p className="text-gray-600">
              <span className="font-medium">Base prediction only.</span>{" "}
              {refinement.reason}
            </p>
          )}
          <p className="mt-1 leading-snug text-gray-400">{refinement.note}</p>
        </div>
      )}

      <p className="mt-2 text-[11px] leading-snug text-gray-400">{es.note}</p>
    </div>
  );
}
