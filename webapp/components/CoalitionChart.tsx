// CoalitionChart — bar chart with three layers:
//   baseline (solid):        pre-shock loyalty μ_i — NOT YET IN SSE PAYLOAD
//                            (the "deltas" event carries Δ, not μ; baseline=null
//                            until the backend widens the stream to include μ)
//   shifted (opaque bars):   post-shock loyalty μ̃_i = μ + Δ — from "deltas" event
//   rebalanced (translucent overlay): optimizer weights w̃_i — from "equilibrium" event
//
// rebalanced is passed as a separate prop so it can transition null → non-null
// without remounting the chart or re-animating the shifted bars. Use conditional
// layer rendering inside the chart, NOT a key change on the outer element.
// TODO: implement with recharts or d3

import type { Party } from "@/lib/types";

interface CoalitionChartProps {
  baseline: Record<string, number> | null;   // μ_i pre-shock (null until backend sends it)
  shifted: Record<string, number> | null;    // μ̃_i post-shock — from "deltas" SSE event
  rebalanced: Record<string, number> | null; // w̃_i optimizer weights — from "equilibrium"
  feasible: boolean;
  party: Party;
  loading: boolean;
}

export default function CoalitionChart(_props: CoalitionChartProps) {
  return null;
}
