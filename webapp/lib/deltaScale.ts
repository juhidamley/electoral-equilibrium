// Shared delta-bin axis/formatting constants — single source of truth for
// every chart that plots a Δ loyalty value (deltas_race/religion/gender).
//
// MUST STAY IN SYNC WITH THE PYTHON DECODE TABLE:
//   electoral/core/types.py::BIN_MIDPOINTS / DELTA_BINS (bin ceiling ±0.03)
//   electoral/llm/inference.py's post-intensity clip (hard ceiling ±0.0375)
// There is no automated link between this file and those — this is the same
// duplication risk already flagged on the Python side (BIN_MIDPOINTS is
// separately copied in scripts/generate_synthetic.py and
// electoral/nlp/elasticity.py; see DECISIONS.md "Step 2.1"). If the decode
// table is ever rescaled again, this file must be updated by hand at the
// same time — it will NOT happen automatically.
//
// DELTA_AXIS_MAX is the CLIP ceiling (0.0375), not the raw bin ceiling
// (0.03): `intensity` can scale a bin's midpoint above 0.03 (up to the
// API's max intensity of 3.0) before the backend clips it, so a real value
// can legitimately land anywhere up to 0.0375. Using the smaller 0.03 as
// the axis domain would let a real, valid value render off-chart.

export const DELTA_AXIS_MAX = 0.0375;
export const DELTA_AXIS_DOMAIN: [number, number] = [-DELTA_AXIS_MAX, DELTA_AXIS_MAX];

// Whole-percentage-point ticks, not forced to the exact domain edge
// (±3.75pp) — round numbers read faster than an odd fraction, and Recharts
// draws ticks fine when they don't touch the domain boundary. Matches the
// old axis's 7-tick density (was 5pp steps over ±15pp; this is ~1pp steps
// over ±3.75pp, the same proportional coverage of the domain).
export const DELTA_AXIS_TICKS = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03];

// One decimal place is required, not cosmetic: at this scale, Math.round()
// on a whole percentage point collapses slight_pos/slight_neg (±0.003, i.e.
// ±0.3pp) to the same "0pp" as neutral (0.000). toFixed(1) keeps every one
// of the 9 bins visually and textually distinct (see webapp/lib/__tests__).
export function formatDeltaPP(v: number): string {
  return `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}pp`;
}
