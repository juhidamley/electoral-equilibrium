// ============================================================================
// WHAT THIS FILE IS (frontend data contracts)
// ============================================================================
// This is the TypeScript MIRROR of electoral/artifacts.py. The Python backend
// sends JSON; the browser receives it. TypeScript can't read Python, so we
// re-declare the same shapes here as `interface`s/`type`s. That gives us
// compile-time safety: if you try to read `simulation.win_prob` when the field
// is actually `win_probability`, the TypeScript compiler catches it before the
// app ever runs.
//
// TWO LAYERS, DON'T CONFUSE THEM:
//   • types.ts (this file) = COMPILE-TIME shapes. They vanish at runtime — they
//     only help the editor and the type-checker. They do NOT verify that data
//     actually arriving from the network matches.
//   • schemas.ts (Zod)     = RUNTIME validation. It actually inspects incoming
//     JSON and throws if a field is missing or out of range. Use that at the
//     network boundary (see lib/api.ts).
//
// Mirror of electoral/artifacts.py frozen dataclasses.
// These types MUST be kept in sync with the Python definitions manually —
// there is no codegen step. Any field added to artifacts.py must be added here.
//
// EstimateResponse assumes the /estimate endpoint returns all three artifacts
// bundled. The current endpoint (electoral/llm/inference.py) returns only
// ShockResponseData. To use EstimateResponse the endpoint must be widened to
// also run the optimizer and Monte Carlo and return a composite payload, OR
// the frontend must call three separate endpoints sequentially. This widening
// is tracked as a separate task.

export type Party = "democrat" | "republican";

export type DeltaBin =
  | "strong_neg"
  | "mod_neg"
  | "mild_neg"
  | "slight_neg"
  | "neutral"
  | "slight_pos"
  | "mild_pos"
  | "mod_pos"
  | "strong_pos";

export type RaceBloc =
  | "african_american"
  | "asian"
  | "latino"
  | "other_race"
  | "white";

export type ReligionBloc =
  | "evangelical"
  | "catholic"
  | "protestant"
  | "secular"
  | "jewish"
  | "muslim"
  | "other_rel";

export type GenderBloc = "women" | "men" | "other_gender";

export interface ShockResponseData {
  shock: string;
  cycle: number;
  party: Party;
  delta_bins_race: Record<RaceBloc, DeltaBin>;
  delta_bins_religion: Record<ReligionBloc, DeltaBin>;
  delta_bins_gender: Record<GenderBloc, DeltaBin>;
  deltas_race: Record<RaceBloc, number>;
  deltas_religion: Record<ReligionBloc, number>;
  deltas_gender: Record<GenderBloc, number>;
  delta_eff: number;
  covariance: number[][]; // 5×5 race-only covariance matrix
  source: string;
}

export interface EquilibriumData {
  method: string;
  party: Party;
  shock: string | null;
  weights: Record<RaceBloc, number>;
  mu_shifted: Record<RaceBloc, number>;   // per-bloc post-shock loyalty μ̃_i (for bar rendering)
  mu_eff_shifted: number;                 // λ-weighted scalar across all three strata (for gap display)
  feasible: boolean;
  target_met: boolean;
  target: number;
}

export interface SimulationData {
  n_simulations: number;
  seed: number;
  win_probability: number;
  win_probability_low: number; // bootstrap CI lower bound (5th percentile)
  win_probability_high: number; // bootstrap CI upper bound (95th percentile)
  percentiles: Record<RaceBloc, number[]>; // [p5, p25, p50, p75, p95]
}

// Composite response type — requires endpoint widening (see file-top note).
export interface EstimateResponse {
  shock: ShockResponseData;
  equilibrium: EquilibriumData;
  simulation: SimulationData;
}

// ── Dashboard-only types (no FastAPI counterpart) ─────────────────────────────

export interface AuditEntry {
  id: string;
  timestamp: string; // ISO 8601
  event_text: string;
  party: Party;
  intensity: number;
  win_prob: number;
  feasible: boolean;
  llm_ms: number;
  optimizer_ms: number;
  montecarlo_ms: number;
}

export interface CoverageCell {
  bloc_id: RaceBloc | ReligionBloc | GenderBloc;
  cycle: number;
  quality: "real" | "imputed" | "missing";
  vote_share: number | null;
  sources: string[];
}

export interface TrainingRun {
  run_id: string;
  epochs: number;
  train_loss: number;
  val_loss: number | null; // null when in-loop eval is disabled (eval_mae=inf case)
  mae: number | null;
}
