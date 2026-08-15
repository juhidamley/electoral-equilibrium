// Runtime validation schemas for all FastAPI response shapes.
//
// WHAT IS ZOD / WHY THIS EXISTS:
// types.ts gives COMPILE-TIME types, but those disappear at runtime — they can't
// check that data actually arriving over the network is well-formed. If the
// backend has a bug and sends win_probability = 5, or omits a bloc, TypeScript
// won't notice; the bad value just flows into the charts and corrupts the UI.
// Zod fixes that: each `z.object({...})` below is a SCHEMA you can run against
// real data at runtime. `Schema.parse(json)` returns the value if it's valid or
// THROWS a detailed ZodError if not. We run these at the network boundary (see
// lib/api.ts), so malformed responses are rejected loudly at the door instead of
// silently breaking something three components deep. It's the TypeScript analog
// of the Python validate() methods in artifacts.py.
//
// These mirror webapp/lib/types.ts and electoral/artifacts.py — all three files
// must be updated together whenever a field is added, removed, or retyped.
// See the Week 8 contract-sync task in docs/tasks.tex.
//
// The constraints here deliberately duplicate the Python validate() invariants
// (covariance is 5×5, win_probability ordering) so a malformed backend response
// throws a ZodError at the API boundary rather than propagating NaN or undefined
// into the UI.

import { z } from "zod";

export const PartySchema = z.enum(["democrat", "republican"]);

export const DeltaBinSchema = z.enum([
  "strong_neg",
  "mod_neg",
  "mild_neg",
  "slight_neg",
  "neutral",
  "slight_pos",
  "mild_pos",
  "mod_pos",
  "strong_pos",
]);

const RACE_BLOCS = [
  "african_american",
  "asian",
  "latino",
  "other_race",
  "white",
] as const;

const RELIGION_BLOCS = [
  "evangelical",
  "catholic",
  "protestant",
  "secular",
  "jewish",
  "muslim",
  "other_rel",
] as const;

const GENDER_BLOCS = ["women", "men", "other_gender"] as const;

// Build a record schema requiring exactly the given keys — any missing or extra
// key will cause a parse error, matching Python's assert_required_keys checks.
const raceRecord = <T extends z.ZodTypeAny>(v: T) =>
  z
    .object(
      Object.fromEntries(RACE_BLOCS.map((b) => [b, v])) as Record<
        (typeof RACE_BLOCS)[number],
        T
      >,
    )
    .strict();

const religionRecord = <T extends z.ZodTypeAny>(v: T) =>
  z
    .object(
      Object.fromEntries(RELIGION_BLOCS.map((b) => [b, v])) as Record<
        (typeof RELIGION_BLOCS)[number],
        T
      >,
    )
    .strict();

const genderRecord = <T extends z.ZodTypeAny>(v: T) =>
  z
    .object(
      Object.fromEntries(GENDER_BLOCS.map((b) => [b, v])) as Record<
        (typeof GENDER_BLOCS)[number],
        T
      >,
    )
    .strict();

export const ShockResponseDataSchema = z.object({
  shock: z.string(),
  cycle: z.number().int(),
  party: PartySchema,
  delta_bins_race: raceRecord(DeltaBinSchema),
  delta_bins_religion: religionRecord(DeltaBinSchema),
  delta_bins_gender: genderRecord(DeltaBinSchema),
  deltas_race: raceRecord(z.number()),
  deltas_religion: religionRecord(z.number()),
  deltas_gender: genderRecord(z.number()),
  delta_eff: z.number(),
  covariance: z
    .array(z.array(z.number()))
    .length(5)
    .refine((rows) => rows.every((r) => r.length === 5), {
      message: "covariance must be 5×5",
    }), // enforces square race-only matrix
  source: z.string(),
});

export const EquilibriumDataSchema = z.object({
  method: z.string(),
  party: PartySchema,
  shock: z.string().nullable(),
  weights: raceRecord(z.number()),
  mu_shifted: raceRecord(z.number()),
  mu_eff_shifted: z.number(),
  feasible: z.boolean(),
  target_met: z.boolean(),
  target: z.number(),
});

export const SimulationDataSchema = z
  .object({
    n_simulations: z.number().int(),
    seed: z.number().int(),
    win_probability: z.number().min(0).max(1),
    win_probability_low: z.number().min(0).max(1),
    win_probability_high: z.number().min(0).max(1),
    percentiles: raceRecord(z.array(z.number()).length(5)), // [p5, p25, p50, p75, p95]
  })
  .refine((d) => d.win_probability_low <= d.win_probability_high, {
    message: "win_probability_low must be <= win_probability_high",
  })
  .refine(
    (d) =>
      d.win_probability_low <= d.win_probability &&
      d.win_probability <= d.win_probability_high,
    {
      message:
        "win_probability must lie within [win_probability_low, win_probability_high]",
    },
  );

// Composite — requires endpoint widening; see EstimateResponse note in types.ts.
export const EstimateResponseSchema = z.object({
  shock: ShockResponseDataSchema,
  equilibrium: EquilibriumDataSchema,
  simulation: SimulationDataSchema,
});

// Inferred types — prefer importing concrete types from types.ts for
// field-level use; use these only when you need the inferred Zod output shape.
export type EstimateResponse = z.infer<typeof EstimateResponseSchema>;
