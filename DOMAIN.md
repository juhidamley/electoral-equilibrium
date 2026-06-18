# DOMAIN.md — Electoral Equilibrium Framework

**Status:** Generalization contract. Version 1.0, SRP Summer 2026.

This document specifies the requirements for porting the Electoral Equilibrium
pipeline to any new application domain. The codebase is a reusable stochastic
optimization engine: given a domain definition, a configuration file, and a
historical event-response dataset, it produces the rebalanced stakeholder
coalition that maximizes the probability of meeting a domain-specific threshold
under uncertainty. The electoral application is the first instantiation.

Every section that describes a contract has a corresponding enforcement point in
the code. Where a contract is violated, `validate()` raises a `ValueError` with
a message that names the class and field — not a silent failure. This is the
"validation as infrastructure" principle established in `AGENTS.md §5`.

All new verticals must also comply with the reproducibility and artifact-first
contracts in `AGENTS.md`. Those contracts are not repeated here; read them first.

---

## 1. Domain Definition

A **domain** is a four-tuple: `(Blocs, Metric, WinCondition, Objective)`.

### 1.1 Blocs — stakeholder groups

A **bloc** is a mutually exclusive, collectively exhaustive partition of the relevant
population into groups whose behavior toward the agent (voter, shareholder, legislator)
can be independently estimated. Blocs are the atomic decision variables of the
optimizer.

Requirements:

- Each bloc must cover a non-overlapping, identifiable segment of the population.
- The union of all blocs must cover approximately 100% of the population.
- Each bloc must have a historical loyalty/interest metric that can be estimated
  from publicly available data over at least five historical observation periods.
- Bloc identifiers must be lowercase snake_case strings registered in `electoral/core/types.py`.
- The framework supports up to three parallel strata, each with an independent
  set of blocs. Strata are parallel marginal tables — NOT nested cross-tabulations.
  See §3.3 for the stratum schema.

The optimizer's decision variables are the weights over **one primary stratum only**
(in the electoral vertical: race/ethnicity). Secondary strata are fixed-weight
contributions to the effective metric scalar.

### 1.2 Metric — loyalty or interest score

The **metric** is a per-bloc scalar in [0, 1] representing the average degree to
which bloc members support the agent (vote share, approval probability, support
fraction on a board resolution, etc.).

The framework computes an **effective metric scalar** by linearly combining three
parallel strata:

```
μ_eff(w) = λ₁ · Σᵢ wᵢ · μᵢ_primary
         + λ₂ · Σᵣ vᵣ · μᵣ_secondary
         + λ₃ · Σᵍ gᵍ · μᵍ_tertiary
```

where:
- `w` is the vector of optimizer decision weights over the primary stratum blocs (sum to 1)
- `v`, `g` are fixed stratum-share weights from the historical panel (not optimized)
- `λ₁ + λ₂ + λ₃ = 1.0`, calibrated by `kernels/raking.py` from historical data,
  stored in `configs/layer_weights.json`

**The additive independence assumption** (that strata contribute independently) is
an approximation. This is the ecological fallacy (Robinson, 1950). Any paper
reporting results from this framework must acknowledge this explicitly and, where
possible, compare additive vs. raked (IPF-calibrated) outputs.

### 1.3 Win condition — V_eq threshold

The **win condition** is a threshold `V_eq` in (0.5, 0.7) such that the agent wins
if `μ_eff(w) ≥ V_eq`. The threshold is domain-specific, empirically derived, and
never hardcoded.

Derivation procedure (mandatory for all new domains):

1. Collect historical observation periods where the agent won.
2. For each winning period `c`, compute `S_c = μ_eff^(c)(w^(c))` using the realized
   bloc weights and metric values.
3. Set `V_eq = mean(S_c)` across all winning periods.
4. Store the result in `configs/party_config.json` (or its domain-equivalent) with
   `V_eq_low` and `V_eq_high` bounds as a ± sensitivity interval.

`PipelineConfig.validate()` enforces `0.5 < target < 0.7`. Any domain where the
empirically derived `V_eq` falls outside this range requires a `DECISIONS.md` entry
explaining why the constraint should be widened.

### 1.4 Objective — maximize probability of meeting V_eq

The framework maximizes the probability that `μ̃_eff(w) ≥ V_eq` after a shock, not
the expected post-shock metric value. This distinction matters: a campaign (or
analogous agent) at a structural deficit cannot win by minimizing variance — it must
maximize the probability of the tail event where the uncertain coalition shift exceeds
the deficit.

The quasi-convex objective in Sharpe-ratio form:

```
max Φ( (μ̃_eff(w) − V_eq) / sqrt(λ₁² · wᵀ Σ_Δ w) )
```

where `Φ` is the standard normal CDF, `μ̃_eff(w)` is the post-shock effective
metric scalar, and `Σ_Δ` is the primary-stratum covariance of shock deltas. This
is solved via CVXPY DQCP (`problem.solve(qcp=True)`). The quasiconvexity proof is
a required deliverable in any paper using this framework; see §5.4.

---

## 2. Configuration Contract

### 2.1 Required fields in base.json

Every domain instantiation requires a `configs/base.json` (or equivalent) with the
following fields. No field in this list may be hardcoded in kernel logic; all kernel
functions receive the config object as an explicit argument.

```jsonc
{
  // Unique identifier for this run. Used in StageArtifact.run_key.
  "run_key": "string — e.g. 'corp_governance_2026'",

  // Global random seed. ALL stochastic operations derive from this
  // via derive_seed(config.seed, stage_name). See AGENTS.md §3.
  "seed": 42,

  // Domain agent identifier. In the electoral vertical: "democrat" | "republican".
  // For a new domain, define an analogous type in electoral/core/types.py
  // and update PipelineConfig.validate() to enforce the valid value set.
  "party": "string — lowercase, no spaces",

  // V_eq threshold. Must be in (0.5, 0.7); enforced by PipelineConfig.validate().
  // Derived empirically via the procedure in §1.3.
  "target": 0.535,

  // Root path for panel/archive input data.
  "data_path": "data/",

  // Root path for artifact outputs (JSON + Parquet).
  "output_dir": "artifacts/",

  // Controls which pipeline tasks run.
  // "historical": full rebuild from raw data (used during initial development).
  // "continuous": nightly incremental update; skips panel rebuild.
  "pipeline_mode": "historical",

  // Primary stratum bloc IDs. These must match CANONICAL_RACES (or equivalent)
  // in electoral/core/types.py. The optimizer decision variables are over this list.
  "races": ["bloc_a", "bloc_b", "bloc_c", "bloc_d", "bloc_e"],

  // Secondary stratum bloc IDs (fixed-weight; not optimized).
  "religions": ["group_1", "group_2", "..."],

  // Tertiary stratum bloc IDs (fixed-weight; not optimized).
  "genders": ["segment_x", "segment_y", "segment_z"],

  // Bio classifier endpoint (Pi Hailo NPU in the electoral vertical).
  // For domains without a bio classifier, set to null and disable the kernel.
  "pi_bio_server": "http://100.x.x.x:9000",
  "pi_npu_enabled": false
}
```

### 2.2 Companion config files

Each field in `configs/layer_weights.json`, `configs/party_config.json`,
`configs/bin_uncertainty.json`, and `configs/market_contracts.json` is domain-specific.
Copy these files, rename them or keep them as-is, and populate them for your domain
before running any kernel. The pipeline will not accept `null` sigma values in
`bin_uncertainty.json` after Week 3 without a `DECISIONS.md` entry.

### 2.3 What must never be hardcoded

The following must never appear as literals in `electoral/` kernel code. They must
always be read from `PipelineConfig` or the companion configs:

- Bloc ID strings (no `if bloc == "evangelical":` in a kernel)
- The `V_eq` threshold (no `if mu_eff > 0.535:`)
- The agent type / party string
- Layer weight values (`λ₁`, `λ₂`, `λ₃`)
- The number of strata or blocs per stratum
- The random seed

`PipelineConfig` is a frozen dataclass; treat it as the single source of truth for
all of the above throughout the pipeline execution.

---

## 3. Data Requirements

### 3.1 Historical event-response dataset (shocks catalog)

The LLM fine-tuning dataset is assembled from historical events where the agent
experienced a measurable shock to its stakeholder coalition. The minimum viable
catalog for fine-tuning is **25 events** with the following properties:

- Events span at least **three distinct observation periods** (election cycles, fiscal
  years, legislative sessions — whatever constitutes a "cycle" in the domain).
- Each event has a verifiable date and a documented outcome (election result, board
  vote, legislative vote).
- Events are drawn from at least **three of the canonical shock categories** defined
  in `configs/shock_taxonomy.json`.

Each event in `configs/shocks.json` must conform to this schema:

```jsonc
{
  "id": "snake_case_event_identifier",        // unique; used as shock_id throughout
  "date": "YYYY-MM-DD",                       // event date (UTC)
  "category": "one of configs/shock_taxonomy.json categories",
  "description": "one-sentence plain-text description",
  "keywords": ["keyword_1", "keyword_2"],     // used for social/news matching
  "archive_ids": ["dataset_slug"],            // HuggingFace or local archive IDs
  "target_blocs": ["bloc_id_1", "bloc_id_2"],// which blocs are primarily affected
  "shock_window_days": 14,                    // collection window around event date
  "active": true
}
```

All `target_blocs` values must be members of the union of `CANONICAL_RACES`,
`CANONICAL_RELIGIONS`, and `CANONICAL_GENDERS` (or their domain equivalents).

### 3.2 Prediction market contract mapping

`configs/market_contracts.json` maps each `shock_id` to prediction market contract
identifiers (Polymarket, PredictIt, or domain-equivalent). Prediction market prices
are **calibration benchmarks only** — they are never training inputs and never enter
the optimizer or Monte Carlo. Their sole use is post-hoc comparison of the model's
win-probability estimate against market consensus.

Any new domain must identify an equivalent calibration signal (prediction markets,
FiveThirtyEight-style aggregates, betting odds) and populate `market_contracts.json`
before the paper's evaluation section can be written.

### 3.3 Panel data schema — parallel stratum architecture

The historical panel provides per-bloc, per-cycle estimates of (i) the loyalty metric,
(ii) the stratum share, and (iii) turnout (or participation rate). It must be supplied
as three separate flat tables — one per stratum — in the following schema:

```
data/panel/panel_primary.parquet
data/panel/panel_secondary.parquet
data/panel/panel_tertiary.parquet
```

Each Parquet file has these columns (types are strict; `validate()` rejects deviations):

| Column | Type | Constraint | Description |
|---|---|---|---|
| `cycle` | `int` | YYYY ≥ 1990 | Observation period identifier |
| `stratum` | `str` | `"race"` \| `"religion"` \| `"gender"` | Stratum label |
| `bloc` | `str` | member of canonical set | Bloc identifier (snake_case) |
| `vote_share` | `float` | [0.0, 1.0] | Loyalty/interest metric for this bloc × cycle |
| `stratum_share` | `float` | [0.0, 1.0] | Population fraction (rows sum to 1.0 per cycle) |
| `turnout` | `float` | [0.0, 1.0] | Participation rate (can be 1.0 if not applicable) |
| `source` | `str` | non-empty | Data provenance (e.g. `"ARDA"`, `"GSS"`, `"NEP"`) |

**Invariant enforced by `VoterPanelData.validate()`:** within each stratum, the sum
of `stratum_share` across all blocs within a single `cycle` must equal `1.0 ± 1e-6`.
Any panel that violates this invariant fails validation before entering the pipeline.

For a new domain, rename the stratum label column value to match your domain (the
string `"race"` is just a label; what matters is the shape of the table). Update
`CANONICAL_RACES` (and equivalents) in `electoral/core/types.py` to match your blocs.

---

## 4. Fine-Tuning Schema

The LLM fine-tuning schema is **domain-agnostic by design**. The model learns the
mapping from a structured event description plus per-stratum sentiment signals to
per-stratum discrete delta bins. It does not learn the names of political parties,
demographic groups, or specific historical events.

### 4.1 Training record schema (`data/finetune/train.jsonl`)

Each line is a JSON object with this schema:

```jsonc
{
  // Event metadata — not seen by the model as labeled fields,
  // but included for traceability and future filtering.
  "shock_id": "kavanaugh_2018",
  "cycle": 2018,
  "party": "democrat",

  // Input features: free-text event description (the model's prompt).
  "event_description": "Brett Kavanaugh Supreme Court nomination hearings; ...",

  // Input features: per-stratum RoBERTa sentiment elasticity scores.
  // Keys are stratum-level; values are floats in [-1, 1].
  // These are ElasticityScore scalars from electoral/nlp/elasticity.py.
  "sentiment_scores": {
    "primary": {
      "bloc_a": -0.31,
      "bloc_b": 0.12,
      "bloc_c": 0.08,
      "bloc_d": -0.19,
      "bloc_e": 0.04
    },
    "secondary": { "group_1": 0.45, "group_2": -0.22, "...": 0.0 },
    "tertiary":  { "segment_x": 0.18, "segment_y": -0.07, "segment_z": 0.0 }
  },

  // Target outputs: one delta bin per stratum cell.
  // Must be members of DELTA_BINS from electoral/core/types.py.
  // The LLM is trained to produce exactly these tokens via constrained decoding.
  "delta_bins": {
    "primary":   { "bloc_a": "mild_neg", "bloc_b": "slight_pos", "...": "neutral" },
    "secondary": { "group_1": "mod_pos", "group_2": "slight_neg", "...": "neutral" },
    "tertiary":  { "segment_x": "slight_pos", "segment_y": "neutral", "segment_z": "neutral" }
  },

  // Numeric delta values (midpoints of delta_bins, from BIN_MIDPOINTS).
  // Used for regression baselines; not seen by the LLM.
  "deltas": {
    "primary":   { "bloc_a": -0.035, "blob_b": 0.012, "...": 0.0 },
    "secondary": { "group_1": 0.070, "group_2": -0.012, "...": 0.0 },
    "tertiary":  { "segment_x": 0.012, "segment_y": 0.0, "segment_z": 0.0 }
  },

  // Data source tag — controls inclusion in covariance estimation.
  // "llm_unified" | "roberta_news_only" | "roberta_social_only"
  // Posts with inference_method="language_prior" must NOT appear in this file.
  "source": "llm_unified",

  // Dataset split assignment. Set by the train/eval split script.
  "split": "train"  // "train" | "eval"
}
```

### 4.2 The 9-token delta bin vocabulary

All domains use the same nine constrained output tokens. The token vocabulary is
registered in `DELTA_BINS` in `electoral/core/types.py` and must not be altered:

| Token | Numeric range | Midpoint | Conservative σ prior |
|---|---|---|---|
| `strong_neg` | [−0.15, −0.09) | −0.120 | 0.030 |
| `mod_neg` | [−0.09, −0.05) | −0.070 | 0.020 |
| `mild_neg` | [−0.05, −0.02) | −0.035 | 0.015 |
| `slight_neg` | [−0.02, −0.005) | −0.012 | 0.007 |
| `neutral` | [−0.005, +0.005] | 0.000 | 0.002 |
| `slight_pos` | (+0.005, +0.02] | +0.012 | 0.007 |
| `mild_pos` | (+0.02, +0.05] | +0.035 | 0.015 |
| `mod_pos` | (+0.05, +0.09] | +0.070 | 0.020 |
| `strong_pos` | (+0.09, +0.15] | +0.120 | 0.030 |

The conservative σ prior (last column) is used in `configs/bin_uncertainty.json`
for bins with fewer than five historical examples. Empirically calibrated σ values
replace the prior as historical data accumulates.

The token `weak_neg` / `weak_pos` are not valid. Use `slight_neg` / `slight_pos`.
This was standardized in DECISIONS.md §4. Do not deviate.

### 4.3 Validation independence from domain content

The pipeline validation logic is fully domain-agnostic. `ShockResponseData.validate()`
checks that:
- All delta bin values are members of `DELTA_BINS` (the vocabulary is fixed)
- All bloc keys match the blocs registered in `PipelineConfig` (not hardcoded strings)
- The covariance matrix is square and matches the primary stratum bloc count
- `source` is one of `{"llm_unified", "roberta_news_only", "roberta_social_only"}`

No validation logic references specific bloc names ("evangelical"), party names
("democrat"), or domain concepts. A new domain passes through identical validation
code; only the values in `PipelineConfig` change.

---

## 5. Implementation Guide for New Verticals

Work through these steps sequentially. Each step has a gate criterion; do not
proceed until the criterion is met.

### Step 1 — Map stakeholder groups to the stratum schema

Identify three independent, non-nested population dimensions for your domain.
These become the primary, secondary, and tertiary strata.

**Gate criterion:** For each dimension, you can define 3–10 mutually exclusive blocs
such that: (a) each bloc is identifiable in available public data, (b) the bloc's
loyalty/interest metric has been measured in at least five historical periods, and
(c) the stratum shares sum to approximately 1.0 across blocs within each period.

Register the blocs in `electoral/core/types.py` following the pattern of
`CANONICAL_RACES`, `CANONICAL_RELIGIONS`, and `CANONICAL_GENDERS`. Update
`PipelineConfig` defaults and add a `DECISIONS.md` entry documenting why each
bloc boundary was drawn where it is.

**Example — shareholder coalition domain:**

| Stratum | Blocs | Metric |
|---|---|---|
| Primary (investor type) | `activist_fund`, `pension_fund`, `index_fund`, `retail`, `insider` | Fraction of shares voted FOR management resolution |
| Secondary (holding duration) | `long_term`, `medium_term`, `short_term` | Same metric, averaged by duration cohort |
| Tertiary (geography) | `domestic_us`, `domestic_uk`, `international` | Same metric, by domicile |

### Step 2 — Derive V_eq from historical data

Run `kernels/data.py` → `kernels/baseline.py` on your historical panel to execute
`build_constraint_spec()`. The function computes:

```python
S_c = sum(w_i^(c) * mu_i_eff^(c) for i in primary_blocs)  # per winning period
V_eq = mean(S_c for c in winning_periods)
```

**Gate criterion:** `V_eq` falls in (0.5, 0.7) and its standard deviation across
winning periods is < 0.05. High variance indicates the win condition is unstable
and the domain definition may need revision. Write the derived values to
`configs/party_config.json` (or its domain equivalent) and commit the file.

### Step 3 — Develop the keyword lexicon and proxy mapping

The bio classifier (`electoral/nlp/bio_classifier.py`) uses a three-stage inference
pipeline: (1) SetFit model, (2) keyword lexicon fallback, (3) language prior fallback.
For a new domain, the lexicon must be rebuilt from scratch.

Populate `configs/race_lexicon.json`, `configs/religion_lexicon.json`, and
`configs/gender_lexicon.json` with domain-specific keywords mapping to your strata.
Each entry maps a keyword string to a `{bloc_id: weight}` dict summing to 1.0:

```json
{
  "keywords": {
    "pension fund": {"pension_fund": 1.0},
    "calpers":      {"pension_fund": 0.9, "index_fund": 0.1},
    "vanguard":     {"index_fund": 1.0},
    "activist":     {"activist_fund": 1.0}
  }
}
```

For the proxy mapping: if your domain has no natural "bio" text equivalent, substitute
with whatever author-level signal is available (job title, affiliation, account type).
The classifier endpoint (`pi_bio_server`) remains the architectural interface — only
the model weights and lexicon change.

**Gate criterion:** Manual evaluation of 100 randomly sampled author profiles from
your corpus shows ≥ 70% correct stratum assignment by the lexicon+SetFit pipeline.
Document the evaluation method and accuracy in `DECISIONS.md`.

### Step 4 — Configure the CVXPY optimizer objective

The optimizer in `electoral/portfolios/cvx.py` must be instantiated with the
domain's objective function. For the standard max-P(win) objective, no changes
to the optimizer code are needed — only the inputs change (different bloc IDs,
different `Σ_Δ`, different `V_eq`).

If the domain requires a **modified objective** (e.g., minimize capital exposure
subject to a win-probability constraint, rather than maximize win probability), the
new objective must pass the following quasi-convexity verification before the code
is committed:

```python
problem = cp.Problem(cp.Maximize(new_objective_expression))
assert problem.is_dcp(qcp=True), (
    "Objective is not quasi-convex — revise formulation before proceeding. "
    "See DECISIONS.md for the quasiconvexity proof requirement."
)
problem.solve(qcp=True)
```

**Quasi-convexity proof — required paper deliverable:** For any objective of the
Sharpe-ratio form `f(w) = g(w) / h(w)`, the proof proceeds as follows:

1. **Numerator** `g(w) = μ̃_eff(w) − V_eq = λ₁ · wᵀμ_Δ − V_eq + constants`.
   This is affine in `w`, hence both convex and concave.
2. **Denominator** `h(w) = sqrt(λ₁² · wᵀ Σ_Δ w)`.
   `Σ_Δ` is positive semi-definite by construction (Ledoit-Wolf shrinkage guarantees
   strict positive-definiteness). Therefore `wᵀ Σ_Δ w` is a convex quadratic form,
   `sqrt(·)` is a monotone concave transformation of a convex function, and `h(w)`
   is convex. For any feasible `w` (simplex interior), `h(w) > 0`.
3. **Ratio** A ratio of an affine function to a positive convex function is
   quasi-convex (Diamond & Boyd, "Disciplined Quasi-Convex Programming," 2019,
   arXiv:1905.00562, Proposition 1). Therefore `f(w)` is quasi-convex.
4. **Φ composition** The standard normal CDF `Φ` is monotone increasing, so
   maximizing `Φ(f(w))` is equivalent to maximizing `f(w)`. Quasi-convexity is
   preserved under monotone composition.
5. **CVXPY DQCP** `assert problem.is_dcp(qcp=True)` verifies the formulation
   is recognized by CVXPY's quasi-convex program (QCP) checker at runtime.

This proof must appear verbatim (with domain-specific variable substitutions) in the
methodology section of any paper using this framework.

---

## 6. Electoral Vertical — Worked Example

The electoral application instantiates this framework as follows.

### Blocs and strata

Three parallel strata cover the U.S. electorate independently:

| Stratum | Blocs (n) | IDs | Source |
|---|---|---|---|
| Primary — Race/Ethnicity | 5 | `african_american`, `latino`, `asian`, `white`, `other_race` | ARDA, NEP |
| Secondary — Religion | 7 | `evangelical`, `catholic`, `protestant`, `secular`, `jewish`, `muslim`, `other_rel` | GSS, ARDA |
| Tertiary — Gender | 3 | `women`, `men`, `other_gender` | NEP, ANES |

Population coverage: each stratum independently sums to ~100% of the electorate.
No cross-tabulations are required. The additive independence approximation is
acknowledged in the paper as the ecological fallacy and tested by comparing additive
vs. raked (IPF-calibrated) outputs from `kernels/raking.py`.

### V_eq derivation

`V_eq` is derived from National Election Pool (NEP) exit polls for 2000, 2004, 2008,
2012, 2016, 2020 using `kernels/baseline.py → build_constraint_spec()`. Winning
cycles are identified per party and `V_eq = mean(S_c)` computed over those cycles.
Results stored in `configs/party_config.json`:

```json
{
  "democrat": { "V_eq": 0.525, "V_eq_low": 0.520, "V_eq_high": 0.535 },
  "republican": { "V_eq": 0.500, "V_eq_low": 0.490, "V_eq_high": 0.510 }
}
```

Republican `V_eq` is lower because Republicans won (2000, 2016) while losing the
national popular vote — a structural feature of the Electoral College not present in
other democracies. New domains may have an analogous structural asymmetry; document it.

### Layer weights

Calibrated via `kernels/raking.py` from historical NEP and GSS data:

```json
{ "lambda_1": 0.50, "lambda_2": 0.30, "lambda_3": 0.20 }
```

`λ₁ + λ₂ + λ₃ = 1.0`. Race/ethnicity carries the largest weight because it has the
highest historical cross-cycle variance in the electoral context. Other domains may
calibrate different weight distributions.

### LLM — QLoRA Mistral 7B adapter

Base model: `mistralai/Mistral-7B-v0.3`. Fine-tuned via QLoRA (rank 16, α=32) on
25+ historical shock events using 4-bit NF4 quantization. Training runs on CMC HPC
A100 via `scripts/hpc_submit.sh`. The adapter is stored in
`adapters/mistral-7b-electoral/` and referenced by `LLMFineTuneData.adapter_path`.

The adapter teaches the model the relationship between political shock events and
demographic loyalty shifts for the U.S. electorate. A new domain requires a new
adapter trained on domain-specific events. The base model is reused; only the adapter
weights change.

### Optimizer and Monte Carlo

The CVXPY DQCP optimizer (`electoral/portfolios/cvx.py`) solves for the coalition
weight vector `w*` that maximizes P(win) using the quasi-convex objective defined
in §1.4. The quasi-convexity proof (§5.4) appears in the paper's methodology.

The Monte Carlo (`electoral/simulation/montecarlo.py`) propagates the 5×5 race-level
covariance `Σ_Δ` through 10,000 Logistic-Normal ILR draws to produce the 90% CI
on win probability (`SimulationData.win_probability_low` to `win_probability_high`).
Dirichlet sampling is explicitly rejected (see `DECISIONS.md §Monte Carlo`).

### Calibration

Post-shock model win-probability estimates are compared against Polymarket and
PredictIt market prices using `markets/aggregator.py`. Market prices are displayed
in the web app as "Market consensus: X%" alongside the model output. They are never
fed back into the model as training signal.

---

## Appendix A: New Vertical Checklist

Use this checklist when starting a new domain implementation. Every item maps to
a section of this document and a specific code location.

```
DOMAIN DEFINITION
- [ ] Three parallel strata identified with 3-10 blocs each (§1.1)
- [ ] Bloc IDs registered in electoral/core/types.py as CANONICAL_* lists (§1.1)
- [ ] Loyalty/interest metric defined and bounded to [0, 1] (§1.2)
- [ ] Layer weights λ₁, λ₂, λ₃ calibrated and written to configs/layer_weights.json (§1.2)
- [ ] V_eq derived from historical data and written to configs/party_config.json (§1.3)
- [ ] V_eq confirmed in (0.5, 0.7); if not, DECISIONS.md entry written (§1.3)

CONFIGURATION
- [ ] configs/base.json created for the new domain; no field hardcoded in kernels (§2)
- [ ] configs/shocks.json populated with ≥25 historical events (§3.1)
- [ ] configs/market_contracts.json populated with calibration signal contracts (§3.2)
- [ ] configs/bin_uncertainty.json populated (or conservative σ priors accepted) (§4.2)

DATA
- [ ] Three panel Parquet files created in data/panel/ (§3.3)
- [ ] stratum_share sums to 1.0 ± 1e-6 per cycle verified (§3.3)
- [ ] data/finetune/train.jsonl and eval.jsonl assembled (§4.1)
- [ ] No language_prior posts in fine-tuning files (§4.1)

LEXICON AND CLASSIFIER
- [ ] configs/race_lexicon.json adapted for primary stratum blocs (§5.3)
- [ ] configs/religion_lexicon.json adapted for secondary stratum blocs (§5.3)
- [ ] configs/gender_lexicon.json adapted for tertiary stratum blocs (§5.3)
- [ ] Manual evaluation: ≥70% lexicon+SetFit accuracy on 100 samples (§5.3)

OPTIMIZER
- [ ] problem.is_dcp(qcp=True) asserted in cvx.py (§5.4)
- [ ] Quasi-convexity proof written for paper methodology section (§5.4)

REPRODUCIBILITY (AGENTS.md §3)
- [ ] All stochastic operations use derive_seed(config.seed, stage_name) (AGENTS.md §3)
- [ ] just smoke completes without error (AGENTS.md §2)
- [ ] Two smoke runs with same seed produce byte-identical artifacts (AGENTS.md §3)
- [ ] DECISIONS.md updated with any domain-specific architectural choices
```

---

## Appendix B: Artifacts Produced Per Domain Run

Every domain run produces the same artifact sequence, regardless of content.
The artifact schema in `electoral/artifacts.py` is fully domain-agnostic.

| Stage | Artifact class | File |
|---|---|---|
| 1 | `VoterPanelData` | `artifacts/{run_key}/voter_panel.json` |
| 2 | `BaselinePortfolioData` | `artifacts/{run_key}/baseline_portfolio.json` |
| 3 | `SentimentData` | `artifacts/{run_key}/sentiment_data.json` |
| 4 | `LLMFineTuneData` | `artifacts/{run_key}/llm_finetune.json` |
| 5 | `ShockResponseData` | `artifacts/{run_key}/shock_response.json` |
| 6 | `EquilibriumData` | `artifacts/{run_key}/optimization.json` |
| 7 | `SimulationData` | `artifacts/{run_key}/simulation.json` |
| 8 | `MetricsTablesData` | `artifacts/{run_key}/metrics_tables.json` |

All artifacts are written by `electoral/core/io.py → write_artifact()`. All are
wrapped in a `StageArtifact` envelope with `stage`, `run_key`, `metadata`, and
`data` fields. The envelope schema is version-stable; downstream consumers
(webapp, audit log, paper export scripts) read from these files, not from
in-memory objects. This is the artifact-first contract.
