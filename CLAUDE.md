# Electoral Equilibrium — Claude Code Context

7-week SRP project. Supervised by Prof. Gaston Espinosa at Claremont McKenna College.
Full devplan in `docs/devplan.pdf`. Canonical decisions in `DECISIONS.md`.

---

## What this project is

A stochastic optimization framework that estimates how voter coalitions must rebalance
after political shocks. User types any hypothetical event → system outputs the new optimal
coalition weights and a 90% CI on win probability.

Three-stage pipeline:
1. **LLM** (Mistral 7B QLoRA): shock text → delta bins per stratum
2. **Optimizer** (CVXPY DQCP): delta bins → optimal race coalition weights + win probability
3. **NLP** (RoBERTa + SetFit): social media / news → training data for Stage 1

---

## Compute stack — which machine does what

| Machine | Primary role | Notes |
|---|---|---|
| M5 MacBook Pro (48GB) | Interactive dev, cleaning model, Claude Code, Bluesky collection | Main dev machine |
| Intel MacBook (2019) | News scraper 24/7, nightly pipeline | Always on, launchd 2am |
| Raspberry Pi 5 + Hailo NPU | SetFit bio classifier | Accessible via Tailscale at $PI_TAILSCALE_IP:9000 |
| CMC HPC (A100) | QLoRA fine-tuning, SLURM scoring | Data in/out via rsync |

**Syncthing** ties M5, Intel Mac, and Pi together. Never install on HPC.
**Tailscale** for Pi connectivity — mDNS blocked on CMC enterprise WiFi.

---

## Repo structure

```
electoral-equilibrium/
├── CLAUDE.md                    # this file
├── DECISIONS.md                 # architectural decisions log
├── Justfile                     # task runner
├── configs/
│   ├── base.json                # PipelineConfig — global seed, paths, party, pi IP
│   ├── party_config.json        # V_eq thresholds per party (~0.52-0.53 Dem, ~0.49-0.51 Rep)
│   ├── layer_weights.json       # lambda_1/2/3 + raked outputs from build_constraint_spec
│   ├── shock_taxonomy.json      # canonical shock categories + keywords
│   ├── mmd_config.json          # MMD lambda (median heuristic, 1/2σ²)
│   ├── bin_uncertainty.json     # per-bin empirical σ_b for Logistic-Normal MC
│   ├── language_priors.json     # Pew-based priors for language fallback
│   ├── race_lexicon.json        # bio keyword → race weights
│   ├── religion_lexicon.json    # bio keyword → religion weights
│   ├── gender_lexicon.json      # bio keyword → gender signal
│   └── market_contracts.json   # shock_id → Polymarket/PredictIt contract IDs
├── electoral/
│   ├── artifacts.py             # ALL frozen dataclasses (VoterPanelData, etc.)
│   ├── config.py                # PipelineConfig + derive_seed
│   ├── stages.py                # one stub function per pipeline stage
│   ├── pipeline.py              # Prefect @flow wiring all stages
│   ├── core/
│   │   ├── schema.py            # JSON schema validators
│   │   ├── types.py             # type aliases: RaceId, ReligionId, GenderId, BlocWeight
│   │   ├── io.py                # read_artifact / write_artifact (Parquet + JSON)
│   │   └── rng.py               # make_rng(seed), derive_seed(base, stage_name)
│   ├── data/
│   │   ├── loaders.py           # load panel_race/religion/gender parquets
│   │   ├── cleaning.py          # normalise, dedup, validate panel rows
│   │   └── panel.py             # build_constraint_spec → ConstraintSpec
│   ├── kernels/
│   │   ├── data.py              # panel ingestion kernel
│   │   ├── baseline.py          # moment estimation + GP classifier
│   │   ├── raking.py            # iterative proportional fitting (optional calibration)
│   │   └── sentiment.py         # orchestrates bio + scoring pipelines
│   ├── models/
│   │   └── ml_baseline.py       # GP classifier, XGBoost baseline, LOCO CV
│   ├── portfolios/
│   │   ├── weights.py           # equal-weight + value-weight baselines
│   │   ├── constraints.py       # ConstraintSpec dataclass
│   │   └── cvx.py               # CVXPY max-P(win) optimizer (DQCP)
│   ├── simulation/
│   │   └── montecarlo.py        # Logistic-Normal ILR Monte Carlo
│   ├── nlp/
│   │   ├── archive.py           # HistoricalArchiveLoader
│   │   ├── news_loader.py       # 3DLNews2 + Webhose loaders
│   │   ├── social.py            # platform-agnostic collector interface
│   │   ├── scraper.py           # live news scraper (Windows)
│   │   ├── bio_classifier.py    # SetFit + keyword lexicon (3-stage inference)
│   │   ├── scorer.py            # RoBERTa with bloc-weighted aggregation
│   │   └── elasticity.py        # regression + fine-tuning dataset assembly
│   ├── markets/
│   │   ├── collector.py         # Polymarket/PredictIt price collector
│   │   └── aggregator.py        # volume-weighted multi-market aggregator
│   └── optimization/
│       └── cvx.py               # re-exported from portfolios/cvx.py
├── collectors/                  # Intel Mac always-on collectors
│   ├── bluesky.py               # AT Protocol firehose → rawdata/social/bluesky/
│   ├── apify.py                 # X scraper via Apify API → rawdata/social/apify/
│   └── reddit.py                # Reddit API → rawdata/social/reddit/
├── scripts/
│   ├── sample_archives.py       # stratified sampling (HPC)
│   ├── clean_with_llm.py        # LLM cleaning with local open-weight model (M5)
│   ├── score_array.sh           # SLURM array job for RoBERTa scoring
│   ├── generate_synthetic.py    # Gemini synthetic data + MMD/PCA/PCD diagnostics
│   └── pi_bio_server.py         # FastAPI server on Pi (SetFit endpoint)
├── data/
│   ├── panel/
│   │   ├── panel_race.parquet
│   │   ├── panel_religion.parquet
│   │   └── panel_gender.parquet
│   ├── archives/
│   │   ├── README.md
│   │   ├── discord/
│   │   ├── reddit/
│   │   └── news/
│   ├── finetune/
│   │   ├── new_events.jsonl
│   │   ├── synthetic.jsonl
│   │   └── synthetic_diagnostics.json
│   └── embeddings/              # RoBERTa embedding cache (Parquet)
├── rawdata/
│   └── social/{platform}/{shock_id}/{machine_id}_posts.jsonl
└── tests/
    ├── fixtures/toy_panel.csv
    ├── test_artifact_roundtrip.py
    ├── test_data_cleaning.py
    ├── test_baseline.py
    ├── test_rng.py
    └── test_mc_convergence.py
```

---

## Demographic architecture — THREE PARALLEL STRATA

No nested cross-tabulations. Each stratum independently covers ~100% of the electorate.

**Stratum 1 — Race/Ethnicity** (optimizer decision variables w_i, sum to 1):
- african_american (~12%), latino (~11%), asian (~5%), white (~62%), other_race (~10%)

**Stratum 2 — Religion** (fixed weights v_R from voter panel, sum to 1):
- evangelical (~24%), catholic (~21%), protestant (~13%), secular (~26%), jewish (~2%), muslim (~1%), other_rel (~13%)

**Stratum 3 — Gender** (fixed weights g_G from voter panel, sum to 1):
- women (~52%), men (~47%), other_gender (~1%)

**Effective loyalty scalar:**
```
mu_eff(w) = lambda_1 * sum(w_i * mu_race_i)
          + lambda_2 * sum(v_R * mu_rel_R)
          + lambda_3 * sum(g_G * mu_gen_G)
```
λ₁ + λ₂ + λ₃ = 1, calibrated from historical election data in build_constraint_spec.
Stored in configs/layer_weights.json as {"lambda_1": x, "lambda_2": x, "lambda_3": x, "raked": {...}}.

**Win condition:** mu_eff(w) >= V_eq
V_eq derived empirically from voter panel: ~0.52-0.53 for Democrats, ~0.49-0.51 for Republicans.

**Additive independence is an approximation** — acknowledge ecological fallacy in paper.
Raking (raking.py) runs as optional calibration; paper compares additive vs raked outputs.

---

## Artifact schemas (electoral/artifacts.py)

All frozen dataclasses. Every stage produces one; every stage reads one.

```python
@dataclass(frozen=True)
class VoterPanelData:
    cycles: list[int]
    races: list[str]       # 5 values
    religions: list[str]   # 7 values
    genders: list[str]     # 3 values
    n_rows_race: int
    n_rows_religion: int
    n_rows_gender: int
    layer_weights: dict[str, float]   # lambda_1, lambda_2, lambda_3
    source: str | None

@dataclass(frozen=True)
class BaselinePortfolioData:
    method: str
    party: str                        # "democrat" or "republican"
    weights: dict[str, float]         # race_id → weight, 5 keys, sums to 1.0
    mu_race: dict[str, float]         # race_id → vote_share
    mu_religion: dict[str, float]     # religion_id → vote_share
    mu_gender: dict[str, float]       # gender_id → vote_share
    mu_eff: float                     # scalar effective loyalty
    layer_weights: dict[str, float]
    target: float                     # V_eq

@dataclass(frozen=True)
class ShockResponseData:
    shock: str
    cycle: int
    party: str
    delta_bins_race: dict[str, str]      # race_id → 9-token bin
    delta_bins_religion: dict[str, str]  # religion_id → 9-token bin
    delta_bins_gender: dict[str, str]    # gender_id → 9-token bin
    deltas_race: dict[str, float]
    deltas_religion: dict[str, float]
    deltas_gender: dict[str, float]
    delta_eff: float                     # scalar effective delta
    covariance: list[list[float]]        # 5x5 race-level
    source: str   # "llm_unified" | "roberta_news_only" | "roberta_social_only"

@dataclass(frozen=True)
class EquilibriumData:
    method: str
    party: str
    shock: str | None
    weights: dict[str, float]        # race_id → weight, 5 keys, sums to 1.0
    mu_eff_shifted: float            # post-shock effective loyalty scalar
    feasible: bool
    target_met: bool
    target: float

@dataclass(frozen=True)
class SimulationData:
    n_simulations: int
    seed: int
    win_probability: float           # point estimate
    win_probability_low: float       # 5th percentile
    win_probability_high: float      # 95th percentile
    percentiles: dict[str, list[float]]
```

---

## Optimizer — max P(win) via DQCP (portfolios/cvx.py)

```python
# Objective (quasi-convex, Sharpe-ratio form):
# max Φ( (mu_eff_tilde(w) - V_eq) / sqrt(lambda_1² * w^T Σ_Δ w) )
#
# CRITICAL: declare DQCP explicitly
problem = cp.Problem(cp.Maximize(cp.ratio(numerator, denominator)))
assert problem.is_dcp(qcp=True), "Objective is not quasi-convex — check formulation"
problem.solve(qcp=True)
```

Never use min-variance (wrong for deficit scenarios). The quasiconvexity proof is a paper deliverable.

---

## Monte Carlo — Logistic-Normal ILR (simulation/montecarlo.py)

**NOT Dirichlet** (forces negative off-diagonal covariances — cannot model wave elections).
**NOT delta method** (floor w_i >= 0.01 distorts Aitchison geometry non-linearly).

Use ILR (isometric log-ratio) with Helmert contrast matrix:
1. Map w* to ILR coords: z* = V^T log(w*)  where V is Helmert matrix
2. Propagate Σ_Δ to ILR space via Jacobian: Σ_ILR = J Σ_Δ J^T
3. Draw y^(n) ~ N(z*, Σ_ILR) in R^(K-1)
4. Back-transform: w^(n) = softmax(V y^(n))
5. Compute win flags and 5th/95th percentile CI bounds

Zero-weight blocs → report as infeasible_bloc, never floor to 0.01.

---

## Collectors

### Intel Mac — news scraper (always-on, launchd 2am)

The Intel Mac runs the news scraper continuously, previously the Windows laptop's job.

**News scraper** (electoral/nlp/scraper.py):
- Targets: Christianity Today, CBN (Evangelical), Univision (Latino),
  NYT, WaPo (secular), Fox (conservative)
- Output: rawdata/news/{outlet}/{YYYY-MM-DD}.jsonl
- Run as launchd daemon, nightly

### M5 MacBook Pro — Bluesky collection + dev

**Bluesky** (collectors/bluesky.py):
- **Primary source: pre-existing HuggingFace archival datasets** (historical coverage)
  Download at start of project; do not re-scrape what already exists.
  Known datasets to check: search HuggingFace for "bluesky posts" filtered by size/date.
  Output after download: rawdata/social/bluesky/archive/{dataset_id}.jsonl
- **Secondary: AT Protocol firehose** (live collection going forward)
  Via atproto Python library. Filter: English, political keywords from configs/shock_taxonomy.json
  Output: rawdata/social/bluesky/live/{YYYY-MM-DD}.jsonl
- Bio inference: POST to http://$PI_TAILSCALE_IP:9000/classify

**Apify/X** (collectors/apify.py):
- Apify X scraper, free tier (500 results/run)
- Run per shock event, not continuously
- Output: rawdata/social/apify/{shock_id}/{YYYY-MM-DD}.jsonl

**Reddit** (collectors/reddit.py):
- Reddit API OAuth (100 req/min free tier)
- Subreddits: r/Catholicism, r/Christianity, r/exchristian, r/Conservative,
  r/progressive, r/BlackPeopleTwitter, r/LatinoPeopleTwitter, r/Jewish, r/islam
- Output: rawdata/social/reddit/{subreddit}/{YYYY-MM-DD}.jsonl
- inference_method: "subreddit_proxy" (exclude from Σ_Δ covariance estimation)

All posts written as JSONL with envelope:
```json
{
  "schema_version": "1.0",
  "created_at": "ISO8601",
  "stage": "collect",
  "seed": 42,
  "payload": { "text": "...", "created_at": "...", "lang": "...", "platform": "..." }
}
```

---

## Deterministic seed contract (electoral/core/rng.py)

```python
GLOBAL_SEED = config.seed  # from configs/base.json
make_rng(seed) → np.random.Generator  # seeded generator
derive_seed(base_seed, stage_name) → int  # deterministic sub-seed
```

Never call np.random directly. Never call random.seed() globally.
Every stochastic operation takes a seeded generator as a parameter.

---

## LLM output schema — 9-token bins

Stratum delta bins: strong_neg, mod_neg, mild_neg, slight_neg, neutral,
                    slight_pos, mild_pos, mod_pos, strong_pos

Constrained decoding via outlines library. MUST assert problem.is_dcp(qcp=True) for optimizer.

---

## Cleaning model

Use local open-weight model — NEVER Gemini API (breaks reproducibility).
Recommended: Qwen2.5-7B-Instruct or Mistral-7B-Instruct-v0.3 via mlx_lm on M5.
Record HuggingFace revision hash in DECISIONS.md.

---

## Language fallback rule

Posts assigned via language fallback (inference_method: "language_prior"):
- Exclude from BOTH mean μ and covariance Σ_Δ estimation
- Use as held-out validation set only
- Bifurcating mean vs covariance datasets violates optimizer assumptions

---

## Prediction markets

$\Delta\pi$ ≠ $\Delta\mu$ — probability ≠ vote-share margin.
Prediction markets are CALIBRATION BENCHMARKS ONLY, never training inputs.
Post-hoc: compare model win-probability against market price after each shock resolves.
Display in app as "Market consensus: X%" alongside model output.

---

## Task list conventions

**`/done` is a visual marker only — it does NOT mean a task is complete.**
A checkbox rendered by `/done` simply formats the line as a checked item for readability.
Do not infer that any work has been finished, merged, tested, or verified based on `/done` appearing next to a task.
Always verify actual completion by reading code, running tests, or asking the user explicitly.

---

## Key invariants

- Voter panel: sum(stratum_share) = 1.0 ± 1e-6 within each cycle per table
- Coalition weights: sum(w_i) = 1.0 ± 1e-9 (5 race blocs)
- All bloc ID strings: lowercase snake_case (e.g. african_american, not AfricanAmerican)
- party field: "democrat" or "republican" (lowercase)
- All cycles: int YYYY (e.g. 2020, not "2020")
