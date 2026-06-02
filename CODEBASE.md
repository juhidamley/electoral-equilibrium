# Codebase Reference — Electoral Equilibrium

Every source file documented, organized by directory. For each file: what it does,
what it reads/writes, and what stage of the pipeline it belongs to.

**Total source files:** ~130 Python modules + configs + tests + frontend  
**Test status:** 384 passing, 21 skipped (skipped = Week 2+ kernels not yet implemented)  
**Implemented:** Week 1 (data pipeline) + `ml_baseline.py` (Week 2 moment estimation). Remaining Weeks 2–7 kernels are stubs that raise `NotImplementedError`.

---

## Top-Level Files

| File | What it does |
|------|-------------|
| `CLAUDE.md` | Instructions for Claude Code: project overview, compute stack, repo structure, artifact schemas, optimizer formula, Monte Carlo algorithm, key invariants. Not part of the application — it is agent context. |
| `README.md` | User-facing project documentation: what the system does, pipeline overview, setup instructions, compute stack map. |
| `DECISIONS.md` | Log of every architectural decision with rationale. Consult before modifying any module. Every design choice that isn't obvious from the code is explained here. |
| `ESPINOSA.md` | Open questions for supervisor check-ins. Grouped by topic and urgency. |
| `DOMAIN.md` | Generalization contract for porting the framework to non-electoral domains. Defines the interface a new domain must implement. |
| `AGENTS.md` | Contributor conventions: branch naming, code style, PR requirements, validation discipline, artifact-first design. Read this before contributing. |
| `QWEN.md` | Qwen2.5-7B documentation for the local cleaning model on the M5 MacBook. Records recommended inference parameters and revision hash once selected. |
| `Justfile` | Task runner (similar to Makefile). Key targets: `just smoke` (run full pipeline with smoke config), `just test` (pytest), `just clean` (rm artifacts), `just score` / `just train` (HPC SLURM submission). |
| `pyproject.toml` | Python project metadata and dependency declarations. Core: `numpy`, `pandas`, `pyarrow`. Optional groups: `data` (survey processing), `ml` (sklearn, cvxpy), `llm` (transformers, peft), `dev` (pytest, ruff, black, mypy). |
| `main.py` | Entry point alias — calls `electoral.pipeline.main()`. |
| `.env` | Local environment variables (not committed). Stores Tailscale IP, API keys. |

---

## `electoral/` — Core Package

### Package Root

#### `electoral/__init__.py`
Empty package marker.

#### `electoral/config.py`
**Purpose:** Immutable pipeline configuration loaded from JSON.

`PipelineConfig` is a frozen dataclass. Key fields:
- `run_key` — unique string identifying this run (e.g. `"base_2026"`)
- `seed` — global RNG seed; all stochastic operations derive from this
- `party` — `"democrat"` or `"republican"`
- `target` — V_eq win threshold (~0.535 for Democrats)
- `pipeline_mode` — `"historical"` (full rebuild) or `"continuous"` (nightly incremental)
- `races/religions/genders` — canonical bloc lists; defaults to the 5/7/3 canonical sets
- `pi_bio_server` — Tailscale URL for Raspberry Pi SetFit endpoint
- `pi_npu_enabled` — whether Hailo NPU is available

`from_json(path)` loads from JSON and silently ignores unknown keys, allowing forward compatibility. `derive_seed(stage_name)` delegates to `electoral.core.rng.derive_seed`.

#### `electoral/artifacts.py`
**Purpose:** All frozen dataclass schemas for every pipeline stage. This is the single source of truth for what each stage produces and consumes.

Every dataclass has `to_dict()` / `from_dict()` / `validate()`. No pandas or numpy in payload fields — all native Python types.

| Class | Stage | Key fields |
|-------|-------|-----------|
| `StageArtifact` | Envelope | `stage`, `run_key`, `metadata`, `data` |
| `VoterPanelData` | Stage 1 | `cycles`, `races`, `religions`, `genders`, `layer_weights`, `source` |
| `BaselinePortfolioData` | Stage 2 | `weights` (race→float), `mu_race/religion/gender`, `mu_eff`, `target` |
| `SentimentData` | Stage 3a | `model`, `shocks`, `scores[bloc][shock]=elasticity` |
| `SocialMediaSentimentData` | Stage 3b | `shock`, `platforms`, `scores[platform][proxy]`, `lagged_delta` |
| `LLMFineTuneData` | Stage 3c | `base_model`, `lora_rank`, `n_examples`, `cycles_used`, `adapter_path` |
| `PredictionMarketData` | Stage 3d | `pre_shock_prob`, `post_shock_*`, `delta_prob`, `sources` |
| `ShockResponseData` | Stage 4 | `shock`, `deltas[bloc]=Δμ`, `covariance` (N×N), `source` |
| `EquilibriumData` | Stage 5 | `weights[bloc]`, `mu_shifted[bloc]`, `feasible`, `target_met` |
| `SimulationData` | Stage 6 | `n_simulations`, `win_probability`, `percentiles[bloc]=[p5,p25,p50,p75,p95]` |
| `MetricsTablesData` | Stage 7 | `tables[key]=JSON payload` |

`ShockResponseData.deltas` values are clamped to `[-0.15, +0.15]` (validated). `EquilibriumData.weights` must sum to 1.0 and be keyed by race blocs only (religion/gender weights are fixed, not optimized).

#### `electoral/stages.py`
**Purpose:** One function per pipeline stage. Currently only Stage 1 (`build_voter_panel`) is fully implemented; the rest return valid placeholder artifacts. Each function:
1. Calls the corresponding kernel
2. Writes the panel parquet files (Stage 1) or artifact JSON
3. Returns the typed payload dataclass

Stage dependency graph:
```
build_voter_panel ──► build_baseline_portfolio
                 └──► build_sentiment_data ──► build_llm_finetune
                                          └──► build_shock_response
                                                    └──► build_optimization
                                                               └──► run_simulations
```

#### `electoral/pipeline.py`
**Purpose:** Prefect DAG orchestrating all stage functions.

Uses `@flow` and `@task` decorators. Falls back to plain Python execution if Prefect is not installed (so CI works without the full Prefect stack). Tasks share a retry policy of 2 retries with 30-second delay. `task_baseline_portfolio` and `task_sentiment_data` both take the same `panel` future — Prefect executes them concurrently.

CLI: `python -m electoral.pipeline --config configs/base.json --shock "tariff" --intensity 0.7`

---

### `electoral/core/` — Infrastructure

#### `electoral/core/types.py`
**Purpose:** All type aliases and canonical constants. Single source of truth for demographic bloc identifiers.

Key exports:
- `Race`, `Religion`, `Gender` — `str`-mixin enums (members are valid strings and work as dict keys)
- `CANONICAL_RACES` — `["african_american","latino","asian","white","other_race"]`
- `CANONICAL_RELIGIONS` — `["evangelical","catholic","protestant","secular","jewish","muslim","other_rel"]`
- `CANONICAL_GENDERS` — `["women","men","other_gender"]`
- `DELTA_BINS` — 9-token vocabulary for LLM constrained decoding
- `BIN_MIDPOINTS` — numeric midpoint for each bin (used by `bin_to_delta()`)
- `LAYER_WEIGHT_KEYS` — `("lambda_1","lambda_2","lambda_3")`
- `VALID_SOURCES` — `{"llm_unified","roberta_news_only","roberta_social_only"}`

Use `Race.WHITE` not the string literal `"white"` in kernel code.

#### `electoral/core/schema.py`
**Purpose:** Five validation helpers used by `validate()` methods in `artifacts.py`.

| Function | What it checks |
|----------|---------------|
| `assert_required_keys(d, keys, context)` | All keys present in dict |
| `assert_shares_sum_to_one(d, context)` | Dict float values sum to 1.0 ± 1e-6 |
| `assert_valid_share(v, name, context)` | Float in [0, 1] |
| `assert_sorted_unique(lst, name, context)` | List is sorted and deduplicated |
| `assert_unique(lst, name, context)` | List has no duplicates |

#### `electoral/core/rng.py`
**Purpose:** Deterministic RNG contract. Ensures reproducibility across runs.

- `derive_seed_tokens(tokens: list[str]) → int` — SHA-256 hash of `":"` -joined tokens, modded to `[0, 2**31)`
- `derive_seed(base_seed, stage_name) → int` — convenience wrapper: `derive_seed_tokens([str(base_seed), stage_name])`
- `make_rng(seed) → np.random.Generator` — PCG64 generator

**Contract:** Never call `np.random` directly. Never call `random.seed()` globally. Every stochastic operation must receive a `make_rng(derive_seed(config.seed, stage_name))` generator.

#### `electoral/core/io.py`
**Purpose:** Artifact I/O: write/read JSON envelopes and Parquet tables.

- `write_artifact(path, d)` — writes JSON; creates parent directories automatically
- `write_parquet(path, df)` — writes DataFrame to Parquet (no index)
- `read_artifact(path) → dict` — reads JSON; unwraps `StageArtifact` envelope if present
- `read_parquet(path) → pd.DataFrame`

---

### `electoral/data/` — Survey Data Processing

#### `electoral/data/loaders.py`
**Purpose:** Source-specific loaders for each survey dataset. Each loader reads a raw or labeled file and returns an uncleaned DataFrame with canonical column names applied.

| Function | Survey | File |
|----------|--------|------|
| `load_arda(path)` | ANES CDF (mislabeled "ARDA" in devplan) | `anes_labeled_subset.parquet` |
| `load_gss(path)` | General Social Survey | `gss_labeled_subset.parquet` |
| `load_gallup(path)` | Democracy Fund VOTER Panel (mislabeled "Gallup") | `voter_panel.csv` |
| `load_nep(path)` | CNN/SSRS National Exit Polls | `nep_{year}_exit_poll.csv` |
| `load_pew(path)` | NPORS (mislabeled "Pew") | `npors_2024_labeled.parquet` |
| `load_ces(path)` | CES cumulative 2006–2024 | `ces_cumulative_labeled.parquet` |
| `load_ces_2024(path)` | CES 2024 single-year | `ces_2024_labeled.parquet` |
| `load_csv_panel(path)` | Generic preprocessed panel CSV | any panel CSV |

Column remapping uses `configs/column_maps.json` (raw source column names → canonical names). Each loader applies the map via `_apply_map()`, which skips unmappable columns and prevents many-to-one collisions.

**Important naming corrections** (devplan used wrong names):
- "ARDA" in devplan = ANES CDF
- "Gallup" in devplan = Democracy Fund VOTER Panel
- "Pew" in devplan = NPORS (Pew's National Public Opinion Reference Survey)

#### `electoral/data/cleaning.py`
**Purpose:** Cleans a raw survey panel DataFrame. Applied after loading, before aggregation.

`clean_raw_panel(df)` runs six steps in order:
1. Coerce `cycle` to nullable Int64 (YYYY)
2. Normalise `bloc` to lowercase snake_case via `_normalize_bloc()`
3. Map normalised bloc strings to canonical IDs via `normalize_bloc()`
4. Coerce `vote_share` and `turnout` to nullable Float64
5. Drop rows where `cycle` or `bloc` is null; log count
6. Sort by `(cycle, bloc)`; raise on duplicate `(cycle, bloc, source)` tuples

`normalize_bloc(raw: str) → str` is the public function for scalar normalisation. It handles ~80 survey alias strings (e.g. `"Black/African American"` → `"african_american"`, `"White Evangelical"` → `"evangelical"`).

`impute_missing_cells(panel) → pd.DataFrame` fills structurally absent (cycle, bloc) cells using rules from DECISIONS.md §Coverage Gap Imputation:
- `other_gender` in [2004, 2008, 2012, 2020, 2024]: constant 0.76 (Pew LGBTQ lean)
- `muslim` 2004: carry-backward from 2008 value
- `other_race` 1948: carry-forward from 1952 value

Called inside `build_voter_panel` after `clean_raw_panel`. Also callable directly; does not mutate the input.

`_BLOC_MAP` contains the complete alias table — update this when adding new source aliases.

#### `electoral/data/panel.py`
**Purpose:** Validates a panel DataFrame against five structural invariants.

`validate_panel(df, required_cols, context)` enforces:
1. All required columns present
2. No nulls in required columns
3. `cycle` is int YYYY in [1900, 2100]
4. `bloc` IDs are lowercase snake_case (regex: `^[a-z][a-z0-9]*(_[a-z0-9]+)*$`)
5. `vote_share` and `turnout` (when present) are in [0, 1]

---

### `electoral/kernels/` — Stage Implementations

#### `electoral/kernels/data.py` ✅ IMPLEMENTED (Week 1)
**Purpose:** Voter panel kernel — loads all survey sources, resolves conflicts, and returns `VoterPanelData`.

`build_voter_panel(config) → (VoterPanelData, pd.DataFrame)` orchestrates:
1. `_from_nep(paths)` — loads all 4 NEP exit-poll CSVs; computes binomial SE from `n_total × stratum_share`
2. `_from_anes(path)` — ANES labeled subset; maps vote_indicator encoding (1=Dem, 2=Rep, 3=Other); Kish effective-sample-size SE
3. `_from_gss(path)` — GSS retrospective vote columns (`pres16`/`pres20`); handles per-wave election mapping
4. `_from_ces(path)` — CES cumulative; presidential cycles only (year % 4 == 0); `weight_cumulative` weighting
5. `resolve_conflicts(panel)` — inverse-SE weighted merge for (cycle, bloc) pairs in multiple sources; merged `source` = sorted "+" join

Source-specific bloc remaps (`_ANES_RACE`, `_GSS_RELIGION`, `_CES_GENDER`, etc.) are hardcoded dicts at the top of the file.

`_agg_stratum()` is the shared aggregation helper: maps raw bloc labels → canonical IDs, drops unmapped/null rows, computes weighted vote_share and Kish SE per (cycle, bloc).

`resolve_conflicts()` logs every conflict at INFO level. Missing (cycle, bloc) cells are logged as warnings.

#### `electoral/kernels/baseline.py` 🔲 STUB (Week 2)
Will implement: GP classifier for win probability from historical loyalty estimates; `mu_eff` computation from layer weights; V_eq derivation from winning cycles; LOCO-CV evaluation.

#### `electoral/kernels/raking.py` 🔲 STUB (Week 2)
Will implement: Iterative Proportional Fitting (IPF) across the three marginal stratum tables. Optional calibration step. Paper will compare additive vs. raked outputs.

#### `electoral/kernels/sentiment.py` 🔲 STUB (Week 3)
Will implement: orchestrates bio classifier (Pi endpoint) + RoBERTa scorer on social/news posts. Outputs `SentimentData` with per-bloc elasticity scores.

#### `electoral/kernels/shock.py` 🔲 STUB (Week 4)
Will implement: Mistral 7B + constrained decoding via `outlines`. Input: shock text string. Output: `ShockResponseData` with delta bins and Δμ per bloc.

#### `electoral/kernels/optimize.py` 🔲 STUB (Week 5)
Will implement: CVXPY DQCP optimizer. Objective: `max Φ((μ̃_eff(w) − V_eq) / sqrt(λ₁² wᵀΣ_Δw))`. Must assert `problem.is_dcp(qcp=True)`. Decision variables: race bloc weights w (5-simplex). Fixed: religion weights `v_R`, gender weights `g_G`.

#### `electoral/kernels/finetune.py` 🔲 STUB (Week 4)
Will implement: assembles instruction-completion pairs from RoBERTa-scored posts and historical events; calls QLoRA trainer on HPC.

#### `electoral/kernels/metrics_tables.py` 🔲 STUB (Week 6+)
Will implement: manuscript-ready performance tables for bio classifier, RoBERTa scorer, and LLM delta predictions.

---

### `electoral/models/` — ML Models

#### `electoral/models/ml_baseline.py` ✅ PARTIAL (Week 2 — moment estimation implemented)
**Purpose:** Estimates bloc-level vote-share moments from the voter panel. These are the primary inputs to the CVXPY optimizer.

**Exports:**

`MomentEstimates` — frozen dataclass returned by `estimate_moments`:
- `mu_race / mu_religion / mu_gender` — `dict[str, float]`, mean vote share for party P across winning cycles; `float("nan")` for blocs absent from the panel
- `Sigma` — `np.ndarray` shape `(5, 5)`, empirical race-bloc covariance across **all** cycles (not just winning)
- `winning_cycles` — `list[int]`, cycles where national race-weighted share > 0.50
- `race_blocs` — `list[str]` = `CANONICAL_RACES`; authoritative row/col order for Σ

`estimate_moments(df, party, *, epsilon=1e-6) → MomentEstimates`:
- For `party="democrat"`: uses `vote_share` directly
- For `party="republican"`: uses `1 − vote_share` throughout (including winning-cycle identification)
- Winning cycles identified by computing a race-bloc weighted national vote share using approximate electorate shares from CLAUDE.md (AA 12%, Latino 11%, Asian 5%, White 62%, Other 10%), then checking `> 0.50`
- Σ NaN entries (blocs in fewer than 2 cycles, ddof=1) are zeroed before the PSD safeguard
- PSD safeguard: `Sigma += (-min_eig + epsilon) * I` if `min_eig < 0`

`psd_repair(cov, eps=1e-6) → np.ndarray`:
- Public helper: adds `(-min_eig + eps) * I` if the minimum eigenvalue is negative; returns the input unchanged (zero-copy) when already PSD
- Used by `estimate_moments` and available for direct use by the CVXPY kernel

**Known limitation (see ESPINOSA.md §10.3):** The panel-derived winning-cycle criterion misclassifies 4 of 20 cycles (1952, 1988, 2004 as Democrat-winning; 1992 as Republican-winning). Root causes: survey-weight mismatch for pre-modern electorates, and three-party race contamination in 1992. Fix: pass an external `actual_vote_shares` lookup table.

**Not yet implemented:** GP classifier (RBF kernel), XGBoost baseline, leave-one-cycle-out (LOCO) cross-validation — these are the remaining Week 2 tasks in `kernels/baseline.py`.

#### `electoral/models/bootstrap.py` 🔲 STUB
Will implement: bootstrap resampling for confidence intervals on GP predictions.

#### `electoral/models/regression.py` 🔲 STUB
Will implement: OLS/Ridge regression for Δμ estimation from sentiment scores.

---

### `electoral/portfolios/` — Optimization Building Blocks

#### `electoral/portfolios/constraints.py` 🔲 STUB
Will implement: `ConstraintSpec` dataclass encoding V_eq, layer weights (λ₁/λ₂/λ₃), party, `mu_eff` baseline, and the 5×5 race covariance matrix `Σ_Δ`.

#### `electoral/portfolios/cvx.py` 🔲 STUB
Will implement: CVXPY DQCP optimizer. The quasi-convex Sharpe-ratio objective. Key implementation note: must call `problem.solve(qcp=True)` and assert `problem.is_dcp(qcp=True) == True`.

#### `electoral/portfolios/weights.py` 🔲 STUB
Will implement: equal-weight baseline (uniform 1/5 race allocation) and value-weight baseline (proportional to electorate share).

---

### `electoral/optimization/`

#### `electoral/optimization/cvx.py`
Re-exports from `electoral/portfolios/cvx.py`. Exists to provide a consistent import path.

#### `electoral/optimization/simplex.py`
Simplex projection utilities for the CVXPY optimizer.

---

### `electoral/simulation/`

#### `electoral/simulation/montecarlo.py` 🔲 STUB (Week 5)
Will implement: Logistic-Normal ILR Monte Carlo.

Algorithm:
1. Map optimal weights w* to ILR coordinates: `z* = Vᵀ log(w*)` (Helmert contrast matrix V)
2. Propagate Σ_Δ to ILR space via Jacobian: `Σ_ILR = J Σ_Δ Jᵀ`
3. Draw `y^(n) ~ N(z*, Σ_ILR)` in R^(K-1) for n = 1..N
4. Back-transform: `w^(n) = softmax(V y^(n))`
5. Compute `win_flag^(n) = 1[μ̃_eff(w^(n)) ≥ V_eq]`
6. Report: `win_probability = mean(win_flags)`, `percentiles[bloc]` = [p5,p25,p50,p75,p95] of w distribution

Zero-weight draws → flag as `infeasible_bloc`, never floor.

---

### `electoral/nlp/` — NLP Pipeline

#### `electoral/nlp/archive.py`
Loads pre-downloaded HuggingFace historical archives (Bluesky, Discord-Unveiled, USC Telegram, Kavanaugh 56M posts). These are the primary training data source — do not re-scrape what already exists on HuggingFace.

#### `electoral/nlp/news_loader.py`
Loaders for 3DLNews2 (14k outlets, 1995–2024) and Webhose news datasets.

#### `electoral/nlp/scraper.py`
Live news scraper targeting: Christianity Today (evangelical), CBN (evangelical), Univision (Latino), NYT, WaPo (secular), Fox (conservative). Runs nightly on the Intel Mac as a launchd daemon. Output: `rawdata/news/{outlet}/{YYYY-MM-DD}.jsonl`.

#### `electoral/nlp/bio_classifier.py` 🔲 STUB (Week 3)
Will implement: 3-stage demographic inference from social media bios:
1. SetFit fine-tuned model (runs on Raspberry Pi via Tailscale)
2. Keyword lexicon fallback (race/religion/gender lexicons in `configs/`)
3. Language prior fallback (Pew-based priors from `configs/language_priors.json`)

Posts using stage 3 get `inference_method: "language_prior"` and are excluded from Σ_Δ estimation.

#### `electoral/nlp/scorer.py` 🔲 STUB (Week 3)
Will implement: RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) scorer with bloc-weighted aggregation. Computes per-shock elasticity Δμ per demographic bloc. Embedding cache written to `data/embeddings/`.

#### `electoral/nlp/elasticity.py` 🔲 STUB (Week 4)
Will implement: fine-tuning dataset assembly from scored posts + historical (shock, outcome) pairs.

#### `electoral/nlp/social.py` 🔲 STUB
Will implement: platform-agnostic collector interface. Defines the JSONL envelope schema:
```json
{"schema_version":"1.0","created_at":"ISO8601","stage":"collect","seed":42,
 "payload":{"text":"...","created_at":"...","lang":"...","platform":"..."}}
```

#### `electoral/nlp/collectors/schema.py`
Post schema definitions shared across Bluesky, Apify/X, and Reddit collectors.

#### `electoral/nlp/collectors/bluesky_firehose.py` 🔲 STUB
Will implement: AT Protocol firehose listener with English + political keyword filtering.

#### `electoral/nlp/collectors/apify_x_scraper.py` 🔲 STUB
Will implement: Apify X (Twitter) free-tier scraper (500 results/run), triggered per shock event.

---

### `electoral/markets/`

#### `electoral/markets/collector.py` 🔲 STUB
Will implement: price poller for Polymarket and PredictIt contracts. Contract IDs are in `configs/market_contracts.json`.

#### `electoral/markets/aggregator.py` 🔲 STUB
Will implement: volume-weighted multi-market price aggregator. Output: `"Market consensus: X%"` display value. **Markets are calibration benchmarks only** — never used as training inputs.

---

### `electoral/llm/` — LLM Fine-Tuning

#### `electoral/llm/trainer.py` 🔲 STUB
Will implement: QLoRA trainer (HuggingFace PEFT, 4-bit NF4, rank-16 LoRA) for Mistral 7B. Runs on HPC A100.

#### `electoral/llm/inference.py` 🔲 STUB
Will implement: constrained decoding via `outlines` library. Constrains output to 9-token bin vocabulary per stratum.

#### `electoral/llm/eval.py` 🔲 STUB
Will implement: evaluation against held-out historical (shock, delta_bin) pairs.

---

### `electoral/api/` — FastAPI Endpoints

#### `electoral/api/shock_endpoint.py` 🔲 STUB
Will implement: `POST /estimate` — accepts shock text + party, streams Server-Sent Events (SSE) with stage-by-stage results (deltas → equilibrium → simulation → error).

#### `electoral/api/audit.py` 🔲 STUB
Will implement: `GET /audit_log` and `POST /log_inference` — audit trail for all inference runs.

---

### `electoral/metrics/` and `electoral/reporting/`

#### `electoral/metrics/performance.py` / `tables.py` 🔲 STUBS
Will implement: precision/recall/F1/AUC for bio classifier and scorer; manuscript-ready table formatting.

#### `electoral/reporting/export.py` 🔲 STUB
Will implement: export of `EquilibriumData` + `SimulationData` to frontend-consumable JSON.

---

## `collectors/` — Always-On Collectors (Intel Mac / M5)

#### `collectors/bluesky.py` 🔲 STUB
AT Protocol firehose listener. Primary data source is pre-existing HuggingFace Bluesky archives. Secondary: live firehose going forward. Output: `rawdata/social/bluesky/live/{YYYY-MM-DD}.jsonl`.

#### `collectors/apify.py` 🔲 STUB
Apify X scraper (free tier, 500 results/run). Run per shock event. Output: `rawdata/social/apify/{shock_id}/{YYYY-MM-DD}.jsonl`.

#### `collectors/reddit.py` 🔲 STUB
Reddit API OAuth collector. Subreddits: r/Catholicism, r/Christianity, r/exchristian, r/Conservative, r/progressive, r/BlackPeopleTwitter, r/LatinoPeopleTwitter, r/Jewish, r/islam. Output: `rawdata/social/reddit/{subreddit}/{YYYY-MM-DD}.jsonl`. Uses `inference_method: "subreddit_proxy"` — excluded from Σ_Δ.

---

## `scripts/` — Standalone Operational Scripts

| Script | Machine | What it does |
|--------|---------|-------------|
| `audit_panel.py` | M5 | Coverage matrix, vote-share summary stats, outlier flags for the processed voter panel. Output: `rawdata/audit_report.txt`. |
| `clean_with_llm.py` | M5 | LLM cleaning of raw social posts using Qwen2.5-7B-Instruct via `mlx_lm`. Never Gemini API (reproducibility). |
| `generate_synthetic.py` | M5 | Synthetic training example generation + MMD/PCA/PCD diagnostic validation. Soft gate before adding to fine-tuning set. |
| `sample_archives.py` | HPC | Stratified sampling of HuggingFace archives for human annotation. SLURM array job. |
| `score_array.sh` | HPC | SLURM array job wrapper for RoBERTa scoring at scale. |
| `hpc_submit.sh` | HPC | Generic SLURM job submission wrapper. |
| `cnn_ssrs_poll_scraper.py` | M5 | PDF scraper for CNN/SSRS exit polls (2004, 2016, 2020, 2024). |
| `compile_hailo.py` | Pi | Compiles SetFit model + Hailo runtime for Raspberry Pi 5 NPU. |
| `pi_bio_server.py` | Pi | FastAPI server running on Pi; POST `/classify` endpoint for bio inference. Listens on `$PI_TAILSCALE_IP:9000`. |
| `verify_artifacts.py` | M5 | Validates all artifacts in `artifacts/` directory against their schemas. |
| `sensitivity_analysis.py` | M5 | Perturbation analysis on λ₁/λ₂/λ₃ and V_eq. |

---

## `configs/` — Configuration Files

| File | What it configures |
|------|--------------------|
| `base.json` | Default `PipelineConfig`. `seed=42`, `party="democrat"`, `target=0.535`. |
| `smoke.json` | Minimal config for `just smoke`. Same structure; optimized for fast runtime. |
| `party_config.json` | V_eq thresholds per party once empirically derived from panel. Placeholder until `build_constraint_spec` is implemented. |
| `layer_weights.json` | λ₁=0.50, λ₂=0.30, λ₃=0.20. Placeholder — must be recomputed from historical election data via `build_constraint_spec`. Also contains a `raked` section for IPF-calibrated weights. |
| `shock_taxonomy.json` | Canonical shock categories (economy, immigration, healthcare, scandal, etc.) + keywords used by collectors for filtering. |
| `mmd_config.json` | MMD kernel, bandwidth (median heuristic, `1/(2σ²)`), and acceptance threshold for synthetic data quality gate. |
| `bin_uncertainty.json` | Per-bin empirical `σ_b` values for Logistic-Normal covariance propagation. Must be populated by `generate_synthetic.py`. |
| `language_priors.json` | Pew-based demographic priors used as language fallback when bio inference is below confidence threshold. |
| `race_lexicon.json` | Bio keyword → race bloc probability weights (e.g. `"latinx" → {latino: 0.95}`). |
| `religion_lexicon.json` | Bio keyword → religion bloc probability weights (e.g. `"evangelical" → {evangelical: 0.95}`). |
| `gender_lexicon.json` | Bio keyword → gender signal (e.g. `"she/her" → {women: 0.9}`). |
| `market_contracts.json` | Shock event ID → Polymarket/Kalshi/PredictIt contract URLs. Calibration only. |
| `column_maps.json` | Survey dataset column name mappings. Format: `{SOURCE_NAME: {raw_col: canonical_col}}`. Consumed by `data/loaders.py`. |
| `shocks.json` | Catalog of historical shock events with metadata (date, description, category). |

---

## `tests/` — Test Suite

**384 passing, 21 skipped** as of Week 1–2.

| File | What it tests |
|------|--------------|
| `conftest.py` | Shared fixtures: `LAYER_WEIGHTS`, canonical lists, toy `PipelineConfig`, sample `VoterPanelData` and `BaselinePortfolioData`. |
| `test_config.py` | `PipelineConfig.from_json()`, field validation, `derive_seed` determinism. |
| `test_io.py` | JSON/Parquet round-trips, parent dir creation, dtype preservation. |
| `test_schema.py` | All five `assert_*` helpers with valid and invalid inputs. |
| `test_rng.py` | `make_rng` seeding, `derive_seed` determinism, verify no `np.random` global calls in codebase. |
| `test_data_cleaning.py` | `clean_raw_panel` all 6 steps; `normalize_bloc` alias table; `impute_missing_cells` all 3 rules + 11 edge cases. 40 tests total. |
| `test_loaders.py` | Generic CSV loader; ANES/CES/GSS/NEP/NPORS wrapper behavior. |
| `test_panel.py` | `validate_panel` all 5 invariants; error messages include context. |
| `test_artifact_roundtrip.py` | `to_dict()` → `from_dict()` → `validate()` for all 10 dataclasses. |
| `test_ml_baseline.py` | `estimate_moments` winning cycles, μ values, Republican vote-share flip, Σ shape/PSD/symmetry/dtype, `race_blocs` order, empty winning cycles; `psd_repair` known near-singular matrix, zero-copy for already-PSD inputs, shift respects eps. 28 tests. |
| `test_baseline.py` | ⏸ Skipped (Week 2): GP classifier, mu_eff, V_eq derivation, LOCO-CV. |
| `test_mc_convergence.py` | ⏸ Skipped (Week 5): ILR back-transform, win probability convergence, Helmert matrix orthogonality, seed reproducibility. |
| `test_bio_classifier.py` | ⏸ Skipped (Week 3): 3-stage inference, SetFit/lexicon/prior fallback logic. |
| `test_sentiment.py` | ⏸ Skipped (Week 3): RoBERTa scorer, bloc-weighted aggregation, embedding cache. |
| `test_social.py` | ⏸ Skipped (Week 3): JSONL envelope schema, post deduplication, platform merge. |
| `test_collectors.py` | ⏸ Skipped (Week 3): Bluesky/Apify/Reddit output format. |
| `test_stages.py` | Stage function inputs/outputs; artifact envelope validation; smoke pipeline end-to-end. |

**Fixtures:**
- `tests/fixtures/toy_panel.csv` — 20-row panel: 2 cycles × 10 blocs, includes one dirty row (missing vote_share, string turnout)
- `tests/fixtures/smoke_panel.csv` — full-scale smoke test panel
- `tests/fixtures/surveys/cnn_ssrs_polls/nep_2020_exit_poll.csv` — NEP 2020 fixture

---

## `data/` — Source Data

### Survey Files

| Source | Location | Cycles | Notes |
|--------|----------|--------|-------|
| ANES CDF | `data/surveys/anes_timeseries_cdf_csv_20260205/` | 1948–2020 | Use labeled subset (parquet). Weight col null for 2020 — see ESPINOSA.md Q1.1 |
| CES cumulative | `data/surveys/CES_2006_2024/` | 2006–2024 | Use `race_h` (any-part Hispanic), not `race` |
| CES 2024 | `data/surveys/CES_2024/` | 2024 | Verify `CC24_410` coding against codebook PDF |
| GSS | `data/surveys/GSS_stata (1)/` | 1972–2024 | Retrospective vote columns: `pres16`, `pres20`, `whovote24` |
| NPORS 2024 | `data/surveys/NPORS-2024-Data-Release/` | 2024 | Single cross-section; cycle hardcoded to 2024 |
| VOTER Panel | `data/surveys/VOTER Panel Data Files/` | 2012–2020 | Longitudinal (not cross-section); Windows-1252 encoding |
| NEP | `data/surveys/cnn_ssrs_polls/` | 2004, 2016, 2020, 2024 | Gold standard for race/gender vote shares; no 2008/2012 |

### Processed Panel Files (generated by `build_voter_panel`)

Output location: `artifacts/panel/` (not `data/panel/` — see `stages.py:39`).

| File | Content |
|------|---------|
| `panel_race.parquet` | Rows with bloc ∈ CANONICAL_RACES |
| `panel_religion.parquet` | Rows with bloc ∈ CANONICAL_RELIGIONS |
| `panel_gender.parquet` | Rows with bloc ∈ CANONICAL_GENDERS |

Schema: `cycle (Int64), bloc (string), vote_share (Float64), source (str)`.

---

## `artifacts/smoke/` — Smoke Test Artifacts

Pre-generated artifacts for fast CI smoke testing. Each JSON follows the `StageArtifact` envelope format with placeholder payload values. Used by `test_stages.py`.

---

## `rawdata/` — Raw Working Files

| File | Content |
|------|---------|
| `SOURCE NOTES.md` | Data provenance documentation: source names, URLs, download dates, license restrictions. |
| `audit_report.txt` | Output of `scripts/audit_panel.py` — row counts, NA rates, outlier flags. |
| `spot_check.md` | 10-cell manual cross-validation against raw source documents. 4 discrepancies flagged for Week 2. |
| `anes_vcf_labels.json` | ANES variable codebook: VCF variable IDs → human-readable labels, valid categories, and numeric-to-string mappings. |
| `column_maps.json` | Working copy of source-to-canonical column name mappings (synced to `configs/column_maps.json`). |

---

## `deploy/` — Deployment

#### `deploy/modal_app.py` 🔲 STUB
Modal GPU deployment (A100 serverless). Exposes `/estimate` endpoint. Target: Week 6.

#### `deploy/backend_router.py` 🔲 STUB
Unified FastAPI router aggregating `/estimate`, `/audit_log`, `/market_prices`, `/shock_taxonomy`.

---

## `webapp/` — Next.js 16 Frontend

| File | What it renders |
|------|----------------|
| `app/page.tsx` | Landing page: hero, shock input form, party selector |
| `app/dashboard/page.tsx` | Results dashboard: deltas, coalition chart, win gauge, market consensus |
| `components/ShockInput.tsx` | Party + shock text + intensity slider form |
| `components/CoalitionChart.tsx` | Baseline → optimal race weight visualization |
| `components/WinGauge.tsx` | Win probability gauge with 90% CI band; "No feasible path" / "Uncertain path" states |
| `components/ShockNarrative.tsx` | LLM-generated explanation of the delta bins and rebalancing |
| `components/AuditLog.tsx` | Inference audit trail with all intermediate artifacts |
| `lib/api.ts` | Typed FastAPI SSE client |

---

## `elec-equilibrium/` — Reddit Devvit App

A separate TypeScript application built with Devvit (Reddit's developer platform). Enables users to interact with the shock estimator directly within a subreddit post.

| Directory | Purpose |
|-----------|---------|
| `src/client/` | React components rendered in the Reddit iframe (game UI, splash screen) |
| `src/server/` | Devvit server handlers (tRPC router, post triggers, vote counting) |
| `src/shared/` | Type definitions shared between client and server |

Build output in `dist/`. Not part of the core Python pipeline.

---

## `adapters/`

`adapters/mistral-7b-electoral/` — placeholder directory for QLoRA fine-tuned Mistral 7B adapter weights (LoRA rank-16). Empty until Week 4 HPC training run completes.

---

## `docs/`

| File | Content |
|------|---------|
| `docs/electoral_equilibrium_devplan.pdf` | 73-page 7-week development plan with per-day milestones |
| `docs/electoral_task_instructions.pdf` | 50-page SRP task specification and success criteria |