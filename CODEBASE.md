# Codebase Reference — Electoral Equilibrium

Every source file documented, organized by directory. For each file: what it does,
what it reads/writes, and what stage of the pipeline it belongs to.

**Total source files:** ~130 Python modules + configs + tests + frontend  
**Test status:** 623 passing, 17 skipped  
**Implemented:** Weeks 1–5 (data pipeline through NLP sentiment pipeline + bio classifier training). Week 5 optimizer/MC stubs remain.

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
- `pi_npu_enabled` — whether Hailo NPU is available; **confirmed `false`** (NPU deferred indefinitely — see DECISIONS.md)

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

- `write_artifact(path, envelope, df=None)` — writes the JSON envelope; if `df` given, also writes a sibling `.parquet`. Creates parent dirs.
- `write_json(path, payload)` — stable formatting (`indent=2`, `sort_keys=True`); runs `sanitize_floats` first.
- `sanitize_floats(obj)` — recursively maps non-finite floats (inf/-inf/NaN) → `null` so artifacts are always valid JSON (TS frontend / DuckDB safe). Shared sanitizer; also applied at the MC CLI, SSE frames, and fine-tune serializer.
- `read_artifact(path) → (envelope, df|None)` — reads JSON; loads the sibling `.parquet` if present.
- `read_json` / `write_parquet` / `read_parquet` (lazy pandas/pyarrow import).

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

#### `electoral/kernels/baseline.py` ✅ IMPLEMENTED (Week 2)
`build_baseline_portfolio(config, panel_df) → BaselinePortfolioData`. Estimates per-bloc moments (`estimate_moments` in `models/ml_baseline.py`), sets baseline race weights by **NEP×loyalty** (`w0_i ∝ nep_share_i · loyalty_i`, normalized — not an optimization), and computes the steady-state `mu_eff` via the three-stratum formula using the REAL religion/gender vote shares (equal-weighted, the v_R/g_G placeholder until raking). NaN bloc loyalties imputed to neutral 0.5 with a warning.

#### `electoral/kernels/raking.py` ✅ IMPLEMENTED (Week 2)
`rake_layer_weights(panel)` calibrates the λ layer weights by IPF / regularized least-squares against historical national 2-party Dem vote share (ridge-regularized to handle the strata's collinearity; simplex-projected each pass). `write_raked_weights()` writes them under the `"raked"` key in `layer_weights.json`, leaving the additive λ's untouched so the paper can compare additive vs. raked. Optional calibration step.

#### `electoral/kernels/sentiment.py` ✅ IMPLEMENTED (Week 4)
**Purpose:** Orchestrates the full NLP sentiment pipeline — loads social posts and archives, runs bio classification and RoBERTa scoring, writes `SentimentData` and `SocialMediaSentimentData` artifacts.

`run_sentiment_pipeline(config_path, posts_root, archive_root, news_root, shock_ids, model_name, window_hours, output_dir, shocks_path)` → `(SentimentData, list[SocialMediaSentimentData])`. Builds the nested `shock_id → platform → list[post]` dict via `_load_social_posts()`, then delegates to `score_social_for_shock()` and `score_news_for_shocks()` from `nlp/scorer.py`. Writes JSON artifacts to `output_dir` if provided.

#### `electoral/kernels/shock.py` ✅ IMPLEMENTED (Week 4–6)
`build_shock_response(config, event, intensity) → (ShockResponseData, EquilibriumData)`, the full chain: (1) `_estimate_shock` runs the fine-tuned LLM via `outlines` constrained decoding; (2) `_load_baseline` (race μ) + `_load_baseline_fixed_strata` (religion/gender μ) read `baseline_portfolio.json`; (3) `mu_tilde = clip(μ_baseline + Δ_race)`; (3b) `compute_fixed_loyalty(...)` builds the real religion+gender contribution from baseline shifted by the shock's religion/gender deltas; (4) `_build_sigma_delta` computes the Ledoit-Wolf 5×5 Σ_Δ from panel first-differences (diagonal fallback if no panel); (5) `solve_rebalanced(..., fixed_loyalty=...)`; (6) writes per-id `shock_{id}.json` / `equilibrium_{id}.json`. Validates both artifacts.

#### `electoral/kernels/optimize.py` ✅ IMPLEMENTED (Week 5)
`build_optimization(config, shock) → EquilibriumData` — thin stage wrapper that calls `optimization.cvx.solve_rebalanced` (the canonical optimizer) and writes `optimization.json`. The objective/solve lives in `optimization/dqcp.py`.

#### `electoral/kernels/finetune.py` ✅ IMPLEMENTED (Week 5)
`build_llm_finetune(config) → LLMFineTuneData`. **Idempotent:** if a trained LoRA adapter (`adapter_config.json`) already exists at the configured path, returns its metadata without retraining; otherwise calls `train_hpc` (`llm/trainer.py`). Logs eval MAE from `trainer_state.json`.

#### `electoral/kernels/metrics_tables.py` 🔲 STUB (Week 6+)
Will implement: manuscript-ready performance tables for bio classifier, RoBERTa scorer, and LLM delta predictions.

---

### `electoral/models/` — ML Models

#### `electoral/models/ml_baseline.py` ✅ IMPLEMENTED
Moment estimation + a win-probability classifier. `estimate_moments(panel, party)` → μ (per stratum) and the 5×5 race Σ; `ground_truth_winning_cycles`, `derive_ec_veq(party)` (EC-adjusted V_eq via logistic fit over 20 cycles — Rep < 0.5), `psd_repair(cov)` (used by the optimizer/MC). Lower half: `fit_gp_classifier` (Gaussian Process) with leave-one-cycle-out (LOCO) CV and `platt_scale_loco` calibration.

#### `electoral/models/bootstrap.py` ✅ IMPLEMENTED
`ledoit_wolf_cov(delta_matrix)` (shrinkage covariance for the p>n regime — the production Σ_Δ estimator) plus `bootstrap_cov_matrix` / `bootstrap_cov_weighted` (kept for the paper's covariance-comparison table). Uses `make_rng` for reproducibility.

#### `electoral/models/regression.py` 🔲 STUB
Placeholder (raises `NotImplementedError`). Δμ-from-sentiment regression lives in `nlp/elasticity.py`.

---

### `electoral/portfolios/` — Optimization Building Blocks

#### `electoral/portfolios/constraints.py` ✅ IMPLEMENTED
`ConstraintSpec` dataclass: `blocs`, per-bloc `lower`/`upper` weight bounds. `from_bounds(...)` / `default(...)` factories; `validate()` checks bounds ∈ [0,1], lower ≤ upper, Σlower ≤ 1 ≤ Σupper. Consumed by `solve_dqcp` (enforced exactly in the SOCP) and `solve_baseline`.

#### `electoral/portfolios/cvx.py` ✅ IMPLEMENTED
`solve_baseline(mu, cov, target, blocs, spec)` — the **min-variance** QP baseline (`min wᵀΣw s.t. μᵀw ≥ target`, simplex + bounds), used only as the paper's comparison against the max-P(win) optimizer. Relax-and-retry on infeasibility. (This is NOT the production optimizer — that's `optimization/dqcp.py`.)

#### `electoral/portfolios/weights.py` ✅ IMPLEMENTED
`equal_weight_baseline` (uniform 1/n) and `value_weight_baseline` (∝ electorate share) — trivial benchmark portfolios for paper comparison; no solver.

---

### `electoral/optimization/`

#### `electoral/optimization/dqcp.py` ✅ IMPLEMENTED — the single canonical optimizer
Max-P(win) Sharpe-ratio optimizer. `solve_dqcp(mu_race, cov_race, target, lambda_1, fixed_loyalty, *, blocs, spec, solver=CLARABEL)` solves the SOCP form (normalized Charnes–Cooper / Lobo et al.) and returns a weights dict. Maximizes `(λ₁·μ·w + fixed_loyalty − V_eq) / √(wᵀΣw)`. Per-bloc `ConstraintSpec` bounds are enforced EXACTLY inside the cone via the linear pair `z_i ≥ lo_i·s`, `z_i ≤ hi_i·s` (no post-solve clipping). Helpers: `compute_mu_eff(weights, mu_race, λ₁, fixed_loyalty)` = `λ₁·Σ(w·μ) + fixed_loyalty`; `compute_fixed_loyalty(mu_religion, mu_gender, λ₂, λ₃, deltas_religion, deltas_gender)` = the real religion+gender contribution (equal-weighted v_R/g_G, post-shock-shifted).

#### `electoral/optimization/cvx.py` ✅ IMPLEMENTED — EquilibriumData wrapper
`solve_rebalanced(mu_tilde, cov_delta, target, party, shock, floor=0.05, ceiling=0.60, fixed_loyalty=None)` is the canonical entry point used by BOTH the kernel and the API. It delegates the solve to `dqcp.solve_dqcp` (passing a `[floor, ceiling]` `ConstraintSpec`), computes `mu_eff_shifted`, and returns a validated `EquilibriumData`. `fixed_loyalty=None` → neutral `(λ₂+λ₃)·0.5` placeholder. On infeasible/solver failure → `solve_equal_weight_rebalanced` (equal weights, `feasible=False`, `method="equal_weight_fallback"`). Also re-exports `solve_baseline` from `portfolios/cvx.py`.

#### `electoral/optimization/simplex.py` ✅ IMPLEMENTED
Euclidean projection onto the probability simplex. `project_simplex(v)` (Duchi et al. 2008, O(n log n)) snaps any vector to the nearest non-negative, sums-to-1 vector; `project_simplex_batch(W)` applies it per row. A repair utility, not the main MC path.

---

### `electoral/simulation/`

#### `electoral/simulation/montecarlo.py` ✅ IMPLEMENTED — Logistic-Normal ILR Monte Carlo
`run_ilr_montecarlo(equilibrium, config, n_simulations=10_000, sigma_default=0.02, cov_delta=None, fixed_loyalty=None) → SimulationData`.

Algorithm:
1. Map weights w* to ILR coords: `z* = Vᵀ log(w*)` (Helmert matrix `helmert_matrix(k)`, `ilr(w, V)`)
2. Propagate Σ_Δ to ILR space via Jacobian: `Σ_ILR = J Σ_Δ Jᵀ`, then PSD-repair
3. Draw `y^(n) ~ N(z*, Σ_ILR)` via Cholesky (vectorized over N)
4. Back-transform: `w^(n) = softmax(V y^(n))` (`ilr_inv(z, V)`)
5. `win_flag = 1[μ_eff(w^(n)) ≥ V_eq]`; `win_probability = mean`; per-bloc percentiles [p5,p25,p50,p75,p95]
6. 90% CI on win_probability via Bernoulli/binomial bootstrap (not percentile of the 0/1 flags)

Σ_Δ: uses `cov_delta` (the shock's real covariance) when supplied, else the isotropic diagonal `sigma_default²·I`. `fixed_loyalty`: the religion+gender contribution; when `None` it is DERIVED from `equilibrium.mu_eff_shifted − λ₁·Σ(w·μ̃)` so the MC win-check always matches the optimizer (falls back to neutral on unset/fallback equilibria). Zero-weight blocs → `ValueError` (ILR undefined at the simplex boundary), never floored. Also exposes a `python -m electoral.simulation.montecarlo` CLI for SLURM runs.

---

### `electoral/nlp/` — NLP Pipeline

#### `electoral/nlp/archive.py` ✅ IMPLEMENTED (Week 4, updated Week 5)
**Purpose:** Loads historical archives from JUHIDRIVE and repo archive directories into the canonical post schema.

`HistoricalArchiveLoader` class: `load_reddit()`, `load_news(source="all")`, `load_discord()`, `load_all()`. Private normalizers: `_normalize_reddit()`, `_normalize_webhose()`, `_normalize_3dlnews()`, `_normalize_discord()`. Routes posts to shock slugs via keyword matching against `configs/shocks.json`. Reddit posts assigned `inference_method="subreddit_proxy"` via the `SUBREDDIT_PROXY` dict. Archive paths driven by `configs/archives.json`.

Week 5 additions: (1) `.sav` file support — `_load_sav_file()` reads SPSS SAV files via `pyreadstat`, maps `Tweet → text` and `Event` integer codes → shock IDs (`{1: "metoo_2017", 2/3/4: "kavanaugh_2018"}`). Tested on `metoo_liwc`: 3,683 records. (2) `TelegramLoader._load_csv_file()` handles CSV archives via pandas (`dtype=str`, `keep_default_na=True`). Maps `content/message_id/date/language/from_id/post_author` columns; packs `toxicity`, `severe_toxicity`, `identity_attack` into a `metadata` dict. Tested on `telegram_2024`: 3,122 posts.

#### `electoral/nlp/news_loader.py` ✅ IMPLEMENTED (Week 4, updated Week 5)
**Purpose:** Loaders for scraped news and HuggingFace news datasets.

`ScrapedNewsLoader`: reads `rawdata/news/{outlet}/{YYYY-MM-DD}.jsonl`. `load_huggingface_news(dataset_name, shocks_path, max_records, split)`: graceful `ImportError` when `datasets` not installed. `OUTLET_DEMO_PROXY`: per-outlet readership composition dict serialised into `author_description` as JSON, so `RoBERTaScorer` extracts outlet-level demographic weights without calling the bio classifier on news articles.

Week 5 addition: `__main__` CLI block (`python -m electoral.nlp.news_loader --loader 3dlnews|scraped|webhose --shock-id STR --since DATE --until DATE --states ABBR... --max-articles N --verbose`). Saves output to `/Volumes/JUHIDRIVE/electoralData/rawdata/articles/` by default. Uses `3dlnews_parsed` directory for the 3DLNews archive (pre-parsed JSONL).

#### `electoral/nlp/scraper.py` ✅ REWRITTEN (Week 5)
Live news scraper. **Week 5 rewrite:** Replaced Google News RSS (broken base64-encoded `CBMi...` URLs) with direct site RSS feeds via `feedparser>=6.0`. Sources: BBC (`feeds.bbci.co.uk/news/world/rss.xml`), Guardian (`world/iran/rss`), NPR, Christianity Today, CBN, Fox News. Each source has `rss_url` and optionally `rss_fallback_url` (CBN, Fox News). Fallback fires only when primary returns `bozo=True` with no entries. Sources without direct RSS (NYT, WaPo, Univision, Politico) are skipped with a warning. Runs nightly on the Intel Mac as a launchd daemon. Output: `rawdata/news/{outlet}/{YYYY-MM-DD}.jsonl`.

#### `electoral/nlp/bio_classifier.py` ✅ IMPLEMENTED (Week 4)
**Purpose:** 3-stage demographic inference from social media author bios.

`BioClassification` dataclass: `inference_method`, `race_weights`, `religion_weights`, `gender_weights`; methods `has_signal()`, `is_estimable()`. `BioClassifier.classify(bio, lang, inference_method)` short-circuits in order:
1. `_stage1_keyword()`: scans `bio.lower()` against all three lexicon configs; accumulates weights, normalises
2. `_stage2_language_prior()`: maps lang prefix (es→spanish, ko→korean, zh→chinese, ar→arabic) via `_LANG_TO_PRIOR`
3. `_stage3_setfit()`: POST to Pi server; returns `None` if Pi returns `bloc=null` (CPU mode until Week 4 Day 3)

`inference_method="language_prior"` → excluded from both μ and Σ_Δ. `_UPSTREAM_METHODS = frozenset(["platform_proxy","subreddit_proxy"])` bypasses all three stages.

#### `electoral/nlp/scorer.py` ✅ IMPLEMENTED (Week 4)
**Purpose:** RoBERTa sentiment scorer with bloc-weighted aggregation across 15 demographic blocs.

Model: `cardiffnlp/twitter-roberta-base-sentiment` (LABEL_0=neg, LABEL_1=neu, LABEL_2=pos). Score = P(pos) − P(neg) ∈ [−1, 1]. Truncates to 128 words before tokenisation. Auto-selects MPS/CUDA/CPU. Batch size 32. Key functions:
- `RoBERTaScorer.score_posts_for_shock(posts, bio_classifier, shock_id)` → `dict[str, float]` (15 blocs)
- `score_news_for_shocks(articles, scorer, shocks, model_name)` → `SentimentData`
- `score_social_for_shock(posts_by_platform, scorer, bio_classifier, shock_id, window_hours=72)` → `SocialMediaSentimentData`

`_extract_outlet_proxy(post)` reads `author_description` JSON (set by `news_loader.py`) to skip bio classifier for news. `_aggregate_scores()`: weighted average per bloc; language_prior excluded by default.

#### `electoral/nlp/elasticity.py` ✅ IMPLEMENTED (Week 4)
**Purpose:** Maps RoBERTa scores to 9-token delta bins; assembles LLM fine-tuning dataset.

`_BIN_THRESHOLDS`: 9 bins strong_neg (−1, −0.50) through strong_pos (0.50, 1); boundaries are `lo ≤ score < hi`. `score_to_bin(score)`: maps float ∈ [−1, 1] to bin label. `estimate_elasticity(sentiment_data, vote_deltas)`: OLS via `numpy.lstsq`; skips blocs with <3 observations (NaN). `assemble_finetune_dataset()` → `LLMFineTuneData`: writes JSONL where each example is `{"instruction": "Shock event: ...", "output": "Race blocs:\n  african_american: slight_pos\n..."}`.

#### `electoral/nlp/social.py` ✅ IMPLEMENTED (Week 5)
Platform-agnostic collector interface. `SocialCollector` ABC with `collect()` abstract method. Subclasses: `BlueSkyCollector` (AT Protocol firehose + HuggingFace archive download), `ApifyCollector` (X free-tier 500 results/run), `TruthSocialCollector` (returns 403 on unauthenticated requests — no public API), `FacebookCollector`. JSONL envelope schema:
```json
{"schema_version":"1.0","created_at":"ISO8601","stage":"collect","seed":42,
 "payload":{"text":"...","created_at":"...","lang":"...","platform":"..."}}
```

#### `electoral/nlp/collectors/schema.py`
Post schema definitions shared across Bluesky, Apify/X, and Reddit collectors.

#### `electoral/nlp/collectors/bluesky_firehose.py` ✅ IMPLEMENTED
AT Protocol firehose listener with English + political-keyword filtering; writes the canonical post envelope. (Delegated to by `social.BlueSkyCollector`.)

#### `electoral/nlp/collectors/apify_x_scraper.py` ✅ IMPLEMENTED
Apify X (Twitter) free-tier scraper (500 results/run), triggered per shock event. (Delegated to by `social.ApifyCollector`.)

---

### `electoral/markets/`

#### `electoral/markets/collector.py` ✅ IMPLEMENTED
Price poller for Polymarket / PredictIt / Metaculus / Manifold. Contract IDs read from `configs/market_contracts.json` (never resolved live → reproducible). **Calibration display only — never a training input** (a market price is P(win)=Δπ, not the vote-share margin Δμ).

#### `electoral/markets/aggregator.py` ✅ IMPLEMENTED
`volume_weighted_price(...)` → volume-weighted consensus across platforms (equal-weight when a platform reports no volume); wraps the result in `PredictionMarketData` for the `"Market consensus: X%"` display.

---

### `electoral/llm/` — LLM Fine-Tuning

#### `electoral/llm/trainer.py` ✅ IMPLEMENTED
QLoRA trainer (HuggingFace PEFT, 4-bit NF4, rank-16 LoRA) for Mistral 7B; HPC-only deps. `TrainConfig`, `format_prompt`/`format_completion` (fine-tune record → `[INST]…[/INST]` + JSON target, sanitized), `train_hpc`, `_eval_mae`. Writes the adapter + `trainer_state.json`.

#### `electoral/llm/inference.py` ✅ IMPLEMENTED
`ShockEstimator` — constrained decoding via `outlines`, constraining output to the `ShockResponseSchema` (9-token bins per stratum). `.estimate(event_dict, intensity)` → `ShockResponseData`.

#### `electoral/llm/eval.py` ✅ IMPLEMENTED
Evaluation against held-out historical (shock, delta_bin) pairs.

---

### `electoral/api/` — FastAPI Endpoints

#### `electoral/api/shock_endpoint.py` ✅ IMPLEMENTED
The web backend. Loads the LLM once via FastAPI `lifespan`; holds the executor pools on `app.state`. **Concurrency contract:** the CVXPY optimizer runs in a `ProcessPoolExecutor(max_workers=1)` (CVXPY's C solvers are NOT thread-safe — threads segfault); the LLM and Monte Carlo run in a `ThreadPoolExecutor` (GIL released). Routes: `POST /estimate`; `GET /estimate/stream` (SSE: `deltas → equilibrium → simulation → done`, frames sanitized via `sanitize_floats`); `GET /api/market-prior`; the dashboard routes `GET /api/{audit,audit/count,coverage,sentiment-dist,bio-coverage,training-logs,convergence}` (all behind `_require_dashboard_auth`); `GET /health`, `GET /blocs`. The module-level `solve_rebalanced(...)` is a picklable adapter that delegates to `optimization.cvx.solve_rebalanced` (passing the real `fixed_loyalty` computed from nominal religion/gender priors). Auth: HMAC-SHA256 signed session cookie (`_verify_session_token`, constant-time, fails closed).

#### `electoral/api/audit.py` ✅ IMPLEMENTED
`AuditLogger` — append-only estimate log backed by a single DuckDB file (`data/audit.duckdb`), guarded by a `threading.Lock`. `log_estimate(...)` (incl. `party`), `recent(limit, search=None)` (parameterized `ILIKE` search, limit-clamped), `count()`. Non-finite floats stored as NULL; failures swallowed so logging never breaks a request.

---

### `electoral/metrics/` and `electoral/reporting/`

#### `electoral/metrics/performance.py` / `tables.py` ✅ IMPLEMENTED
`performance.py`: `win_probability(sim_data)`, `equilibrium_gap(weights, mu, target)` (= wᵀμ − target; intersects keys so race-only dicts can't leak), `bloc_delta_summary(baseline, rebalanced)` (signed Δ per bloc, sorted by |Δ|) — all JSON-native. `tables.py`: manuscript-ready table builders (e.g. `build_shock_summary_table`).

#### `electoral/reporting/export.py` ✅ IMPLEMENTED
Exports artifacts to CSV / LaTeX / JSON (`export_csv`, `export_latex_table`, `export_json`) for the paper.

---

## `collectors/` — Always-On Collectors (Intel Mac / M5)

> ⚠️ These top-level modules are intentional PLACEHOLDERS (they `raise
> NotImplementedError`). The working collectors live in the package:
> `electoral/nlp/social.py` + `electoral/nlp/collectors/{bluesky_firehose,apify_x_scraper}.py`,
> and Reddit is ingested via `electoral/nlp/archive.py` (`subreddit_proxy`). These
> files are kept only so the original devplan layout resolves.

#### `collectors/bluesky.py` 🔲 PLACEHOLDER → see `electoral/nlp/collectors/bluesky_firehose.py`
#### `collectors/apify.py` 🔲 PLACEHOLDER → see `electoral/nlp/collectors/apify_x_scraper.py`
#### `collectors/reddit.py` 🔲 PLACEHOLDER → Reddit handled in `electoral/nlp/archive.py` (`subreddit_proxy`, excluded from Σ_Δ)

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
| `compile_hailo.py` | Pi | Hailo NPU compilation script — **deferred indefinitely** (SDK gated behind vendor registration; compiler runs on host Mac, not Pi). CPU fallback is active and sufficient. |
| `train_setfit.py` | M5 | Trains three SetFit bio classifiers (race, religion, gender) on `data/bio_labels/labeled_bios.jsonl` (2,191 labeled bios, 80/20 split). Uses new `Trainer` API (not deprecated `SetFitTrainer`). `trainer.evaluate()` is NOT used — replaced with sklearn `f1_score` directly due to matmul overflow in the HF evaluate library. Final macro-F1: race 0.97, religion 0.85, gender 0.93. Models saved to `models/setfit_race/`, `models/setfit_religion/`, `models/setfit_gender/`. |
| `pi_bio_server.py` | M5 (SRP batch phase) / Pi (continuous collection) | FastAPI server for bio inference. `ClassifyRequest(BaseModel)` is at module level (moved from inside `_build_app` — FastAPI requires module-level models for request body recognition). **SRP batch phase:** Running at `http://localhost:9000` on M5 (`configs/base.json: pi_bio_server`, `pi_npu_enabled: false`). Pi server (`http://100.125.90.19:9000`) preserved for post-SRP continuous collection. CPU mode: `all-MiniLM-L6-v2` via sentence-transformers, ~20 ms/bio. |
| `verify_artifacts.py` | M5 | Validates all artifacts in `artifacts/` directory against their schemas. |
| `sensitivity_analysis.py` | M5 | Perturbation analysis on λ₁/λ₂/λ₃ and V_eq. |
| `hpc/discord_download.slurm` | HPC | Download `SaisExperiments/Discord-Unveiled-Compressed` via HuggingFace Hub to Hopper (`/hopper/home/jdamley28/discord/dataset.zst`). |
| `hpc/discord_metadata.slurm` | HPC | Decompress first 100MB of dataset.zst; extract `server_list.txt` with all guild IDs and server names. Depends on discord_download. |
| `hpc/discord_extract.slurm` | HPC | SLURM array job: one task per guild ID in `target_servers.txt`; extracts that server's messages from dataset.zst to `extracted/{guild_id}.json`. |
| `hpc/discord_sample.slurm` | HPC | Filters messages to 2019-01-01–2024-11-30, samples ≤50k per server (seed=42), converts to canonical posts.jsonl schema. Depends on discord_extract array. |
| `hpc/target_servers.txt` | HPC | Placeholder guild ID list organised by demographic category. Fill in after reviewing `server_list.txt` from discord_metadata. |

---

## `configs/` — Configuration Files

| File | What it configures |
|------|--------------------|
| `base.json` | Default `PipelineConfig`. `seed=42`, `party="democrat"`, `target=0.5066` (updated from empirical V_eq derivation). `pi_bio_server="http://localhost:9000"` (M5 localhost; SRP batch phase), `pi_npu_enabled=false`. `data_root="/Volumes/JUHIDRIVE/electoralData/archives/"`, `pi_data_root="/mnt/juhidrive/electoralData/archives/"`. |
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
| `shocks.json` | Catalog of shock events with metadata (date, description, category, keywords, target_blocs, shock_window_days, date_window). Includes `iran_war_2026` (2026-02-28, Geopolitical, target_blocs: muslim/jewish/evangelical/secular) added Week 5 as live-scraping test shock. |
| `archives.json` | Machine-readable registry of all 15 archive datasets: slug, platform, mac/pi paths, tier, inference_method, shock coverage with `window_start`/`shock_date`/`window_end` per shock. Drives `HistoricalArchiveLoader` and SLURM sampling jobs. |

---

## `tests/` — Test Suite

**623 passing, 17 skipped** as of Week 4.

| File | What it tests |
|------|--------------|
| `conftest.py` | Shared fixtures: `LAYER_WEIGHTS`, canonical lists, toy `PipelineConfig`, sample `VoterPanelData` and `BaselinePortfolioData`. |
| `test_config.py` | `PipelineConfig.from_json()`, field validation, `derive_seed` determinism. |
| `test_io.py` | JSON/Parquet round-trips, parent dir creation, dtype preservation. |
| `test_schema.py` | All five `assert_*` helpers with valid and invalid inputs. |
| `test_rng.py` | `make_rng` seeding, `derive_seed` determinism, verify no `np.random` global calls in codebase. |
| `test_data_cleaning.py` | `clean_raw_panel` all 6 steps; `normalize_bloc` alias table; edge cases. |
| `test_loaders.py` | Generic CSV loader; ANES/CES/GSS/NEP/NPORS wrapper behavior. |
| `test_panel.py` | `validate_panel` all 5 invariants; error messages include context. |
| `test_artifact_roundtrip.py` | `to_dict()` → `from_dict()` → `validate()` for all 10 dataclasses. |
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