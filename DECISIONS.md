# DECISIONS.md — Electoral Equilibrium

Log of every architectural decision, constraint, and non-obvious choice.
Update this file whenever you make a decision that would confuse a future reader.
Format: date, decision, and reason.

---

## Demographic Architecture

**[2026-06] Parallel three-stratum architecture (not nested layers)**
Race, Religion, and Gender are three independent parallel strata, not nested.
Each stratum independently covers ~100% of the electorate using only marginal tables.
Reason: nested architecture (race → religion → gender) required race × religion × gender
cross-tabulations, producing sparse cells (Asian Muslim: ~0.1% of electorate) that
contaminate the covariance matrix via imputation. Parallel strata need only marginal tables.

**[2026-06] Additive independence is acknowledged as an approximation**
mu_eff = λ₁Σw_i·μ_race + λ₂Σv_R·μ_rel + λ₃Σg_G·μ_gen assumes demographic identities
contribute independently. This is the ecological fallacy (Robinson, 1950). A Latino
Evangelical's behavior is not perfectly reconstructed by summing the Latino, Evangelical,
and male marginals. Theoretically correct alternative is MRP (Multilevel Regression with
Poststratification). MRP deferred because it requires full joint cross-tab data and
individual-level survey responses not available in the 7-week SRP timeline.
Paper must state this assumption explicitly and test it on held-out cycles.

**[2026-06] Raking as optional calibration (not MRP)**
raking.py implements iterative proportional fitting across all three marginal tables.
It enforces joint consistency without requiring full joint distribution.
Paper must compare additive vs raked outputs. Large divergence = important finding
about where intersectional interactions matter most.

**[2026-06] White voters and Men added as explicit strata cells**
Previous designs treated these as implicit. White: non-Hispanic White only (~62%).
Men: explicit alongside Women and Other. Both carry their own loyalty estimates.

**[2026-06] Five race blocs (added Other)**
african_american, latino, asian, white, other_race.
"Other" absorbs multiracial and unclassified voters (~10% of electorate).

**[2026-06] Seven religion groups (added Other)**
evangelical, catholic, protestant, secular, jewish, muslim, other_rel.
"Other" absorbs all remaining (~13% of electorate).

---

## Optimization

**[2026-06] Maximize P(win) not minimize variance (DQCP)**
min-variance is wrong for campaigns at a structural deficit — it flees from the only
mechanism (high-variance bets) that could produce a tail-event victory.
Correct objective: max Φ((μ_eff(w) - V_eq) / sqrt(λ₁²·w^T·Σ_Δ·w))
This is quasi-convex (Sharpe-ratio form). Proof: numerator is linear in w, denominator
is convex and strictly positive under LedoitWolf. Ratio of linear to positive convex
is quasi-convex (Diamond & Boyd, DQCP 2019, arXiv:1905.00562).
Solved via CVXPY DQCP: problem.solve(qcp=True). Assert problem.is_dcp(qcp=True) == True.
Quasiconvexity proof is a required methodology section deliverable in the paper.

**[2026-06] V_eq is empirically derived from voter panel, not hardcoded**
Computed by build_constraint_spec: for each cycle where party won, compute
S_c = Σ w_i^(c) μ_i^eff,(c), then V_eq = mean(S_c) across winning cycles.
Expected range: ~0.52-0.53 for Democrats, ~0.49-0.51 for Republicans.
Republican range is lower because Republicans won (2000, 2016) while losing popular vote.
Stored in configs/party_config.json. Recomputed when new panel data is added.

**[2026-06] 5×5 covariance matrix (race blocs only)**
Σ_Δ is derived from historical cycle-to-cycle variance in μ_i^race.
Religion and gender do NOT contribute to the covariance matrix.
LedoitWolf shrinkage applied to guarantee positive-definiteness at n=6-8 cycles.
This is regularization not signal synthesis — paper must acknowledge small-n limitation.

---

## Monte Carlo

**[2026-06] Logistic-Normal ILR (not Dirichlet, not delta method)**
Dirichlet: off-diagonal covariances always negative (Cov(w_i,w_j) = -α_iα_j/(α_0²(α_0+1)) < 0).
Cannot model wave elections where blocs move in the same direction simultaneously.
Delta method floor (w_i >= 0.01): distorts Aitchison geometry non-linearly near simplex
boundaries. A 0.01 perturbation in probability space is NOT 0.01 in log-ratio space.

Correct approach: ILR (isometric log-ratio) with Helmert contrast matrix.
ILR coordinates are orthonormal, well-conditioned throughout simplex interior.
Zero-weight blocs → report as infeasible_bloc, never floor.

**[2026-06] 90% CI band displayed on WinGauge**
win_probability_low = 5th percentile, win_probability_high = 95th percentile.
"No feasible path" fires when upper bound cannot reach V_eq.
"Uncertain path" fires when lower bound < V_eq <= upper bound.

---

## LLM and Fine-Tuning

**[2026-06] Mistral 7B selected for fine-tuning**
Smallest model that reliably handles nuanced political/religious language.
QLoRA: 4-bit NF4 quantization + low-rank adaptation (rank 16, α 32).
Constrained decoding via outlines library, 9-token vocabulary per stratum.

**[2026-06] Cleaning model: local open-weight model, NOT Gemini API**
Gemini Pro via API undergoes continuous silent weight updates. Non-deterministic
outputs break the downstream seed contract. Cleaning output feeds directly into
RoBERTa scorer and fine-tuning dataset — any non-determinism propagates everywhere.
Use: Qwen2.5-7B-Instruct or Mistral-7B-Instruct-v0.3 via mlx_lm on M5.
Record the exact HuggingFace revision hash here when selected:
  Model: [TO BE FILLED IN WEEK 0]
  Revision hash: [TO BE FILLED IN WEEK 0]

**[2026-06] Synthetic data: Gemini 2.5 Pro for generation, local model for cleaning**
Gemini 2.5 Pro used only for synthetic shock scenario generation (long-context, one-time).
Three diagnostics required before accepting any synthetic batch:
  1. MMD (RBF kernel, λ = 1/(2σ²) via median heuristic — NOT bootstrap)
  2. PCA alignment (>70% of synthetic variance explained by top-2 real PCs)
  3. PCD (pairwise correlation difference, Frobenius norm)
Soft MMD weight: w = 0.5 × exp(-λ·MMD²). Hard rejection gate prohibited.
λ stored in configs/mmd_config.json. Bootstrap unstable at N<30 events in RKHS.

**[2026-06] MMD λ: fixed theoretical constant, not bootstrap-estimated**
With N<30 historical events, bootstrapping λ in infinite-dimensional RKHS produces
extreme variance. Use median heuristic: σ² = median(||x_i - x_j||²) over real data pairs,
λ = 1/(2σ²). Deterministic, interpretable, no bootstrap required.

---

## Data Sources

**[2026-06] Machine role swap — Intel Mac takes over news scraping from Windows**
Windows laptop not currently available. Intel Mac (always-on) now runs the news
scraper (Christianity Today, CBN, Univision, NYT, WaPo, Fox) as a launchd daemon.
This was previously the Windows laptop's role. M5 takes over Bluesky collection.
Update if/when Windows laptop becomes available.

**[2026-06] Bluesky: pre-existing HuggingFace datasets preferred over scraping from scratch**
Multiple research groups have archived Bluesky firehose data to HuggingFace datasets
(AT Protocol is public and openly archived). Downloading existing datasets is faster,
covers more historical events, and avoids re-collecting what others have already done.
Strategy: (1) search HuggingFace for "bluesky" datasets at project start, download
relevant ones covering 2023-present; (2) run live AT Protocol firehose from M5 going
forward for events not covered by archives. The live firehose runs via the atproto
Python library. Both feed into rawdata/social/bluesky/ with archive/ and live/ subdirs.
Bluesky demographic note: far-left, post-Twitter migration, chronically online —
NOT broadly secular/diverse. This is a known, narrow demographic slice.

**[2026-06] Prediction markets: calibration benchmark ONLY, not training input**
Δπ (win probability change) ≠ Δμ (vote-share margin change).
A Polymarket price aggregates traders' estimates of national electorate behavior —
second-order forecasting, not first-order ideological sentiment.
Using Δπ to supervise a Δμ estimator overstates demographic elasticity.
Markets used for: (1) post-hoc calibration audit, (2) live display in app.
prediction_market block retained in schema as metadata only.

**[2026-06] Bluesky: far-left chronically online, not broadly secular/diverse**
Bluesky is a post-Twitter migration platform dominated by far-left politically
activated users. NOT a good proxy for "secular/younger voters" broadly.
More accurate: far-left mirror image of Truth Social.

**[2026-06] Language fallback: validation only, exclude from mean AND covariance**
Including language-prior posts in mean estimation while excluding from covariance
bifurcates the data generating process — μ and Σ would come from different corpora,
violating the mean-variance optimizer's foundational assumption.
All language-prior posts: inference_method = "language_prior".
Excluded from both μ and Σ_Δ estimation. Used as held-out validation set.

**[2026-06] Reddit: subreddit-proxy posts excluded from Σ_Δ covariance**
Server/subreddit-level proxy forces artificial within-community correlation.
Same logic as language-prior exclusion.
inference_method = "subreddit_proxy" for Reddit posts.
inference_method = "server_proxy" for Discord posts.

**[2026-06] Discord-Unveiled-Compressed added as young male signal**
Dataset: SaisExperiments/Discord-Unveiled-Compressed (HuggingFace)
Paper: arXiv:2502.00627
2.05B messages, 3,167 public servers, 2015-2024.
Collected via Discord public API — publishable, no ToS concerns.
Filter to politically relevant servers before downloading (do NOT download full corpus).
Demographic proxy: young male voters across racial/religious blocs.

**[2026-06] Prediction market post-2012 data only**
IEM (pre-2012) has position limits incompatible with modern platform liquidity.
Naive aggregation across eras introduces microstructure noise.
Polymarket: 2020-present. PredictIt: 2014-present. Metaculus: 2015-present.

---

## Infrastructure

**[2026-06] Tailscale for Pi connectivity (not mDNS)**
CMC enterprise WiFi blocks mDNS/Bonjour at wireless controller level.
pi.local will NOT resolve. Static IP on DHCP equally unreliable.
Tailscale installed in Week 0. Pi accessible at stable 100.x.x.x Tailscale IP.
Update configs/base.json: "pi_bio_server": "http://100.x.x.x:9000"
Pi Tailscale IP: [TO BE FILLED IN WEEK 0]

**[2026-06] PostgreSQL (Supabase) for serverless write sink, DuckDB read-only for dashboard**
Modal spins multiple isolated containers under load. DuckDB file-lock collisions
across containers drop audit records. Two-tier architecture:
  - Live writes: PostgreSQL (Supabase free tier, 500MB)
  - Analyst dashboard: DuckDB read-only, querying exports from PostgreSQL
asyncio.Queue single-writer prevents intra-process locks only — not cross-container.

**[2026-06] ProcessPoolExecutor(max_workers=1) for CVXPY**
CVXPY's C solvers (SCS, ECOS, OSQP) are not thread-safe. Concurrent ThreadPoolExecutor
requests corrupt solver state → segfaults. Use isolated subprocess.

**[2026-06] CVXPY DQCP explicit declaration required**
problem.solve(qcp=True) is mandatory.
assert problem.is_dcp(qcp=True) must be a unit test — not optional.
Without DQCP, quasi-convex objective will silently return local/wrong optima.

**[2026-06] Syncthing on local machines only (NOT HPC)**
HPC compute nodes are ephemeral — allocated for job, reclaimed when done.
Syncthing daemons killed on reclaim, sync state lost.
HPC data transfer: explicit rsync before each SLURM job submission.

---

## Paper Requirements

The methodology section must include:
1. Formal quasiconvexity proof for the DQCP objective
2. Explicit additive independence assumption statement + empirical fit test on held-out cycles
3. Additive vs raked model comparison and discussion of divergence
4. MMD + PCA alignment + PCD synthetic data validation report
5. Ecological fallacy acknowledgment + MRP deferred rationale
6. Social media as weak supervisory signal — not median voter proxy
7. Prediction market calibration methodology

---

## Open Items (fill in during Week 0)

- [ ] Pi Tailscale IP confirmed: ___________
- [ ] Cleaning model selected and revision hash recorded: ___________
- [ ] V_eq thresholds computed from panel and written to configs/party_config.json
- [ ] λ₁, λ₂, λ₃ initial estimates written to configs/layer_weights.json
- [ ] MMD λ computed from real data median heuristic, written to configs/mmd_config.json
- [ ] HuggingFace revision hash for cleaning model recorded above
- [ ] Meta Content Library application submitted
- [ ] Reddit API OAuth credentials obtained
- [ ] Syncthing confirmed running on M5, Intel Mac, Windows, Pi
- [ ] ARDA/GSS/NEP cross-tab data confirmed accessible at race × religion × gender marginal level
