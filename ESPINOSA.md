# Questions for Prof. Espinosa — Electoral Equilibrium SRP

**Project:** Electoral Equilibrium — Stochastic Coalition Optimization  
**Student:** Juhi Damley  
**Supervisor:** Prof. Gaston Espinosa, Claremont McKenna College  
**Document updated:** 2026-06-10 (Week 5, Day 1)

Questions are grouped by topic and tagged with urgency: **[Week 2]** must be resolved before the next sprint begins, **[Paper]** are methodology decisions that must appear explicitly in the paper, and **[Open]** can be discussed whenever.

---

## 1. Data Quality — Spot-Check Findings

These questions arise directly from `rawdata/spot_check.md`, which cross-checked 10 panel cells against raw source documents.

### Q1.1 — ANES 2020 missing survey weights [Week 2]
The ANES 2020 subset shipped in our labeled parquet has a fully null `weight` column (8,280 rows, zero non-null weights). The pipeline falls back to equal-weighting, and unweighted ANES 2020 produces a white Dem vote share of 51.7% vs. the NEP exit-poll gold standard of 41%. The merged panel value for white 2020 is 44.6% — 3.6 pp above NEP.

**Question:** Should ANES 2020 be excluded entirely from the race/gender merge for 2020 (since no survey weights = SE undefined), or is there a correct ANES 2020 post-stratification weight file we should be using? The ANES 2020 dataset exists in two forms (fresh cross-section + panel re-interview); the labeled subset may have been built from the panel re-interview wave, which has different weight variable names.

### Q1.2 — Systematic ~4–5 pp ANES–NEP gap for white voters [Paper]
ANES weighted white Dem share in 2016: 41.7% vs. NEP 37%. This pattern likely holds across cycles. ANES is a pre/post-election panel that over-represents civically engaged participants; NEP captures at-the-polling-place voters. The inverse-SE merge moderates toward ~40%, but the gap is never fully closed.

**Question:** Do you want the final paper to use NEP as the primary source for race blocs in cycles where NEP is available, treating ANES/CES as supporting evidence? Or should the merged value stand with an explicit "survey/exit-poll divergence" limitation note?

### Q1.3 — CES 2024 understates African American Dem support [Week 2]
CES 2024 African American Dem share: 81.9% vs. NEP 2024: 86%. The pipeline merged value is 83.8% — 2.2 pp below the exit poll. This appears to be a known post-election CES recall artifact where Black respondents slightly underreport Democratic presidential vote.

**Question:** Is the 4 pp CES–NEP gap within an acceptable range for this project, or should we add a cycle-specific correction or use NEP-only for 2024 african_american?

### Q1.4 — NEP 2016 duplicate Protestant rows [Open]
NEP 2016 has two rows under `category=religion, sub_category=protestant` with different sub_pct (27% and 52%) and different dem_pct (36% and 39%). The current pipeline averages them, producing a merged 37.5% Dem share. The 27% row likely refers to White Evangelical/Born-Again Protestant; the 52% row to all Protestants.

**Question:** Is it valid to conflate these two rows into a single `protestant` bloc average, or should the 27% row be routed to `evangelical` and the 52% row to `protestant`? That would affect the relative weight of the evangelical vs. protestant blocs in 2016.

---

## 2. Methodology — Demographic Architecture

### Q2.1 — Ecological fallacy disclosure [Paper]
DECISIONS.md §Demographic Architecture acknowledges that `mu_eff = λ₁Σw_i·μ_race + λ₂Σv_R·μ_rel + λ₃Σg_G·μ_gen` assumes demographic identities contribute additively and independently (Robinson 1950 ecological fallacy). A Latino Evangelical man's behavior is not reconstructable from the three separate marginals.

**Question:** Should we include a held-out test where we simulate the additive model against a known cross-tabulated source (e.g. NEP race × gender) to bound the magnitude of the error? If the additive model produces a bias of <2 pp on held-out cells, that's a publishable robustness check. If not, MRP becomes necessary.

### Q2.2 — Layer weights λ₁/λ₂/λ₃ are placeholder values [Week 2]
`configs/layer_weights.json` currently has `λ₁=0.50, λ₂=0.30, λ₃=0.20` with the note "Calibrated by build_constraint_spec from historical NEP data." These are not yet empirically derived — `build_constraint_spec` is not yet implemented.

**Question:** For Week 2 baseline, should we use the placeholder weights (50/30/20) as a working assumption, or does the paper need these to be computed before the baseline results are meaningful? What is the intended calibration procedure for λ — ordinary least squares regressing historical mu_eff against actual election outcomes?

### Q2.3 — V_eq empirical derivation [Week 2]
DECISIONS.md §Optimization says V_eq is derived as the mean of `Σ w_i^(c) μ_i^eff,(c)` across winning cycles. `configs/base.json` currently hardcodes `target=0.535` for Democrats.

**Question:** How many winning cycles should enter the V_eq average? Using all cycles where the party won (e.g. 1992, 1996, 2008, 2012, 2020 for Democrats) vs. just the most recent 3 would produce meaningfully different V_eq values. Which window does the devplan specify?

### Q2.4 — "Evangelical" is both a race-stratum alias and a religion bloc [Open]
In `data.py`, NEP rows with `sub_category="white evangelical"` are routed to `evangelical` (religion stratum), not race. But in the voter panel, `evangelical` can only have vote_share estimates from sources that include religion questions (CES, GSS). ANES does not have evangelical as a category.

**Question:** Is the coverage gap for the evangelical bloc (missing from ~15 of 20 cycles) acceptable for the paper, or do we need to back-fill using NEP evangelicals as the primary source with CES/GSS for post-2008 cycles?

---

## 3. Methodology — Optimization

### Q3.1 — DQCP quasiconvexity proof as paper deliverable [Paper]
DECISIONS.md §Optimization states: "Quasiconvexity proof is a required methodology section deliverable in the paper." The proof rests on: numerator `μ̃_eff(w) - V_eq` is linear in w, denominator `sqrt(λ₁² wᵀΣ_Δw)` is convex and positive under LedoitWolf → ratio is quasi-convex per Diamond & Boyd (2019).

**Question:** Is it sufficient to cite Diamond & Boyd (2019, arXiv:1905.00562) and state the Sharpe-ratio quasi-convexity result, or does the paper need an original algebraic derivation showing that our specific formulation satisfies the DQCP conditions? Are there reviewers likely to question this?

### Q3.2 — Feasibility when mu_eff is already above V_eq [Open]
The optimizer maximizes P(win). If the current mu_eff already exceeds V_eq (incumbent party in good standing), the optimizer trivially succeeds with equal weights. The paper needs to distinguish between:
- Party already above V_eq, shock pushes it below: rebalancing required
- Party already below V_eq (structural deficit): rebalancing required
- Party above V_eq, shock insufficient to threaten it: no rebalancing needed

**Question:** Should the optimizer always run and report the optimal weights regardless of starting position, or should it short-circuit and return baseline weights when `mu_eff_pre_shock > V_eq + shock_delta_eff`? What is the correct display behavior on the web app in this case?

### Q3.3 — Race-only covariance matrix Σ_Δ [Paper]
DECISIONS.md states: "Religion and gender do NOT contribute to the covariance matrix." The 5×5 Σ_Δ is estimated from historical cycle-to-cycle variance in `μ_i^race` only. With n=6–8 observed cycles, LedoitWolf shrinkage is applied.

**Question:** LedoitWolf shrinkage on a 5×5 matrix from n=6–8 cycles gives a well-conditioned matrix, but the shrinkage intensity will be very high (close to the identity). The paper will likely be questioned on this: are we estimating Σ_Δ or constructing it? Should we report the average shrinkage coefficient (ρ) as a transparency metric, and if ρ > 0.8, discuss what that implies for the optimizer's behavior?

---

## 4. Methodology — Monte Carlo

### Q4.1 — ILR back-transform and zero weights [Paper]
DECISIONS.md mandates: "Zero-weight blocs → report as infeasible_bloc, never floor to 0.01." With n=10,000 draws from a Logistic-Normal, some draws will place negligible weight on small blocs (asian ~5%, other_race ~10%). The `infeasible_bloc` flag fires when a simulated weight is exactly zero, which is probability zero under a continuous distribution.

**Question:** What is the correct numerical threshold for `infeasible_bloc`? Suggested: flag draws where any `w_i < 1e-4` (i.e., below rounding to 4 significant figures). Is there a more principled threshold tied to the survey-implied 95% lower bound on bloc size?

### Q4.2 — "No feasible path" display logic [Open]
DECISIONS.md §Monte Carlo says: "'No feasible path' fires when upper bound cannot reach V_eq" — i.e., win_probability_high < V_eq. But `SimulationData` in `artifacts.py` only stores `win_probability` (point estimate), `percentiles` per bloc, and no `win_probability_low/high` fields (those fields were described in the devplan but not yet in the dataclass).

**Question:** Should `SimulationData` be extended to include `win_probability_low` and `win_probability_high` as explicit float fields, or are they computed from `percentiles` at display time? If computed at display time, what is the mapping from bloc-level percentiles to the scalar CI bounds shown on the WinGauge?

---

## 5. Methodology — LLM and NLP

### Q5.1 — Training data size for QLoRA fine-tuning [Week 2]
The LLM fine-tuning dataset will consist of historical (shock, delta_bin) pairs assembled from RoBERTa-scored social media + news posts. For a 7-week SRP, the realistic number of training examples is ~500–2,000.

**Question:** Is 500–2,000 fine-tuning examples sufficient for meaningful QLoRA adaptation on Mistral 7B for this specific task (9-token bin prediction per stratum)? Should we plan for synthetic data augmentation (`generate_synthetic.py`) to supplement the real examples, and if so, what is the target ratio of real to synthetic?

### Q5.2 — MMD validation threshold [Open]
`configs/mmd_config.json` defines the acceptance threshold for synthetic data quality via Maximum Mean Discrepancy. The current file exists but the threshold values have not been set from real data.

**Question:** What MMD threshold is appropriate for accepting synthetic examples into the fine-tuning set? The devplan mentions a "soft gate" but does not specify the threshold. Should we set it empirically from a pilot run, or do you have a prior from similar NLP fine-tuning work?

### Q5.3 — Language fallback rule and held-out set composition [Paper]
DECISIONS.md says posts assigned via `inference_method: "language_prior"` are excluded from both mean μ and covariance Σ_Δ estimation, and used only as a held-out validation set. This bifurcates the dataset by inference quality.

**Question:** Is using language-prior posts as validation correct if they are also the posts where we have least confidence in the demographic assignment? A model validated only on low-confidence assignments may look good while failing on the high-confidence training examples. Should we instead hold out a random stratified sample across all inference methods?

---

## 6. Data Sources and Collection

### Q6.1 — Evangelical coverage gap (no ANES evangelical data) [Week 2]
The evangelical bloc (religion stratum, ~24% of electorate) has no data before 2008 in our panel because ANES does not report evangelical as a distinct category (it collapses into "Protestant"). NEP begins reporting evangelical shares starting with 2004.

**Question:** For pre-2004 cycles where evangelical data is unavailable, should we impute using a regression on available protestant data (assuming evangelical ≈ f(protestant)), use the language-prior fallback, or simply mark those cells as missing and train the LLM only on 2004+ data?

### Q6.2 — VOTER Panel (Democracy Fund) usage [Open]
A VOTER Panel CSV is present at `data/surveys/VOTER Panel Data Files/voter_panel.csv`, and a loader exists (`load_gallup`). However, the VOTER Panel is a longitudinal panel tracking the same respondents over time (not a repeated cross-section), which means cycle-to-cycle correlated errors within-respondent would violate the independence assumption in the Kish effective-sample-size formula.

**Question:** Should the VOTER Panel be used to estimate vote_share point estimates (using the Kish formula), or only to estimate within-respondent volatility (i.e., how individual-level vote intention changes across waves)? The current kernel treats it as a cross-section, which may be technically incorrect.

### Q6.3 — Reddit `inference_method: "subreddit_proxy"` validity [Paper]
DECISIONS.md states Reddit posts are classified using "subreddit_proxy" (r/Catholicism → catholic, r/BlackPeopleTwitter → african_american, etc.) and excluded from Σ_Δ covariance estimation. The proxy assumes subreddit membership ≈ demographic membership, which is not validated.

**Question:** Is subreddit proxy valid enough to include in the μ point estimates (but not covariance), or should it be treated as a weak prior only — useful for direction of effect but not magnitude? The paper needs to take a clear position on this.

### Q6.4 — BLM Twitter coverage dropped; Reddit Pushshift is sole social source [Paper]
All Twitter dataset candidates for `blm_george_floyd_2020` were dropped (2026-06-03): echen102 and SMAPP NYU are IDs-only (API-gated). Reddit Pushshift (r/BlackLivesMatter, r/BlackPeopleTwitter, r/BlackPeopleofReddit, filtered May–Jul 2020) is now the only social media source for this shock.

**Question:** Is Reddit-only coverage acceptable for the BLM/George Floyd shock in the paper? The subreddit communities are high-signal for African American sentiment, but the hashtag spread across demographics on Twitter in ways Reddit does not capture. Does relying on community-membership-as-proxy limit the external validity of the coalition delta estimate for this event?

### Q6.5 — Discord data via SaisExperiments/Discord-Unveiled-Compressed [Paper]
SLURM pipeline written (2026-06-03) to download and process the Discord-Unveiled dataset from HuggingFace on Hopper. Discord political and identity servers could provide subreddit-proxy-style signal for evangelical, conservative, catholic, and progressive blocs beyond what Reddit covers.

**Question:** Is Discord data methodologically defensible in the paper? Discord is semi-private (invite-gated) and communities differ structurally from Reddit (lower public accountability, more ideologically homogeneous per server, no user bio field). Should Discord be Tier 2 (subreddit_proxy equivalent) included in μ estimates, or restricted to qualitative validation only?

---

## 7. Compute Infrastructure

### Q7.1 — Pi Tailscale IP ~~not yet recorded~~ **RESOLVED (2026-06-03), UPDATED (2026-06-10)**
~~`configs/base.json` has `pi_bio_server: "http://100.x.x.x:9000"` — a placeholder IP.~~

**Resolved (2026-06-03):** Bio server confirmed running at `http://100.125.90.19:9000/health` in CPU fallback
mode. Response: `{"status":"ok","mode":"cpu","inference_backend":"cpu","model":"all-MiniLM-L6-v2"}`.
Static IP recorded in `configs/base.json`. Hailo NPU compilation deferred indefinitely — SDK
gated behind vendor registration; CPU at ~20 ms/bio is sufficient for the SRP timeline.

**Updated (2026-06-10):** SetFit training completed on M5. For the SRP batch processing phase,
the bio server is now running locally on M5 at `http://localhost:9000` (`pi_npu_enabled=false`).
The Pi server (`100.125.90.19:9000`) is preserved for post-SRP continuous collection when M5 may
not be running, but `configs/base.json` now points to localhost for all current batch jobs.
No question outstanding.

### Q7.2 — HPC access for QLoRA fine-tuning [Week 2]
`scripts/hpc_submit.sh` and `scripts/score_array.sh` target CMC's HPC cluster with A100 GPUs. Week 4 of the devplan requires QLoRA fine-tuning of Mistral 7B.

**Question:** Has HPC access been provisioned? If the account/project allocation is not yet active, the fine-tuning timeline could slip. Should we test the training loop on the M5 (lower batch size, fewer steps) to verify the code before submitting the full run to HPC?

---

## 8. Paper / Publication

### Q8.1 — Scope of quasiconvexity contribution [Paper]
The DQCP optimizer applying max P(win) to voter coalition rebalancing appears to be novel — no prior political science work applies quasi-convex portfolio optimization to demographic coalition weights. The devplan describes this as a paper deliverable.

**Question:** What venue is the target for the SRP paper — a CMC senior thesis submission only, or a conference/journal (e.g., Political Analysis, APSA)? The depth of the quasiconvexity proof section would change substantially depending on the audience's mathematical background.

### Q8.2 — Comparison baseline [Paper]
The optimizer is evaluated against some baseline. The devplan mentions "equal-weight" and "value-weight" baselines (`portfolios/weights.py`). These are trivial. A more meaningful baseline would be the historical winning coalition weights from NEP exit polls (i.e., how did the party actually allocate resources, measured by actual vote-share outcomes).

**Question:** Should the comparison use (a) equal-weight, (b) the last-cycle's actual NEP-derived weights, or (c) both? For reviewers, comparison against only equal-weight would be weak.

### Q8.3 — Prediction market calibration as a paper result [Paper]
`PredictionMarketData` and `markets/` exist to compare model win-probability against Polymarket/PredictIt prices post-hoc. DECISIONS.md is explicit that market prices are calibration benchmarks only, not training inputs.

**Question:** Is the market calibration result intended to be a named section or just a footnote in the paper? If it's a section, we need at least 3–5 resolved shock events where both the model output and the market price are available. Do we have or expect to collect that many events during the SRP?

---

## 9. Open Items from DECISIONS.md (not yet resolved)

The following items remain unchecked in `DECISIONS.md §Open Items`:

| Item | Status | Urgency |
|------|--------|---------|
| Pi Tailscale IP confirmed | **CONFIRMED: 100.125.90.19:9000 (CPU mode, Week 4). Updated Week 5: bio server moved to M5 localhost:9000 for SRP batch phase.** | Done |
| Cleaning model revision hash | Not recorded | Week 2 |
| V_eq computed from panel → party_config.json | Not done (placeholder 0.535) | Week 2 |
| λ₁, λ₂, λ₃ computed from data → layer_weights.json | Not done (placeholder 50/30/20) | Week 2 |
| MMD λ computed from real data | Not done | Week 3 |
| bin_uncertainty.json σ values populated | Not done | Week 3 |
| CES 2024 `CC24_410` presidential vote column verified | Needs codebook check | Week 2 |
| VOTER Panel race/religion numeric codes per-wave | Not decoded | Week 2 |
| NEP sub_category → canonical bloc lookup table | Built in code (cleaning.py) | Done |
| Meta Content Library application | Unknown status | Week 3 |
| Reddit API OAuth credentials | Unknown status | Week 3 |
| Syncthing running on all machines | **REPLACED (2026-06-03):** Syncthing had silent conflicts and multi-hour stalls across three OS types. Switched to external HD (4TB WD) + rsync over SSH for HPC. | Done |
| configs/archives.json — 15-archive machine-readable registry | Written 2026-06-03; drives HistoricalArchiveLoader and SLURM sampling | Done |
| Discord SLURM pipeline (4 scripts on Hopper) | Written 2026-06-03; awaiting HPC account confirmation before submitting | Week 4 |
| HPC Hopper account (jdamley28) active? | Scripts ready; confirm account + allocation before discord_download submit | Week 4 |
| BLM Twitter coverage dropped — Reddit sole source | Decision logged 2026-06-03; see Q6.4 | Paper |
| CSES dataset (657 MB) found on JUHIDRIVE | Leave in place; post-SRP cross-national V_eq calibration only | Post-SRP |
| TikTok Whisper transcripts (tiktok_2024, 1.73M rows) | Tier 2 assigned, platform proxy; no shock-window filtering possible | Week 5 |
| SetFit bio classifier training complete | **RESOLVED 2026-06-10.** Race 0.97, religion 0.85, gender 0.93 macro-F1 on 2,191 labeled bios (80/20 split). Models saved to `models/setfit_race/`, `models/setfit_religion/`, `models/setfit_gender/`. `trainer.evaluate()` removed — HF evaluate library produces 0.0 (matmul overflow in LR head); sklearn `f1_score` used directly. | Done |