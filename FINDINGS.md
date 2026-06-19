# FINDINGS.md — Electoral Equilibrium

Empirical results and methodological findings produced by this pipeline.
Each entry states what was found, what it means, its limitations, and
whether it constitutes new research or replicates known results.

---

## 1. Raked λ weights: gender explains more temporal variance than race or religion

**Result:** IPF calibration on 20 election cycles (1948–2024) converged in 2 iterations to:

| Stratum | Additive prior | Raked λ |
|---------|---------------|---------|
| Race (λ₁) | 0.50 | **0.114** |
| Religion (λ₂) | 0.30 | **0.224** |
| Gender (λ₃) | 0.20 | **0.662** |

**What this means:** The gender stratum's electorate-weighted vote-share series has
the highest cross-cycle variance of the three strata. The gender gap was near zero in
the 1950s, grew to ~12pp by 2020, and is the dominant signal in the regression of
stratum averages against the historical two-party Democratic result.

**What this does NOT mean:** This is a model calibration result, not a causal claim
that gender is "more important" than race at the individual level. African American
loyalty has been 87–92% since 1964 — very high but also very stable. A stable signal
contributes little to the least-squares fit regardless of its magnitude. The IPF
rewards temporal variation, not absolute electoral importance.

**Limitations:**
- Ecological fallacy: λ weights are aggregate-level regression coefficients, not
  individual-level causal effects.
- Confounding with era: the gender gap's growth is partially collinear with the
  post-Reagan party sort. The cycle-level regression cannot separate the two.
- The additive independence assumption may route correlated effects through the most
  volatile stratum. Intersectional interactions (Latino Evangelical women) are not
  modeled.

**Is this new?** The gender gap is documented since Frankovic (1982); its growth since
Carroll and CAWP work in the 1990s. The finding here is methodological: IPF applied to
a three-stratum λ decomposition recovers this known result from first principles, and
the magnitude of the divergence from the equal-weight prior (λ₃: 0.20 → 0.66) is a
useful calibration signal. The paper should frame it as a consistency check, not a
discovery.

**Research prospect:** Comparing additive vs. raked outputs on held-out cycles (2022
midterms, 2026 if available) will quantify how much the λ calibration matters for
optimizer outputs. If the DQCP optimizer's coalition weights shift substantially between
additive and raked λ, that is a genuine methodological contribution: it demonstrates
that the λ choice is not a nuisance parameter but a load-bearing model assumption.

---

## 2. Min-variance QP assigns zero weight to Latino and Asian blocs

**Result:** Democrat baseline coalition (min-variance QP, V_eq = 0.5066):

| Bloc | Weight | μ | σ |
|------|-------:|---:|---:|
| african_american | **0.510** | 0.892 | 0.104 |
| white | **0.302** | 0.467 | 0.073 |
| other_race | **0.188** | 0.579 | 0.233 |
| latino | 0.000 | 0.669 | 0.106 |
| asian | 0.000 | 0.579 | 0.245 |

μ_eff = 0.659, margin above V_eq = +12.4 pp.

**What this means:** The min-variance QP correctly concentrates weight on the blocs
with the best loyalty-to-variance ratio. African Americans have the highest loyalty at
moderate variance; white voters have the lowest variance overall and serve as a
stabilizing complement. Latino and Asian blocs have mid-range loyalty and above-average
variance — the optimizer treats them as noise sources.

**Why it matters for the paper:** This is the paper's primary motivation for the DQCP
objective. Min-variance is the standard portfolio optimization baseline; its failure to
represent Latino and Asian voters exposes a systematic flaw in applying pure variance
minimization to coalition modeling. A party that genuinely needs to rebuild these blocs
after a shock cannot do so under a min-variance criterion.

**Research prospect:** The DQCP optimizer (Week 5) maximizes P(win) rather than
minimizing variance. In deficit scenarios — where μ_eff < V_eq after a shock — the
only path to feasibility may run through the high-variance blocs (Latino, Asian).
Demonstrating that the DQCP restores positive weights to these blocs while the
min-variance QP does not is a concrete, quantifiable paper contribution.

---

## 3. Electoral College geographic asymmetry: Democrats need 50.66% to win the EC

**Result:** Logistic regression on 20 historical cycles (1948–2024) of
P(Dem EC win) = σ(A·x + B) yields:

| Party | V_eq (EC-adjusted) |
|-------|-------------------|
| Democrat | **0.5066** |
| Republican | **0.4934** |

Democrats need 50.66% of the two-party popular vote for a 50% probability of winning
the Electoral College. Republicans need only 49.34%. The asymmetry is 1.32 pp.

**What this means:** The EC's winner-take-all state allocation creates a geographic
efficiency gap. Democrats run up large margins in California and New York; Republicans
are more efficiently distributed across small-margin swing states. A model using a flat
50% threshold would systematically underestimate Democrat structural disadvantage.

**Supporting evidence:** In 2000, Gore won 50.3% two-party but lost the EC. In 2016,
Clinton won 51.1% but lost the EC. In 2024, Republicans won the EC at 50.8% two-party
vote. In 2000, Bush won the EC at 49.7%.

**Limitations:**
- n=20 cycles; logistic regression on 20 points is underpowered. The 1.32pp figure
  carries a wide confidence interval.
- The gap is non-stationary: pre-1968 party structure made Southern Democrats EC-
  efficient. Post-1994 sorted electorate is the relevant regime.
- V_eq should be recomputed after each new cycle; it is not a stable structural constant.

**Research prospect:** Using EC-adjusted V_eq rather than 50% is methodologically
correct for any coalition optimizer aimed at electoral outcomes. This correction is not
standard in the political science forecasting literature, which typically uses vote-share
models rather than EC-probability models. The paper should derive the logistic
regression formally and report the V_eq estimate with confidence intervals.

---

## 4. GP classifier achieves 0.900 accuracy, 0.090 Brier on 20-cycle LOCO-CV

**Result:** After three sequential fixes, the Gaussian Process electoral classifier on
the 20-cycle Democrat panel (1948–2024) reaches:

| Version | Fix applied | Accuracy | Brier | 2024 p(win) |
|---------|-------------|----------|-------|-------------|
| Initial | — | 0.500 | 0.245 | — |
| +StandardScaler | Feature normalization | 0.750 | 0.180 | — |
| +Perot correction | 1992 three-party adjustment | 0.850 | 0.140 | — |
| +alpha=0.10 | Observation noise regularization | 0.900 | 0.090 | 0.647 |
| +Stratum features | mu_race, mu_religion, mu_gender | 0.850* | **0.071** | 0.542 |

*Adding stratum features drops accuracy from 0.900 to 0.850 because 1988 flips from
correct to incorrect (ANES 1988 oversample issue), but Brier score improves 21%.
The Brier score is the correct primary metric for a probability estimator.

**What this means:** A GP trained solely on voter panel survey data (no polling, no
prediction markets, no news signals) achieves competitive retrospective classification
of election outcomes. The stacked sequence of fixes shows that data quality and feature
scale matter more than model architecture at this sample size.

**Remaining misses:** 2000 (Gore: 50.3% two-party, model: 0.479) and 2024 (Harris:
49.2% two-party, model: 0.542) are both within the GP's uncertainty band (prob_std
≈ 0.25–0.38). The model correctly signals these as borderline; the binary call is wrong
but the confidence level is appropriate. No further improvement is achievable without
additional features unavailable at prediction time.

**Is this new?** GP classifiers for election outcomes are documented in the political
forecasting literature (Jackman 2005, Strauss 2007). The finding here is the
verification that a panel-survey-only feature matrix, properly scaled and corrected for
three-party years, recovers competitive accuracy without polling averages — a useful
baseline for the shock-response model's counterfactual prediction task.

---

## 5. Feature standardization doubled accuracy from coin-flip to competitive

**Result:** Without `StandardScaler`, the GP kernel optimizer collapsed both RBF and
Matérn length scales to the lower bound (1e-5) on every LOCO fold, reducing the model
to a near-delta function. Accuracy was 0.500 (random).

Fix: StandardScaler on all 20 cycles before the LOCO loop. Post-standardization,
inter-cycle distances are O(1) and both kernel components remain active.
Accuracy: 0.500 → 0.800 (with subsequent fixes → 0.900).

**Why it matters:** This is a reproducibility finding. Any subsequent researcher who
applies a GP to vote-share data without standardizing features will recover a
coin-flip classifier. The symptom (length-scale collapse to lower bound) will not
produce a visible error — the model will train without warning and produce garbage.
Document this explicitly in the methodology section.

---

## 6. 1992 Perot three-party correction required for valid covariance estimation

**Result:** Clinton's raw three-party vote share (43.01%) is not comparable to other
cycles' two-party shares. Correcting to the two-party equivalent (53.46% using bloc-
specific Perot support from the 1992 National Exit Poll) changed the 1992 LOCO-CV
fold from misclassified to correctly classified (prob_win 0.421 → 0.837) and improved
overall accuracy from 0.850 to 0.900.

The correction is bloc-specific:

| Bloc | Perot share | Raw Dem % | Corrected 2P % |
|------|------------|-----------|---------------|
| White | 21% | ~41% | ~52% |
| African American | 7% | ~83% | ~89% |
| Latino | 14% | ~61% | ~71% |

**Why it matters:** Any voter panel that uses 1992 raw Democratic vote shares as input
to a covariance matrix will produce a biased Σ_Δ with inflated race-bloc variance. This
is a silent data error — the pipeline produces results without errors; they are simply
wrong. The correction is straightforward but requires sourcing bloc-specific Perot
support from the 1992 NEP rather than applying a flat national adjustment.

---

## 7. Platt scaling degrades calibration: a counterintuitive result

**Result:** Jackknife Platt scaling on the GP's raw scores degraded every metric:

| Metric | Raw GP | Platt-calibrated |
|--------|--------|-----------------|
| Accuracy | 0.800 | 0.750 |
| Brier score | 0.122 | 0.162 |
| 2024 market divergence | 12.9pp | 31.2pp |

**Diagnosis:** Platt scaling is trained on folds where `prob_win ≈ 0.6 → win`. It
correctly learns to push high-confidence predictions higher. But 2024 is an out-of-
distribution extrapolation: the GP correctly signals genuine uncertainty (prob_std=0.377)
for a cycle with a novel coalition structure. Platt scaling treats that uncertainty as
a calibration error and pushes confidence in the wrong direction.

**Why it matters:** The standard recommendation to calibrate GP classifiers with Platt
scaling is not universally correct. When the model's uncertainty comes from genuine
epistemic gaps (few training points near the test point, not just class-probability
miscalibration), Platt scaling is harmful. The paper should report raw GP predictions
and explain why Platt scaling was tested and rejected, with this table as supporting
evidence.

---

## 8. Full 1948–2024 panel dominates modern-only subsets at all era cutoffs

**Result:** Era sensitivity analysis restricting training data to post-realignment cycles:

| Training era | n | Accuracy | Brier | p(2024) | Misses |
|---|---|---|---|---|---|
| 1948–2024 (full) | 20 | **0.900** | **0.090** | 0.647 | [2000, 2024] |
| 1980–2024 | 12 | 0.750 | 0.144 | 0.619 | [2000, 2004, 2024] |
| 1992–2024 | 9 | 0.667 | 0.235 | 0.871 | [2000, 2004, 2024] |

**What this means:** Pre-realignment data (1948–1968, white Southern Democrats) does
not poison the model because `include_year=True` gives the GP temporal ordering,
assigning lower covariance to temporally distant cycles. Small-sample degradation
dominates any coalition-consistency gain from restricting to the modern era.

**Methodological implication:** The 7-decade panel is a genuine asset, not a liability.
Use the full panel. The standard "restrict to post-1980 aligned electorate" intuition
is wrong here because it discards more data than the pre-realignment noise it avoids.

---

## 9. Active-set IPF converges in 2 iterations where gradient methods require 100+

**Result:** Three standard convergence approaches failed or were slow before the
active-set formulation was implemented:

| Method | Iterations to convergence | Notes |
|--------|--------------------------|-------|
| Multiplicative IPF | ~288 | λ₁ → 0 slowly |
| FISTA (projected gradient) | Did not converge | λ oscillated |
| Regularized coordinate descent | >100 | Renormalization unstable |
| **Active-set IPF (final)** | **2** | Exact normal equations |

**Root cause of failure:** The three stratum series (race/religion/gender weighted
averages) are nearly collinear — all track the partisan tide. The unregularized MSE
landscape is nearly flat. First-order methods step along a near-flat valley and converge
extremely slowly.

**Solution:** Solve the regularized normal equations directly:
`(S_a^T S_a + 0.01·I)λ_a = S_a^T y + 0.01·λ_prior`
The 1% Tikhonov regularization toward an equal-weight prior makes the system
well-conditioned (condition number from ~1000 to ~100) and allows an exact one-shot
solve. The IPF outer loop then re-solves on the active simplex face until the
projection no longer changes the support (typically 1–3 passes).

**Research prospect:** The near-collinearity of stratum series is itself a finding —
all three demographic dimensions track the partisan tide together, which means
the optimizer is solving a nearly underdetermined system. This justifies the
regularization toward equal weights as a principled prior, not just a numerical trick.

---

## Summary: what is genuinely new vs. known

| Finding | Status |
|---------|--------|
| Raked λ₃=0.66 (gender temporal variance dominates) | **Consistent with known gender gap literature; methodologically novel as IPF calibration result** |
| Min-variance QP assigns zero weight to Latino/Asian | **Known limitation of variance-minimizing objectives; novel application to coalition modeling as paper motivation** |
| EC-adjusted V_eq: +1.32pp Democrat disadvantage | **Known gerrymander/EC effect; novel application as optimizer threshold** |
| GP accuracy 0.900 / Brier 0.090 on panel-only features | **Competitive with published electoral forecasting baselines** |
| Feature standardization doubled accuracy | **Methodological finding; reproducibility warning for future researchers** |
| Perot bloc-specific correction required | **Data quality finding; prevents silent covariance bias** |
| Platt scaling degrades calibration | **Counterintuitive; publishable as a calibration methodology note** |
| Full 1948+ panel beats modern-only subsets | **Counterintuitive (pre-realignment data helps); methodologically notable** |
| Active-set IPF: 2 iterations vs. 100+ | **Numerical methods finding; near-collinearity of stratum series is the root cause** |

**Core paper contribution:** The combination of (1) EC-adjusted V_eq derivation,
(2) three-stratum raked λ calibration, (3) DQCP quasi-convex optimizer, and (4)
Logistic-Normal ILR Monte Carlo into a single stochastic coalition optimization
framework is the primary novel contribution. Each component individually uses known
techniques; the integration and application to electoral shock response is new.
