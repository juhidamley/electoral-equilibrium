# Per-Platform Sentiment–Favorability Correlations

**Purpose**: document the Pearson correlation between each platform's baseline-adjusted RoBERTa sentiment scores and 14-day lagged favorability deltas (from survey/panel data), and record the down-weighting or exclusion decisions that follow.

Prediction markets are calibration benchmarks only and are not included here.

---

## Methodology

### Sentiment scores (X)

For each platform and each shock event, `score_posts_for_shock()` produces a per-bloc sentiment score ∈ [−1, 1] (P(pos) − P(neg) under the RoBERTa classifier). Scores are baseline-adjusted: the pre-shock window score is subtracted so the resulting Δ sentiment is shock-specific.

Platforms in scope: `bluesky`, `apify_x`, `reddit`, `3dlnews`, `webhose`.

### Favorability deltas (Y)

14-day lagged favorability is taken from the survey panel (ANES/CES/GSS/YouGov), averaging the two-week window following each shock date. The baseline is the 30-day pre-shock average. Each (shock, bloc) pair produces one scalar Δ favorability value.

### Correlation

For each platform and each bloc, Pearson r is computed across all shocks where both Δ sentiment and Δ favorability are available. A minimum of 5 (shock, bloc) pairs is required; otherwise the entry is marked as insufficient data.

**Near-zero threshold**: |r| < 0.10 triggers a down-weighting or exclusion review. The decision is documented in the table below and encoded in `fit_elasticity()` via the `alpha_grid` or by dropping the platform column.

---

## Existing exclusions (already encoded in codebase)

These decisions predate the correlation computation and are grounded in methodological constraints rather than empirical correlation:

| Platform / inference method | Decision | Reason | Where encoded |
|---|---|---|---|
| `subreddit_proxy` (Reddit) | Excluded from Σ_Δ covariance | Subreddit membership is a group-level proxy, not individual bio inference; including in covariance conflates group and individual-level effects | `bio_classifier.py` `is_estimable()`, `_aggregate_scores()` |
| `language_prior` (any platform) | Excluded from μ **and** Σ_Δ | Bifurcates the data-generating process: μ and Σ would come from different corpora, violating optimizer mean-variance assumptions | `bio_classifier.py` `is_estimable()` |
| `platform_proxy` (3dlnews, Discord) | Excluded from Σ_Δ | State-level geographic proxy only; no individual bio signal; cross-sectional ecological fallacy if included in covariance | `sample_archives.py` `inference_method` field |
| Discord (invite-gated) | Pending methodological review | Self-selected community membership creates selection bias incompatible with representative electoral assumptions | `ESPINOSA.md` Q6.5 |

---

## Per-platform correlation table

*To be filled after first scoring run. Run `scripts/compute_platform_correlations.py` once rawdata/sampled/ is populated.*

| Platform | Blocs with r ≥ 0.10 | Median |r| | Min |r| | Max |r| | Decision |
|---|---|---|---|---|---|
| `bluesky` | TBD | TBD | TBD | TBD | TBD |
| `apify_x` | TBD | TBD | TBD | TBD | TBD |
| `reddit` (μ only) | TBD | TBD | TBD | TBD | TBD |
| `3dlnews` (μ only) | TBD | TBD | TBD | TBD | TBD |
| `webhose` | TBD | TBD | TBD | TBD | TBD |

### Decision rule

| Condition | Action |
|---|---|
| Median \|r\| ≥ 0.20 across blocs | Include at full weight in `fit_elasticity()` |
| 0.10 ≤ median \|r\| < 0.20 | Include; note lower predictive power in paper |
| Median \|r\| < 0.10 | Down-weight by 0.5 in `fit_elasticity()` alpha grid, or exclude if near-zero for all blocs; document below |

---

## Near-zero correlation decisions

*To be filled after correlation computation.*

| Platform | Median \|r\| | Decision | Justification |
|---|---|---|---|
| — | — | — | — |

---

## Notes for paper

- Report per-platform Pearson r (Table in Appendix B) alongside bootstrap 95% CI computed over held-out shocks.
- If `3dlnews` shows near-zero correlation for most blocs: expected given geographic-proxy-only signal; retain for geographic heterogeneity analysis, exclude from Σ_Δ (already enforced).
- `reddit` μ-only inclusion: subreddit demographic proxy is coarser than individual bio inference; correlation vs. favorability is expected to be weaker than `bluesky`/`apify_x`. Document as a limitation.
- Additive independence assumption for the three strata means per-bloc correlations are computed independently; cross-stratum correlations are not used in the regression.
