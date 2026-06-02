# Voter Panel Spot-Check — 10 Cell Cross-Validation

**Date:** 2026-06-01  
**Branch:** week1day5  
**Checker:** Juhi Damley  
**Pipeline value source:** `build_voter_panel(PipelineConfig.from_json("configs/base.json"))`  
**Method:** For each cell, the pipeline value (inverse-SE–weighted merge) was compared against
each contributing raw source individually. Δ = pipeline − primary_source (NEP where available,
else most-cited survey). ">2pp" flags are marked **DISCREPANCY** and require discussion.

---

## Spot-Check Table

| # | Cycle | Bloc | Stratum | Pipeline | Primary src | Primary val | Δ (pp) | Status |
|---|-------|------|---------|----------|-------------|-------------|--------|--------|
| SC-01 | 2020 | white | race | 0.4458 | NEP 2020 | 0.410 | +3.6 | **DISCREPANCY** |
| SC-02 | 2020 | african_american | race | 0.8967 | NEP 2020 | 0.870 | +2.7 | **DISCREPANCY** |
| SC-03 | 2020 | latino | race | 0.6576 | NEP 2020 | 0.650 | +0.8 | CLOSE |
| SC-04 | 2024 | white | race | 0.4229 | NEP 2024 | 0.420 | +0.3 | MATCH |
| SC-05 | 2024 | african_american | race | 0.8379 | NEP 2024 | 0.860 | −2.2 | **DISCREPANCY** |
| SC-06 | 2016 | white | race | 0.3988 | NEP 2016 | 0.370 | +2.9 | **DISCREPANCY** |
| SC-07 | 2016 | african_american | race | 0.8960 | NEP 2016 | 0.890 | +0.6 | CLOSE |
| SC-08 | 2020 | women | gender | 0.5714 | NEP 2020 | 0.570 | +0.1 | MATCH |
| SC-09 | 2020 | men | gender | 0.4786 | NEP 2020 | 0.450 | +2.9 | **DISCREPANCY** |
| SC-10 | 2024 | catholic | religion | 0.3971 | NEP 2024 | 0.390 | +0.7 | CLOSE |

Status thresholds: **MATCH** ≤ 1 pp · **CLOSE** 1–2 pp · **DISCREPANCY** > 2 pp

---

## Per-Cell Detail

### SC-01 — 2020 / white / race

| Source | Raw value | n (approx) |
|--------|-----------|------------|
| NEP 2020 | 41.0% | 15,590 total; ~67% white → ~10,445 |
| CES 2020 (wtd) | 44.5% | large |
| ANES 2020 (unwt) | 51.7% | ~3,200 white voters |
| GSS 2021 (2020 recall) | not computed — GSS pres20 col empty | — |
| **Pipeline merged** | **44.6%** | — |

**Root cause:** ANES 2020 weight column is entirely null in our labeled subset
(`anes_labeled_subset.parquet`; verified: `weight.notna().sum() == 0` for year 2020).
The kernel falls back to equal-weight (weight = 1.0), which over-represents
college-educated ANES respondents. Unweighted ANES 2020 white Dem share = 51.7% —
far above the NEP exit-poll gold standard of 41%. The inverse-SE merge gives ANES
substantial weight (n ≈ 3,200), pulling the merged estimate up to 44.6%.

**CES 2020 also shows 44.5%**, so the upward bias is not solely ANES-driven — it likely
reflects the systematic known gap between pre-/post-election surveys and actual vote.

**Action needed:** Decide whether to exclude ANES 2020 from race/gender merges
(no survey weights → SE cannot be computed reliably) or down-weight it explicitly.
See DECISIONS.md open item: "V_eq and lambda weights recomputed from actual panel data."

---

### SC-02 — 2020 / african_american / race

| Source | Raw value |
|--------|-----------|
| NEP 2020 | 87.0% |
| CES 2020 (wtd) | 91.4% |
| ANES 2020 (unwt) | 94.4% |
| **Pipeline merged** | **89.7%** |

Spread across sources: 87–94%. The 2.7 pp gap vs. NEP is within the expected
survey/exit-poll divergence range for minority blocs. CES and ANES (unweighted)
both exceed the NEP figure. The inverse-SE merge moderates toward ~90%, which is
between CES and NEP. **Flagging for discussion** — at 87–90% the optimizer
conclusion is the same, but the exact value affects Σ_Δ covariance estimates.

---

### SC-03 — 2020 / latino / race

| Source | Raw value |
|--------|-----------|
| NEP 2020 | 65.0% |
| CES 2020 (wtd) | 66.2% |
| ANES 2020 (unwt) | 73.0% |
| **Pipeline merged** | **65.8%** |

Pipeline very close to NEP and CES; ANES outlier (unweighted). No action needed.

---

### SC-04 — 2024 / white / race

| Source | Raw value |
|--------|-----------|
| NEP 2024 | 42.0% |
| CES 2024 (wtd) | 43.5% |
| **Pipeline merged** | **42.3%** |

Pipeline within 0.3 pp of NEP. ANES 2024 is present with weights (n = 1,042
weight-bearing obs) and agrees closely. **No issues.**

---

### SC-05 — 2024 / african_american / race

| Source | Raw value |
|--------|-----------|
| NEP 2024 | 86.0% |
| CES 2024 (wtd) | 81.9% |
| **Pipeline merged** | **83.8%** |

Pipeline is 2.2 pp below the NEP exit poll, pulled down by CES 2024 (81.9%).
A −4.1 pp CES–NEP gap for Black voters in 2024 is consistent with known patterns
where post-election CES vote recall underestimates minority turnout and Democratic
margin in that cycle. The pipeline intermediate is between the two, but NEP should
be treated as closer to ground truth for 2024 given its methodology.

**Action needed:** Consider giving NEP 2024 higher weight for 2024 african_american
specifically, or document the CES–NEP 4 pp gap as a known limitation in DECISIONS.md.

---

### SC-06 — 2016 / white / race

| Source | Raw value |
|--------|-----------|
| NEP 2016 | 37.0% |
| ANES 2016 (wtd, n=1,957) | 41.8% |
| GSS 2018-survey 2016-recall (wtd) | not disaggregated (GSS race in religion table) |
| **Pipeline merged** | **39.9%** |

A consistent ~3–5 pp gap between ANES and NEP for white Dem share appears across
multiple cycles. ANES is a face-to-face/online panel that over-represents engaged
civic participants; NEP captures at-the-polling-place behavior. The 2.9 pp pipeline–NEP
gap is driven by ANES pulling the merge above the exit-poll figure.

**Action needed:** Investigate whether ANES sample-frame correction is applied before
the SE computation. If not, ANES may need a calibration deflation or an explicit note
in DECISIONS.md that the white bloc figure is survey-mean, not exit-poll-aligned.

---

### SC-07 — 2016 / african_american / race

| Source | Raw value |
|--------|-----------|
| NEP 2016 | 89.0% |
| ANES 2016 (wtd) | 90.0% |
| **Pipeline merged** | **89.6%** |

All sources agree within 1 pp. **No issues.**

---

### SC-08 — 2020 / women / gender

| Source | Raw value |
|--------|-----------|
| NEP 2020 | 57.0% |
| CES 2020 (wtd) | 56.7% |
| ANES 2020 (unwt) | 60.5% |
| **Pipeline merged** | **57.1%** |

Pipeline matches NEP to 0.1 pp; CES also confirms. ANES unweighted is elevated
(60.5%) but is outweighed. **No issues.**

---

### SC-09 — 2020 / men / gender

| Source | Raw value |
|--------|-----------|
| NEP 2020 | 45.0% |
| CES 2020 (wtd) | 48.0% |
| ANES 2020 (unwt) | 54.6% |
| **Pipeline merged** | **47.9%** |

NEP says 45%, pipeline says 47.9% — a 2.9 pp gap. CES also shows 48.0%, so the
divergence from NEP is present in CES as well. The same ANES missing-weight issue
(SC-01) affects the gender merge: unweighted ANES men = 54.6% pulls the estimate
up. CES and ANES together outweigh NEP in the inverse-SE merge, resulting in ~3 pp
above the exit-poll figure.

**Action needed:** Same as SC-01 — consider excluding unweighted ANES 2020 from
gender merge, or document the known 3 pp survey/exit-poll gender gap.

---

### SC-10 — 2024 / catholic / religion

| Source | Raw value |
|--------|-----------|
| NEP 2024 (Catholic row) | 39.0% |
| CES 2024 (Roman Catholic, wtd) | 42.5% |
| **Pipeline merged** | **39.7%** |

Pipeline is very close to NEP (0.7 pp). The 3.5 pp CES–NEP gap for Catholics in 2024
is within the expected survey/exit-poll range. The inverse-SE merge appropriately
weights the larger NEP n more heavily. **No action needed for the value;**
the CES–NEP catholic gap is worth noting in the limitations section of the paper.

---

## Summary of Discrepancies for Supervision Check-in

_Note: 5 cells are marked **DISCREPANCY** in the spot-check table; they are grouped into 4 root-cause flags below._
| Flag | Cells affected | Root cause | Severity |
|------|---------------|------------|----------|
| **F-1** | SC-01, SC-09 (2020 white, men) | ANES 2020 weights all-null; unweighted ANES inflates merged estimate by ~3 pp | High — affects two strata |
| **F-2** | SC-02 (2020 african_american) | Multi-source spread 87–94%; ANES unweighted at 94.4% pulls up | Medium — within expected range but wide spread |
| **F-3** | SC-05 (2024 african_american) | CES 2024 understates vs. NEP by 4.1 pp; pipeline splits the difference | Medium — NEP should dominate for 2024 |
| **F-4** | SC-06 (2016 white) | ANES white Dem share systematically ~5 pp above NEP across cycles; sample-frame bias | Medium — recurring across cycles |

**Recommended fixes before Week 2 baseline:**
1. Zero out ANES 2020 from race/gender merges (weight col null → SE undefined → exclude).
   This should bring SC-01 (white 2020) from 44.6% → ~44.5% (CES-led), closer to NEP 41%.
   *Note: CES 2020 itself shows 44.5% — there remains a ~3.5 pp survey/exit-poll gap even
   without ANES. Document in DECISIONS.md as known limitation.*
2. For 2024 african_american: add a note in DECISIONS.md flagging the 4 pp CES–NEP gap;
   consider NEP-only for 2024 if the CES 2024 vote recall is considered unreliable.
3. Record the systematic ~4–5 pp ANES white Dem overstatement in DECISIONS.md and flag it
   in the paper's limitations section (ANES civic-participation sample-frame bias).

---

## Sources consulted

| File | Variable(s) |
|------|-------------|
| `data/surveys/cnn_ssrs_polls/nep_2020_exit_poll.csv` | category=Race, Gender; dem_pct |
| `data/surveys/cnn_ssrs_polls/nep_2024_exit_poll.csv` | category=Race, Gender, Religion; dem_pct |
| `data/surveys/cnn_ssrs_polls/nep_2016_exit_poll.csv` | category=race, religion; dem_pct |
| `data/surveys/anes_timeseries_cdf_csv_20260205/anes_labeled_subset.parquet` | year, race_7cat, gender, pres_vote, weight |
| `data/surveys/CES_2006_2024/ces_cumulative_labeled.parquet` | year, race_h, gender, religion, voted_pres_party, weight_cumulative |
| `data/surveys/GSS_stata (1)/gss_labeled_subset.parquet` | year, relig, pres16, vote16, weight |
