# Baseline Portfolio — Interpretation

**Run date:** 2026-06-02  
**Party:** Democrat  
**Method:** min-variance QP (`solve_baseline`)  
**V_eq target:** 0.535 (placeholder; see open question below)

---

## Coalition weights

| Bloc | Weight | μ (winning cycles) | σ (historical) |
|------|-------:|-------------------:|---------------:|
| african_american | **0.510** | 0.892 | 0.104 |
| white | **0.302** | 0.467 | 0.073 |
| other_race | **0.188** | 0.579 | 0.233 |
| latino | 0.000 | 0.669 | 0.106 |
| asian | 0.000 | 0.579 | 0.245 |

**μ_eff = 0.659 — margin above V_eq: +12.4 pp**

---

## What the optimizer is doing

The min-variance QP concentrates weight on the two blocs that maximize the loyalty-to-variance ratio. African Americans have the highest loyalty (89%) at moderate variance (σ = 0.10); White voters have the lowest variance in Σ (σ = 0.07) and serve as the stabilizing complement. Other race fills the remainder of the simplex (19%).

Latino and Asian voters receive zero weight. Both have mid-range loyalty (0.58–0.67) and above-average variance, so they contribute noise without enough expected gain over the African American + White anchor to justify the variance cost. This is mathematically correct for a min-variance objective. It is almost certainly not a realistic coalition strategy.

The DQCP optimizer (Week 5) will change this. By maximizing P(win) rather than minimizing variance, it is forced to consider whether the additional expected loyalty from a larger Latino or Asian share outweighs the variance penalty — especially in deficit scenarios where the only path to V_eq runs through high-variance blocs. The zero-weight result here is best understood as a baseline that exposes the limitation of the min-variance criterion, not as a practical recommendation.

---

## Cross-reference against Espinosa's research

All race-stratum μ values are within ±0.07 of published reference ranges (Pew/NEP 2016–2024). Five borderline values are documented in ESPINOSA.md §10.1; all are historically explainable rather than data errors:

- **White μ = 0.467** (ref: 0.37–0.45): marginally elevated because winning cycles include 1948–1964, when Democratic white support was structurally higher than today. Any paper claim about the white bloc baseline should restrict to post-1980 cycles or note the historical drift.
- **Asian σ = 0.245**: three times higher than the next-most-volatile bloc (other_race). Partly artifactual — sparse pre-1968 ANES data, 1965 Immigration Act break, 1992 three-party contamination — rather than genuine swing volatility. Asian CIs from the MC simulation will be unreliable; see ESPINOSA.md §10.4 for the question about restricting to post-1968.

Religion and gender strata broadly align with published figures. Evangelical μ = 0.291 sits within the ~24% reference (the slight elevation reflects pre-Moral-Majority cycles where the bloc boundary was less distinct). Muslim μ = 0.851 is above the 0.80 ceiling because the data is restricted to 2000–2020, entirely missing the documented 2024 erosion.

---

## Open questions for supervision check-in

**Q1 — Zero weights for Latino and Asian.** 24M and 10M voters respectively receive w = 0 in the min-variance baseline. For the paper: is the correct framing (a) "impose a floor (e.g. w_i ≥ 0.05) so the baseline reflects a plausible coalition," or (b) "the zero-weight result is the paper's finding — it shows why min-variance is the wrong objective for coalition building, and the DQCP paper contribution is what restores these blocs"?

**Q2 — V_eq = 0.535 is Democrat-specific.** The Republican baseline produces μ_eff = 0.502, which is 3.3 pp below this target. Republicans win at lower effective loyalty thresholds (~0.49–0.51 per CLAUDE.md). The per-party V_eq derivation from `build_constraint_spec` is required before the optimizer runs are valid for Republicans.

**Q3 — Popular vote vs Electoral College.** In 2000 and 2016, the popular-vote winner lost the Electoral College. The current winning-cycle criterion uses popular vote (Gore/Clinton counted as Democratic wins). If V_eq should reflect the Electoral College threshold, both the winning-cycle set and the target value change.
