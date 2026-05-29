# QWEN.md — Electoral Equilibrium (Quick Reference)

You are a second pair of eyes on this project alongside Claude Code.
Your job: code review, catch bugs, suggest improvements, answer questions.
Do NOT read the full devplan — everything you need is here.

---

## What this project is

A Python + Next.js system that takes a hypothetical political event as input
and outputs how a party's voter coalition must rebalance to maintain a winning
position, plus the win probability with a 90% CI.

Three pipeline stages:
1. **LLM** (Mistral 7B, QLoRA fine-tuned): shock text → demographic shift estimates
2. **Optimizer** (CVXPY): shifts → optimal coalition weights
3. **Monte Carlo**: weights → win probability + confidence interval

---

## Demographic model

Three independent strata. Each covers ~100% of the electorate from marginal tables.
No cross-tabulations. No intersection cells.

**Stratum 1 — Race** (optimizer rebalances these, 5 weights sum to 1):
`african_american, latino, asian, white, other_race`

**Stratum 2 — Religion** (fixed from voter panel, 7 weights sum to 1):
`evangelical, catholic, protestant, secular, jewish, muslim, other_rel`

**Stratum 3 — Gender** (fixed from voter panel, 3 weights sum to 1):
`women, men, other_gender`

**Effective loyalty scalar:**
```python
mu_eff = (lambda_1 * sum(w[i] * mu_race[i] for i in races)
        + lambda_2 * sum(v[R] * mu_rel[R] for R in religions)
        + lambda_3 * sum(g[G] * mu_gen[G] for G in genders))
# lambda_1 + lambda_2 + lambda_3 = 1, calibrated from historical data
# stored in configs/layer_weights.json
```

**Win condition:** `mu_eff >= V_eq`
V_eq ~ 0.52–0.53 Democrat, 0.49–0.51 Republican (from voter panel, not hardcoded).

---

## Optimizer (electoral/portfolios/cvx.py)

**Objective: maximize P(win), NOT minimize variance.**
Min-variance is wrong when baseline loyalty is below V_eq — it locks you into a loss.

```python
# Sharpe-ratio form, quasi-convex
numerator   = mu_eff_tilde - V_eq          # linear in w
denominator = cp.sqrt(cp.quad_form(w, Sigma_delta))  # convex, > 0

problem = cp.Problem(cp.Maximize(cp.ratio(numerator, denominator)))
assert problem.is_dcp(qcp=True)   # MUST pass — if not, the objective is wrong
problem.solve(qcp=True)           # DQCP, not standard solve
```

Covariance matrix `Sigma_delta` is 5×5 (race blocs only), LedoitWolf-regularized.

---

## Monte Carlo (electoral/simulation/montecarlo.py)

**NOT Dirichlet** (forces negative off-diagonal covariances).
**NOT Gaussian + project** (distorts simplex geometry at boundaries).
**NOT delta method with floor** (0.01 floor distorts log-ratio space non-linearly).

Use **Logistic-Normal with ILR (isometric log-ratio) parameterization:**
```python
# V = Helmert contrast matrix (K-1 x K)
z_star = V.T @ np.log(w_star)                  # map w* to ILR space
J = jacobian_of_ilr(w_star)                    # Jacobian at w*
Sigma_ILR = J @ Sigma_delta @ J.T              # propagate covariance
y_samples = rng.multivariate_normal(z_star, Sigma_ILR, size=N)
w_samples = softmax(V @ y_samples.T, axis=0).T # back to simplex
```

Zero-weight blocs → `infeasible_bloc` flag, never floor to 0.01.
Output: `win_probability` (point), `win_probability_low` (p5), `win_probability_high` (p95).

---

## Artifacts (electoral/artifacts.py)

All frozen dataclasses. Every stage reads one, produces one.
Key fields to know:

```python
ShockResponseData:
    delta_bins_race:     dict[str, str]   # race_id -> 9-token bin
    delta_bins_religion: dict[str, str]   # religion_id -> 9-token bin
    delta_bins_gender:   dict[str, str]   # gender_id -> 9-token bin
    delta_eff:           float            # scalar effective delta
    covariance:          list[list[float]] # 5x5

EquilibriumData:
    weights:         dict[str, float]   # race_id -> weight, 5 keys, sum=1
    mu_eff_shifted:  float              # post-shock scalar
    feasible:        bool
    target_met:      bool

SimulationData:
    win_probability:      float
    win_probability_low:  float   # p5
    win_probability_high: float   # p95
```

---

## Canonical IDs

All strings are **lowercase snake_case**. Never deviate.

Race: `african_american`, `latino`, `asian`, `white`, `other_race`
Religion: `evangelical`, `catholic`, `protestant`, `secular`, `jewish`, `muslim`, `other_rel`
Gender: `women`, `men`, `other_gender`
Party: `"democrat"` or `"republican"` (lowercase)
Cycles: `int` YYYY (e.g. `2020`, not `"2020"`)

---

## LLM output bins

9 tokens: `strong_neg, mod_neg, mild_neg, slight_neg, neutral, slight_pos, mild_pos, mod_pos, strong_pos`

Constrained decoding via `outlines` library. Three separate dicts output per inference call:
`delta_bins_race`, `delta_bins_religion`, `delta_bins_gender`.

---

## Seed contract (electoral/core/rng.py)

```python
make_rng(seed) -> np.random.Generator
derive_seed(base_seed, stage_name) -> int
```

Never call `np.random` directly. Never call `random.seed()` globally.
Every stochastic function takes a seeded generator as a parameter.

---

## Key don'ts

- **Don't** use Gemini API for cleaning — breaks reproducibility. Use local Qwen2.5-7B or Mistral-7B-Instruct via mlx_lm.
- **Don't** use prediction market Δπ as a training feature. Δπ ≠ Δμ (probability ≠ vote-share margin). Markets are calibration benchmarks only.
- **Don't** use `problem.solve()` without `qcp=True` on the optimizer.
- **Don't** include language-fallback posts (`inference_method: "language_prior"`) in mean or covariance estimation — validation only.
- **Don't** floor Monte Carlo weights to 0.01 — use `infeasible_bloc` instead.
- **Don't** install Syncthing on the HPC.

---

## Machines

| Machine | Role |
|---|---|
| M5 MacBook Pro | Dev, cleaning, Bluesky collection |
| Intel MacBook | News scraper 24/7 (launchd 2am), Apify, Reddit |
| Raspberry Pi 5 (Tailscale) | SetFit bio classifier at `$PI_TAILSCALE_IP:9000` |
| CMC HPC | QLoRA fine-tuning, SLURM scoring |

---

## Files to know

```
electoral/artifacts.py       # all dataclass schemas
electoral/core/rng.py        # seed contract
electoral/portfolios/cvx.py  # CVXPY optimizer
electoral/simulation/montecarlo.py  # ILR Monte Carlo
electoral/kernels/raking.py  # optional marginal calibration
configs/layer_weights.json   # lambda_1, lambda_2, lambda_3
configs/base.json            # PipelineConfig, seed, party, pi IP
DECISIONS.md                 # full rationale for every design choice
```

---

## If you need more context

Read `DECISIONS.md` for the why behind any decision.
Read `electoral/artifacts.py` for exact schemas.
Do NOT read the devplan PDF — it will fill your context window.
