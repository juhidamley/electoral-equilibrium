#!/usr/bin/env python3
"""Local end-to-end pipeline verification — NO Modal, NO network.

Verifies the un-deployed backend changes against their invariants and prints
PASS/FAIL per check with the actual numbers, before a Modal deploy:

  1. COVARIANCE     — Σ_Δ is Ledoit-Wolf, 1990+ window, symmetric/PSD/non-diagonal,
                      per-bloc std in [0.05, 0.13].
  2. BASELINE       — offline estimate_moments uses the 1990+ cutoff; thin-bloc σ
                      collapsed to ~0.10–0.13; μ ∈ [0, 1].
  3. NOMINALS       — _NOMINAL_MU_RACE == documented _DEM_BENCHMARKS.
  4. WEIGHT BOUNDS  — optimizer weights stay in [0.5·w0, 1.5·w0] or correctly
                      report infeasible (equal_weight_fallback); ≥1 case infeasible.
  5. END-TO-END     — LLM → optimizer → Monte Carlo on 3 events; no NaN/Inf,
                      win_prob ∈ [0,1], CI low ≤ point ≤ high, some neutral bins.
  6. SUMMARY        — total PASS/FAIL, with every FAIL listed.

Run from the repo root (where artifacts/panel/panel_race.parquet lives). The
end-to-end stage needs the fine-tuned adapter; if it can't load, that stage is
SKIPPED (not failed) so the static checks still run everywhere.

    python scripts/verify_pipeline.py
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

# Repo root on path so `import electoral` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from electoral.core.types import (  # noqa: E402
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)
from electoral.kernels.shock import COVARIANCE_MIN_CYCLE, build_sigma_delta_from_panel  # noqa: E402

SEED = 42
RACES = list(CANONICAL_RACES)
PANEL = Path("artifacts/panel/panel_race.parquet")

# ── PASS/FAIL harness ─────────────────────────────────────────────────────────

_RESULTS: list[tuple[str, str, str, str]] = []  # (section, name, status, detail)


def record(section: str, name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    _RESULTS.append((section, name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def skip(section: str, name: str, detail: str) -> None:
    _RESULTS.append((section, name, "SKIP", detail))
    print(f"  [SKIP] {name} — {detail}")


def _flatten(obj):
    """Yield every scalar inside nested dict/list/ndarray/scalar."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for v in np.asarray(obj).ravel().tolist() if isinstance(obj, np.ndarray) else obj:
            yield from _flatten(v)
    else:
        yield obj


def all_finite(obj) -> bool:
    try:
        return all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in _flatten(obj))
    except (TypeError, ValueError):
        return False


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ── 1. COVARIANCE ─────────────────────────────────────────────────────────────


def check_covariance() -> np.ndarray | None:
    header("1. COVARIANCE — Σ_Δ (Ledoit-Wolf, 1990+ window)")
    if not PANEL.exists():
        record("COVARIANCE", "panel parquet present", False, f"{PANEL} not found")
        return None

    sd = np.asarray(build_sigma_delta_from_panel(PANEL), dtype=float)
    np.set_printoptions(precision=5, suppress=True)
    print("  Σ_Δ (5×5):")
    for row in sd:
        print("   ", row)
    stds = np.sqrt(np.diag(sd))
    print("  per-bloc std:", {b: round(float(s), 4) for b, s in zip(RACES, stds)})

    # Independently confirm the window the builder used (1990+, ~9 cycles, not 20).
    import pandas as pd

    piv = (
        pd.read_parquet(PANEL)
        .query("cycle >= @COVARIANCE_MIN_CYCLE")
        .pivot_table(index="cycle", columns="bloc", values="vote_share", aggfunc="mean")
        .reindex(columns=RACES)
        .dropna()
    )
    cyc = [int(c) for c in piv.index]
    print(f"  cycles used: {len(cyc)} ({cyc[0]}–{cyc[-1]}) → {len(cyc) - 1} first-differences")

    eig = np.linalg.eigvalsh(sd)
    off_max = float(np.abs(sd - np.diag(np.diag(sd))).max())

    record("COVARIANCE", "symmetric", bool(np.allclose(sd, sd.T)),
           f"max|Σ-Σᵀ|={float(np.abs(sd - sd.T).max()):.2e}")
    record("COVARIANCE", "PSD (all eigenvalues > 0)", bool((eig > 1e-10).all()),
           f"min eig={float(eig.min()):.3e}")
    record("COVARIANCE", "non-diagonal (max off-diag > 0)", off_max > 1e-6,
           f"max|off-diag|={off_max:.5f}")
    record("COVARIANCE", "per-bloc std ∈ [0.05, 0.13]",
           bool(((stds >= 0.05) & (stds <= 0.13)).all()),
           f"std={np.round(stds, 4).tolist()}")
    record("COVARIANCE", "window is 1990+ (~9 cycles, not 20)",
           cyc[0] >= COVARIANCE_MIN_CYCLE and 6 <= len(cyc) <= 12,
           f"{len(cyc)} cycles from {cyc[0]}")
    return sd


# ── 2. BASELINE ───────────────────────────────────────────────────────────────


def check_baseline() -> None:
    header("2. BASELINE — offline estimate_moments (1990+ cutoff)")
    import pandas as pd

    from electoral.kernels.baseline import DEFAULT_NEP_SHARES
    from electoral.models.ml_baseline import estimate_moments, ground_truth_winning_cycles

    if not PANEL.exists():
        record("BASELINE", "panel parquet present", False, f"{PANEL} not found")
        return

    panel_df = pd.read_parquet(PANEL)
    win = ground_truth_winning_cycles("democrat")

    def run(min_cycle):
        m = estimate_moments(panel_df, "democrat", winning_cycles=win, min_cycle=min_cycle)
        mu = m.mu_race
        sig = {b: float(np.sqrt(max(0.0, m.Sigma[i, i]))) for i, b in enumerate(RACES)}
        raw = {b: DEFAULT_NEP_SHARES[b] * mu[b] for b in RACES}
        tot = sum(raw.values())
        w = {b: raw[b] / tot for b in RACES}
        return mu, sig, w

    mu_no, sig_no, _ = run(None)            # full panel (proves the cutoff bites)
    mu, sig, w = run(COVARIANCE_MIN_CYCLE)  # 1990+ (production)

    print(f"  {'bloc':<18}{'weight':>9}{'μ':>9}{'σ(1990+)':>11}{'σ(full)':>10}")
    for b in RACES:
        print(f"  {b:<18}{w[b]:>9.4f}{mu[b]:>9.4f}{sig[b]:>11.4f}{sig_no[b]:>10.4f}")

    # The cutoff is "used" iff the thin-bloc σ collapsed vs the full panel.
    record("BASELINE", "cutoff active: asian σ collapsed (full>0.20 → cut<0.13)",
           sig_no["asian"] > 0.20 and sig["asian"] < 0.13,
           f"asian σ {sig_no['asian']:.3f} → {sig['asian']:.3f}")
    record("BASELINE", "asian σ ∈ ~[0.10, 0.13]", 0.08 <= sig["asian"] <= 0.13,
           f"{sig['asian']:.4f}")
    record("BASELINE", "other_race σ ∈ ~[0.08, 0.13]", 0.08 <= sig["other_race"] <= 0.13,
           f"{sig['other_race']:.4f}")
    record("BASELINE", "all μ ∈ [0, 1]",
           all(0.0 <= mu[b] <= 1.0 for b in RACES),
           f"min={min(mu.values()):.3f} max={max(mu.values()):.3f}")
    record("BASELINE", "no NaN/Inf in μ or σ", all_finite(mu) and all_finite(sig))


# ── 3. NOMINALS ───────────────────────────────────────────────────────────────


def check_nominals() -> None:
    header("3. NOMINALS — _NOMINAL_MU_RACE == documented _DEM_BENCHMARKS")
    from electoral.api.shock_endpoint import _NOMINAL_MU_RACE
    from electoral.models.benchmarks import DEM_RACE_BENCHMARKS

    dem = _NOMINAL_MU_RACE["democrat"]
    print(f"  {'bloc':<18}{'nominal':>9}{'benchmark':>11}")
    for b in RACES:
        print(f"  {b:<18}{dem[b]:>9.2f}{DEM_RACE_BENCHMARKS[b]:>11.2f}")
    expected = {"african_american": 0.87, "latino": 0.62, "asian": 0.66,
                "white": 0.39, "other_race": 0.58}
    record("NOMINALS", "live nominal == _DEM_BENCHMARKS",
           all(abs(dem[b] - DEM_RACE_BENCHMARKS[b]) < 1e-9 for b in RACES))
    record("NOMINALS", "values match expected NEP 2000–2024",
           all(abs(dem[b] - expected[b]) < 1e-9 for b in RACES))


# ── 4. OPTIMIZER WEIGHT BOUNDS ────────────────────────────────────────────────


def check_weight_bounds(sd: np.ndarray | None) -> None:
    header("4. OPTIMIZER WEIGHT BOUNDS — w ∈ [0.5·w0, 1.5·w0] or infeasible")
    from electoral.api.shock_endpoint import (
        _NOMINAL_MU_GENDER,
        _NOMINAL_MU_RELIGION,
        _load_party_config,
        _target_for_party,
    )
    from electoral.kernels.baseline import DEFAULT_NEP_SHARES
    from electoral.optimization.cvx import (
        WEIGHT_LOWER_MULT,
        WEIGHT_UPPER_MULT,
        solve_rebalanced,
    )
    from electoral.optimization.dqcp import compute_fixed_loyalty
    from electoral.simulation.montecarlo import _load_layer_weights

    cov = sd.tolist() if sd is not None else [[0.01 if i == j else 0.0 for j in range(5)]
                                              for i in range(5)]
    lw = _load_layer_weights()
    target = _target_for_party("democrat", _load_party_config())
    fixed = compute_fixed_loyalty(_NOMINAL_MU_RELIGION["democrat"],
                                  _NOMINAL_MU_GENDER["democrat"],
                                  lw["lambda_2"], lw["lambda_3"])
    lo = {b: DEFAULT_NEP_SHARES[b] * WEIGHT_LOWER_MULT for b in RACES}
    hi = {b: min(1.0, DEFAULT_NEP_SHARES[b] * WEIGHT_UPPER_MULT) for b in RACES}

    cases = {
        "easy":    {"african_american": 0.95, "latino": 0.70, "asian": 0.70,
                    "white": 0.55, "other_race": 0.65},
        "hard":    {b: 0.52 for b in RACES},
        "extreme": {b: 0.05 for b in RACES},  # below any in-band μ_eff → infeasible
    }
    n_infeasible = 0
    for label, mu in cases.items():
        eq = solve_rebalanced(mu, cov, target=target, party="democrat",
                              shock=f"bounds_{label}", fixed_loyalty=fixed)
        print(f"\n  case={label}  feasible={eq.feasible}  method={eq.method}  target={target:.4f}")
        if eq.feasible:
            within = True
            print(f"    {'bloc':<18}{'lo':>8}{'w':>8}{'hi':>8}")
            for b in RACES:
                w = eq.weights[b]
                ok = lo[b] - 1e-6 <= w <= hi[b] + 1e-6
                within &= ok
                print(f"    {b:<18}{lo[b]:>8.3f}{w:>8.3f}{hi[b]:>8.3f}{'' if ok else '  <<OUT'}")
            record("WEIGHT BOUNDS", f"{label}: feasible weights within [0.5·w0,1.5·w0]", within)
        else:
            n_infeasible += 1
            record("WEIGHT BOUNDS", f"{label}: infeasible → equal_weight_fallback",
                   eq.method == "equal_weight_fallback",
                   f"method={eq.method}")
    record("WEIGHT BOUNDS", "≥1 case correctly infeasible (bounds bite)", n_infeasible >= 1,
           f"{n_infeasible} infeasible")


# ── 5. END-TO-END ─────────────────────────────────────────────────────────────


def check_end_to_end(sd: np.ndarray | None) -> None:
    header("5. END-TO-END — LLM → optimizer → Monte Carlo (seed=42)")
    import dataclasses

    section = "END-TO-END"
    try:
        import torch

        from electoral.api.shock_endpoint import (
            _NOMINAL_MU_GENDER,
            _NOMINAL_MU_RACE,
            _NOMINAL_MU_RELIGION,
            _load_party_config,
            _target_for_party,
        )
        from electoral.config import PipelineConfig
        from electoral.llm.inference import load_model, predict_delta_bins
        from electoral.optimization.cvx import solve_rebalanced
        from electoral.optimization.dqcp import compute_fixed_loyalty
        from electoral.simulation.montecarlo import _load_layer_weights, run_ilr_montecarlo
    except Exception as exc:  # pragma: no cover - import guard
        skip(section, "imports", f"{type(exc).__name__}: {exc}")
        return

    adapter = os.environ.get("ADAPTER_PATH", "models/mistral-r16")
    base_model = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.3")
    if not Path(adapter).exists():
        skip(section, "adapter load", f"adapter not found at {adapter} (set ADAPTER_PATH)")
        return
    try:
        torch.manual_seed(SEED)
        model, tokenizer = load_model(adapter, base_model)
    except Exception as exc:
        skip(section, "adapter load", f"{type(exc).__name__}: {exc}")
        return

    cov = sd.tolist() if sd is not None else None
    lw = _load_layer_weights()
    party_config = _load_party_config()

    events = [
        ("dem_scandal", "democrat",
         "A major corruption scandal implicates the sitting Democratic administration"),
        ("rep_scandal", "republican",
         "A major corruption scandal implicates the sitting Republican administration"),
        ("neutral", "democrat",
         "Congress passes a routine procedural rule change on committee scheduling"),
    ]

    for label, party, text in events:
        try:
            torch.manual_seed(SEED)
            bins = predict_delta_bins(text, party, model, tokenizer,
                                      use_constrained=True, seed=SEED)
            d_race = {b: max(-0.15, min(0.15, BIN_MIDPOINTS[bins[b]])) for b in CANONICAL_RACES}
            d_rel = {b: max(-0.15, min(0.15, BIN_MIDPOINTS[bins[b]])) for b in CANONICAL_RELIGIONS}
            d_gen = {b: max(-0.15, min(0.15, BIN_MIDPOINTS[bins[b]])) for b in CANONICAL_GENDERS}

            nom = _NOMINAL_MU_RACE[party]
            delta_eff = (
                lw["lambda_1"] * sum(nom[b] * d_race[b] for b in CANONICAL_RACES)
                + lw["lambda_2"] * sum(_NOMINAL_MU_RELIGION[party][r] * d_rel[r]
                                       for r in CANONICAL_RELIGIONS)
                + lw["lambda_3"] * sum(_NOMINAL_MU_GENDER[party][g] * d_gen[g]
                                       for g in CANONICAL_GENDERS)
            )
            mu_tilde = {b: float(max(0.01, min(0.99, nom[b] + d_race[b]))) for b in CANONICAL_RACES}
            fixed = compute_fixed_loyalty(_NOMINAL_MU_RELIGION[party], _NOMINAL_MU_GENDER[party],
                                          lw["lambda_2"], lw["lambda_3"],
                                          deltas_religion=d_rel, deltas_gender=d_gen)
            target = _target_for_party(party, party_config)
            eq = solve_rebalanced(mu_tilde, cov, target=target, party=party,
                                  shock=label, fixed_loyalty=fixed)
            cfg = dataclasses.replace(PipelineConfig.from_json("configs/base.json"),
                                      seed=SEED, party=party, target=target)
            sim = run_ilr_montecarlo(eq, cfg, n_simulations=10_000, cov_delta=cov,
                                     fixed_loyalty=fixed)

            ci_width = sim.win_probability_high - sim.win_probability_low
            n_neutral = sum(1 for v in bins.values() if v == "neutral")
            print(f"\n  event={label} ({party})")
            print(f"    bins: {bins}")
            print(f"    delta_eff={delta_eff:+.4f}  feasible={eq.feasible}  "
                  f"win_prob={sim.win_probability:.4f}  "
                  f"CI=[{sim.win_probability_low:.4f}, {sim.win_probability_high:.4f}] "
                  f"(width {ci_width:.4f})  neutral_bins={n_neutral}/15")

            finite = all_finite([list(d_race.values()), list(d_rel.values()),
                                 list(d_gen.values()), delta_eff,
                                 list(eq.weights.values()), list(eq.mu_shifted.values()),
                                 sim.win_probability, sim.win_probability_low,
                                 sim.win_probability_high, sim.percentiles])
            record(section, f"{label}: no NaN/Inf anywhere", finite)
            record(section, f"{label}: win_prob ∈ [0,1]",
                   0.0 <= sim.win_probability <= 1.0, f"{sim.win_probability:.4f}")
            record(section, f"{label}: CI low ≤ point ≤ high",
                   sim.win_probability_low <= sim.win_probability <= sim.win_probability_high,
                   f"[{sim.win_probability_low:.3f}, {sim.win_probability_high:.3f}]")
            record(section, f"{label}: ≥1 neutral delta bin (not all forced ±)",
                   n_neutral >= 1, f"{n_neutral} neutral")
        except Exception as exc:
            record(section, f"{label}: pipeline ran without error", False,
                   f"{type(exc).__name__}: {exc}")


# ── 6. SUMMARY ────────────────────────────────────────────────────────────────


def summary() -> int:
    header("6. SUMMARY")
    n_pass = sum(1 for *_, s, _ in _RESULTS if s == "PASS")
    n_fail = sum(1 for *_, s, _ in _RESULTS if s == "FAIL")
    n_skip = sum(1 for *_, s, _ in _RESULTS if s == "SKIP")
    print(f"  PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  (total {len(_RESULTS)})")
    if n_fail:
        print("\n  FAILURES:")
        for section, name, status, detail in _RESULTS:
            if status == "FAIL":
                print(f"    - [{section}] {name}" + (f" — {detail}" if detail else ""))
    if n_skip:
        print("\n  SKIPPED:")
        for section, name, status, detail in _RESULTS:
            if status == "SKIP":
                print(f"    - [{section}] {name}" + (f" — {detail}" if detail else ""))
    verdict = "ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED"
    print(f"\n  {verdict}")
    return 1 if n_fail else 0


def main() -> int:
    # Surface the builder's INFO logs (cycle range/count) so they're visible.
    logging.basicConfig(level=logging.INFO, format="    LOG %(name)s: %(message)s")
    print(f"Local pipeline verification (seed={SEED}, no Modal, no network)\n"
          f"cwd={os.getcwd()}")
    sd = check_covariance()
    check_baseline()
    check_nominals()
    check_weight_bounds(sd)
    check_end_to_end(sd)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
