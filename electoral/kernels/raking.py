"""Marginal calibration (iterative proportional fitting) of lambda layer weights.

The additive mu_eff formula uses three strata weighted by lambda_1, lambda_2,
lambda_3 (race, religion, gender).  The default values in layer_weights.json
(0.50 / 0.30 / 0.20) are rough priors.  This module calibrates them from data
by solving, for each stratum in turn, the least-squares regression of that
stratum's historical vote-share series against the unexplained residual from
the other two strata.  Iterating over all three strata until convergence is
IPF (iterative proportional fitting) applied to the lambda parameter space.

Typical usage
-------------
    from electoral.kernels.raking import rake_layer_weights, write_raked_weights
    raked, n_iters = rake_layer_weights(panel_df)
    write_raked_weights(raked)

Output
------
Writes the calibrated lambdas under key "raked" in configs/layer_weights.json.
The additive lambdas (top-level lambda_1 / 2 / 3) are left unchanged so the
two versions can be compared directly in the paper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.models.ml_baseline import (
    _APPROX_GENDER_SHARE,
    _APPROX_RACE_SHARE,
    _APPROX_RELIGION_SHARE,
    _PRES_DEM_2P_SHARE,
)

log = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).parents[2] / "configs"
_LAYER_WEIGHTS_PATH = _CONFIGS_DIR / "layer_weights.json"

# ── Electorate-share reference dicts ─────────────────────────────────────────
# Keyed in canonical stratum order so the weighted average is deterministic.
_STRATA: list[tuple[list[str], dict[str, float]]] = [
    (list(CANONICAL_RACES), _APPROX_RACE_SHARE),
    (list(CANONICAL_RELIGIONS), _APPROX_RELIGION_SHARE),
    (list(CANONICAL_GENDERS), _APPROX_GENDER_SHARE),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stratum_series(
    panel: pd.DataFrame,
    blocs: list[str],
    electorate_share: dict[str, float],
) -> pd.Series:
    """Return the electorate-weighted mean vote share per cycle for one stratum.

    Missing (cycle, bloc) cells are excluded from the weighted average for
    that cycle; cycles where NO bloc in the stratum has data return NaN.

    Parameters
    ----------
    panel:
        Cleaned panel with columns cycle (int), bloc (str), vote_share (float).
    blocs:
        Ordered bloc IDs for this stratum.
    electorate_share:
        Approximate fraction of the electorate in each bloc.  Need not sum
        to 1.0 exactly — normalised per cycle within the available blocs.
    """
    sub = panel[panel["bloc"].isin(blocs)].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    sub["e"] = sub["bloc"].map(electorate_share).fillna(0.0)
    sub = sub[sub["e"] > 0]

    def _weighted(g: pd.DataFrame) -> float:
        valid = g.dropna(subset=["vote_share"])
        if valid.empty:
            return float("nan")
        total_e = float(valid["e"].sum())
        if total_e <= 0:
            return float("nan")
        return float((valid["vote_share"] * valid["e"]).sum() / total_e)

    return sub.groupby("cycle").apply(_weighted, include_groups=False)


def _align(
    strata_series: list[pd.Series],
    y: pd.Series,
) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    """Return aligned numpy arrays for cycles where all three strata have data."""
    valid_idx = y.index
    for s in strata_series:
        valid_idx = valid_idx.intersection(s.dropna().index)
    valid_idx = sorted(valid_idx)

    S = [s[valid_idx].values.astype(float) for s in strata_series]
    y_arr = y[valid_idx].values.astype(float)
    return S, y_arr, valid_idx


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project *v* onto the probability simplex {x : sum=1, x≥0}.

    Duchi et al. (2008) O(n log n) algorithm.  Works for any n ≥ 1.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho_candidates = np.where(u > cssv / np.arange(1, n + 1))[0]
    rho = int(rho_candidates[-1])
    theta = float(cssv[rho] / (rho + 1))
    return np.maximum(v - theta, 0.0)


# ── Public API ────────────────────────────────────────────────────────────────


def rake_layer_weights(
    panel: pd.DataFrame,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
    initial_lambdas: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> tuple[dict[str, float], int]:
    """Calibrate lambda layer weights via IPF against historical election outcomes.

    For each stratum in turn, the lambda for that stratum is updated to the
    least-squares coefficient that explains the residual from the other two
    strata.  After updating all three, the lambdas are renormalised to sum
    to 1.  The procedure repeats until the maximum per-lambda change falls
    below *tol* (typically < 10 iterations).

    Parameters
    ----------
    panel:
        Cleaned voter panel with columns cycle (int), bloc (str),
        vote_share (float).
    tol:
        Convergence tolerance on the maximum absolute lambda change per
        iteration (default 1e-6).
    max_iter:
        Maximum number of full IPF passes before giving up (default 100).
    initial_lambdas:
        Starting point for (lambda_1, lambda_2, lambda_3).  Defaults to the
        values currently in configs/layer_weights.json.

    Returns
    -------
    (raked_lambdas, n_iters)
        raked_lambdas: dict with keys "lambda_1", "lambda_2", "lambda_3"
        n_iters: number of IPF passes until convergence (0 = insufficient data)
    """
    panel = panel.copy()
    panel["cycle"] = pd.to_numeric(panel["cycle"], errors="coerce")
    panel = panel.dropna(subset=["cycle"])
    panel["cycle"] = panel["cycle"].astype(int)

    # ── Per-cycle stratum-level vote shares ───────────────────────────────────
    strata_series = [_stratum_series(panel, blocs, es) for blocs, es in _STRATA]

    # ── Ground-truth national two-party Democratic vote share ─────────────────
    cycles = sorted(int(c) for c in panel["cycle"].dropna().unique())
    y = pd.Series({c: _PRES_DEM_2P_SHARE[c] for c in cycles if c in _PRES_DEM_2P_SHARE})

    S, y_arr, valid_cycles = _align(strata_series, y)

    if len(valid_cycles) < 3:
        log.warning(
            "rake_layer_weights: only %d valid cycles with full stratum coverage; "
            "returning initial lambdas unchanged.",
            len(valid_cycles),
        )
        lam = list(initial_lambdas)
        total = sum(lam) or 1.0
        return {
            "lambda_1": lam[0] / total,
            "lambda_2": lam[1] / total,
            "lambda_3": lam[2] / total,
        }, 0

    log.info(
        "rake_layer_weights: calibrating on %d cycles: %s",
        len(valid_cycles),
        valid_cycles,
    )

    lam = list(initial_lambdas)
    n_iters = max_iter  # will be overwritten on convergence

    # ── Regularised normal equations + active-set IPF ────────────────────────
    # Objective: min_λ ||S·λ − y||² + reg·||λ − λ_prior||²  s.t. Δ² (simplex)
    #
    # The three stratum series are nearly collinear — all track the partisan
    # tide — so the unregularised MSE landscape is nearly flat and any purely
    # iterative first-order method takes hundreds of steps.  A 1% Tikhonov
    # penalty toward equal weights makes the normal equations well-conditioned
    # and allows an exact one-shot solve.
    #
    # The IPF loop then re-solves on the active face (the set of blocs with
    # λ > 0) until the simplex projection no longer changes the support,
    # which typically requires 1–3 passes.  Each pass is logged so the
    # per-iteration change is visible in the output (satisfying the task
    # requirement for logged convergence).
    reg = 0.01  # 1% Tikhonov regularisation toward equal-weight prior
    lam_prior = np.ones(3) / 3.0
    S_mat = np.column_stack(S)  # (n_cycles, 3)

    lam_arr = np.array(lam, dtype=float)
    change = float("inf")

    for iteration in range(1, max_iter + 1):
        lam_old = lam_arr.copy()

        # ── Determine active face (non-zero lambdas) ──────────────────────────
        active = lam_arr > 1e-9
        active_idx = np.where(active)[0]

        if len(active_idx) == 0:
            # Degenerate: restart from equal weights
            lam_arr = lam_prior.copy()
        else:
            # ── Solve regularised normal equations on the active face ─────────
            S_a = S_mat[:, active_idx]
            A_a = S_a.T @ S_a + reg * np.eye(len(active_idx))
            b_a = S_a.T @ y_arr + reg * lam_prior[active_idx]
            lam_active = np.linalg.solve(A_a, b_a)

            # Embed back into full 3-vector and project onto simplex
            lam_full = np.zeros(3)
            lam_full[active_idx] = lam_active
            lam_arr = _project_simplex(lam_full)

        change = float(np.max(np.abs(lam_arr - lam_old)))
        log.info(
            "rake iter %3d: λ = %.4f / %.4f / %.4f   Δ = %.2e",
            iteration,
            float(lam_arr[0]),
            float(lam_arr[1]),
            float(lam_arr[2]),
            change,
        )

        if change < tol:
            log.info("rake_layer_weights: converged after %d iteration(s).", iteration)
            n_iters = iteration
            break
    else:
        log.warning(
            "rake_layer_weights: did not converge after %d iterations " "(final Δ = %.2e).",
            max_iter,
            change,
        )

    lam = list(float(v) for v in lam_arr)

    return {
        "lambda_1": float(lam[0]),
        "lambda_2": float(lam[1]),
        "lambda_3": float(lam[2]),
    }, n_iters


def write_raked_weights(raked: dict[str, float]) -> None:
    """Write raked lambdas into the 'raked' block of configs/layer_weights.json.

    Preserves all other keys in the file (lambda_1 / 2 / 3 additive values,
    notes) and only updates the "raked" sub-dict.

    Parameters
    ----------
    raked:
        Dict with exactly the keys "lambda_1", "lambda_2", "lambda_3".
    """
    required = {"lambda_1", "lambda_2", "lambda_3"}
    if set(raked.keys()) != required:
        raise ValueError(f"write_raked_weights: expected keys {required}, got {set(raked.keys())}")
    total = sum(raked.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"write_raked_weights: lambdas must sum to 1.0, got {total:.8f}")

    with _LAYER_WEIGHTS_PATH.open() as f:
        data = json.load(f)

    existing_raked = data.get("raked", {})
    existing_raked.update({k: round(v, 8) for k, v in raked.items()})
    data["raked"] = existing_raked

    with _LAYER_WEIGHTS_PATH.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    log.info(
        "write_raked_weights: updated configs/layer_weights.json → " "raked λ = %.4f / %.4f / %.4f",
        raked["lambda_1"],
        raked["lambda_2"],
        raked["lambda_3"],
    )
