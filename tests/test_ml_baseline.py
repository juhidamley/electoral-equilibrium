"""Tests for electoral/models/ml_baseline.py — estimate_moments() and fit_gp_classifier()."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.models.ml_baseline import (
    GPBaselineResult,
    LocoFoldResult,
    MomentEstimates,
    estimate_moments,
    fit_gp_classifier,
    ground_truth_winning_cycles,
    psd_repair,
    save_loco_json,
)

TOY_PANEL = Path("tests/fixtures/toy_panel.csv")

# Approximate national two-party vote shares in the toy panel (both cycles have
# all 5 race blocs present, so no renormalization needed):
#   2016 Dem: 0.12*0.89 + 0.11*0.66 + 0.05*0.65 + 0.62*0.37 + 0.10*0.56 = 0.4973 < 0.50
#   2020 Dem: 0.12*0.87 + 0.11*0.63 + 0.05*0.72 + 0.62*0.41 + 0.10*0.55 = 0.5189 > 0.50
# → Democrat winning cycles: [2020]
# → Republican (flipped shares) winning cycles: [2016]


@pytest.fixture
def toy() -> pd.DataFrame:
    return pd.read_csv(TOY_PANEL)


# ── panel-derived winning cycle identification (no override) ───────────────────


def test_dem_winning_cycles(toy):
    result = estimate_moments(toy, "democrat")
    assert result.winning_cycles == [2020]


def test_rep_winning_cycles(toy):
    result = estimate_moments(toy, "republican")
    assert result.winning_cycles == [2016]


# ── ground_truth_winning_cycles ───────────────────────────────────────────────


def test_ground_truth_dem_includes_known_wins():
    cycles = ground_truth_winning_cycles("democrat")
    for c in [1948, 1960, 1964, 1976, 1992, 1996, 2008, 2012, 2020]:
        assert c in cycles, f"expected Democrat win {c} missing from ground truth"


def test_ground_truth_dem_excludes_known_losses():
    cycles = ground_truth_winning_cycles("democrat")
    for c in [1952, 1956, 1972, 1980, 1984, 1988, 2004, 2024]:
        assert c not in cycles, f"expected Democrat loss {c} incorrectly in ground truth"


def test_ground_truth_1992_is_dem_win():
    # 1992: Clinton won; panel misclassifies as R due to Perot 1-flip artifact.
    assert 1992 in ground_truth_winning_cycles("democrat")
    assert 1992 not in ground_truth_winning_cycles("republican")


def test_ground_truth_1988_is_rep_win():
    # 1988: Bush won; panel incorrectly flags as D.
    assert 1988 not in ground_truth_winning_cycles("democrat")
    assert 1988 in ground_truth_winning_cycles("republican")


def test_ground_truth_rep_includes_known_wins():
    cycles = ground_truth_winning_cycles("republican")
    for c in [1952, 1956, 1968, 1972, 1980, 1984, 1988, 2004, 2024]:
        assert c in cycles, f"expected Republican win {c} missing from ground truth"


def test_ground_truth_raises_on_invalid_party():
    with pytest.raises(ValueError, match="party must be"):
        ground_truth_winning_cycles("libertarian")  # type: ignore[arg-type]


def test_ground_truth_dem_rep_are_complementary():
    dem = set(ground_truth_winning_cycles("democrat"))
    rep = set(ground_truth_winning_cycles("republican"))
    from electoral.models.ml_baseline import _PRES_DEM_2P_SHARE
    all_cycles = set(_PRES_DEM_2P_SHARE.keys())
    assert dem | rep == all_cycles
    assert dem & rep == set()


# ── winning_cycles override parameter ─────────────────────────────────────────


def test_winning_cycles_override_used_for_mu(toy):
    # Force both cycles to be "winning" — mu should average over 2016 + 2020.
    result = estimate_moments(toy, "democrat", winning_cycles=[2016, 2020])
    expected_aa = (0.89 + 0.87) / 2
    assert result.mu_race["african_american"] == pytest.approx(expected_aa)
    assert result.winning_cycles == [2016, 2020]


def test_winning_cycles_override_empty_gives_nan_mu(toy):
    result = estimate_moments(toy, "democrat", winning_cycles=[])
    assert result.winning_cycles == []
    assert all(np.isnan(v) for v in result.mu_race.values())


def test_winning_cycles_override_does_not_affect_sigma(toy):
    # Sigma is always derived from all cycles regardless of winning_cycles.
    result_default = estimate_moments(toy, "democrat")
    result_override = estimate_moments(toy, "democrat", winning_cycles=[2016])
    np.testing.assert_array_equal(result_default.Sigma, result_override.Sigma)


# ── mu_race ───────────────────────────────────────────────────────────────────


def test_dem_mu_race_equals_2020_vote_shares(toy):
    result = estimate_moments(toy, "democrat")
    assert result.mu_race["african_american"] == pytest.approx(0.87)
    assert result.mu_race["latino"] == pytest.approx(0.63)
    assert result.mu_race["asian"] == pytest.approx(0.72)
    assert result.mu_race["white"] == pytest.approx(0.41)
    assert result.mu_race["other_race"] == pytest.approx(0.55)


def test_rep_mu_race_uses_flipped_vote_share(toy):
    result = estimate_moments(toy, "republican")
    # Winning cycle is 2016; Republican share = 1 - Democrat share
    assert result.mu_race["african_american"] == pytest.approx(1.0 - 0.89)
    assert result.mu_race["white"] == pytest.approx(1.0 - 0.37)
    assert result.mu_race["latino"] == pytest.approx(1.0 - 0.66)


def test_mu_race_has_all_canonical_keys(toy):
    result = estimate_moments(toy, "democrat")
    assert set(result.mu_race.keys()) == set(CANONICAL_RACES)


# ── mu_religion ───────────────────────────────────────────────────────────────


def test_dem_mu_religion_over_winning_cycles(toy):
    result = estimate_moments(toy, "democrat")
    # 2020 data in toy panel: evangelical=0.22, catholic=0.50, secular=0.68
    assert result.mu_religion["evangelical"] == pytest.approx(0.22)
    assert result.mu_religion["catholic"] == pytest.approx(0.50)
    assert result.mu_religion["secular"] == pytest.approx(0.68)


def test_missing_religion_blocs_produce_nan(toy):
    result = estimate_moments(toy, "democrat")
    # protestant, jewish, muslim, other_rel are absent from toy panel
    for bloc in ("protestant", "jewish", "muslim", "other_rel"):
        assert np.isnan(result.mu_religion[bloc]), f"expected NaN for {bloc}"


def test_mu_religion_has_all_canonical_keys(toy):
    result = estimate_moments(toy, "democrat")
    assert set(result.mu_religion.keys()) == set(CANONICAL_RELIGIONS)


# ── mu_gender ─────────────────────────────────────────────────────────────────


def test_dem_mu_gender_women(toy):
    result = estimate_moments(toy, "democrat")
    # 2020 women = 0.57 (winning cycle for Dem)
    assert result.mu_gender["women"] == pytest.approx(0.57)


def test_dem_mu_gender_men_nan(toy):
    # 2020/men has empty vote_share in toy panel → dropped → no data for winning cycle 2020
    result = estimate_moments(toy, "democrat")
    assert np.isnan(result.mu_gender["men"])


def test_mu_gender_other_gender_nan(toy):
    result = estimate_moments(toy, "democrat")
    # other_gender absent from toy panel
    assert np.isnan(result.mu_gender["other_gender"])


def test_mu_gender_has_all_canonical_keys(toy):
    result = estimate_moments(toy, "democrat")
    assert set(result.mu_gender.keys()) == set(CANONICAL_GENDERS)


# ── Sigma ─────────────────────────────────────────────────────────────────────


def test_sigma_shape_is_5x5(toy):
    result = estimate_moments(toy, "democrat")
    assert result.Sigma.shape == (5, 5)


def test_sigma_is_symmetric(toy):
    result = estimate_moments(toy, "democrat")
    np.testing.assert_allclose(result.Sigma, result.Sigma.T, atol=1e-12)


def test_sigma_is_psd(toy):
    result = estimate_moments(toy, "democrat")
    eigenvalues = np.linalg.eigvalsh(result.Sigma)
    assert np.all(eigenvalues >= -1e-12), f"min eigenvalue = {eigenvalues.min():.3e}"


def test_sigma_uses_all_cycles_not_just_winning(toy):
    # With only 1 winning cycle (2020) for Democrat, Sigma computed from 1 row
    # would be NaN (ddof=1) → zeroed to all-zeros. Sigma from both cycles
    # should be nonzero.
    result = estimate_moments(toy, "democrat")
    assert not np.allclose(result.Sigma, np.zeros((5, 5)))


def test_sigma_dtype_is_float64(toy):
    result = estimate_moments(toy, "democrat")
    assert result.Sigma.dtype == np.float64


# ── race_blocs label order ────────────────────────────────────────────────────


def test_race_blocs_matches_canonical_order(toy):
    result = estimate_moments(toy, "democrat")
    assert result.race_blocs == list(CANONICAL_RACES)


# ── psd_repair ────────────────────────────────────────────────────────────────


def test_psd_repair_near_singular_matrix():
    # Rank-1 outer product with a tiny negative perturbation on the diagonal
    # — a known near-singular matrix that is NOT positive semi-definite.
    v = np.array([1.0, 0.5, 0.2, 0.8, 0.3])
    M = np.outer(v, v)  # rank-1, PSD
    M[0, 0] -= 0.05  # push smallest eigenvalue negative

    assert np.linalg.eigvalsh(M).min() < 0, "precondition: M must be non-PSD"

    repaired = psd_repair(M, eps=1e-6)

    eigenvalues = np.linalg.eigvalsh(repaired)
    assert np.all(eigenvalues >= 0.0), f"still non-PSD after repair: {eigenvalues}"


def test_psd_repair_already_psd_unchanged():
    # A strictly positive diagonal matrix is already PSD — psd_repair must
    # return the identical object (no copy, no shift).
    M = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    result = psd_repair(M)
    assert result is M


def test_psd_repair_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    M = A @ A.T  # guaranteed PSD
    result = psd_repair(M)
    assert result.shape == (5, 5)
    assert result.dtype == np.float64


def test_psd_repair_shift_amount_respects_eps():
    # With eps=0.01, all eigenvalues of the repaired matrix must be >= 0,
    # and the minimum should be close to eps (since we added exactly
    # (-min_eig + eps) * I).
    v = np.array([1.0, 0.5, 0.2, 0.8, 0.3])
    M = np.outer(v, v)
    M[0, 0] -= 0.05  # force negative eigenvalue
    eps = 0.01
    repaired = psd_repair(M, eps=eps)
    min_eig = np.linalg.eigvalsh(repaired).min()
    assert min_eig >= 0.0
    assert min_eig == pytest.approx(eps, abs=1e-10)


# ── PSD safeguard (via estimate_moments) ──────────────────────────────────────


def test_psd_safeguard_triggers_on_degenerate_panel():
    # Build a panel where bloc "asian" only appears in 1 cycle → NaN variance
    # → zeroed diagonal entry. Combined with variation in other blocs the
    # resulting Sigma may be non-PSD; estimate_moments must still return PSD Sigma.
    rows = []
    for bloc in CANONICAL_RACES:
        base = {
            "african_american": 0.87,
            "latino": 0.63,
            "asian": 0.72,
            "white": 0.41,
            "other_race": 0.55,
        }[bloc]
        for i, cycle in enumerate([2016, 2020, 2024]):
            if bloc == "asian" and cycle != 2016:
                # asian absent in 2020 and 2024 → single-cycle bloc
                continue
            rows.append({"cycle": cycle, "bloc": bloc, "vote_share": base + i * 0.01})
    df = pd.DataFrame(rows)
    result = estimate_moments(df, "democrat")
    eigenvalues = np.linalg.eigvalsh(result.Sigma)
    assert np.all(eigenvalues >= -1e-12), f"PSD safeguard failed: min eig = {eigenvalues.min():.3e}"


# ── empty winning cycles ──────────────────────────────────────────────────────


def test_empty_winning_cycles_all_mu_nan():
    rows = [
        {"cycle": cycle, "bloc": bloc, "vote_share": 0.20}
        for cycle in [2016, 2020]
        for bloc in CANONICAL_RACES
    ]
    df = pd.DataFrame(rows)
    result = estimate_moments(df, "democrat")
    assert result.winning_cycles == []
    assert all(np.isnan(v) for v in result.mu_race.values())
    assert all(np.isnan(v) for v in result.mu_religion.values())
    assert all(np.isnan(v) for v in result.mu_gender.values())


# ── return type ───────────────────────────────────────────────────────────────


def test_returns_moment_estimates_instance(toy):
    result = estimate_moments(toy, "democrat")
    assert isinstance(result, MomentEstimates)


# ── error handling ────────────────────────────────────────────────────────────


def test_raises_on_invalid_party(toy):
    with pytest.raises(ValueError, match="party must be"):
        estimate_moments(toy, "libertarian")  # type: ignore[arg-type]


def test_raises_on_missing_vote_share_column():
    df = pd.DataFrame({"cycle": [2020], "bloc": ["white"]})
    with pytest.raises(ValueError, match="missing columns"):
        estimate_moments(df, "democrat")


def test_raises_on_missing_cycle_column():
    df = pd.DataFrame({"bloc": ["white"], "vote_share": [0.4]})
    with pytest.raises(ValueError, match="missing columns"):
        estimate_moments(df, "democrat")


# ══════════════════════════════════════════════════════════════════════════════
# fit_gp_classifier — GP win-probability model with LOCO-CV
# ══════════════════════════════════════════════════════════════════════════════

# ── shared fixtures ───────────────────────────────────────────────────────────


def _gp_panel(cycles: list[int], seed: int = 7) -> pd.DataFrame:
    """Return a synthetic panel with varied vote_share for all CANONICAL_RACES."""
    rng = np.random.default_rng(seed)
    rows = [
        {"cycle": c, "bloc": b, "vote_share": float(rng.uniform(0.3, 0.85))}
        for c in cycles
        for b in CANONICAL_RACES
    ]
    return pd.DataFrame(rows)


_CYCLES_6 = [2004, 2008, 2012, 2016, 2020, 2024]
_WIN_6 = [2008, 2012, 2020]  # 3 wins, 3 losses


@pytest.fixture(scope="module")
def gp_result_6() -> GPBaselineResult:
    df = _gp_panel(_CYCLES_6)
    return fit_gp_classifier(
        df, "democrat", winning_cycles=_WIN_6, rng=np.random.default_rng(0)
    )


# ── fold count and structure ──────────────────────────────────────────────────


def test_gp_loco_fold_count(gp_result_6):
    assert len(gp_result_6.folds) == len(_CYCLES_6)


def test_gp_fold_cycle_is_python_int(gp_result_6):
    for f in gp_result_6.folds:
        assert type(f.cycle) is int, f"expected int, got {type(f.cycle).__name__}"


def test_gp_fold_y_true_is_binary(gp_result_6):
    for f in gp_result_6.folds:
        assert f.y_true in (0, 1)


def test_gp_fold_cycles_match_panel(gp_result_6):
    assert [f.cycle for f in gp_result_6.folds] == sorted(_CYCLES_6)


def test_gp_result_is_frozen():
    r = GPBaselineResult(party="x", folds=(), accuracy=0.5, brier_score=0.1)
    with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
        r.party = "y"  # type: ignore[misc]


# ── prob_win range ────────────────────────────────────────────────────────────


def test_gp_prob_win_in_unit_interval(gp_result_6):
    for f in gp_result_6.folds:
        if not math.isnan(f.prob_win):
            assert 0.0 <= f.prob_win <= 1.0, f"cycle {f.cycle}: prob_win={f.prob_win}"


def test_gp_prob_std_nonneg(gp_result_6):
    for f in gp_result_6.folds:
        if not math.isnan(f.prob_std):
            assert f.prob_std >= 0.0, f"cycle {f.cycle}: prob_std={f.prob_std}"


def test_gp_prob_std_is_not_bernoulli_std(gp_result_6):
    # GPR posterior std can exceed 0.5; Bernoulli std is always <= 0.5.
    # Verify at least one valid fold has prob_std > 0.5, which proves we are
    # using GPR posterior uncertainty rather than sqrt(p*(1-p)).
    large_stds = [f.prob_std for f in gp_result_6.folds if not math.isnan(f.prob_std) and f.prob_std > 0.5]
    assert large_stds, (
        "No fold had prob_std > 0.5; this suggests sqrt(p*(1-p)) is being used "
        "instead of the GPR posterior std."
    )


# ── single-class training fold → NaN ─────────────────────────────────────────


def test_gp_single_class_fold_is_nan():
    # 3 cycles, first two are winning → holding out cycle 2016 leaves
    # train=[2008(win), 2012(win)], constant labels → NaN fold.
    df = _gp_panel([2008, 2012, 2016])
    result = fit_gp_classifier(df, "democrat", winning_cycles=[2008, 2012])
    held_out = next(f for f in result.folds if f.cycle == 2016)
    assert math.isnan(held_out.prob_win)
    assert math.isnan(held_out.prob_std)


def test_gp_single_class_fold_y_true_correct():
    df = _gp_panel([2008, 2012, 2016])
    result = fit_gp_classifier(df, "democrat", winning_cycles=[2008, 2012])
    held_out = next(f for f in result.folds if f.cycle == 2016)
    assert held_out.y_true == 0  # 2016 is not in winning_cycles


# ── aggregate metrics ─────────────────────────────────────────────────────────


def test_gp_accuracy_is_in_unit_interval(gp_result_6):
    assert 0.0 <= gp_result_6.accuracy <= 1.0


def test_gp_brier_score_is_nonneg(gp_result_6):
    assert gp_result_6.brier_score >= 0.0


def test_gp_accuracy_matches_manual_count(gp_result_6):
    valid = [f for f in gp_result_6.folds if not math.isnan(f.prob_win)]
    expected = sum(1 for f in valid if (f.prob_win >= 0.5) == bool(f.y_true)) / len(valid)
    assert gp_result_6.accuracy == pytest.approx(expected, abs=1e-12)


def test_gp_brier_score_matches_manual(gp_result_6):
    valid = [f for f in gp_result_6.folds if not math.isnan(f.prob_win)]
    expected = sum((f.prob_win - f.y_true) ** 2 for f in valid) / len(valid)
    assert gp_result_6.brier_score == pytest.approx(expected, abs=1e-12)


# ── reproducibility ───────────────────────────────────────────────────────────


def test_gp_reproducible_with_same_seed():
    df = _gp_panel(_CYCLES_6)
    r1 = fit_gp_classifier(df, "democrat", winning_cycles=_WIN_6, rng=np.random.default_rng(99))
    r2 = fit_gp_classifier(df, "democrat", winning_cycles=_WIN_6, rng=np.random.default_rng(99))
    for f1, f2 in zip(r1.folds, r2.folds):
        if not math.isnan(f1.prob_win):
            assert f1.prob_win == pytest.approx(f2.prob_win, abs=1e-12)
            assert f1.prob_std == pytest.approx(f2.prob_std, abs=1e-12)


# ── party flip ────────────────────────────────────────────────────────────────


def test_gp_republican_result_differs_from_democrat():
    df = _gp_panel(_CYCLES_6)
    dem = fit_gp_classifier(df, "democrat", winning_cycles=_WIN_6, rng=np.random.default_rng(0))
    rep = fit_gp_classifier(df, "republican", winning_cycles=_WIN_6, rng=np.random.default_rng(0))
    dem_probs = [f.prob_win for f in dem.folds if not math.isnan(f.prob_win)]
    rep_probs = [f.prob_win for f in rep.folds if not math.isnan(f.prob_win)]
    # With the same winning_cycles list but flipped vote_share, predictions differ.
    assert dem_probs != rep_probs


# ── error handling ────────────────────────────────────────────────────────────


def test_gp_invalid_party_raises():
    with pytest.raises(ValueError, match="party must be"):
        fit_gp_classifier(_gp_panel(_CYCLES_6), "libertarian")  # type: ignore[arg-type]


def test_gp_missing_column_raises():
    df = pd.DataFrame({"cycle": [2020], "bloc": ["white"]})  # no vote_share
    with pytest.raises(ValueError, match="missing columns"):
        fit_gp_classifier(df, "democrat")


def test_gp_too_few_cycles_raises():
    df = _gp_panel([2016, 2020])  # only 2 cycles
    with pytest.raises(ValueError, match="3 cycles"):
        fit_gp_classifier(df, "democrat", winning_cycles=[2020])


# ── save_loco_json ────────────────────────────────────────────────────────────


def test_save_loco_json_schema(gp_result_6):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    try:
        save_loco_json(gp_result_6, p)
        data = json.loads(p.read_text())
        assert set(data.keys()) == {"party", "accuracy", "brier_score", "folds"}
        assert data["party"] == "democrat"
        fold_keys = set(data["folds"][0].keys())
        assert fold_keys == {"cycle", "y_true", "prob_win", "prob_std"}
    finally:
        p.unlink(missing_ok=True)


def test_save_loco_json_cycle_types(gp_result_6):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    try:
        save_loco_json(gp_result_6, p)
        data = json.loads(p.read_text())
        for fold in data["folds"]:
            assert isinstance(fold["cycle"], int)
            assert isinstance(fold["y_true"], int)
    finally:
        p.unlink(missing_ok=True)


def test_save_loco_json_nan_serialised_as_null():
    df = _gp_panel([2008, 2012, 2016])
    result = fit_gp_classifier(df, "democrat", winning_cycles=[2008, 2012])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    try:
        save_loco_json(result, p)
        data = json.loads(p.read_text())
        nan_folds = [fd for fd in data["folds"] if fd["prob_win"] is None]
        assert nan_folds, "Expected at least one null prob_win in JSON"
    finally:
        p.unlink(missing_ok=True)


def test_save_loco_json_creates_parent_dirs(gp_result_6):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sub" / "nested" / "out.json"
        save_loco_json(gp_result_6, p)
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["party"] == "democrat"
