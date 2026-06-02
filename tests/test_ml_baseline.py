"""Tests for electoral/models/ml_baseline.py — estimate_moments()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.models.ml_baseline import MomentEstimates, estimate_moments

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


# ── winning cycle identification ───────────────────────────────────────────────


def test_dem_winning_cycles(toy):
    result = estimate_moments(toy, "democrat")
    assert result.winning_cycles == [2020]


def test_rep_winning_cycles(toy):
    result = estimate_moments(toy, "republican")
    assert result.winning_cycles == [2016]


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


# ── PSD safeguard ─────────────────────────────────────────────────────────────


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
