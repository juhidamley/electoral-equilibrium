import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from electoral.artifacts import EquilibriumData, SimulationData
from electoral.core.types import CANONICAL_RACES
from electoral.simulation.montecarlo import run_ilr_montecarlo

RACES = list(CANONICAL_RACES)


def _make_equilibrium(mu_val: float, target: float = 0.535) -> EquilibriumData:
    return EquilibriumData(
        method="test",
        party="democrat",
        shock="test_shock",
        weights={r: 0.2 for r in RACES},
        mu_shifted={r: mu_val for r in RACES},
        feasible=True,
        target_met=mu_val >= target,
        target=target,
    )


def _make_config(seed: int = 42) -> MagicMock:
    config = MagicMock()
    config.derive_seed.return_value = seed
    return config


def test_smoke_completes_fast():
    t0 = time.time()
    run_ilr_montecarlo(_make_equilibrium(0.55), _make_config(), n_simulations=100)
    assert time.time() - t0 < 2.0


def test_win_probability_in_range():
    result = run_ilr_montecarlo(_make_equilibrium(0.55), _make_config(), n_simulations=1000)
    assert 0.0 <= result.win_probability <= 1.0


def test_high_mu_high_win_prob():
    result = run_ilr_montecarlo(_make_equilibrium(0.8), _make_config(), n_simulations=1000)
    assert result.win_probability > 0.95


def test_low_mu_low_win_prob():
    result = run_ilr_montecarlo(_make_equilibrium(0.2), _make_config(), n_simulations=1000)
    assert result.win_probability < 0.05


def test_same_seed_deterministic():
    eq = _make_equilibrium(0.55)
    result1 = run_ilr_montecarlo(eq, _make_config(seed=42), n_simulations=500)
    result2 = run_ilr_montecarlo(eq, _make_config(seed=42), n_simulations=500)
    assert result1.to_dict() == result2.to_dict()


def test_percentiles_shape():
    result = run_ilr_montecarlo(_make_equilibrium(0.55), _make_config(), n_simulations=100)
    for bloc, pcts in result.percentiles.items():
        assert len(pcts) == 5, f"{bloc}: expected 5 percentiles, got {len(pcts)}"
        for p in pcts:
            assert 0.0 <= p <= 1.0, f"{bloc}: percentile {p} out of [0, 1]"


# ── Tests retained from earlier spec ─────────────────────────────────────────


def test_montecarlo_degenerate():
    """Highly concentrated weights (w0=0.99) must still produce valid output."""
    races = RACES
    weights = {races[0]: 0.99, **{r: 0.0025 for r in races[1:]}}
    eq = EquilibriumData(
        method="test",
        party="democrat",
        shock="degenerate_test",
        weights=weights,
        mu_shifted={r: 0.50 for r in races},
        feasible=True,
        target_met=False,
        target=0.51,
    )
    result = run_ilr_montecarlo(eq, _make_config(), n_simulations=1000)
    assert 0.0 <= result.win_probability <= 1.0
    assert 0.0 <= result.win_probability_low <= result.win_probability_high <= 1.0
    for bloc, pcts in result.percentiles.items():
        for p in pcts:
            assert np.isfinite(p) and 0.0 <= p <= 1.0, f"{bloc}: bad percentile {p}"


def test_montecarlo_zero_weight_raises():
    """Exact zero weight must raise ValueError, not silently floor."""
    races = RACES
    eq = EquilibriumData(
        method="test",
        party="democrat",
        shock="zero_weight",
        weights={races[0]: 1.0, **{r: 0.0 for r in races[1:]}},
        mu_shifted={r: 0.50 for r in races},
        feasible=True,
        target_met=False,
        target=0.51,
    )
    with pytest.raises(ValueError, match="zero-weight blocs"):
        run_ilr_montecarlo(eq, _make_config(), n_simulations=100)
