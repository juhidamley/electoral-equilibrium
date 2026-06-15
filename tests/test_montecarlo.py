import numpy as np
import pytest
from unittest.mock import MagicMock

from electoral.artifacts import EquilibriumData
from electoral.core.types import CANONICAL_RACES
from electoral.simulation.montecarlo import run_ilr_montecarlo


def _make_config(seed: int = 42) -> MagicMock:
    config = MagicMock()
    config.derive_seed.return_value = seed
    return config


def test_montecarlo_degenerate():
    """With degenerate weights (w0=0.99, rest=0.0025), all samples
    must still sum to 1.0 and win_probability must be in [0, 1]."""
    races = list(CANONICAL_RACES)
    weights = {races[0]: 0.99}
    for r in races[1:]:
        weights[r] = 0.0025

    equilibrium = EquilibriumData(
        method="test",
        party="democrat",
        shock="degenerate_test",
        weights=weights,
        mu_shifted={r: 0.50 for r in races},
        feasible=True,
        target_met=False,
        target=0.51,
    )

    result = run_ilr_montecarlo(equilibrium, _make_config(), n_simulations=1000)

    assert 0.0 <= result.win_probability <= 1.0
    assert 0.0 <= result.win_probability_low <= result.win_probability_high <= 1.0

    for bloc, pcts in result.percentiles.items():
        for p in pcts:
            assert np.isfinite(p), f"Non-finite percentile for {bloc}: {p}"
            assert 0.0 <= p <= 1.0, f"Percentile out of [0,1] for {bloc}: {p}"


def test_montecarlo_zero_weight_raises():
    """Exact zero weight must raise ValueError, not silently floor."""
    races = list(CANONICAL_RACES)
    weights = {races[0]: 1.0}
    for r in races[1:]:
        weights[r] = 0.0

    equilibrium = EquilibriumData(
        method="test",
        party="democrat",
        shock="zero_weight",
        weights=weights,
        mu_shifted={r: 0.50 for r in races},
        feasible=True,
        target_met=False,
        target=0.51,
    )

    with pytest.raises(ValueError, match="zero-weight blocs"):
        run_ilr_montecarlo(equilibrium, _make_config(), n_simulations=100)
