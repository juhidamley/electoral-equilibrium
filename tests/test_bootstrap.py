import numpy as np
import pandas as pd
import pytest

from electoral.models.bootstrap import bootstrap_cov_weighted, ledoit_wolf_cov


@pytest.fixture
def delta_matrix():
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 5))


def test_symmetric_psd(delta_matrix):
    cov = ledoit_wolf_cov(delta_matrix)
    assert np.allclose(cov, cov.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10), f"Non-PSD eigenvalues: {eigvals[eigvals < -1e-10]}"


def test_condition_number(delta_matrix):
    cov = ledoit_wolf_cov(delta_matrix)
    cond = np.linalg.cond(cov)
    assert np.isfinite(cond), "Condition number is not finite"
    assert cond < 1e6, f"Condition number too large: {cond:.2e}"


def test_deterministic(delta_matrix):
    cov1 = ledoit_wolf_cov(delta_matrix)
    cov2 = ledoit_wolf_cov(delta_matrix)
    assert np.allclose(cov1, cov2, atol=1e-12)


def test_shape(delta_matrix):
    cov = ledoit_wolf_cov(delta_matrix)
    n = delta_matrix.shape[1]
    assert cov.shape == (n, n)


def test_input_validation_1d():
    with pytest.raises(ValueError, match="2D"):
        ledoit_wolf_cov(np.array([0.1, 0.2, 0.3]))


def test_input_validation_single_cycle():
    with pytest.raises(ValueError, match=">= 2"):
        ledoit_wolf_cov(np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))


def test_weighted_symmetric_psd():
    rng = np.random.default_rng(7)
    cycles = [2000, 2004, 2008, 2012, 2016, 2020]
    blocs = ["african_american", "asian", "latino", "other_race", "white"]

    rows = [
        {"cycle": cycle, "bloc": bloc, "delta": float(rng.normal(0, 0.05))}
        for cycle in cycles
        for bloc in blocs
    ]
    panel_df = pd.DataFrame(rows)
    elasticities = {c: float(abs(rng.normal(1.0, 0.5))) for c in cycles}

    cov = bootstrap_cov_weighted(panel_df, elasticities, n_bootstrap=500, seed=42)

    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10), f"Non-PSD eigenvalues: {eigvals[eigvals < -1e-10]}"


def test_weighted_higher_elasticity_dominates():
    # High-E cycles: african_american and asian move together (positive correlation).
    # Low-E cycles: african_american and asian move opposite (negative correlation).
    # Giving high elasticity weight to the correlated cycles should produce
    # cov[aa, asian] > 0, while uniform weighting cancels to ≈ 0.
    cycles_high = [2000, 2004, 2008]
    cycles_low = [2012, 2016, 2020]
    blocs = ["african_american", "asian", "latino", "other_race", "white"]

    high_deltas = [(0.10, 0.10), (0.30, 0.30), (0.50, 0.50)]  # positively correlated
    low_deltas = [(0.50, 0.10), (0.30, 0.30), (0.10, 0.50)]   # negatively correlated

    rows = []
    for (aa, asian), cycle in zip(high_deltas, cycles_high):
        rows += [
            {"cycle": cycle, "bloc": "african_american", "delta": aa},
            {"cycle": cycle, "bloc": "asian", "delta": asian},
            *[{"cycle": cycle, "bloc": b, "delta": 0.0} for b in blocs[2:]],
        ]
    for (aa, asian), cycle in zip(low_deltas, cycles_low):
        rows += [
            {"cycle": cycle, "bloc": "african_american", "delta": aa},
            {"cycle": cycle, "bloc": "asian", "delta": asian},
            *[{"cycle": cycle, "bloc": b, "delta": 0.0} for b in blocs[2:]],
        ]

    panel_df = pd.DataFrame(rows)

    # High weights on positively-correlated cycles
    elasticities_high = {c: 10.0 for c in cycles_high} | {c: 0.01 for c in cycles_low}
    # Uniform weights → correlated and anti-correlated cycles cancel → cov[0,1] ≈ 0
    elasticities_uniform = {c: 1.0 for c in cycles_high + cycles_low}

    cov_weighted = bootstrap_cov_weighted(panel_df, elasticities_high, n_bootstrap=2000, seed=42)
    cov_uniform = bootstrap_cov_weighted(panel_df, elasticities_uniform, n_bootstrap=2000, seed=42)

    # blocs sorted alphabetically: african_american=0, asian=1
    assert cov_weighted[0, 1] > cov_uniform[0, 1], (
        f"Weighted cov[aa,asian]={cov_weighted[0,1]:.4f} should exceed "
        f"uniform cov[aa,asian]={cov_uniform[0,1]:.4f}"
    )
