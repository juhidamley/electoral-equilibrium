import time

import numpy as np

from electoral.optimization.simplex import project_simplex, project_simplex_batch


def test_already_valid_simplex():
    v = np.array([0.3, 0.5, 0.2])
    result = project_simplex(v)
    assert np.allclose(result, v, atol=1e-9)


def test_negative_components_zeroed():
    v = np.array([1.5, -0.3, -0.2])
    result = project_simplex(v)
    assert all(r >= 0 for r in result)
    assert result[1] == 0.0 and result[2] == 0.0
    assert abs(result.sum() - 1.0) < 1e-9


def test_output_sums_to_one():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        v = rng.uniform(-2, 2, size=10)
        result = project_simplex(v)
        assert abs(result.sum() - 1.0) < 1e-9
        assert all(r >= -1e-12 for r in result)


def test_batch_rows_sum_to_one():
    rng = np.random.default_rng(0)
    W = rng.uniform(-2, 2, size=(100, 5))
    result = project_simplex_batch(W)
    sums = result.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)


def test_batch_non_negative():
    rng = np.random.default_rng(1)
    W = rng.uniform(-2, 2, size=(100, 5))
    result = project_simplex_batch(W)
    assert np.all(result >= -1e-12)


def test_batch_already_valid_unchanged():
    rng = np.random.default_rng(2)
    raw = rng.dirichlet(np.ones(5), size=20)
    result = project_simplex_batch(raw)
    assert np.allclose(result, raw, atol=1e-9)


def test_batch_10k_under_5s():
    rng = np.random.default_rng(42)
    W = rng.uniform(-2, 2, size=(10_000, 10))
    t0 = time.time()
    result = project_simplex_batch(W)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"10k rows took {elapsed:.2f}s (>5s threshold)"
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-9)
