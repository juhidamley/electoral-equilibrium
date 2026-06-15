import numpy as np

from electoral.optimization.simplex import project_simplex


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
