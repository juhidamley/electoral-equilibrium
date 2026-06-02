"""Tests for electoral/portfolios/constraints.py — ConstraintSpec."""

from __future__ import annotations

import numpy as np
import pytest

from electoral.core.types import CANONICAL_RACES
from electoral.portfolios.constraints import ConstraintSpec
from electoral.portfolios.cvx import solve_baseline

_BLOCS = list(CANONICAL_RACES)  # 5 canonical race blocs


# ── ConstraintSpec.default ────────────────────────────────────────────────────


def test_default_has_empty_bounds():
    spec = ConstraintSpec.default(_BLOCS)
    assert spec.lower == {}
    assert spec.upper == {}


def test_default_lower_bound_is_zero():
    spec = ConstraintSpec.default(_BLOCS)
    for b in _BLOCS:
        assert spec.lower_bound(b) == 0.0


def test_default_upper_bound_is_one():
    spec = ConstraintSpec.default(_BLOCS)
    for b in _BLOCS:
        assert spec.upper_bound(b) == 1.0


# ── ConstraintSpec.from_bounds ────────────────────────────────────────────────


def test_from_bounds_lower_only():
    spec = ConstraintSpec.from_bounds(_BLOCS, lower={"african_american": 0.05})
    assert spec.lower_bound("african_american") == pytest.approx(0.05)
    assert spec.lower_bound("white") == 0.0


def test_from_bounds_upper_only():
    spec = ConstraintSpec.from_bounds(_BLOCS, upper={"white": 0.40})
    assert spec.upper_bound("white") == pytest.approx(0.40)
    assert spec.upper_bound("latino") == 1.0


def test_from_bounds_raises_on_unknown_bloc():
    with pytest.raises(ValueError, match="not in blocs"):
        ConstraintSpec.from_bounds(_BLOCS, lower={"zoroastrian": 0.10})


def test_from_bounds_raises_when_lb_gt_ub():
    with pytest.raises(ValueError, match="lower.*>.*upper"):
        ConstraintSpec.from_bounds(_BLOCS, lower={"white": 0.50}, upper={"white": 0.30})


def test_from_bounds_raises_when_sum_lb_gt_one():
    # 5 blocs × 0.25 = 1.25 > 1
    with pytest.raises(ValueError, match="sum of lower bounds"):
        ConstraintSpec.from_bounds(_BLOCS, lower={b: 0.25 for b in _BLOCS})


def test_from_bounds_raises_when_sum_ub_lt_one():
    # 5 blocs × 0.15 = 0.75 < 1
    with pytest.raises(ValueError, match="sum of upper bounds"):
        ConstraintSpec.from_bounds(_BLOCS, upper={b: 0.15 for b in _BLOCS})


def test_from_bounds_raises_on_lb_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ConstraintSpec.from_bounds(_BLOCS, lower={"white": -0.01})


def test_from_bounds_raises_on_ub_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ConstraintSpec.from_bounds(_BLOCS, upper={"white": 1.01})


# ── max_achievable_loyalty ────────────────────────────────────────────────────


def test_max_achievable_no_bounds_equals_max_mu():
    mu = {b: v for b, v in zip(_BLOCS, [0.89, 0.67, 0.58, 0.47, 0.58])}
    spec = ConstraintSpec.default(_BLOCS)
    # Greedy: all weight to the highest-loyalty bloc → max mu
    assert spec.max_achievable_loyalty(mu) == pytest.approx(max(mu.values()))


def test_max_achievable_with_upper_cap():
    mu = {
        "african_american": 0.89,
        "white": 0.47,
        "latino": 0.67,
        "asian": 0.58,
        "other_race": 0.58,
    }
    # Cap african_american at 0.30 → greedy puts 0.30 there, rest on latino
    spec = ConstraintSpec.from_bounds(list(mu), upper={"african_american": 0.30})
    val = spec.max_achievable_loyalty(mu)
    # 0.30 * 0.89 + 0.70 * 0.67 = 0.267 + 0.469 = 0.736
    assert val == pytest.approx(0.30 * 0.89 + 0.70 * 0.67, abs=1e-6)


# ── solve_baseline integration ────────────────────────────────────────────────


def test_solve_baseline_respects_lower_bound():
    """A lower bound floors the allocation for that bloc."""
    mu = {"a": 0.80, "b": 0.60}
    cov = np.eye(2)
    spec = ConstraintSpec.from_bounds(["a", "b"], lower={"b": 0.20})
    weights = solve_baseline(mu, cov, target=0.65, blocs=["a", "b"], spec=spec)
    assert weights["b"] >= 0.20 - 1e-6


def test_solve_baseline_respects_upper_bound():
    """An upper bound caps the allocation for a high-loyalty bloc."""
    mu = {"a": 0.90, "b": 0.50}
    cov = np.eye(2)
    spec = ConstraintSpec.from_bounds(["a", "b"], upper={"a": 0.60})
    weights = solve_baseline(mu, cov, target=0.65, blocs=["a", "b"], spec=spec)
    assert weights["a"] <= 0.60 + 1e-6


def test_solve_baseline_default_spec_unchanged():
    """Passing spec=None and passing ConstraintSpec.default produce identical weights."""
    mu = {b: v for b, v in zip(_BLOCS, [0.89, 0.67, 0.58, 0.47, 0.58])}
    cov = np.eye(5) * 0.01
    w_none = solve_baseline(mu, cov, target=0.535)
    w_spec = solve_baseline(mu, cov, target=0.535, spec=ConstraintSpec.default(_BLOCS))
    for b in _BLOCS:
        assert w_none[b] == pytest.approx(w_spec[b], abs=1e-6)


def test_solve_baseline_spec_blocs_override_blocs_arg():
    """When spec is provided its blocs take precedence over the blocs keyword."""
    mu = {"a": 0.80, "b": 0.60}
    cov = np.eye(2)
    spec = ConstraintSpec.default(["a", "b"])
    weights = solve_baseline(mu, cov, target=0.65, blocs=["b", "a"], spec=spec)
    # Result should be keyed by spec.blocs order, not the blocs arg
    assert set(weights.keys()) == {"a", "b"}


def test_solve_baseline_infeasible_due_to_upper_cap():
    """If upper bounds prevent reaching target, a clear ValueError is raised."""
    mu = {"a": 0.80, "b": 0.60}
    cov = np.eye(2)
    # Cap a at 0.10 → max loyalty = 0.10*0.80 + 0.90*0.60 = 0.62 < 0.70
    spec = ConstraintSpec.from_bounds(["a", "b"], upper={"a": 0.10})
    with pytest.raises(ValueError, match="target"):
        solve_baseline(mu, cov, target=0.70, blocs=["a", "b"], spec=spec)
