"""Tests for electoral/kernels/raking.py — rake_layer_weights and write_raked_weights."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from electoral.kernels.raking import (
    _project_simplex,
    rake_layer_weights,
    write_raked_weights,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _synthetic_panel(n_cycles: int = 8) -> pd.DataFrame:
    """Minimal 3-stratum panel with predictable structure."""
    import random

    random.seed(0)
    rows = []
    blocs = {
        "race": ["african_american", "latino", "asian", "white", "other_race"],
        "religion": [
            "evangelical",
            "catholic",
            "protestant",
            "secular",
            "jewish",
            "muslim",
            "other_rel",
        ],
        "gender": ["women", "men", "other_gender"],
    }
    cycles = list(range(2000, 2000 + n_cycles * 4, 4))
    for cycle in cycles:
        for stratum_blocs in blocs.values():
            for bloc in stratum_blocs:
                rows.append(
                    {
                        "cycle": cycle,
                        "bloc": bloc,
                        "vote_share": 0.5 + 0.05 * math.sin(cycle / 4),
                        "source": "test",
                    }
                )
    return pd.DataFrame(rows)


# ── _project_simplex ──────────────────────────────────────────────────────────


def test_project_simplex_already_valid():
    v = np.array([0.5, 0.3, 0.2])
    out = _project_simplex(v)
    assert np.allclose(out, v, atol=1e-9)


def test_project_simplex_sums_to_one():
    v = np.array([2.0, -1.0, 0.5])
    out = _project_simplex(v)
    assert pytest.approx(out.sum(), abs=1e-9) == 1.0


def test_project_simplex_nonneg():
    v = np.array([2.0, -1.0, 0.5])
    out = _project_simplex(v)
    assert (out >= 0).all()


def test_project_simplex_uniform():
    v = np.ones(3) / 3
    out = _project_simplex(v)
    assert np.allclose(out, v, atol=1e-9)


# ── rake_layer_weights ────────────────────────────────────────────────────────


def test_rake_returns_three_keys():
    panel = _synthetic_panel()
    raked, _ = rake_layer_weights(panel)
    assert set(raked.keys()) == {"lambda_1", "lambda_2", "lambda_3"}


def test_rake_sums_to_one():
    panel = _synthetic_panel()
    raked, _ = rake_layer_weights(panel)
    assert pytest.approx(sum(raked.values()), abs=1e-6) == 1.0


def test_rake_all_nonneg():
    panel = _synthetic_panel()
    raked, _ = rake_layer_weights(panel)
    for k, v in raked.items():
        assert v >= 0.0, f"{k} = {v} < 0"


def test_rake_converges():
    panel = _synthetic_panel()
    _, n_iters = rake_layer_weights(panel, max_iter=50)
    assert n_iters < 50, f"Expected convergence before 50 iters, got {n_iters}"


def test_rake_fewer_than_3_valid_cycles_returns_initial():
    # Panel with only 2 cycles that have ground-truth data → falls back to initial
    panel = pd.DataFrame(
        [
            {"cycle": 2000, "bloc": "african_american", "vote_share": 0.90, "source": "t"},
            {"cycle": 2000, "bloc": "white", "vote_share": 0.42, "source": "t"},
            {"cycle": 2002, "bloc": "african_american", "vote_share": 0.88, "source": "t"},
        ]
    )
    initial = (0.5, 0.3, 0.2)
    raked, n_iters = rake_layer_weights(panel, initial_lambdas=initial)
    assert n_iters == 0
    assert pytest.approx(raked["lambda_1"], abs=1e-6) == 0.5
    assert pytest.approx(raked["lambda_2"], abs=1e-6) == 0.3
    assert pytest.approx(raked["lambda_3"], abs=1e-6) == 0.2


# ── write_raked_weights ───────────────────────────────────────────────────────


def test_write_raked_weights_updates_file(tmp_path, monkeypatch):
    cfg = tmp_path / "layer_weights.json"
    cfg.write_text(
        json.dumps(
            {
                "lambda_1": 0.5,
                "lambda_2": 0.3,
                "lambda_3": 0.2,
                "raked": {"lambda_1": 0.0, "lambda_2": 0.0, "lambda_3": 0.0},
            }
        )
    )
    import electoral.kernels.raking as raking_module

    monkeypatch.setattr(raking_module, "_LAYER_WEIGHTS_PATH", cfg)

    raked = {"lambda_1": 0.20, "lambda_2": 0.30, "lambda_3": 0.50}
    write_raked_weights(raked)

    data = json.loads(cfg.read_text())
    assert pytest.approx(data["raked"]["lambda_1"], abs=1e-6) == 0.20
    assert pytest.approx(data["raked"]["lambda_3"], abs=1e-6) == 0.50


def test_write_raked_weights_preserves_additive(tmp_path, monkeypatch):
    cfg = tmp_path / "layer_weights.json"
    cfg.write_text(
        json.dumps(
            {
                "lambda_1": 0.5,
                "lambda_2": 0.3,
                "lambda_3": 0.2,
                "raked": {},
            }
        )
    )
    import electoral.kernels.raking as raking_module

    monkeypatch.setattr(raking_module, "_LAYER_WEIGHTS_PATH", cfg)

    write_raked_weights({"lambda_1": 0.4, "lambda_2": 0.3, "lambda_3": 0.3})
    data = json.loads(cfg.read_text())
    # Top-level additive values must not change
    assert data["lambda_1"] == 0.5
    assert data["lambda_2"] == 0.3
    assert data["lambda_3"] == 0.2


def test_write_raked_weights_raises_on_bad_sum():
    with pytest.raises(ValueError, match="sum to 1.0"):
        write_raked_weights({"lambda_1": 0.4, "lambda_2": 0.4, "lambda_3": 0.4})


def test_write_raked_weights_raises_on_wrong_keys():
    with pytest.raises(ValueError, match="keys"):
        write_raked_weights({"lambda_1": 0.5, "lambda_2": 0.5})
