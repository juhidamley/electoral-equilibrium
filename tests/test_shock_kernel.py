"""Integration tests for the build_shock_response kernel.

Covers the two untested fallback seams in the production chain:
  - _load_baseline:     missing baseline_portfolio.json → mu=0.5 + equal weights
  - _build_sigma_delta: missing panel_race.parquet     → diagonal Σ = 0.02²·I

_estimate_shock is patched in every test to avoid loading the LLM adapter.

Note on target: PipelineConfig enforces target ∈ (0.5, 0.7). For tests that
use the fallback mu=0.5, a slight positive delta (0.02) on the stub shock
pushes mu_tilde to 0.52, clearing the target=0.51 threshold and keeping the
optimizer feasible without violating the config constraint.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from electoral.artifacts import ShockResponseData
from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.kernels.shock import build_shock_response

_RACE_IDS = list(CANONICAL_RACES)
_RELIGION_IDS = list(CANONICAL_RELIGIONS)
_GENDER_IDS = list(CANONICAL_GENDERS)
_COV_5X5 = [[0.01 if i == j else 0.0 for j in range(5)] for i in range(5)]


def _stub_shock(party: str = "democrat", delta: float = 0.0) -> ShockResponseData:
    """Stub ShockResponseData with uniform delta across all race blocs."""
    bin_token = "slight_pos" if delta > 0 else "neutral"
    return ShockResponseData(
        shock="test_shock",
        cycle=2020,
        party=party,
        delta_bins_race={r: bin_token for r in _RACE_IDS},
        delta_bins_religion={r: "neutral" for r in _RELIGION_IDS},
        delta_bins_gender={g: "neutral" for g in _GENDER_IDS},
        deltas_race={r: delta for r in _RACE_IDS},
        deltas_religion={r: 0.0 for r in _RELIGION_IDS},
        deltas_gender={g: 0.0 for g in _GENDER_IDS},
        delta_eff=delta,
        covariance=_COV_5X5,
        source="llm_unified",
    )


def _run(config: PipelineConfig, delta: float = 0.0):
    """Patch _estimate_shock and run the full kernel."""
    stub = _stub_shock(party=config.party, delta=delta)
    with patch("electoral.kernels.shock._estimate_shock", return_value=stub):
        return build_shock_response(config, event="test event", intensity=0.5)


@pytest.fixture
def fallback_config(tmp_path) -> PipelineConfig:
    """Config with target=0.51 for fallback-seam tests.

    mu_baseline (fallback) = 0.50; delta=0.02 stub gives mu_tilde=0.52 > 0.51.
    """
    cfg_data = {
        "run_key": "test_fallback",
        "seed": 42,
        "party": "democrat",
        "target": 0.51,
        "data_path": "tests/fixtures/",
        "output_dir": str(tmp_path / "artifacts"),
        "pipeline_mode": "historical",
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg_data))
    cfg = PipelineConfig.from_json(str(p))
    cfg.validate()
    return cfg


class TestBuildShockResponseFallbacks:
    def test_both_fallbacks_produce_valid_artifacts(self, fallback_config):
        """No baseline and no panel — both warning fallbacks fire; output still valid.

        Stub delta=0.02 → mu_tilde=0.52 > target=0.51 → optimizer feasible.
        """
        shock, eq = _run(fallback_config, delta=0.02)

        shock.validate()
        eq.validate()
        assert eq.party == fallback_config.party
        assert set(eq.weights.keys()) == set(_RACE_IDS)
        assert abs(sum(eq.weights.values()) - 1.0) < 1e-6

    def test_load_baseline_reads_mu_from_json(self, smoke_config):
        """_load_baseline uses mu_race from baseline_portfolio.json when present.

        mu_race=0.55 > target=0.535 → clearly feasible with neutral delta=0.
        """
        out = Path(smoke_config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        baseline_payload = {
            "stage": "baseline_portfolio",
            "run_key": "test_run",
            "metadata": {},
            "data": {
                "method": "cvxpy_dqcp",
                "party": "democrat",
                "weights": {r: 0.20 for r in _RACE_IDS},
                "mu_race": {r: 0.55 for r in _RACE_IDS},
                "mu_religion": {r: 0.50 for r in _RELIGION_IDS},
                "mu_gender": {r: 0.50 for r in _GENDER_IDS},
                "mu_eff": 0.535,
                "layer_weights": {"lambda_1": 0.50, "lambda_2": 0.30, "lambda_3": 0.20},
                "target": 0.535,
            },
        }
        (out / "baseline_portfolio.json").write_text(json.dumps(baseline_payload))

        shock, eq = _run(smoke_config, delta=0.0)

        shock.validate()
        eq.validate()
        assert set(eq.weights.keys()) == set(_RACE_IDS)
        assert abs(sum(eq.weights.values()) - 1.0) < 1e-6

    def test_sigma_delta_from_panel_parquet(self, fallback_config):
        """_build_sigma_delta computes Ledoit-Wolf Σ_Δ when panel_race.parquet is present.

        Stub delta=0.02 keeps the optimizer feasible at target=0.51.
        """
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet I/O")
        panel_dir = Path(fallback_config.output_dir) / "panel"
        panel_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        rows = [
            {"cycle": cycle, "bloc": bloc, "vote_share": float(0.50 + rng.normal(0, 0.02))}
            for cycle in [2012, 2016, 2020, 2024]
            for bloc in _RACE_IDS
        ]
        pd.DataFrame(rows).to_parquet(panel_dir / "panel_race.parquet", index=False)

        shock, eq = _run(fallback_config, delta=0.02)

        shock.validate()
        eq.validate()
        cov = np.array(shock.covariance)
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_artifacts_written_to_output_dir(self, fallback_config):
        """build_shock_response writes shock_{id}.json and equilibrium_{id}.json."""
        _run(fallback_config, delta=0.02)

        out = Path(fallback_config.output_dir)
        # stub hardcodes shock="test_shock" → _write_artifacts uses that as the id
        assert (out / "shock_test_shock.json").exists()
        assert (out / "equilibrium_test_shock.json").exists()
