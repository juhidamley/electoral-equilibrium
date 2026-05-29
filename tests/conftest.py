"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from electoral.artifacts import (
    BaselinePortfolioData,
    VoterPanelData,
)
from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

# ── Canonical data shared across tests ───────────────────────────────────────

LAYER_WEIGHTS = {"lambda_1": 0.50, "lambda_2": 0.30, "lambda_3": 0.20}

RACE_WEIGHTS = {
    "african_american": 0.15,
    "latino": 0.12,
    "asian": 0.06,
    "white": 0.57,
    "other_race": 0.10,
}


@pytest.fixture
def minimal_voter_panel() -> VoterPanelData:
    return VoterPanelData(
        cycles=[2016, 2020],
        races=list(CANONICAL_RACES),
        religions=list(CANONICAL_RELIGIONS),
        genders=list(CANONICAL_GENDERS),
        n_rows_race=200,
        n_rows_religion=200,
        n_rows_gender=200,
        layer_weights=LAYER_WEIGHTS,
        source="ARDA+GSS+NEP",
    )


@pytest.fixture
def minimal_baseline() -> BaselinePortfolioData:
    return BaselinePortfolioData(
        method="cvxpy_dqcp",
        party="democrat",
        weights=RACE_WEIGHTS,
        mu_race={r: 0.50 for r in CANONICAL_RACES},
        mu_religion={r: 0.50 for r in CANONICAL_RELIGIONS},
        mu_gender={r: 0.50 for r in CANONICAL_GENDERS},
        mu_eff=0.535,
        layer_weights=LAYER_WEIGHTS,
        target=0.535,
    )


@pytest.fixture
def smoke_config(tmp_path) -> PipelineConfig:
    """PipelineConfig loaded from a minimal in-memory JSON for tests."""
    config_data = {
        "run_key": "test_run",
        "seed": 42,
        "party": "democrat",
        "target": 0.535,
        "data_path": "tests/fixtures/",
        "output_dir": str(tmp_path / "artifacts"),
        "pipeline_mode": "historical",
    }
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(config_data))
    cfg = PipelineConfig.from_json(str(config_path))
    cfg.validate()
    return cfg


@pytest.fixture
def artifacts_dir(tmp_path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    return d
