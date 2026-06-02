"""Tests for electoral/config.py — PipelineConfig loading and validation.

Coverage:
  - from_json: smoke.json and base.json load cleanly with correct field values
  - from_json: optional fields receive canonical defaults when absent from JSON
  - from_json: extra JSON keys are silently ignored
  - validate: rejects invalid party and out-of-range target
  - derive_seed: deterministic, consistent with rng.derive_seed, stage-isolated
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from electoral.config import PipelineConfig
from electoral.core.rng import derive_seed as rng_derive_seed
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

# Repo root so we can reference the real config files
REPO_ROOT = Path(__file__).parent.parent
SMOKE_JSON = REPO_ROOT / "configs" / "smoke.json"
BASE_JSON = REPO_ROOT / "configs" / "base.json"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    return p


def _minimal(tmp_path: Path, **overrides) -> PipelineConfig:
    defaults = {
        "run_key": "test_run",
        "seed": 42,
        "party": "democrat",
        "target": 0.535,
    }
    defaults.update(overrides)
    return PipelineConfig.from_json(_write_config(tmp_path, defaults))


# ── from_json: real config files ─────────────────────────────────────────────


class TestFromJsonRealFiles:
    def test_smoke_json_loads(self):
        cfg = PipelineConfig.from_json(SMOKE_JSON)
        cfg.validate()
        assert cfg.run_key == "smoke_test"
        assert cfg.seed == 7
        assert cfg.party == "democrat"
        assert cfg.target == pytest.approx(0.535)
        assert cfg.data_path == "tests/fixtures/"
        assert cfg.output_dir == "artifacts/smoke/"

    def test_smoke_json_blocs(self):
        cfg = PipelineConfig.from_json(SMOKE_JSON)
        # Smoke uses two blocs per stratum for fast iteration
        assert "african_american" in cfg.races
        assert "white" in cfg.races
        assert "evangelical" in cfg.religions
        assert "secular" in cfg.religions
        assert "women" in cfg.genders
        assert "men" in cfg.genders

    def test_base_json_loads(self):
        cfg = PipelineConfig.from_json(BASE_JSON)
        cfg.validate()
        assert cfg.run_key == "base_2026"
        assert cfg.seed == 42
        assert cfg.party == "democrat"
        assert cfg.target == pytest.approx(0.5066)  # EC-adjusted Dem V_eq

    def test_base_json_full_blocs(self):
        cfg = PipelineConfig.from_json(BASE_JSON)
        assert cfg.races == list(CANONICAL_RACES)
        assert cfg.religions == list(CANONICAL_RELIGIONS)
        assert cfg.genders == list(CANONICAL_GENDERS)


# ── from_json: defaults and tolerance ────────────────────────────────────────


class TestFromJsonDefaults:
    def test_races_default_to_canonical(self, tmp_path):
        cfg = _minimal(tmp_path)
        assert cfg.races == list(CANONICAL_RACES)

    def test_religions_default_to_canonical(self, tmp_path):
        cfg = _minimal(tmp_path)
        assert cfg.religions == list(CANONICAL_RELIGIONS)

    def test_genders_default_to_canonical(self, tmp_path):
        cfg = _minimal(tmp_path)
        assert cfg.genders == list(CANONICAL_GENDERS)

    def test_pipeline_mode_defaults_to_historical(self, tmp_path):
        cfg = _minimal(tmp_path)
        assert cfg.pipeline_mode == "historical"

    def test_output_dir_defaults(self, tmp_path):
        cfg = _minimal(tmp_path)
        assert cfg.output_dir == "artifacts/"

    def test_extra_json_keys_ignored(self, tmp_path):
        cfg = _minimal(tmp_path, unknown_future_field="ignored", another_extra=99)
        cfg.validate()  # must not raise

    def test_accepts_path_object(self, tmp_path):
        p = _write_config(
            tmp_path, {"run_key": "r", "seed": 1, "party": "republican", "target": 0.52}
        )
        cfg = PipelineConfig.from_json(p)  # Path object, not str
        cfg.validate()


# ── validate ─────────────────────────────────────────────────────────────────


class TestValidate:
    def test_democrat_passes(self, tmp_path):
        _minimal(tmp_path, party="democrat").validate()

    def test_republican_passes(self, tmp_path):
        _minimal(tmp_path, party="republican", target=0.52).validate()

    def test_invalid_party_raises(self, tmp_path):
        with pytest.raises(ValueError, match="party"):
            _minimal(tmp_path, party="independent").validate()

    def test_empty_party_raises(self, tmp_path):
        with pytest.raises(ValueError, match="party"):
            _minimal(tmp_path, party="").validate()

    def test_target_at_lower_boundary_raises(self, tmp_path):
        # 0.5 is excluded (strictly greater than 0.5)
        with pytest.raises(ValueError, match="target"):
            _minimal(tmp_path, target=0.5).validate()

    def test_target_at_upper_boundary_raises(self, tmp_path):
        # 0.7 is excluded (strictly less than 0.7)
        with pytest.raises(ValueError, match="target"):
            _minimal(tmp_path, target=0.7).validate()

    def test_target_below_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="target"):
            _minimal(tmp_path, target=0.3).validate()

    def test_target_above_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="target"):
            _minimal(tmp_path, target=0.9).validate()

    def test_democrat_typical_target_valid(self, tmp_path):
        _minimal(tmp_path, party="democrat", target=0.535).validate()

    def test_republican_typical_target_valid(self, tmp_path):
        _minimal(tmp_path, party="republican", target=0.52).validate()

    def test_negative_seed_raises(self, tmp_path):
        with pytest.raises(ValueError, match="seed"):
            _minimal(tmp_path, seed=-1).validate()


# ── derive_seed ───────────────────────────────────────────────────────────────


class TestDeriveSeed:
    def test_deterministic(self, tmp_path):
        cfg = _minimal(tmp_path)
        assert cfg.derive_seed("voter_panel") == cfg.derive_seed("voter_panel")

    def test_consistent_with_rng_module(self, tmp_path):
        cfg = _minimal(tmp_path)
        for stage in ("voter_panel", "monte_carlo", "llm_finetune"):
            assert cfg.derive_seed(stage) == rng_derive_seed(cfg.seed, stage)

    def test_different_stages_produce_different_seeds(self, tmp_path):
        cfg = _minimal(tmp_path)
        seeds = {
            cfg.derive_seed(s) for s in ("voter_panel", "monte_carlo", "llm_finetune", "setfit")
        }
        assert len(seeds) == 4

    def test_different_base_seeds_produce_different_seeds(self, tmp_path):
        cfg_a = _minimal(tmp_path, seed=42)
        cfg_b = _minimal(tmp_path, seed=99)
        assert cfg_a.derive_seed("voter_panel") != cfg_b.derive_seed("voter_panel")

    def test_seed_in_numpy_range(self, tmp_path):
        cfg = _minimal(tmp_path)
        s = cfg.derive_seed("monte_carlo")
        assert 0 <= s < 2**31
