"""Tests for electoral/stages.py — one test per stub function.

Each test:
  1. Builds a PipelineConfig pointing at tmp_path for output
  2. Calls the stub with minimal valid inputs
  3. Asserts the returned payload is the correct type and passes validate()
  4. Asserts the JSON envelope was written to disk
  5. Checks key fields are derived from the config (party, seed, etc.)

Stubs produce zero-value placeholder payloads — no real computation occurs.
These tests will be replaced by integration tests as each stub is implemented.
"""

from __future__ import annotations

import json
from pathlib import Path

import importlib

import pytest

from electoral.artifacts import (
    BaselinePortfolioData,
    EquilibriumData,
    LLMFineTuneData,
    SentimentData,
    ShockResponseData,
    SimulationData,
    VoterPanelData,
)
from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.stages import (
    build_baseline_portfolio,
    build_llm_finetune,
    build_optimization,
    build_shock_response,
    build_sentiment_data,
    build_voter_panel,
    run_simulations,
)


# ── Shared fixture ────────────────────────────────────────────────────────────


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    """Minimal two-bloc-per-stratum config writing to a temp directory."""
    import json as _json

    data = {
        "run_key": "test_run",
        "seed": 7,
        "party": "democrat",
        "target": 0.535,
        "data_path": "tests/fixtures/",
        "output_dir": str(tmp_path / "artifacts"),
        "pipeline_mode": "historical",
        "races": ["african_american", "white"],
        "religions": ["evangelical", "secular"],
        "genders": ["women", "men"],
    }
    p = tmp_path / "cfg.json"
    p.write_text(_json.dumps(data))
    c = PipelineConfig.from_json(p)
    c.validate()
    return c


def _torch_available() -> bool:
    for mod in ("torch", "sentencepiece", "transformers"):
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
    return True


_skip_no_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="torch/transformers/sentencepiece not installed — skip LLM-dependent stages",
)


def _envelope(path: Path) -> dict:
    """Read and return the JSON envelope written by a stub."""
    assert path.exists(), f"Expected artifact at {path}"
    with open(path) as f:
        return json.load(f)


# ── Stage 1: build_voter_panel ────────────────────────────────────────────────


class TestBuildVoterPanel:
    def test_returns_voter_panel_data(self, cfg):
        result = build_voter_panel(cfg)
        assert isinstance(result, VoterPanelData)

    def test_payload_validates(self, cfg):
        result = build_voter_panel(cfg)
        result.validate()

    def test_races_match_config(self, cfg):
        result = build_voter_panel(cfg)
        assert result.races == cfg.races

    def test_religions_match_config(self, cfg):
        result = build_voter_panel(cfg)
        assert result.religions == cfg.religions

    def test_genders_match_config(self, cfg):
        result = build_voter_panel(cfg)
        assert result.genders == cfg.genders

    def test_envelope_written_to_disk(self, cfg):
        build_voter_panel(cfg)
        envelope = _envelope(Path(cfg.output_dir) / "voter_panel.json")
        assert envelope["stage"] == "voter_panel"
        assert envelope["run_key"] == cfg.run_key

    def test_envelope_seed_is_derived(self, cfg):
        build_voter_panel(cfg)
        envelope = _envelope(Path(cfg.output_dir) / "voter_panel.json")
        assert envelope["metadata"]["seed"] == cfg.derive_seed("voter_panel")


# ── Stage 2: build_baseline_portfolio ────────────────────────────────────────


class TestBuildBaselinePortfolio:
    @pytest.fixture
    def panel(self, cfg) -> VoterPanelData:
        return build_voter_panel(cfg)

    def test_returns_baseline_portfolio_data(self, cfg, panel):
        result = build_baseline_portfolio(cfg, panel)
        assert isinstance(result, BaselinePortfolioData)

    def test_payload_validates(self, cfg, panel):
        result = build_baseline_portfolio(cfg, panel)
        result.validate()

    def test_party_matches_config(self, cfg, panel):
        result = build_baseline_portfolio(cfg, panel)
        assert result.party == cfg.party

    def test_target_matches_config(self, cfg, panel):
        result = build_baseline_portfolio(cfg, panel)
        assert result.target == pytest.approx(cfg.target)

    def test_layer_weights_from_panel(self, cfg, panel):
        result = build_baseline_portfolio(cfg, panel)
        assert result.layer_weights == panel.layer_weights

    def test_weights_cover_canonical_races(self, cfg, panel):
        # The kernel always optimises over all five CANONICAL_RACES, not the
        # (potentially reduced) cfg.races used by the old placeholder.
        result = build_baseline_portfolio(cfg, panel)
        assert set(result.weights.keys()) == set(CANONICAL_RACES)

    def test_envelope_written_to_disk(self, cfg, panel):
        build_baseline_portfolio(cfg, panel)
        envelope = _envelope(Path(cfg.output_dir) / "baseline_portfolio.json")
        assert envelope["stage"] == "baseline_portfolio"


# ── Stage 3: build_sentiment_data ────────────────────────────────────────────


class TestBuildSentimentData:
    @pytest.fixture
    def panel(self, cfg) -> VoterPanelData:
        return build_voter_panel(cfg)

    def test_returns_sentiment_data(self, cfg, panel):
        result = build_sentiment_data(cfg, panel)
        assert isinstance(result, SentimentData)

    def test_payload_validates(self, cfg, panel):
        result = build_sentiment_data(cfg, panel)
        result.validate()

    def test_envelope_written_to_disk(self, cfg, panel):
        build_sentiment_data(cfg, panel)
        envelope = _envelope(Path(cfg.output_dir) / "sentiment_data.json")
        assert envelope["stage"] == "sentiment_data"
        assert envelope["run_key"] == cfg.run_key


# ── Stage 4: build_llm_finetune ──────────────────────────────────────────────


class TestBuildLLMFinetune:
    @pytest.fixture
    def sentiment(self, cfg) -> SentimentData:
        panel = build_voter_panel(cfg)
        return build_sentiment_data(cfg, panel)

    def test_returns_llm_finetune_data(self, cfg, sentiment):
        result = build_llm_finetune(cfg, sentiment)
        assert isinstance(result, LLMFineTuneData)

    def test_payload_validates(self, cfg, sentiment):
        result = build_llm_finetune(cfg, sentiment)
        result.validate()

    def test_envelope_written_to_disk(self, cfg, sentiment):
        build_llm_finetune(cfg, sentiment)
        envelope = _envelope(Path(cfg.output_dir) / "llm_finetune.json")
        assert envelope["stage"] == "llm_finetune"


# ── Stage 5: build_shock_response ────────────────────────────────────────────


@_skip_no_torch
class TestBuildShockResponse:
    def test_returns_shock_response_data(self, cfg):
        result = build_shock_response(cfg, "roe_v_wade_2022", 0.8)
        assert isinstance(result, ShockResponseData)

    def test_payload_validates(self, cfg):
        result = build_shock_response(cfg, "roe_v_wade_2022", 0.8)
        result.validate()

    def test_shock_field_matches_event(self, cfg):
        result = build_shock_response(cfg, "kavanaugh_2018", 0.6)
        assert result.shock == "kavanaugh_2018"

    def test_party_matches_config(self, cfg):
        result = build_shock_response(cfg, "event", 0.5)
        assert result.party == cfg.party

    def test_delta_bins_race_covers_canonical_races(self, cfg):
        result = build_shock_response(cfg, "event", 0.5)
        assert set(result.delta_bins_race.keys()) == set(CANONICAL_RACES)

    def test_delta_bins_religion_covers_canonical_religions(self, cfg):
        result = build_shock_response(cfg, "event", 0.5)
        assert set(result.delta_bins_religion.keys()) == set(CANONICAL_RELIGIONS)

    def test_delta_bins_gender_covers_canonical_genders(self, cfg):
        result = build_shock_response(cfg, "event", 0.5)
        assert set(result.delta_bins_gender.keys()) == set(CANONICAL_GENDERS)

    def test_covariance_is_5x5(self, cfg):
        result = build_shock_response(cfg, "event", 0.5)
        assert len(result.covariance) == 5
        assert all(len(row) == 5 for row in result.covariance)

    def test_envelope_written_to_disk(self, cfg):
        build_shock_response(cfg, "event", 0.5)
        # event "event" → shock_id "event" → shock_event.json
        envelope = _envelope(Path(cfg.output_dir) / "shock_event.json")
        assert envelope["stage"] == "shock_response"
        assert envelope["metadata"]["shock"] == "event"


# ── Stage 6: build_optimization ──────────────────────────────────────────────


@_skip_no_torch
class TestBuildOptimization:
    @pytest.fixture
    def shock(self, cfg) -> ShockResponseData:
        return build_shock_response(cfg, "roe_v_wade_2022", 0.8)

    def test_returns_equilibrium_data(self, cfg, shock):
        result = build_optimization(cfg, shock)
        assert isinstance(result, EquilibriumData)

    def test_payload_validates(self, cfg, shock):
        result = build_optimization(cfg, shock)
        result.validate()

    def test_party_matches_config(self, cfg, shock):
        result = build_optimization(cfg, shock)
        assert result.party == cfg.party

    def test_shock_field_propagated(self, cfg, shock):
        result = build_optimization(cfg, shock)
        assert result.shock == shock.shock

    def test_weights_race_blocs_only(self, cfg, shock):
        # weights are the CVXPY decision variables — all 5 canonical race blocs,
        # not religion/gender and not limited to cfg.races (which may be a smoke subset)
        result = build_optimization(cfg, shock)
        assert set(result.weights.keys()) == set(CANONICAL_RACES)

    def test_weights_equal_mu_shifted_keys(self, cfg, shock):
        result = build_optimization(cfg, shock)
        assert set(result.weights.keys()) == set(result.mu_shifted.keys())

    def test_envelope_written_to_disk(self, cfg, shock):
        build_optimization(cfg, shock)
        envelope = _envelope(Path(cfg.output_dir) / "optimization.json")
        assert envelope["stage"] == "optimization"


# ── Stage 7: run_simulations ─────────────────────────────────────────────────


@_skip_no_torch
class TestRunSimulations:
    @pytest.fixture
    def equilibrium(self, cfg) -> EquilibriumData:
        shock = build_shock_response(cfg, "event", 0.5)
        return build_optimization(cfg, shock)

    def test_returns_simulation_data(self, cfg, equilibrium):
        result = run_simulations(cfg, equilibrium)
        assert isinstance(result, SimulationData)

    def test_payload_validates(self, cfg, equilibrium):
        result = run_simulations(cfg, equilibrium)
        result.validate()

    def test_seed_is_derived(self, cfg, equilibrium):
        result = run_simulations(cfg, equilibrium)
        assert result.seed == cfg.derive_seed("monte_carlo")

    def test_n_simulations_default(self, cfg, equilibrium):
        result = run_simulations(cfg, equilibrium)
        assert result.n_simulations == 10_000

    def test_n_simulations_override(self, cfg, equilibrium):
        result = run_simulations(cfg, equilibrium, n_simulations=500)
        assert result.n_simulations == 500

    def test_win_probability_in_range(self, cfg, equilibrium):
        result = run_simulations(cfg, equilibrium)
        assert 0.0 <= result.win_probability <= 1.0

    def test_envelope_written_to_disk(self, cfg, equilibrium):
        run_simulations(cfg, equilibrium)
        envelope = _envelope(Path(cfg.output_dir) / "simulation.json")
        assert envelope["stage"] == "simulation"
        assert envelope["metadata"]["n_simulations"] == 10_000

    def test_seed_reproducible_across_calls(self, cfg, equilibrium, tmp_path):
        """Two runs with the same config seed must produce the same derived seed."""
        r1 = run_simulations(cfg, equilibrium)
        r2 = run_simulations(cfg, equilibrium)
        assert r1.seed == r2.seed
