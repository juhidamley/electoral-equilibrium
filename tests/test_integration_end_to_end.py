"""End-to-end pipeline integration test.

Runs the full stage sequence from a clean state using the smoke config
(tests/fixtures/ with the toy NEP exit poll).  LLM-dependent stages
(build_llm_finetune, build_shock_response) are bypassed with a
synthetic ShockResponseData so optimization and simulation can be
exercised without GPU or model weights.

Assertions:
  - Each stage artifact file exists at the expected path
  - The JSON envelope contains stage, run_key, metadata, data keys
  - The payload parsed from data passes validate() without error
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from electoral.artifacts import (
    BaselinePortfolioData,
    EquilibriumData,
    SentimentData,
    ShockResponseData,
    SimulationData,
    VoterPanelData,
)
from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.stages import (
    build_baseline_portfolio,
    build_optimization,
    build_sentiment_data,
    build_voter_panel,
    run_simulations,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_RACE_IDS = list(CANONICAL_RACES)
_RELIGION_IDS = list(CANONICAL_RELIGIONS)
_GENDER_IDS = list(CANONICAL_GENDERS)
_COV_5X5 = [[0.01 if i == j else 0.0 for j in range(5)] for i in range(5)]

# Payload class for each artifact file — used for parse + validate assertions
_ARTIFACT_PAYLOAD_CLASSES = {
    "voter_panel.json": VoterPanelData,
    "baseline_portfolio.json": BaselinePortfolioData,
    "sentiment_data.json": SentimentData,
    "optimization.json": EquilibriumData,
    "simulation.json": SimulationData,
}

_ENVELOPE_REQUIRED_KEYS = {"stage", "run_key", "metadata", "data"}
_ENVELOPE_KEY_TYPES = {
    "stage": str,
    "run_key": str,
    "metadata": dict,
    "data": dict,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_smoke_shock(party: str) -> ShockResponseData:
    """Synthetic neutral ShockResponseData — no LLM required."""
    return ShockResponseData(
        shock="smoke_test_event",
        cycle=2020,
        party=party,
        delta_bins_race={r: "neutral" for r in _RACE_IDS},
        delta_bins_religion={r: "neutral" for r in _RELIGION_IDS},
        delta_bins_gender={g: "neutral" for g in _GENDER_IDS},
        deltas_race={r: 0.0 for r in _RACE_IDS},
        deltas_religion={r: 0.0 for r in _RELIGION_IDS},
        deltas_gender={g: 0.0 for g in _GENDER_IDS},
        delta_eff=0.0,
        covariance=_COV_5X5,
        source="llm_unified",
    )


def _load_envelope(path: Path) -> dict:
    assert path.exists(), f"Artifact not found at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory) -> tuple[PipelineConfig, Path]:
    """Run the full non-LLM pipeline once and return (config, output_dir)."""
    import json as _json

    tmp = tmp_path_factory.mktemp("e2e")
    data = {
        "run_key": "e2e_smoke",
        "seed": 42,
        "party": "democrat",
        "target": 0.535,
        "data_path": "tests/fixtures/",
        "output_dir": str(tmp / "artifacts"),
        "pipeline_mode": "historical",
    }
    cfg_path = tmp / "e2e_config.json"
    cfg_path.write_text(_json.dumps(data))
    cfg = PipelineConfig.from_json(str(cfg_path))
    cfg.validate()

    # Run stages sequentially — matching the dependency graph in stages.py
    panel = build_voter_panel(cfg)
    build_baseline_portfolio(cfg, panel)
    build_sentiment_data(cfg, panel)
    shock = _make_smoke_shock(cfg.party)
    shock.validate()
    build_optimization(cfg, shock)
    equilibrium = build_optimization(cfg, shock)
    run_simulations(cfg, equilibrium, n_simulations=1_000)

    return cfg, Path(cfg.output_dir)


# ── Envelope structure ────────────────────────────────────────────────────────


class TestEnvelopeStructure:
    @pytest.mark.parametrize("filename", list(_ARTIFACT_PAYLOAD_CLASSES))
    def test_envelope_has_required_keys(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        missing = _ENVELOPE_REQUIRED_KEYS - set(envelope.keys())
        assert not missing, f"{filename}: missing envelope keys {missing}"

    @pytest.mark.parametrize("filename", list(_ARTIFACT_PAYLOAD_CLASSES))
    def test_envelope_key_types(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        for key, expected_type in _ENVELOPE_KEY_TYPES.items():
            assert isinstance(envelope[key], expected_type), (
                f"{filename}: envelope[{key!r}] expected {expected_type.__name__}, "
                f"got {type(envelope[key]).__name__}"
            )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_PAYLOAD_CLASSES))
    def test_run_key_matches_config(self, smoke_run, filename):
        cfg, out = smoke_run
        envelope = _load_envelope(out / filename)
        assert envelope["run_key"] == cfg.run_key

    @pytest.mark.parametrize("filename", list(_ARTIFACT_PAYLOAD_CLASSES))
    def test_stage_is_non_empty_string(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        assert isinstance(envelope["stage"], str) and envelope["stage"]


# ── Artifact files exist ──────────────────────────────────────────────────────


class TestArtifactFilesExist:
    @pytest.mark.parametrize("filename", list(_ARTIFACT_PAYLOAD_CLASSES))
    def test_artifact_file_exists(self, smoke_run, filename):
        _, out = smoke_run
        assert (out / filename).exists(), f"Expected {filename} in {out}"


# ── Parse and validate ────────────────────────────────────────────────────────


class TestParseAndValidate:
    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_PAYLOAD_CLASSES.items()),
    )
    def test_payload_parses_without_error(self, smoke_run, filename, payload_cls):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        payload = payload_cls.from_dict(envelope["data"])
        assert isinstance(payload, payload_cls)

    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_PAYLOAD_CLASSES.items()),
    )
    def test_payload_validates(self, smoke_run, filename, payload_cls):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        payload = payload_cls.from_dict(envelope["data"])
        # validate() must not raise
        payload.validate()

    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_PAYLOAD_CLASSES.items()),
    )
    def test_envelope_is_json_serializable(self, smoke_run, filename, payload_cls):
        _, out = smoke_run
        # Reading the file proves it's valid JSON; re-dump confirms no Python
        # objects leaked into the data field (datetime, numpy, etc.)
        envelope = _load_envelope(out / filename)
        json.dumps(envelope)  # raises TypeError on non-serializable objects


# ── Stage-specific content ────────────────────────────────────────────────────


class TestStageContent:
    def test_voter_panel_stage_name(self, smoke_run):
        _, out = smoke_run
        envelope = _load_envelope(out / "voter_panel.json")
        assert envelope["stage"] == "voter_panel"

    def test_baseline_portfolio_stage_name(self, smoke_run):
        _, out = smoke_run
        envelope = _load_envelope(out / "baseline_portfolio.json")
        assert envelope["stage"] == "baseline_portfolio"

    def test_sentiment_data_stage_name(self, smoke_run):
        _, out = smoke_run
        envelope = _load_envelope(out / "sentiment_data.json")
        assert envelope["stage"] == "sentiment_data"

    def test_optimization_stage_name(self, smoke_run):
        _, out = smoke_run
        envelope = _load_envelope(out / "optimization.json")
        assert envelope["stage"] == "optimization"

    def test_simulation_stage_name(self, smoke_run):
        _, out = smoke_run
        envelope = _load_envelope(out / "simulation.json")
        assert envelope["stage"] == "simulation"

    def test_optimization_party_matches_config(self, smoke_run):
        cfg, out = smoke_run
        envelope = _load_envelope(out / "optimization.json")
        payload = EquilibriumData.from_dict(envelope["data"])
        assert payload.party == cfg.party

    def test_simulation_win_probability_in_range(self, smoke_run):
        _, out = smoke_run
        envelope = _load_envelope(out / "simulation.json")
        payload = SimulationData.from_dict(envelope["data"])
        assert 0.0 <= payload.win_probability <= 1.0

    def test_simulation_ci_is_valid(self, smoke_run):
        _, out = smoke_run
        envelope = _load_envelope(out / "simulation.json")
        payload = SimulationData.from_dict(envelope["data"])
        assert payload.win_probability_low <= payload.win_probability
        assert payload.win_probability <= payload.win_probability_high

    def test_parquet_files_exist(self, smoke_run):
        _, out = smoke_run
        panel_dir = out / "panel"
        for name in ("panel_race.parquet", "panel_religion.parquet", "panel_gender.parquet"):
            assert (panel_dir / name).exists(), f"Expected parquet at {panel_dir / name}"

    def test_voter_panel_seed_in_metadata(self, smoke_run):
        cfg, out = smoke_run
        envelope = _load_envelope(out / "voter_panel.json")
        assert "seed" in envelope["metadata"]
        assert envelope["metadata"]["seed"] == cfg.derive_seed("voter_panel")
