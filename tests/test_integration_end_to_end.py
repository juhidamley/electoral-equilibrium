"""End-to-end pipeline integration test using configs/smoke.json.

Runs the full 7-stage sequence from a clean state, mocking the two LLM-dependent
stages (build_llm_finetune, build_shock_response via _estimate_shock) so the suite
needs no GPU or model weights.

Assertions:
  - Every stage artifact file exists, including BOTH per-shock (shock_{id}.json,
    equilibrium_{id}.json) AND legacy (shock_response.json, equilibrium.json) paths.
  - The JSON envelope contains exactly {stage, run_key, metadata, data} keys with
    the correct Python types.
  - The payload parsed from data passes from_dict() and validate() without error.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

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
    build_sentiment_data,
    build_shock_response,
    build_voter_panel,
    run_simulations,
)

# ── Smoke event ───────────────────────────────────────────────────────────────
# Determines shock_id = "smoke_test" → artifact file names:
#   shock_smoke_test.json, equilibrium_smoke_test.json
_SMOKE_EVENT = "smoke_test"
_SMOKE_SHOCK_ID = _SMOKE_EVENT[:30].lower().replace(" ", "_")

# ── Synthetic payloads for LLM mocks ─────────────────────────────────────────

_ALL_RACES = list(CANONICAL_RACES)
_ALL_RELIGIONS = list(CANONICAL_RELIGIONS)
_ALL_GENDERS = list(CANONICAL_GENDERS)
_COV_5X5 = [[0.0004 if i == j else 0.0 for j in range(5)] for i in range(5)]


def _make_llm_finetune() -> LLMFineTuneData:
    return LLMFineTuneData(
        base_model="mistralai/Mistral-7B-v0.3",
        lora_rank=16,
        n_examples=42,
        cycles_used=[2020],
        adapter_path=None,
    )


def _make_smoke_shock(party: str) -> ShockResponseData:
    """Neutral synthetic shock covering all CANONICAL_* strata (required by shock.py step 3)."""
    return ShockResponseData(
        shock=_SMOKE_SHOCK_ID,
        cycle=2020,
        party=party,
        delta_bins_race={r: "neutral" for r in _ALL_RACES},
        delta_bins_religion={r: "neutral" for r in _ALL_RELIGIONS},
        delta_bins_gender={g: "neutral" for g in _ALL_GENDERS},
        deltas_race={r: 0.0 for r in _ALL_RACES},
        deltas_religion={r: 0.0 for r in _ALL_RELIGIONS},
        deltas_gender={g: 0.0 for g in _ALL_GENDERS},
        delta_eff=0.0,
        covariance=_COV_5X5,
        source="llm_unified",
    )


# ── Artifact registry ─────────────────────────────────────────────────────────

# Stage artifacts: filename → payload class
_STAGE_ARTIFACTS: dict[str, type] = {
    "voter_panel.json": VoterPanelData,
    "baseline_portfolio.json": BaselinePortfolioData,
    "sentiment_data.json": SentimentData,
    "llm_finetune.json": LLMFineTuneData,
    "optimization.json": EquilibriumData,
    "simulation.json": SimulationData,
}

# Shock kernel artifacts: both per-id and legacy paths, both payload classes
_SHOCK_ARTIFACTS: dict[str, type] = {
    f"shock_{_SMOKE_SHOCK_ID}.json": ShockResponseData,
    "shock_response.json": ShockResponseData,
    f"equilibrium_{_SMOKE_SHOCK_ID}.json": EquilibriumData,
    "equilibrium.json": EquilibriumData,
}

_ALL_ARTIFACTS: dict[str, type] = {**_STAGE_ARTIFACTS, **_SHOCK_ARTIFACTS}

_ENVELOPE_REQUIRED_KEYS = {"stage", "run_key", "metadata", "data"}
_ENVELOPE_KEY_TYPES = {"stage": str, "run_key": str, "metadata": dict, "data": dict}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_envelope(path: Path) -> dict:
    assert path.exists(), f"Artifact not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# ── Module-scoped fixture ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory) -> tuple[PipelineConfig, Path]:
    """Run the full pipeline once from configs/smoke.json into a tmp output dir.

    LLM stages are mocked so no GPU or model weights are needed.
    Returns (cfg, output_dir_path).
    """
    tmp = tmp_path_factory.mktemp("e2e")
    cfg = PipelineConfig.from_json("configs/smoke.json")
    cfg = dataclasses.replace(cfg, output_dir=str(tmp / "artifacts"))
    cfg.validate()

    synthetic_finetune = _make_llm_finetune()
    synthetic_shock = _make_smoke_shock(cfg.party)

    with (
        patch("electoral.kernels.finetune.build_llm_finetune", return_value=synthetic_finetune),
        patch("electoral.kernels.shock._estimate_shock", return_value=synthetic_shock),
    ):
        panel = build_voter_panel(cfg)
        build_baseline_portfolio(cfg, panel)
        sentiment = build_sentiment_data(cfg, panel)
        build_llm_finetune(cfg, sentiment)
        shock = build_shock_response(cfg, _SMOKE_EVENT, 0.5)
        equilibrium = build_optimization(cfg, shock)
        run_simulations(cfg, equilibrium, n_simulations=500)

    return cfg, Path(cfg.output_dir)


# ── All artifact files exist ──────────────────────────────────────────────────


class TestArtifactFilesExist:
    @pytest.mark.parametrize("filename", list(_ALL_ARTIFACTS))
    def test_file_exists(self, smoke_run, filename):
        _, out = smoke_run
        assert (out / filename).exists(), (
            f"Expected artifact file {filename!r} in {out}. "
            f"Files present: {[f.name for f in out.iterdir() if f.is_file()]}"
        )

    def test_panel_parquets_exist(self, smoke_run):
        _, out = smoke_run
        panel_dir = out / "panel"
        for name in ("panel_race.parquet", "panel_religion.parquet", "panel_gender.parquet"):
            assert (panel_dir / name).exists(), f"Missing parquet: {panel_dir / name}"


# ── Envelope structure ────────────────────────────────────────────────────────


class TestEnvelopeStructure:
    @pytest.mark.parametrize("filename", list(_ALL_ARTIFACTS))
    def test_has_required_keys(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        missing = _ENVELOPE_REQUIRED_KEYS - set(envelope.keys())
        assert not missing, f"{filename}: missing envelope keys {missing}"

    @pytest.mark.parametrize("filename", list(_ALL_ARTIFACTS))
    def test_no_extra_top_level_keys(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        extra = set(envelope.keys()) - _ENVELOPE_REQUIRED_KEYS
        assert not extra, f"{filename}: unexpected top-level keys {extra}"

    @pytest.mark.parametrize("filename", list(_ALL_ARTIFACTS))
    def test_key_types(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        for key, expected_type in _ENVELOPE_KEY_TYPES.items():
            assert isinstance(envelope[key], expected_type), (
                f"{filename}: envelope[{key!r}] expected {expected_type.__name__}, "
                f"got {type(envelope[key]).__name__}"
            )

    @pytest.mark.parametrize("filename", list(_ALL_ARTIFACTS))
    def test_run_key_matches_config(self, smoke_run, filename):
        cfg, out = smoke_run
        envelope = _load_envelope(out / filename)
        assert envelope["run_key"] == cfg.run_key, (
            f"{filename}: run_key {envelope['run_key']!r} != config.run_key {cfg.run_key!r}"
        )

    @pytest.mark.parametrize("filename", list(_ALL_ARTIFACTS))
    def test_stage_is_non_empty_string(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        assert isinstance(envelope["stage"], str) and envelope["stage"], (
            f"{filename}: 'stage' must be a non-empty string"
        )


# ── Payload parse and validate ────────────────────────────────────────────────


class TestParseAndValidate:
    @pytest.mark.parametrize("filename,payload_cls", list(_ALL_ARTIFACTS.items()))
    def test_from_dict_succeeds(self, smoke_run, filename, payload_cls):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        payload = payload_cls.from_dict(envelope["data"])
        assert isinstance(payload, payload_cls), (
            f"{filename}: from_dict() returned {type(payload).__name__}, "
            f"expected {payload_cls.__name__}"
        )

    @pytest.mark.parametrize("filename,payload_cls", list(_ALL_ARTIFACTS.items()))
    def test_validate_passes(self, smoke_run, filename, payload_cls):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        payload = payload_cls.from_dict(envelope["data"])
        payload.validate()  # must not raise

    @pytest.mark.parametrize("filename", list(_ALL_ARTIFACTS))
    def test_json_serializable(self, smoke_run, filename):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        try:
            json.dumps(envelope)
        except TypeError as exc:
            pytest.fail(f"{filename}: envelope is not JSON-serializable: {exc}")


# ── Stage-specific stage-name assertions ──────────────────────────────────────


class TestStageNames:
    @pytest.mark.parametrize(
        "filename,expected_stage",
        [
            ("voter_panel.json", "voter_panel"),
            ("baseline_portfolio.json", "baseline_portfolio"),
            ("sentiment_data.json", "sentiment_data"),
            ("llm_finetune.json", "llm_finetune"),
            (f"shock_{_SMOKE_SHOCK_ID}.json", "shock_response"),
            ("shock_response.json", "shock_response"),
            (f"equilibrium_{_SMOKE_SHOCK_ID}.json", "equilibrium"),
            ("equilibrium.json", "equilibrium"),
            ("optimization.json", "optimization"),
            ("simulation.json", "simulation"),
        ],
    )
    def test_stage_name(self, smoke_run, filename, expected_stage):
        _, out = smoke_run
        envelope = _load_envelope(out / filename)
        assert envelope["stage"] == expected_stage, (
            f"{filename}: expected stage={expected_stage!r}, got {envelope['stage']!r}"
        )


# ── Per-shock and legacy files contain the same payload ───────────────────────


class TestPerShockVsLegacy:
    def test_shock_files_identical(self, smoke_run):
        """Per-shock and legacy shock files must contain the same payload."""
        _, out = smoke_run
        per_id = _load_envelope(out / f"shock_{_SMOKE_SHOCK_ID}.json")
        legacy = _load_envelope(out / "shock_response.json")
        assert per_id["data"] == legacy["data"], (
            "shock_{id}.json and shock_response.json data payloads diverged"
        )

    def test_equilibrium_files_identical(self, smoke_run):
        """Per-shock and legacy equilibrium files must contain the same payload."""
        _, out = smoke_run
        per_id = _load_envelope(out / f"equilibrium_{_SMOKE_SHOCK_ID}.json")
        legacy = _load_envelope(out / "equilibrium.json")
        assert per_id["data"] == legacy["data"], (
            "equilibrium_{id}.json and equilibrium.json data payloads diverged"
        )


# ── Content sanity checks ─────────────────────────────────────────────────────


class TestContentSanity:
    def test_simulation_win_probability_in_range(self, smoke_run):
        _, out = smoke_run
        payload = SimulationData.from_dict(_load_envelope(out / "simulation.json")["data"])
        assert 0.0 <= payload.win_probability <= 1.0

    def test_simulation_ci_ordered(self, smoke_run):
        _, out = smoke_run
        payload = SimulationData.from_dict(_load_envelope(out / "simulation.json")["data"])
        assert payload.win_probability_low <= payload.win_probability <= payload.win_probability_high

    def test_optimization_party_matches_config(self, smoke_run):
        cfg, out = smoke_run
        payload = EquilibriumData.from_dict(_load_envelope(out / "optimization.json")["data"])
        assert payload.party == cfg.party

    def test_voter_panel_seed_in_metadata(self, smoke_run):
        cfg, out = smoke_run
        envelope = _load_envelope(out / "voter_panel.json")
        assert "seed" in envelope["metadata"]
        assert envelope["metadata"]["seed"] == cfg.derive_seed("voter_panel")

    def test_llm_finetune_reflects_mock(self, smoke_run):
        _, out = smoke_run
        payload = LLMFineTuneData.from_dict(_load_envelope(out / "llm_finetune.json")["data"])
        assert payload.lora_rank == 16
        assert payload.n_examples == 42
