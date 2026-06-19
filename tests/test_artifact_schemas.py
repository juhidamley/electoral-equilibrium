"""Artifact schema validation tests.

For every artifact produced by the smoke pipeline:
  1. Envelope top-level keys have the correct Python types
     (stage: str, run_key: str, metadata: dict, data: dict).
  2. Every dataclass field declared on the payload class appears in the data
     dict — introspected via dataclasses.fields(), NOT a hardcoded list.
  3. No value anywhere in the data dict is a non-finite float (inf/nan).
     This assertion closes the "no artifact may serialize Infinity/NaN" flag:
     if it passes, all artifacts are safe for standard JSON serialization.
  4. round-trip: from_dict(data).to_dict() reproduces data exactly.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Generator
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

# ── Smoke constants ───────────────────────────────────────────────────────────

_SMOKE_EVENT = "smoke_test"
_SMOKE_SHOCK_ID = _SMOKE_EVENT[:30].lower().replace(" ", "_")

_ALL_RACES = list(CANONICAL_RACES)
_ALL_RELIGIONS = list(CANONICAL_RELIGIONS)
_ALL_GENDERS = list(CANONICAL_GENDERS)
_COV_5X5 = [[0.0004 if i == j else 0.0 for j in range(5)] for i in range(5)]

# ── Artifact registry: filename → payload class ────────────────────────────────
# Covers ALL artifacts written by a full smoke run, including both per-id
# (shock_smoke_test.json, equilibrium_smoke_test.json) and legacy
# (shock_response.json, equilibrium.json) paths.

_ARTIFACT_CLASSES: dict[str, type] = {
    "voter_panel.json": VoterPanelData,
    "baseline_portfolio.json": BaselinePortfolioData,
    "sentiment_data.json": SentimentData,
    "llm_finetune.json": LLMFineTuneData,
    f"shock_{_SMOKE_SHOCK_ID}.json": ShockResponseData,
    "shock_response.json": ShockResponseData,
    f"equilibrium_{_SMOKE_SHOCK_ID}.json": EquilibriumData,
    "equilibrium.json": EquilibriumData,
    "optimization.json": EquilibriumData,
    "simulation.json": SimulationData,
}


# ── Synthetic LLM payloads ────────────────────────────────────────────────────


def _make_llm_finetune() -> LLMFineTuneData:
    return LLMFineTuneData(
        base_model="mistralai/Mistral-7B-v0.3",
        lora_rank=16,
        n_examples=42,
        cycles_used=[2020],
        adapter_path=None,
    )


def _make_smoke_shock(party: str) -> ShockResponseData:
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


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def schema_run(tmp_path_factory) -> tuple[PipelineConfig, Path]:
    """Single smoke pipeline run using configs/smoke.json; return (cfg, output_dir)."""
    tmp = tmp_path_factory.mktemp("schema")
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_envelope(path: Path) -> dict:
    assert path.exists(), f"Artifact not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _declared_field_names(cls: type) -> set[str]:
    """Return all dataclass field names declared on cls (introspected, not hardcoded)."""
    return {f.name for f in dataclasses.fields(cls)}


def _walk_nonfinite(obj: object, path: str) -> Generator[str, None, None]:
    """Recursively yield dotted paths to any non-finite float in obj."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            yield f"{path}={obj!r}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_nonfinite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_nonfinite(v, f"{path}[{i}]")


# ── Envelope top-level key assertions ─────────────────────────────────────────


class TestEnvelopeTopLevelKeys:
    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_stage_is_str(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert isinstance(envelope.get("stage"), str), (
            f"{filename}: 'stage' must be str, got {type(envelope.get('stage')).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_run_key_is_str(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert isinstance(envelope.get("run_key"), str), (
            f"{filename}: 'run_key' must be str, got {type(envelope.get('run_key')).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_metadata_is_dict(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert isinstance(envelope.get("metadata"), dict), (
            f"{filename}: 'metadata' must be dict, got {type(envelope.get('metadata')).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_data_is_dict(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert isinstance(envelope.get("data"), dict), (
            f"{filename}: 'data' must be dict, got {type(envelope.get('data')).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_no_extra_keys(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        allowed = {"stage", "run_key", "metadata", "data"}
        extra = set(envelope.keys()) - allowed
        assert not extra, f"{filename}: unexpected top-level keys {extra}"


# ── Payload field completeness (dataclass introspection) ──────────────────────


class TestPayloadFieldCompleteness:
    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_CLASSES.items()),
    )
    def test_all_declared_fields_present(self, schema_run, filename, payload_cls):
        """Every field name from dataclasses.fields(cls) must appear in data."""
        _, out = schema_run
        data = _load_envelope(out / filename)["data"]
        declared = _declared_field_names(payload_cls)
        missing = declared - set(data.keys())
        assert not missing, (
            f"{filename}: data dict is missing fields declared on "
            f"{payload_cls.__name__}: {sorted(missing)}"
        )

    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_CLASSES.items()),
    )
    def test_no_undeclared_fields(self, schema_run, filename, payload_cls):
        """data dict must not contain keys absent from the dataclass declaration."""
        _, out = schema_run
        data = _load_envelope(out / filename)["data"]
        declared = _declared_field_names(payload_cls)
        undeclared = set(data.keys()) - declared
        assert not undeclared, (
            f"{filename}: data dict has keys not declared on "
            f"{payload_cls.__name__}: {sorted(undeclared)}"
        )


# ── Non-finite float guard (closes the Infinity/NaN serialization flag) ───────


class TestNoNonFiniteFloats:
    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_no_inf_or_nan_in_data(self, schema_run, filename):
        """No non-finite float (inf, -inf, nan) may appear anywhere in data.

        Walks the entire data dict recursively.  A single non-finite value
        is enough to break standard JSON serialization (json.dumps raises
        ValueError) and violates the artifact contract.
        """
        _, out = schema_run
        data = _load_envelope(out / filename)["data"]
        bad = list(_walk_nonfinite(data, "data"))
        assert not bad, (
            f"{filename}: non-finite floats found in data — "
            f"these would break JSON serialization:\n"
            + "\n".join(f"  {p}" for p in bad)
        )


# ── Round-trip consistency (from_dict → to_dict) ──────────────────────────────


class TestRoundTrip:
    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_CLASSES.items()),
    )
    def test_from_dict_to_dict_is_identity(self, schema_run, filename, payload_cls):
        """from_dict(data).to_dict() must reproduce data exactly."""
        _, out = schema_run
        data = _load_envelope(out / filename)["data"]
        reconstructed = payload_cls.from_dict(data).to_dict()
        assert reconstructed == data, (
            f"{filename}: round-trip mismatch on {payload_cls.__name__}.\n"
            f"Keys that differ: "
            f"{sorted(k for k in set(data) | set(reconstructed) if data.get(k) != reconstructed.get(k))}"
        )


# ── JSON serializability (belt-and-suspenders) ────────────────────────────────


class TestJsonSerializability:
    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_envelope_dumps_without_error(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        try:
            json.dumps(envelope)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"{filename}: json.dumps() raised {type(exc).__name__}: {exc}")
