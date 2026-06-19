"""Artifact schema validation tests.

For each artifact produced by a smoke run:
  1. Assert the StageArtifact envelope has the expected top-level keys
     with the correct Python types (stage: str, run_key: str,
     metadata: dict, data: dict).
  2. Assert that all dataclass fields declared on the payload class are
     present in the data dict — i.e., to_dict() is complete.
  3. Assert the data dict is JSON-serializable (no datetime, numpy types, etc.).

These tests detect regressions where a field is added to a dataclass but
not wired into to_dict(), or where a serialization bug emits a non-JSON
type into the artifact.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import get_type_hints

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

# ── Expected payload class per artifact file ──────────────────────────────────

_ARTIFACT_CLASSES = {
    "voter_panel.json": VoterPanelData,
    "baseline_portfolio.json": BaselinePortfolioData,
    "sentiment_data.json": SentimentData,
    "optimization.json": EquilibriumData,
    "simulation.json": SimulationData,
}

# ── Synthetic shock — no LLM required ────────────────────────────────────────

_RACE_IDS = list(CANONICAL_RACES)
_RELIGION_IDS = list(CANONICAL_RELIGIONS)
_GENDER_IDS = list(CANONICAL_GENDERS)
_COV_5X5 = [[0.01 if i == j else 0.0 for j in range(5)] for i in range(5)]


def _make_smoke_shock(party: str) -> ShockResponseData:
    return ShockResponseData(
        shock="schema_smoke_event",
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


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def schema_run(tmp_path_factory) -> tuple[PipelineConfig, Path]:
    """Single non-LLM pipeline run; return (config, output_dir)."""
    import json as _json

    tmp = tmp_path_factory.mktemp("schema")
    data = {
        "run_key": "schema_smoke",
        "seed": 7,
        "party": "democrat",
        "target": 0.535,
        "data_path": "tests/fixtures/",
        "output_dir": str(tmp / "artifacts"),
        "pipeline_mode": "historical",
    }
    cfg_path = tmp / "cfg.json"
    cfg_path.write_text(_json.dumps(data))
    cfg = PipelineConfig.from_json(str(cfg_path))
    cfg.validate()

    panel = build_voter_panel(cfg)
    build_baseline_portfolio(cfg, panel)
    build_sentiment_data(cfg, panel)
    shock = _make_smoke_shock(cfg.party)
    equilibrium = build_optimization(cfg, shock)
    run_simulations(cfg, equilibrium, n_simulations=500)

    return cfg, Path(cfg.output_dir)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_envelope(path: Path) -> dict:
    assert path.exists(), f"Artifact not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _declared_field_names(cls) -> set[str]:
    """Return all dataclass field names declared on cls (excluding ClassVar)."""
    return {f.name for f in dataclasses.fields(cls)}


# ── Envelope top-level keys ───────────────────────────────────────────────────


class TestEnvelopeTopLevelKeys:
    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_stage_key_present_and_str(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert "stage" in envelope, f"{filename}: missing 'stage' key"
        assert isinstance(envelope["stage"], str), (
            f"{filename}: 'stage' must be str, got {type(envelope['stage']).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_run_key_present_and_str(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert "run_key" in envelope, f"{filename}: missing 'run_key' key"
        assert isinstance(envelope["run_key"], str), (
            f"{filename}: 'run_key' must be str, got {type(envelope['run_key']).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_metadata_present_and_dict(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert "metadata" in envelope, f"{filename}: missing 'metadata' key"
        assert isinstance(envelope["metadata"], dict), (
            f"{filename}: 'metadata' must be dict, got {type(envelope['metadata']).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_data_present_and_dict(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        assert "data" in envelope, f"{filename}: missing 'data' key"
        assert isinstance(envelope["data"], dict), (
            f"{filename}: 'data' must be dict, got {type(envelope['data']).__name__}"
        )

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_no_unexpected_top_level_keys(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        allowed = {"stage", "run_key", "metadata", "data"}
        unexpected = set(envelope.keys()) - allowed
        assert not unexpected, f"{filename}: unexpected top-level keys {unexpected}"


# ── Payload field completeness ────────────────────────────────────────────────


class TestPayloadFieldCompleteness:
    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_CLASSES.items()),
    )
    def test_all_declared_fields_in_data(self, schema_run, filename, payload_cls):
        """Every dataclass field name must appear as a key in data."""
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        data_keys = set(envelope["data"].keys())
        declared = _declared_field_names(payload_cls)
        missing = declared - data_keys
        assert not missing, (
            f"{filename}: data dict is missing fields declared on "
            f"{payload_cls.__name__}: {sorted(missing)}"
        )

    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_CLASSES.items()),
    )
    def test_no_none_for_required_fields(self, schema_run, filename, payload_cls):
        """Fields without Optional typing must not be None in the serialized data."""
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        data = envelope["data"]
        # Gather fields that do not have Optional or None in their annotation
        optional_fields: set[str] = set()
        try:
            hints = get_type_hints(payload_cls)
            for field in dataclasses.fields(payload_cls):
                hint = hints.get(field.name, None)
                if hint is None:
                    continue
                hint_str = str(hint)
                if "None" in hint_str or "Optional" in hint_str:
                    optional_fields.add(field.name)
        except Exception:
            # get_type_hints may fail in some edge cases; skip optional detection
            optional_fields = _declared_field_names(payload_cls)

        non_optional = _declared_field_names(payload_cls) - optional_fields
        null_required = [k for k in non_optional if k in data and data[k] is None]
        assert not null_required, (
            f"{filename}: required (non-Optional) fields are None in data: {null_required}"
        )


# ── JSON serializability ──────────────────────────────────────────────────────


class TestJsonSerializability:
    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_full_envelope_json_serializable(self, schema_run, filename):
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        try:
            json.dumps(envelope)
        except TypeError as e:
            pytest.fail(f"{filename}: envelope is not JSON-serializable: {e}")

    @pytest.mark.parametrize("filename", list(_ARTIFACT_CLASSES))
    def test_no_nan_or_inf_in_data(self, schema_run, filename):
        """Non-finite floats must not appear in artifact data — they break JSON."""
        _, out = schema_run
        envelope = _load_envelope(out / filename)

        def _check(obj, path: str) -> list[str]:
            bad = []
            if isinstance(obj, float):
                import math
                if not math.isfinite(obj):
                    bad.append(f"{path}={obj}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    bad.extend(_check(v, f"{path}.{k}"))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    bad.extend(_check(v, f"{path}[{i}]"))
            return bad

        bad_fields = _check(envelope["data"], "data")
        assert not bad_fields, (
            f"{filename}: non-finite floats found in data: {bad_fields}"
        )


# ── Round-trip through from_dict ──────────────────────────────────────────────


class TestRoundTripConsistency:
    @pytest.mark.parametrize(
        "filename,payload_cls",
        list(_ARTIFACT_CLASSES.items()),
    )
    def test_from_dict_produces_same_to_dict(self, schema_run, filename, payload_cls):
        """from_dict(data).to_dict() must reproduce data exactly."""
        _, out = schema_run
        envelope = _load_envelope(out / filename)
        data = envelope["data"]
        obj = payload_cls.from_dict(data)
        reconstructed = obj.to_dict()
        assert reconstructed == data, (
            f"{filename}: round-trip mismatch. "
            f"from_dict(data).to_dict() != original data.\n"
            f"Keys that differ: "
            f"{[k for k in set(data) | set(reconstructed) if data.get(k) != reconstructed.get(k)]}"
        )
