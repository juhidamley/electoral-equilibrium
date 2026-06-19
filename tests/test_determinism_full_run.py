"""Determinism test: two pipeline runs under identical config + seed must produce
identical artifacts.

Each stage artifact is loaded from both runs, the data payload is compared
field-by-field, and any difference not in _IGNORE_FIELDS is logged AND
causes the test to fail.  An unexpected diff is a finding, not a maintenance
annoyance — it means some stage violates the seeded-RNG contract.

LLM stages are mocked identically in both runs, so all observable differences
must come from the pipeline's own stochastic operations.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from electoral.artifacts import (
    LLMFineTuneData,
    ShockResponseData,
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

log = logging.getLogger(__name__)

# ── Smoke constants ───────────────────────────────────────────────────────────

_SMOKE_EVENT = "smoke_test"
_SMOKE_SHOCK_ID = _SMOKE_EVENT[:30].lower().replace(" ", "_")

_ALL_RACES = list(CANONICAL_RACES)
_ALL_RELIGIONS = list(CANONICAL_RELIGIONS)
_ALL_GENDERS = list(CANONICAL_GENDERS)
_COV_5X5 = [[0.0004 if i == j else 0.0 for j in range(5)] for i in range(5)]

# ── Explicit ignore-set ───────────────────────────────────────────────────────
# Fields that may legitimately differ between two runs even with the same seed.
# This set is kept minimal and documented.  Any field NOT listed here that
# differs between runs will be logged as a warning AND fail the test.
_IGNORE_FIELDS: frozenset[str] = frozenset(
    # Currently empty: all stage outputs should be bit-for-bit identical when
    # the global seed and input data are fixed.  Add fields here only when a
    # documented source of non-determinism is confirmed (e.g. wall-clock
    # timestamps embedded in metadata values).
)


# ── Synthetic LLM payloads (identical mock in both runs) ──────────────────────


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


# ── Pipeline runner ───────────────────────────────────────────────────────────


def _run_pipeline(cfg: PipelineConfig, n_sim: int = 500) -> None:
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
        run_simulations(cfg, equilibrium, n_simulations=n_sim)


def _load_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["data"]


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> tuple[PipelineConfig, Path, PipelineConfig, Path]:
    """Run the pipeline twice with identical configs/smoke.json + same seed."""
    tmp_a = tmp_path_factory.mktemp("det_a")
    tmp_b = tmp_path_factory.mktemp("det_b")

    cfg_a = PipelineConfig.from_json("configs/smoke.json")
    cfg_a = dataclasses.replace(cfg_a, output_dir=str(tmp_a / "artifacts"))
    cfg_a.validate()

    cfg_b = PipelineConfig.from_json("configs/smoke.json")
    cfg_b = dataclasses.replace(cfg_b, output_dir=str(tmp_b / "artifacts"))
    cfg_b.validate()

    _run_pipeline(cfg_a)
    _run_pipeline(cfg_b)
    return cfg_a, Path(cfg_a.output_dir), cfg_b, Path(cfg_b.output_dir)


# ── Comparison helpers ────────────────────────────────────────────────────────


def _compare_values(v1: object, v2: object, float_tol: float = 1e-9) -> bool:
    """Return True if the two values are equal within float_tol."""
    if isinstance(v1, float) and isinstance(v2, float):
        if math.isnan(v1) and math.isnan(v2):
            return True
        return abs(v1 - v2) <= float_tol
    if isinstance(v1, dict) and isinstance(v2, dict):
        return _dicts_equal(v1, v2, float_tol)
    if isinstance(v1, list) and isinstance(v2, list):
        return len(v1) == len(v2) and all(_compare_values(a, b, float_tol) for a, b in zip(v1, v2))
    return v1 == v2


def _dicts_equal(d1: dict, d2: dict, float_tol: float = 1e-9) -> bool:
    if set(d1) != set(d2):
        return False
    return all(_compare_values(d1[k], d2[k], float_tol) for k in d1)


def _compare_payload(
    d1: dict,
    d2: dict,
    label: str,
    *,
    float_tol: float = 1e-9,
) -> list[str]:
    """Return field names that differ between two data dicts.

    Logs a WARNING for each differing field.  Callers should assert the
    returned list is empty (after subtracting _IGNORE_FIELDS) to fail the test.
    """
    diffs: list[str] = []
    all_keys = sorted(set(d1) | set(d2))
    for k in all_keys:
        if k not in d1 or k not in d2:
            log.warning("%s: field %r present in one run but not the other", label, k)
            diffs.append(k)
            continue
        if not _compare_values(d1[k], d2[k], float_tol):
            log.warning(
                "%s: field %r differs — run_a=%r  run_b=%r",
                label,
                k,
                d1[k],
                d2[k],
            )
            diffs.append(k)
    return diffs


def _assert_deterministic(d1: dict, d2: dict, label: str) -> None:
    """Assert that d1 and d2 agree on all fields outside _IGNORE_FIELDS."""
    all_diffs = _compare_payload(d1, d2, label)
    unexpected = [k for k in all_diffs if k not in _IGNORE_FIELDS]
    assert not unexpected, (
        f"{label}: {len(unexpected)} unexpected non-deterministic field(s): {unexpected}. "
        f"Check logs for per-field values. If non-determinism is intentional, "
        f"add the field name(s) to _IGNORE_FIELDS with a documented reason."
    )


# ── VoterPanelData ─────────────────────────────────────────────────────────────


class TestVoterPanelDeterminism:
    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "voter_panel.json"),
            _load_data(out_b / "voter_panel.json"),
            "voter_panel",
        )

    def test_cycles_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        assert (
            _load_data(out_a / "voter_panel.json")["cycles"]
            == _load_data(out_b / "voter_panel.json")["cycles"]
        )

    def test_layer_weights_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        assert (
            _load_data(out_a / "voter_panel.json")["layer_weights"]
            == _load_data(out_b / "voter_panel.json")["layer_weights"]
        )


# ── BaselinePortfolioData ──────────────────────────────────────────────────────


class TestBaselineDeterminism:
    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "baseline_portfolio.json"),
            _load_data(out_b / "baseline_portfolio.json"),
            "baseline_portfolio",
        )

    def test_weights_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "baseline_portfolio.json")
        db = _load_data(out_b / "baseline_portfolio.json")
        assert da["weights"] == db["weights"]

    def test_mu_eff_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "baseline_portfolio.json")
        db = _load_data(out_b / "baseline_portfolio.json")
        assert abs(da["mu_eff"] - db["mu_eff"]) <= 1e-9


# ── SentimentData ──────────────────────────────────────────────────────────────


class TestSentimentDeterminism:
    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "sentiment_data.json"),
            _load_data(out_b / "sentiment_data.json"),
            "sentiment_data",
        )


# ── LLMFineTuneData ────────────────────────────────────────────────────────────


class TestLLMFineTuneDeterminism:
    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "llm_finetune.json"),
            _load_data(out_b / "llm_finetune.json"),
            "llm_finetune",
        )


# ── ShockResponseData (per-id and legacy) ─────────────────────────────────────


class TestShockResponseDeterminism:
    def test_per_id_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / f"shock_{_SMOKE_SHOCK_ID}.json"),
            _load_data(out_b / f"shock_{_SMOKE_SHOCK_ID}.json"),
            f"shock_{_SMOKE_SHOCK_ID}",
        )

    def test_legacy_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "shock_response.json"),
            _load_data(out_b / "shock_response.json"),
            "shock_response",
        )

    def test_covariance_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "shock_response.json")
        db = _load_data(out_b / "shock_response.json")
        # Covariance from _build_sigma_delta must be deterministic
        assert _compare_values(da["covariance"], db["covariance"]), (
            "shock.covariance differs between runs — _build_sigma_delta is non-deterministic"
        )


# ── EquilibriumData (shock kernel, per-id and legacy) ─────────────────────────


class TestEquilibriumKernelDeterminism:
    def test_per_id_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / f"equilibrium_{_SMOKE_SHOCK_ID}.json"),
            _load_data(out_b / f"equilibrium_{_SMOKE_SHOCK_ID}.json"),
            f"equilibrium_{_SMOKE_SHOCK_ID}",
        )

    def test_legacy_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "equilibrium.json"),
            _load_data(out_b / "equilibrium.json"),
            "equilibrium",
        )


# ── EquilibriumData (stages build_optimization) ───────────────────────────────


class TestOptimizationDeterminism:
    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "optimization.json"),
            _load_data(out_b / "optimization.json"),
            "optimization",
        )

    def test_feasibility_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "optimization.json")
        db = _load_data(out_b / "optimization.json")
        assert da["feasible"] == db["feasible"]
        assert da["target_met"] == db["target_met"]


# ── SimulationData ─────────────────────────────────────────────────────────────


class TestSimulationDeterminism:
    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        _assert_deterministic(
            _load_data(out_a / "simulation.json"),
            _load_data(out_b / "simulation.json"),
            "simulation",
        )

    def test_win_probability_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "simulation.json")
        db = _load_data(out_b / "simulation.json")
        assert abs(da["win_probability"] - db["win_probability"]) <= 1e-9, (
            f"win_probability differs: {da['win_probability']} vs {db['win_probability']}"
        )

    def test_ci_bounds_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "simulation.json")
        db = _load_data(out_b / "simulation.json")
        assert abs(da["win_probability_low"] - db["win_probability_low"]) <= 1e-9
        assert abs(da["win_probability_high"] - db["win_probability_high"]) <= 1e-9

    def test_seed_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "simulation.json")
        db = _load_data(out_b / "simulation.json")
        assert da["seed"] == db["seed"]


# ── Different seed changes seed field ─────────────────────────────────────────


class TestSeedContract:
    def test_different_seed_changes_simulation_seed(self, tmp_path):
        """A changed global seed must propagate to the simulation artifact's seed field."""
        cfg_base = PipelineConfig.from_json("configs/smoke.json")
        cfg_a = dataclasses.replace(cfg_base, seed=7, output_dir=str(tmp_path / "a"))
        cfg_b = dataclasses.replace(cfg_base, seed=99, output_dir=str(tmp_path / "b"))
        cfg_a.validate()
        cfg_b.validate()
        _run_pipeline(cfg_a, n_sim=200)
        _run_pipeline(cfg_b, n_sim=200)
        da = _load_data(Path(cfg_a.output_dir) / "simulation.json")
        db = _load_data(Path(cfg_b.output_dir) / "simulation.json")
        assert da["seed"] != db["seed"], (
            "Simulation seed fields must differ when global seeds differ — "
            "derive_seed() contract is not propagated"
        )
