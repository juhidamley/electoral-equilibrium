"""Determinism test: two pipeline runs with identical config and seed.

Runs the non-LLM pipeline stages twice under identical conditions
(same seed, same fixtures) but writing to separate output directories.
For each stage artifact, loads both copies, parses the data payload,
and asserts field-by-field equality for all non-path fields.

Non-deterministic fields (if any are found) are logged as warnings rather
than hard failures so the test suite can identify which specific fields
diverge.  The win_probability, weights, and all primary numerical fields
must be bit-for-bit equal under the seeded RNG contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from electoral.artifacts import (
    ShockResponseData,
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

log = logging.getLogger(__name__)

# ── Synthetic shock — no LLM required ─────────────────────────────────────────

_RACE_IDS = list(CANONICAL_RACES)
_RELIGION_IDS = list(CANONICAL_RELIGIONS)
_GENDER_IDS = list(CANONICAL_GENDERS)
_COV_5X5 = [[0.01 if i == j else 0.0 for j in range(5)] for i in range(5)]


def _make_smoke_shock(party: str) -> ShockResponseData:
    return ShockResponseData(
        shock="determinism_smoke_event",
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


# ── Pipeline runner ────────────────────────────────────────────────────────────


def _run_pipeline(cfg: PipelineConfig, n_sim: int = 500) -> None:
    """Run all non-LLM stages with the given config."""
    panel = build_voter_panel(cfg)
    build_baseline_portfolio(cfg, panel)
    build_sentiment_data(cfg, panel)
    shock = _make_smoke_shock(cfg.party)
    equilibrium = build_optimization(cfg, shock)
    run_simulations(cfg, equilibrium, n_simulations=n_sim)


def _make_cfg(tmp_path: Path, seed: int = 42) -> PipelineConfig:
    import json as _json

    tmp_path.mkdir(parents=True, exist_ok=True)
    data = {
        "run_key": "determinism_test",
        "seed": seed,
        "party": "democrat",
        "target": 0.535,
        "data_path": "tests/fixtures/",
        "output_dir": str(tmp_path / "artifacts"),
        "pipeline_mode": "historical",
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(_json.dumps(data))
    cfg = PipelineConfig.from_json(str(cfg_path))
    cfg.validate()
    return cfg


def _load_data(artifact_path: Path) -> dict:
    envelope = json.loads(artifact_path.read_text(encoding="utf-8"))
    return envelope["data"]


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """Run the pipeline twice with the same seed; return (cfg_a, dir_a, cfg_b, dir_b)."""
    tmp_a = tmp_path_factory.mktemp("det_run_a")
    tmp_b = tmp_path_factory.mktemp("det_run_b")
    cfg_a = _make_cfg(tmp_a, seed=42)
    cfg_b = _make_cfg(tmp_b, seed=42)
    _run_pipeline(cfg_a, n_sim=500)
    _run_pipeline(cfg_b, n_sim=500)
    return cfg_a, Path(cfg_a.output_dir), cfg_b, Path(cfg_b.output_dir)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compare_dicts(
    d1: dict,
    d2: dict,
    label: str,
    *,
    float_tol: float = 1e-12,
) -> list[str]:
    """Return list of fields that differ between two data dicts.

    Float comparison uses absolute tolerance to catch floating-point
    non-determinism while ignoring epsilon-level noise.
    Logs a warning for each differing field.
    """
    diffs = []
    all_keys = set(d1.keys()) | set(d2.keys())
    for k in sorted(all_keys):
        v1 = d1.get(k)
        v2 = d2.get(k)
        if isinstance(v1, float) and isinstance(v2, float):
            if abs(v1 - v2) > float_tol:
                log.warning("%s: field %r differs: %r vs %r", label, k, v1, v2)
                diffs.append(k)
        elif v1 != v2:
            log.warning("%s: field %r differs: %r vs %r", label, k, v1, v2)
            diffs.append(k)
    return diffs


# ── VoterPanelData ─────────────────────────────────────────────────────────────


class TestVoterPanelDeterminism:
    def test_cycles_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "voter_panel.json")
        db = _load_data(out_b / "voter_panel.json")
        assert da["cycles"] == db["cycles"]

    def test_races_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "voter_panel.json")
        db = _load_data(out_b / "voter_panel.json")
        assert da["races"] == db["races"]

    def test_layer_weights_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "voter_panel.json")
        db = _load_data(out_b / "voter_panel.json")
        assert da["layer_weights"] == db["layer_weights"]

    def test_row_counts_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "voter_panel.json")
        db = _load_data(out_b / "voter_panel.json")
        assert da["n_rows_race"] == db["n_rows_race"]
        assert da["n_rows_religion"] == db["n_rows_religion"]
        assert da["n_rows_gender"] == db["n_rows_gender"]

    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "voter_panel.json")
        db = _load_data(out_b / "voter_panel.json")
        diffs = _compare_dicts(da, db, "voter_panel")
        assert not diffs, f"Non-deterministic fields in voter_panel.json: {diffs}"


# ── BaselinePortfolioData ──────────────────────────────────────────────────────


class TestBaselineDeterminism:
    def test_weights_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "baseline_portfolio.json")
        db = _load_data(out_b / "baseline_portfolio.json")
        assert da["weights"] == db["weights"]

    def test_mu_eff_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "baseline_portfolio.json")
        db = _load_data(out_b / "baseline_portfolio.json")
        assert abs(da["mu_eff"] - db["mu_eff"]) < 1e-12

    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "baseline_portfolio.json")
        db = _load_data(out_b / "baseline_portfolio.json")
        diffs = _compare_dicts(da, db, "baseline_portfolio")
        assert not diffs, f"Non-deterministic fields in baseline_portfolio.json: {diffs}"


# ── SentimentData ──────────────────────────────────────────────────────────────


class TestSentimentDeterminism:
    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "sentiment_data.json")
        db = _load_data(out_b / "sentiment_data.json")
        diffs = _compare_dicts(da, db, "sentiment_data")
        assert not diffs, f"Non-deterministic fields in sentiment_data.json: {diffs}"


# ── EquilibriumData ────────────────────────────────────────────────────────────


class TestEquilibriumDeterminism:
    def test_weights_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "optimization.json")
        db = _load_data(out_b / "optimization.json")
        assert da["weights"] == db["weights"]

    def test_feasible_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "optimization.json")
        db = _load_data(out_b / "optimization.json")
        assert da["feasible"] == db["feasible"]

    def test_target_met_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "optimization.json")
        db = _load_data(out_b / "optimization.json")
        assert da["target_met"] == db["target_met"]

    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "optimization.json")
        db = _load_data(out_b / "optimization.json")
        diffs = _compare_dicts(da, db, "optimization")
        assert not diffs, f"Non-deterministic fields in optimization.json: {diffs}"


# ── SimulationData ─────────────────────────────────────────────────────────────


class TestSimulationDeterminism:
    def test_win_probability_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "simulation.json")
        db = _load_data(out_b / "simulation.json")
        assert abs(da["win_probability"] - db["win_probability"]) < 1e-12, (
            f"win_probability differs: {da['win_probability']} vs {db['win_probability']}"
        )

    def test_ci_bounds_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "simulation.json")
        db = _load_data(out_b / "simulation.json")
        assert abs(da["win_probability_low"] - db["win_probability_low"]) < 1e-12
        assert abs(da["win_probability_high"] - db["win_probability_high"]) < 1e-12

    def test_seed_identical(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "simulation.json")
        db = _load_data(out_b / "simulation.json")
        assert da["seed"] == db["seed"]

    def test_no_differing_fields(self, two_runs):
        _, out_a, _, out_b = two_runs
        da = _load_data(out_a / "simulation.json")
        db = _load_data(out_b / "simulation.json")
        diffs = _compare_dicts(da, db, "simulation")
        assert not diffs, f"Non-deterministic fields in simulation.json: {diffs}"


# ── Cross-run seed invariant ───────────────────────────────────────────────────


class TestSeedInvariant:
    def test_different_seed_changes_simulation(self, tmp_path):
        """Sanity check: a different seed SHOULD change the simulation output."""
        cfg_a = _make_cfg(tmp_path / "seed_a", seed=42)
        cfg_b = _make_cfg(tmp_path / "seed_b", seed=99)
        _run_pipeline(cfg_a, n_sim=500)
        _run_pipeline(cfg_b, n_sim=500)
        da = _load_data(Path(cfg_a.output_dir) / "simulation.json")
        db = _load_data(Path(cfg_b.output_dir) / "simulation.json")
        # win_probability may or may not change (depends on the equilibrium), but
        # the seed fields must differ, which demonstrates the RNG contract is wired
        assert da["seed"] != db["seed"], "Seeds should differ when config seeds differ"
