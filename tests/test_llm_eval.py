"""Tests for electoral/llm/eval.py — delta-bin evaluation metrics."""

from __future__ import annotations

import pytest

from electoral.llm.eval import (
    compute_eval_report,
    direction_accuracy,
    mae_in_delta_units,
    per_stratum_mae,
)
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS


# ── Fixtures ──────────────────────────────────────────────────────────────────

ALL_BLOCS = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)


def _all_neutral() -> dict[str, str]:
    return {b: "neutral" for b in ALL_BLOCS}


def _all_mod_pos() -> dict[str, str]:
    return {b: "mod_pos" for b in ALL_BLOCS}


def _all_mod_neg() -> dict[str, str]:
    return {b: "mod_neg" for b in ALL_BLOCS}


# ── mae_in_delta_units ────────────────────────────────────────────────────────


def test_mae_perfect_prediction_is_zero():
    bins = {"african_american": "mild_pos", "latino": "slight_neg"}
    assert mae_in_delta_units(bins, bins) == pytest.approx(0.0)


def test_mae_neutral_vs_neutral_is_zero():
    bins = _all_neutral()
    assert mae_in_delta_units(bins, bins) == pytest.approx(0.0)


def test_mae_mod_pos_vs_mod_neg():
    # mod_pos midpoint = +0.070, mod_neg midpoint = -0.070 → error = 0.140
    pred = {"african_american": "mod_pos"}
    true = {"african_american": "mod_neg"}
    assert mae_in_delta_units(pred, true) == pytest.approx(0.140, abs=1e-9)


def test_mae_slight_pos_vs_neutral():
    # slight_pos midpoint = +0.012, neutral midpoint = 0.000 → error = 0.012
    pred = {"african_american": "slight_pos"}
    true = {"african_american": "neutral"}
    assert mae_in_delta_units(pred, true) == pytest.approx(0.012, abs=1e-9)


def test_mae_averages_over_blocs():
    # bloc a: mod_pos (0.070) vs mod_neg (−0.070) → 0.140
    # bloc b: neutral (0.000) vs neutral (0.000) → 0.000
    # mean = 0.070
    pred = {"a": "mod_pos", "b": "neutral"}
    true = {"a": "mod_neg", "b": "neutral"}
    assert mae_in_delta_units(pred, true) == pytest.approx(0.070, abs=1e-9)


def test_mae_uses_only_common_keys():
    pred = {"african_american": "mod_pos", "extra_key": "neutral"}
    true = {"african_american": "mod_pos", "other_key": "neutral"}
    # Only "african_american" is shared; extra_key and other_key ignored
    assert mae_in_delta_units(pred, true) == pytest.approx(0.0)


def test_mae_no_common_keys_raises():
    pred = {"african_american": "mod_pos"}
    true = {"latino": "neutral"}
    with pytest.raises(ValueError, match="no common keys"):
        mae_in_delta_units(pred, true)


def test_mae_invalid_pred_bin_raises():
    pred = {"african_american": "super_pos"}  # not a valid token
    true = {"african_american": "neutral"}
    with pytest.raises(ValueError, match="valid delta bin"):
        mae_in_delta_units(pred, true)


def test_mae_symmetry():
    pred = {"a": "mod_pos"}
    true = {"a": "mod_neg"}
    assert mae_in_delta_units(pred, true) == pytest.approx(mae_in_delta_units(true, pred))


# ── direction_accuracy ────────────────────────────────────────────────────────


def test_direction_accuracy_perfect():
    bins = {"african_american": "mod_pos", "latino": "slight_neg", "asian": "neutral"}
    assert direction_accuracy(bins, bins) == pytest.approx(1.0)


def test_direction_accuracy_all_wrong():
    pred = {"african_american": "mod_pos", "latino": "mod_pos"}
    true = {"african_american": "mod_neg", "latino": "mod_neg"}
    assert direction_accuracy(pred, true) == pytest.approx(0.0)


def test_direction_accuracy_neutral_vs_neutral_correct():
    pred = {"african_american": "neutral"}
    true = {"african_american": "neutral"}
    assert direction_accuracy(pred, true) == pytest.approx(1.0)


def test_direction_accuracy_slight_pos_vs_mod_pos_correct():
    # Both positive → same direction
    pred = {"african_american": "slight_pos"}
    true = {"african_american": "mod_pos"}
    assert direction_accuracy(pred, true) == pytest.approx(1.0)


def test_direction_accuracy_slight_pos_vs_slight_neg_wrong():
    pred = {"african_american": "slight_pos"}
    true = {"african_american": "slight_neg"}
    assert direction_accuracy(pred, true) == pytest.approx(0.0)


def test_direction_accuracy_neutral_vs_slight_pos_wrong():
    # neutral = 0; slight_pos = +1 → different signs
    pred = {"african_american": "neutral"}
    true = {"african_american": "slight_pos"}
    assert direction_accuracy(pred, true) == pytest.approx(0.0)


def test_direction_accuracy_half_correct():
    # a: correct (both pos), b: wrong (pos vs neg)
    pred = {"a": "mod_pos", "b": "mod_pos"}
    true = {"a": "slight_pos", "b": "mod_neg"}
    assert direction_accuracy(pred, true) == pytest.approx(0.5)


def test_direction_accuracy_no_common_keys_raises():
    with pytest.raises(ValueError, match="no common keys"):
        direction_accuracy({"a": "neutral"}, {"b": "neutral"})


# ── per_stratum_mae ───────────────────────────────────────────────────────────


def test_per_stratum_mae_returns_three_keys():
    pred = _all_neutral()
    true = _all_neutral()
    result = per_stratum_mae(
        pred,
        true,
        races=list(CANONICAL_RACES),
        religions=list(CANONICAL_RELIGIONS),
        genders=list(CANONICAL_GENDERS),
    )
    assert set(result.keys()) == {"race", "religion", "gender"}


def test_per_stratum_mae_perfect_is_zero():
    pred = _all_mod_pos()
    true = _all_mod_pos()
    result = per_stratum_mae(
        pred,
        true,
        races=list(CANONICAL_RACES),
        religions=list(CANONICAL_RELIGIONS),
        genders=list(CANONICAL_GENDERS),
    )
    for stratum, mae in result.items():
        assert mae == pytest.approx(0.0), f"{stratum} MAE should be 0"


def test_per_stratum_mae_race_nonzero_others_zero():
    pred = {b: "neutral" for b in ALL_BLOCS}
    true = {b: "neutral" for b in ALL_BLOCS}
    for b in CANONICAL_RACES:
        pred[b] = "mod_pos"
    result = per_stratum_mae(
        pred,
        true,
        races=list(CANONICAL_RACES),
        religions=list(CANONICAL_RELIGIONS),
        genders=list(CANONICAL_GENDERS),
    )
    assert result["race"] == pytest.approx(0.070, abs=1e-9)
    assert result["religion"] == pytest.approx(0.0)
    assert result["gender"] == pytest.approx(0.0)


# ── compute_eval_report ───────────────────────────────────────────────────────


def test_compute_eval_report_perfect_predictions():
    bins = {"african_american": "mild_pos", "latino": "mod_neg"}
    examples = [(bins, bins)] * 5
    report = compute_eval_report(examples)
    assert report["mae"] == pytest.approx(0.0)
    assert report["direction_accuracy"] == pytest.approx(1.0)
    assert report["n_examples"] == 5


def test_compute_eval_report_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_eval_report([])


def test_compute_eval_report_single_example():
    pred = {"african_american": "mod_pos"}
    true = {"african_american": "mod_neg"}
    report = compute_eval_report([(pred, true)])
    assert report["mae"] == pytest.approx(0.140, abs=1e-9)
    assert report["direction_accuracy"] == pytest.approx(0.0)
    assert report["n_examples"] == 1


def test_compute_eval_report_averages_mae():
    # Example 1: mod_pos vs mod_neg → MAE = 0.140
    # Example 2: neutral vs neutral → MAE = 0.000
    # Mean MAE = 0.070
    ex1 = ({"a": "mod_pos"}, {"a": "mod_neg"})
    ex2 = ({"a": "neutral"}, {"a": "neutral"})
    report = compute_eval_report([ex1, ex2])
    assert report["mae"] == pytest.approx(0.070, abs=1e-9)
    assert report["n_examples"] == 2


def test_compute_eval_report_averages_direction_accuracy():
    # Example 1: all correct (1.0)
    # Example 2: all wrong (0.0)
    bins_pos = {"a": "mod_pos"}
    bins_neg = {"a": "mod_neg"}
    report = compute_eval_report([(bins_pos, bins_pos), (bins_pos, bins_neg)])
    assert report["direction_accuracy"] == pytest.approx(0.5)
