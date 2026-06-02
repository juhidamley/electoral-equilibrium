"""Tests for electoral/portfolios/weights.py."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from electoral.core.types import CANONICAL_RACES
from electoral.portfolios.weights import (
    _APPROX_ELECTORATE_SHARE,
    equal_weight_baseline,
    value_weight_baseline,
)


# ── equal_weight_baseline ─────────────────────────────────────────────────────


def test_equal_weight_sum_to_one():
    result = equal_weight_baseline(["a", "b", "c", "d", "e"])
    assert sum(result.values()) == pytest.approx(1.0)


def test_equal_weight_all_equal():
    blocs = ["a", "b", "c", "d", "e"]
    result = equal_weight_baseline(blocs)
    values = list(result.values())
    assert all(v == pytest.approx(values[0]) for v in values)


def test_equal_weight_value_is_one_over_n():
    blocs = ["a", "b", "c", "d"]
    result = equal_weight_baseline(blocs)
    assert all(v == pytest.approx(0.25) for v in result.values())


def test_equal_weight_single_bloc():
    result = equal_weight_baseline(["only"])
    assert result == {"only": pytest.approx(1.0)}


def test_equal_weight_keys_match_blocs():
    blocs = ["x", "y", "z"]
    result = equal_weight_baseline(blocs)
    assert list(result.keys()) == blocs


def test_equal_weight_five_canonical_races():
    result = equal_weight_baseline(list(CANONICAL_RACES))
    assert len(result) == 5
    assert all(v == pytest.approx(0.2) for v in result.values())


def test_equal_weight_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        equal_weight_baseline([])


# ── value_weight_baseline — no turnout data ───────────────────────────────────


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["cycle", "bloc", "vote_share"])


def test_value_weight_sum_to_one_no_turnout():
    result = value_weight_baseline(_empty_df(), list(CANONICAL_RACES))
    assert sum(result.values()) == pytest.approx(1.0)


def test_value_weight_keys_match_blocs():
    result = value_weight_baseline(_empty_df(), list(CANONICAL_RACES))
    assert list(result.keys()) == list(CANONICAL_RACES)


def test_value_weight_proportional_to_approx_share_no_turnout():
    # Without turnout data the weights must be proportional to _APPROX_ELECTORATE_SHARE.
    blocs = list(CANONICAL_RACES)
    result = value_weight_baseline(_empty_df(), blocs)
    total_share = sum(_APPROX_ELECTORATE_SHARE[b] for b in blocs)
    for bloc in blocs:
        expected = _APPROX_ELECTORATE_SHARE[bloc] / total_share
        assert result[bloc] == pytest.approx(expected, abs=1e-9), bloc


def test_value_weight_white_is_largest_race_bloc():
    # White (~62%) must have the highest weight among CANONICAL_RACES.
    result = value_weight_baseline(_empty_df(), list(CANONICAL_RACES))
    assert result["white"] == max(result.values())


def test_value_weight_asian_is_smallest_race_bloc():
    result = value_weight_baseline(_empty_df(), list(CANONICAL_RACES))
    assert result["asian"] == min(result.values())


# ── value_weight_baseline — with turnout data ─────────────────────────────────


def _df_with_turnout(blocs: list[str], turnouts: list[float]) -> pd.DataFrame:
    """Single-cycle panel with given per-bloc turnout rates."""
    return pd.DataFrame({"cycle": 2020, "bloc": blocs, "turnout": turnouts})


def test_value_weight_uses_turnout_when_present():
    # Two blocs with no approx share; weights must be proportional to turnout.
    df = _df_with_turnout(["x", "y"], [0.3, 0.6])
    result = value_weight_baseline(df, ["x", "y"])
    assert result["x"] == pytest.approx(0.3 / 0.9, abs=1e-6)
    assert result["y"] == pytest.approx(0.6 / 0.9, abs=1e-6)


def test_value_weight_multiplies_share_by_turnout_for_known_blocs():
    # Known race bloc: raw = pop_share × turnout (priority 1).
    # Use two CANONICAL_RACES blocs with explicit turnout.
    df = _df_with_turnout(
        ["african_american", "white"],
        [0.63, 0.71],
    )
    result = value_weight_baseline(df, ["african_american", "white"])
    raw_aa = _APPROX_ELECTORATE_SHARE["african_american"] * 0.63
    raw_wh = _APPROX_ELECTORATE_SHARE["white"] * 0.71
    total = raw_aa + raw_wh
    assert result["african_american"] == pytest.approx(raw_aa / total, abs=1e-9)
    assert result["white"] == pytest.approx(raw_wh / total, abs=1e-9)


def test_value_weight_mean_turnout_across_cycles():
    # Multiple cycles for a bloc → mean turnout is used.
    df = pd.DataFrame(
        {
            "cycle": [2016, 2020, 2024],
            "bloc": ["african_american"] * 3,
            "turnout": [0.60, 0.63, 0.61],
        }
    )
    result = value_weight_baseline(df, ["african_american"])
    assert result["african_american"] == pytest.approx(1.0)  # single bloc → 100%


def test_value_weight_ignores_null_turnout_rows():
    # Null turnout for one cycle must not skew the mean.
    df = pd.DataFrame(
        {
            "cycle": [2016, 2020],
            "bloc": ["african_american", "african_american"],
            "turnout": [0.60, None],
        }
    )
    result = value_weight_baseline(df, ["african_american"])
    # Only 2016 row contributes; result is still 100% for the single bloc.
    assert result["african_american"] == pytest.approx(1.0)


def test_value_weight_sum_to_one_with_turnout():
    df = _df_with_turnout(list(CANONICAL_RACES), [0.60, 0.53, 0.58, 0.71, 0.56])
    result = value_weight_baseline(df, list(CANONICAL_RACES))
    assert sum(result.values()) == pytest.approx(1.0)


# ── value_weight_baseline — unknown blocs ────────────────────────────────────


def test_value_weight_unknown_bloc_gets_fallback(caplog):
    with caplog.at_level(logging.WARNING, logger="electoral.portfolios.weights"):
        result = value_weight_baseline(_empty_df(), ["unknown_bloc"])
    assert result["unknown_bloc"] == pytest.approx(1.0)
    assert any("unknown_bloc" in msg for msg in caplog.messages)


def test_value_weight_mixed_known_and_unknown():
    # "white" has approx share; "custom_bloc" does not.
    # custom_bloc gets uniform fallback = 1.0; white gets 0.62.
    result = value_weight_baseline(_empty_df(), ["white", "custom_bloc"])
    expected_white = 0.62 / (0.62 + 1.0)
    expected_custom = 1.0 / (0.62 + 1.0)
    assert result["white"] == pytest.approx(expected_white, abs=1e-9)
    assert result["custom_bloc"] == pytest.approx(expected_custom, abs=1e-9)


def test_value_weight_empty_blocs_raises():
    with pytest.raises(ValueError, match="non-empty"):
        value_weight_baseline(_empty_df(), [])


# ── value_weight_baseline — df without turnout column ────────────────────────


def test_value_weight_df_without_turnout_column():
    # df has no 'turnout' column → pure approx-share path for known blocs.
    df = pd.DataFrame({"cycle": [2020], "bloc": ["white"], "vote_share": [0.41]})
    result = value_weight_baseline(df, list(CANONICAL_RACES))
    total = sum(_APPROX_ELECTORATE_SHARE[b] for b in CANONICAL_RACES)
    assert result["white"] == pytest.approx(_APPROX_ELECTORATE_SHARE["white"] / total, abs=1e-9)


# ── review fixes ──────────────────────────────────────────────────────────────


def test_value_weight_zero_total_raises():
    # All blocs have turnout=0 and no approx_pop_share → raw weights all 0.
    df = pd.DataFrame({"cycle": [2020, 2020], "bloc": ["x", "y"], "turnout": [0.0, 0.0]})
    with pytest.raises(ValueError, match="all raw weights are zero"):
        value_weight_baseline(df, ["x", "y"])


def test_value_weight_duplicate_blocs_raises():
    with pytest.raises(ValueError, match="duplicate"):
        value_weight_baseline(_empty_df(), ["white", "latino", "white"])


def test_value_weight_duplicate_error_names_the_offender():
    with pytest.raises(ValueError, match="white"):
        value_weight_baseline(_empty_df(), ["white", "white"])
