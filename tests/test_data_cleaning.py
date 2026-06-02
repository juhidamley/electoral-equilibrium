"""Tests for electoral/data/cleaning.py — clean_raw_panel."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from electoral.data.cleaning import clean_raw_panel


def _df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


# ── Dtype coercions ───────────────────────────────────────────────────────────


def test_cycle_string_coerced_to_int64():
    df = _df(cycle=["2020", "2016"], bloc=["white", "latino"])
    out = clean_raw_panel(df)
    assert out["cycle"].dtype == pd.Int64Dtype()
    assert list(out["cycle"]) == [2016, 2020]  # sorted


def test_cycle_float_coerced_to_int64():
    df = _df(cycle=[2020.0, 2016.0], bloc=["white", "latino"])
    out = clean_raw_panel(df)
    assert out["cycle"].dtype == pd.Int64Dtype()


def test_vote_share_coerced_to_float64():
    df = _df(cycle=[2020], bloc=["white"], vote_share=["0.41"])
    out = clean_raw_panel(df)
    assert out["vote_share"].dtype == pd.Float64Dtype()
    assert float(out["vote_share"].iloc[0]) == pytest.approx(0.41)


def test_turnout_coerced_to_float64():
    df = _df(cycle=[2020], bloc=["white"], turnout=["0.71"])
    out = clean_raw_panel(df)
    assert out["turnout"].dtype == pd.Float64Dtype()


def test_nonnumeric_vote_share_becomes_na():
    df = _df(cycle=[2020], bloc=["white"], vote_share=["high"])
    out = clean_raw_panel(df)
    assert pd.isna(out["vote_share"].iloc[0])


# ── Bloc normalisation ────────────────────────────────────────────────────────


def test_bloc_uppercased_lowercased():
    df = _df(cycle=[2020], bloc=["WHITE"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "white"


def test_bloc_spaces_replaced_with_underscore():
    df = _df(cycle=[2020], bloc=["African American"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "african_american"


def test_bloc_hyphens_replaced_with_underscore():
    df = _df(cycle=[2020], bloc=["non-white"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "non_white"


def test_bloc_slashes_replaced_with_underscore():
    df = _df(cycle=[2020], bloc=["Latino/Hispanic"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "latino_hispanic"


# ── Drop on missing required values ──────────────────────────────────────────


def test_null_cycle_row_dropped(caplog):
    df = _df(cycle=[2020, None], bloc=["white", "latino"])
    with caplog.at_level(logging.INFO, logger="electoral.data.cleaning"):
        out = clean_raw_panel(df)
    assert len(out) == 1
    assert "dropped 1" in caplog.text


def test_null_bloc_row_dropped(caplog):
    df = _df(cycle=[2020, 2020], bloc=["white", None])
    with caplog.at_level(logging.INFO, logger="electoral.data.cleaning"):
        out = clean_raw_panel(df)
    assert len(out) == 1
    assert "dropped 1" in caplog.text


def test_no_log_when_no_rows_dropped(caplog):
    df = _df(cycle=[2020], bloc=["white"])
    with caplog.at_level(logging.INFO, logger="electoral.data.cleaning"):
        clean_raw_panel(df)
    assert "dropped" not in caplog.text


# ── Sort ──────────────────────────────────────────────────────────────────────


def test_output_sorted_by_cycle_then_bloc():
    df = _df(
        cycle=[2020, 2016, 2020],
        bloc=["white", "white", "latino"],
    )
    out = clean_raw_panel(df)
    assert list(out["cycle"]) == [2016, 2020, 2020]
    assert list(out["bloc"]) == ["white", "latino", "white"]


# ── Duplicate detection ───────────────────────────────────────────────────────


def test_duplicate_cycle_bloc_source_raises():
    df = _df(cycle=[2020, 2020], bloc=["white", "white"], source=["GSS", "GSS"])
    with pytest.raises(ValueError, match="duplicate"):
        clean_raw_panel(df)


def test_duplicate_cycle_bloc_without_source_raises():
    df = _df(cycle=[2020, 2020], bloc=["white", "white"])
    with pytest.raises(ValueError, match="duplicate"):
        clean_raw_panel(df)


def test_same_cycle_bloc_different_source_ok():
    df = _df(
        cycle=[2020, 2020],
        bloc=["white", "white"],
        source=["GSS", "ANES"],
        vote_share=[0.41, 0.43],
    )
    out = clean_raw_panel(df)
    assert len(out) == 2


# ── Input not mutated ─────────────────────────────────────────────────────────


def test_input_dataframe_not_modified():
    df = _df(cycle=["2020"], bloc=["WHITE"], vote_share=["0.41"])
    original_dtype = df["cycle"].dtype
    original_bloc = df["bloc"].iloc[0]
    original_vote_share = df["vote_share"].iloc[0]
    clean_raw_panel(df)
    assert df["cycle"].dtype == original_dtype
    assert df["bloc"].iloc[0] == original_bloc
    assert df["vote_share"].iloc[0] == original_vote_share
