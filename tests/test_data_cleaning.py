"""Tests for electoral/data/cleaning.py — clean_raw_panel and normalize_bloc."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from electoral.data.cleaning import CANONICAL_BLOCS, clean_raw_panel, normalize_bloc


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
    # "born-again" is a valid alias → "evangelical"
    df = _df(cycle=[2020], bloc=["born-again"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "evangelical"


def test_bloc_slashes_replaced_with_underscore():
    # "Latino/Hispanic" normalises to "latino_hispanic" then maps to canonical "latino"
    df = _df(cycle=[2020], bloc=["Latino/Hispanic"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "latino"


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


# ── normalize_bloc ────────────────────────────────────────────────────────────


def test_all_canonical_ids_pass_through():
    for bloc in CANONICAL_BLOCS:
        assert normalize_bloc(bloc) == bloc


def test_white_evangelical_to_evangelical():
    # Key example from DECISIONS.md / task spec.
    assert normalize_bloc("White Evangelical") == "evangelical"


def test_nep_religion_label_with_born_again():
    assert normalize_bloc("White evangelical/born-again Christian") == "evangelical"


def test_case_insensitive():
    assert normalize_bloc("WHITE") == "white"
    assert normalize_bloc("CATHOLIC") == "catholic"
    assert normalize_bloc("MEN") == "men"


def test_strips_leading_trailing_whitespace():
    assert normalize_bloc("  women  ") == "women"


def test_race_aliases():
    assert normalize_bloc("Black") == "african_american"
    assert normalize_bloc("Hispanic") == "latino"
    assert normalize_bloc("Hispanic/Latino") == "latino"
    assert normalize_bloc("Asian American") == "asian"
    assert normalize_bloc("All Others") == "other_race"
    assert normalize_bloc("Multiracial") == "other_race"


def test_religion_aliases():
    assert normalize_bloc("No religion") == "secular"
    assert normalize_bloc("Roman Catholic") == "catholic"
    assert normalize_bloc("Non-evangelical Protestant") == "protestant"
    assert normalize_bloc("Protestants (non-evangelical)") == "protestant"
    assert normalize_bloc("Unaffiliated") == "secular"
    assert normalize_bloc("Islam") == "muslim"
    assert normalize_bloc("Jew") == "jewish"


def test_gender_aliases():
    assert normalize_bloc("Female") == "women"
    assert normalize_bloc("Male") == "men"
    assert normalize_bloc("Non-binary") == "other_gender"


def test_ambiguous_other_raises():
    # Bare "Other" has no stratum context — must use other_race / other_rel / other_gender.
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_bloc("Other")


def test_none_input_raises():
    with pytest.raises(ValueError, match="non-null str"):
        normalize_bloc(None)  # type: ignore[arg-type]


def test_na_input_raises():
    with pytest.raises(ValueError, match="non-null str"):
        normalize_bloc(pd.NA)  # type: ignore[arg-type]


def test_empty_string_raises():
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_bloc("")


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_bloc("Zoroastrian")


def test_error_message_names_canonical_blocs():
    with pytest.raises(ValueError, match="evangelical"):
        normalize_bloc("not_a_real_bloc")
