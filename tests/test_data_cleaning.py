"""Tests for electoral/data/cleaning.py — clean_raw_panel and normalize_bloc.

Four integration scenarios (i)–(iv) sit at the top; unit tests follow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from electoral.data.cleaning import (
    CANONICAL_BLOCS,
    clean_raw_panel,
    impute_missing_cells,
    normalize_bloc,
)
from electoral.data.loaders import load_csv_panel

TOY_PANEL = Path("tests/fixtures/toy_panel.csv")


def _df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


# ── (i) Clean path: toy fixture → correct dtypes and sort ─────────────────────


def test_fixture_clean_path_dtypes_and_sort():
    """Load toy_panel.csv, clean it, assert dtypes and ascending (cycle, bloc) order."""
    raw = load_csv_panel(TOY_PANEL)
    out = clean_raw_panel(raw)

    # Row count: all 20 rows survive — dirty row has non-null cycle and bloc
    assert len(out) == 20

    # Dtype contract
    assert out["cycle"].dtype == pd.Int64Dtype()
    assert out["vote_share"].dtype == pd.Float64Dtype()
    assert out["turnout"].dtype == pd.Float64Dtype()
    assert isinstance(out["bloc"].dtype, pd.StringDtype)

    # Every bloc ID is one of the 15 canonical identifiers
    assert set(out["bloc"].dropna()) <= CANONICAL_BLOCS

    # Output is sorted: cycle ascending, then bloc ascending within each cycle
    assert list(out["cycle"]) == sorted(out["cycle"])
    for _, grp in out.groupby("cycle", sort=False):
        assert list(grp["bloc"]) == sorted(grp["bloc"])

    # First row is the lexicographically smallest (cycle, bloc) pair
    assert out.iloc[0]["cycle"] == 2016
    assert out.iloc[0]["bloc"] == "african_american"

    # Dirty row (row 20 in fixture): 2020/men — cycle and bloc survive, numerics → NA
    dirty = out.loc[(out["cycle"] == 2020) & (out["bloc"] == "men")]
    assert len(dirty) == 1
    assert pd.isna(dirty["vote_share"].iloc[0])
    assert pd.isna(dirty["turnout"].iloc[0])


# ── (ii) Dirty row: dropped and logged ───────────────────────────────────────


def test_dirty_row_dropped_and_logged(caplog):
    """A row whose cycle coerces to NA is dropped; the count is logged at INFO."""
    df = _df(
        cycle=[2020, None],
        bloc=["white", "latino"],
        vote_share=[0.41, 0.55],
        turnout=[0.71, 0.60],
    )
    with caplog.at_level(logging.INFO, logger="electoral.data.cleaning"):
        out = clean_raw_panel(df)

    assert len(out) == 1
    assert out["bloc"].iloc[0] == "white"
    assert "dropped 1" in caplog.text
    assert "cycle" in caplog.text  # log message names the offending column


# ── (iii) Unrecognized bloc: raises informatively ────────────────────────────


def test_unrecognized_bloc_raises_informatively():
    """A bloc label absent from the alias map raises ValueError naming the label
    and listing the canonical blocs."""
    df = _df(cycle=[2020], bloc=["Zoroastrian"])
    with pytest.raises(ValueError) as exc_info:
        clean_raw_panel(df)

    msg = str(exc_info.value)
    # Error names the normalized form of the unrecognized label
    assert "zoroastrian" in msg
    # Error lists canonical blocs so the caller knows valid options
    assert "Canonical blocs" in msg
    assert "evangelical" in msg


# ── (iv) Duplicate tuple: raises, never silently aggregates ──────────────────


def test_duplicate_tuple_raises_not_aggregated():
    """Two rows with identical (cycle, bloc, source) raise ValueError.
    The pipeline must never silently average or sum duplicate entries."""
    df = _df(
        cycle=[2020, 2020],
        bloc=["white", "white"],
        source=["GSS", "GSS"],
        vote_share=[0.41, 0.43],  # different values — aggregation would hide the conflict
    )
    with pytest.raises(ValueError, match="duplicate"):
        clean_raw_panel(df)


def test_duplicate_without_source_also_raises():
    """When source is absent, (cycle, bloc) duplicates are equally rejected."""
    df = _df(cycle=[2020, 2020], bloc=["white", "white"])
    with pytest.raises(ValueError, match="duplicate"):
        clean_raw_panel(df)


def test_same_bloc_different_source_is_not_a_duplicate():
    """Two rows sharing (cycle, bloc) but different source are legitimate cross-survey data."""
    df = _df(
        cycle=[2020, 2020],
        bloc=["white", "white"],
        source=["GSS", "ANES"],
        vote_share=[0.41, 0.43],
    )
    out = clean_raw_panel(df)
    assert len(out) == 2


# ── normalize_bloc unit tests ─────────────────────────────────────────────────


def test_all_canonical_ids_pass_through():
    for bloc in CANONICAL_BLOCS:
        assert normalize_bloc(bloc) == bloc


def test_white_evangelical_maps_to_evangelical():
    # Primary example from DECISIONS.md: religion-only stratum, "White" prefix discarded.
    assert normalize_bloc("White Evangelical") == "evangelical"


def test_nep_born_again_label():
    assert normalize_bloc("White evangelical/born-again Christian") == "evangelical"


def test_case_insensitive():
    assert normalize_bloc("WHITE") == "white"
    assert normalize_bloc("CATHOLIC") == "catholic"
    assert normalize_bloc("MEN") == "men"


def test_strips_whitespace():
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
    # "Other" has no stratum context — caller must use other_race / other_rel / other_gender.
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


def test_error_names_canonical_blocs():
    with pytest.raises(ValueError, match="Canonical blocs"):
        normalize_bloc("not_a_real_bloc")


# ── clean_raw_panel unit tests ────────────────────────────────────────────────


def test_cycle_string_coerced_to_int64():
    df = _df(cycle=["2020", "2016"], bloc=["white", "latino"])
    out = clean_raw_panel(df)
    assert out["cycle"].dtype == pd.Int64Dtype()


def test_vote_share_coerced_to_float64():
    df = _df(cycle=[2020], bloc=["white"], vote_share=["0.41"])
    out = clean_raw_panel(df)
    assert out["vote_share"].dtype == pd.Float64Dtype()
    assert float(out["vote_share"].iloc[0]) == pytest.approx(0.41)


def test_nonnumeric_vote_share_becomes_na():
    df = _df(cycle=[2020], bloc=["white"], vote_share=["high"])
    out = clean_raw_panel(df)
    assert pd.isna(out["vote_share"].iloc[0])


def test_bloc_alias_resolved_in_pipeline():
    # "African American" → "african_american" (snake_case + canonical lookup)
    df = _df(cycle=[2020], bloc=["African American"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "african_american"


def test_bloc_hyphenated_alias_resolved():
    # "born-again" is a valid evangelical alias
    df = _df(cycle=[2020], bloc=["born-again"])
    out = clean_raw_panel(df)
    assert out["bloc"].iloc[0] == "evangelical"


def test_null_bloc_row_dropped(caplog):
    df = _df(cycle=[2020, 2020], bloc=["white", None])
    with caplog.at_level(logging.INFO, logger="electoral.data.cleaning"):
        out = clean_raw_panel(df)
    assert len(out) == 1
    assert "dropped 1" in caplog.text


def test_no_log_when_nothing_dropped(caplog):
    df = _df(cycle=[2020], bloc=["white"])
    with caplog.at_level(logging.INFO, logger="electoral.data.cleaning"):
        clean_raw_panel(df)
    assert "dropped" not in caplog.text


def test_output_sorted_by_cycle_then_bloc():
    df = _df(cycle=[2020, 2016, 2020], bloc=["white", "white", "latino"])
    out = clean_raw_panel(df)
    assert list(out["cycle"]) == [2016, 2020, 2020]
    assert list(out["bloc"]) == ["white", "latino", "white"]


def test_input_not_mutated():
    df = _df(cycle=["2020"], bloc=["WHITE"], vote_share=["0.41"])
    original_cycle_dtype = df["cycle"].dtype
    original_bloc = df["bloc"].iloc[0]
    original_vote_share = df["vote_share"].iloc[0]
    clean_raw_panel(df)
    assert df["cycle"].dtype == original_cycle_dtype
    assert df["bloc"].iloc[0] == original_bloc
    assert df["vote_share"].iloc[0] == original_vote_share


# ── impute_missing_cells tests ────────────────────────────────────────────────


def test_impute_other_gender_constant_fills_missing_cycles():
    """2004/2008/2012/2020/2024 other_gender rows are added at 0.76 when absent."""
    df = _df(cycle=[2016], bloc=["other_gender"], vote_share=[1.0], source=["ANES"])
    out = impute_missing_cells(df)
    og = out[out["bloc"].astype(str) == "other_gender"].sort_values("cycle").reset_index(drop=True)
    # 2016 observed + 5 imputed = 6 rows
    assert set(og["cycle"].astype(int)) == {2004, 2008, 2012, 2016, 2020, 2024}
    imputed = og[og["cycle"].astype(int) != 2016]
    assert list(imputed["vote_share"]) == pytest.approx([0.76] * len(imputed))
    assert list(imputed["source"]) == ["imputed_pew_lgbtq"] * len(imputed)


def test_impute_other_gender_does_not_overwrite_observed_2016():
    """The 2016 ANES observation (1.0) is retained even though 2016 is not in the constant list."""
    df = _df(cycle=[2016], bloc=["other_gender"], vote_share=[1.0], source=["ANES"])
    out = impute_missing_cells(df)
    obs = out[(out["bloc"].astype(str) == "other_gender") & (out["cycle"].astype(int) == 2016)]
    assert len(obs) == 1
    assert float(obs["vote_share"].iloc[0]) == pytest.approx(1.0)


def test_impute_other_gender_skips_cycles_already_present():
    """Cycles that already have other_gender data are not duplicated."""
    all_cycles = [2004, 2008, 2012, 2016, 2020, 2024]
    df = _df(
        cycle=all_cycles,
        bloc=["other_gender"] * 6,
        vote_share=[0.76] * 6,
        source=["test"] * 6,
    )
    out = impute_missing_cells(df)
    assert len(out[out["bloc"].astype(str) == "other_gender"]) == 6


def test_impute_muslim_2004_carry_backward_from_2008():
    """2004 muslim value is the 2008 observed value when 2004 is absent."""
    df = _df(cycle=[2008], bloc=["muslim"], vote_share=[0.93], source=["CES"])
    out = impute_missing_cells(df)
    m04 = out[(out["bloc"].astype(str) == "muslim") & (out["cycle"].astype(int) == 2004)]
    assert len(m04) == 1
    assert float(m04["vote_share"].iloc[0]) == pytest.approx(0.93)
    assert m04["source"].iloc[0] == "imputed_carry_2008"


def test_impute_muslim_2004_skipped_when_2008_absent():
    """No 2004 muslim row is added when the 2008 reference is also missing."""
    df = _df(cycle=[2012], bloc=["muslim"], vote_share=[0.85], source=["CES"])
    out = impute_missing_cells(df)
    m04 = out[(out["bloc"].astype(str) == "muslim") & (out["cycle"].astype(int) == 2004)]
    assert len(m04) == 0


def test_impute_muslim_2004_not_overwritten_when_present():
    """When 2004 muslim is observed, imputation does not add a second row."""
    df = _df(
        cycle=[2004, 2008],
        bloc=["muslim", "muslim"],
        vote_share=[0.75, 0.93],
        source=["NEP", "CES"],
    )
    out = impute_missing_cells(df)
    m04 = out[(out["bloc"].astype(str) == "muslim") & (out["cycle"].astype(int) == 2004)]
    assert len(m04) == 1
    assert float(m04["vote_share"].iloc[0]) == pytest.approx(0.75)


def test_impute_other_race_1948_carry_forward_from_1952():
    """1948 other_race value is the 1952 observed value when 1948 is absent."""
    df = _df(cycle=[1952], bloc=["other_race"], vote_share=[1.0], source=["ANES"])
    out = impute_missing_cells(df)
    r48 = out[(out["bloc"].astype(str) == "other_race") & (out["cycle"].astype(int) == 1948)]
    assert len(r48) == 1
    assert float(r48["vote_share"].iloc[0]) == pytest.approx(1.0)
    assert r48["source"].iloc[0] == "imputed_carry_1952"


def test_impute_other_race_1948_skipped_when_1952_absent():
    """No 1948 other_race row is added when the 1952 reference is missing."""
    df = _df(cycle=[1956], bloc=["other_race"], vote_share=[0.8], source=["ANES"])
    out = impute_missing_cells(df)
    r48 = out[(out["bloc"].astype(str) == "other_race") & (out["cycle"].astype(int) == 1948)]
    assert len(r48) == 0


def test_impute_no_rows_added_when_all_targets_present():
    """Panel is returned unchanged (same length) when every imputation target exists."""
    rows = [
        *[
            {"cycle": c, "bloc": "other_gender", "vote_share": 0.76, "source": "test"}
            for c in [2004, 2008, 2012, 2016, 2020, 2024]
        ],
        {"cycle": 2004, "bloc": "muslim", "vote_share": 0.93, "source": "test"},
        {"cycle": 1948, "bloc": "other_race", "vote_share": 1.0, "source": "test"},
        {"cycle": 1952, "bloc": "other_race", "vote_share": 1.0, "source": "test"},
        {"cycle": 2008, "bloc": "muslim", "vote_share": 0.93, "source": "test"},
    ]
    df = pd.DataFrame(rows)
    out = impute_missing_cells(df)
    assert len(out) == len(df)


def test_impute_output_sorted_by_cycle_then_bloc():
    """Output rows are in ascending (cycle, bloc) order regardless of input order."""
    df = _df(
        cycle=[2012, 2008, 2020],
        bloc=["muslim", "african_american", "muslim"],
        vote_share=[0.85, 0.93, 0.83],
        source=["CES", "NEP", "CES"],
    )
    out = impute_missing_cells(df)
    pairs = list(zip(out["cycle"].astype(int), out["bloc"].astype(str)))
    assert pairs == sorted(pairs)


def test_impute_input_not_mutated():
    """The input DataFrame is never modified in-place."""
    df = _df(cycle=[2008], bloc=["muslim"], vote_share=[0.93], source=["CES"])
    original_len = len(df)
    original_cols = list(df.columns)
    impute_missing_cells(df)
    assert len(df) == original_len
    assert list(df.columns) == original_cols
