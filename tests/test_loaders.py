"""Tests for electoral/data/loaders.py — load_csv_panel and source-specific loaders."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from electoral.data.loaders import (
    load_arda,
    load_ces,
    load_ces_2024,
    load_csv_panel,
    load_gallup,
    load_gss,
    load_nep,
    load_pew,
)

TOY_PANEL = Path("tests/fixtures/toy_panel.csv")


# ── dtype correctness ─────────────────────────────────────────────────────────


def test_cycle_dtype_is_int64():
    df = load_csv_panel(TOY_PANEL)
    assert df["cycle"].dtype == pd.Int64Dtype()


def test_bloc_dtype_is_string():
    df = load_csv_panel(TOY_PANEL)
    assert isinstance(df["bloc"].dtype, pd.StringDtype)


def test_vote_share_dtype_is_float64():
    df = load_csv_panel(TOY_PANEL)
    assert df["vote_share"].dtype == pd.Float64Dtype()


def test_turnout_dtype_is_float64():
    df = load_csv_panel(TOY_PANEL)
    assert df["turnout"].dtype == pd.Float64Dtype()


# ── NaN coercion ──────────────────────────────────────────────────────────────


def test_nonnumeric_vote_share_becomes_na(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text(
        textwrap.dedent(
            """\
        cycle,bloc,vote_share,turnout
        2020,white,high,0.65
    """
        )
    )
    df = load_csv_panel(csv)
    assert pd.isna(df.loc[0, "vote_share"])


def test_nonnumeric_turnout_becomes_na(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text(
        textwrap.dedent(
            """\
        cycle,bloc,vote_share,turnout
        2020,white,0.40,unknown
    """
        )
    )
    df = load_csv_panel(csv)
    assert pd.isna(df.loc[0, "turnout"])


def test_empty_numeric_cell_becomes_na(tmp_path):
    csv = tmp_path / "missing.csv"
    csv.write_text(
        textwrap.dedent(
            """\
        cycle,bloc,vote_share,turnout
        2020,white,,0.65
    """
        )
    )
    df = load_csv_panel(csv)
    assert pd.isna(df.loc[0, "vote_share"])


def test_toy_fixture_last_row_both_na():
    # toy_panel.csv last row: vote_share missing, turnout="high"
    df = load_csv_panel(TOY_PANEL)
    last = df.iloc[-1]
    assert pd.isna(last["vote_share"])
    assert pd.isna(last["turnout"])


# ── full-file reads ───────────────────────────────────────────────────────────


def test_returns_dataframe():
    df = load_csv_panel(TOY_PANEL)
    assert isinstance(df, pd.DataFrame)


def test_row_count_matches_fixture():
    df = load_csv_panel(TOY_PANEL)
    assert len(df) == 20


def test_extra_columns_pass_through_as_object(tmp_path):
    csv = tmp_path / "extra.csv"
    csv.write_text(
        textwrap.dedent(
            """\
        cycle,bloc,vote_share,turnout,source,notes
        2020,white,0.40,0.65,NEP,some note
    """
        )
    )
    df = load_csv_panel(csv)
    assert "source" in df.columns
    assert "notes" in df.columns
    assert df["source"].dtype == object
    assert df["notes"].dtype == object


def test_no_date_inference_for_cycle():
    df = load_csv_panel(TOY_PANEL)
    # cycle must be Int64, not datetime
    assert not hasattr(df["cycle"].dtype, "tz")
    assert df["cycle"].dtype == pd.Int64Dtype()


# ── error handling ────────────────────────────────────────────────────────────


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_csv_panel("nonexistent/path/panel.csv")


# ── source-specific loaders ───────────────────────────────────────────────────
# All use minimal temp CSVs with the raw source column names so tests are
# self-contained and don't depend on the actual survey files.


def test_load_gss_renames_raw_columns(tmp_path):
    csv = tmp_path / "gss.csv"
    csv.write_text("year,racecen1,sex,relig,wtssnrps\n2020,2,2,2,1.2\n")
    df = load_gss(csv)
    assert "cycle" in df.columns
    assert "bloc__race" in df.columns
    assert "bloc__gender" in df.columns
    assert "bloc__religion" in df.columns
    assert "weight" in df.columns
    assert "year" not in df.columns


def test_load_gss_many_to_one_vote_cols_preserved(tmp_path):
    # pres16 and pres20 both map to vote_indicator — _apply_map must leave both.
    csv = tmp_path / "gss.csv"
    csv.write_text("year,racecen1,sex,relig,pres16,pres20,vote16,vote20\n2020,1,1,1,1,1,1,1\n")
    df = load_gss(csv)
    assert "pres16" in df.columns
    assert "pres20" in df.columns
    assert "vote16" in df.columns
    assert "vote20" in df.columns
    assert list(df.columns).count("vote_indicator") == 0


def test_load_arda_renames_vcf_codes(tmp_path):
    csv = tmp_path / "anes.csv"
    csv.write_text("VCF0004,VCF0105a,VCF0104,VCF0704,VCF0010x\n2020,1,1,1,0.8\n")
    df = load_arda(csv)
    assert "cycle" in df.columns
    assert "bloc__race" in df.columns
    assert "bloc__gender" in df.columns
    assert "vote_indicator" in df.columns
    assert "weight" in df.columns
    assert "VCF0004" not in df.columns


def test_load_arda_labeled_supplement(tmp_path):
    # labeled parquet has descriptive names instead of VCF codes
    csv = tmp_path / "anes_labeled.csv"
    csv.write_text(
        "year,race_7cat,gender,religion,voted,pres_vote_dem,weight\n2020,1,1,1,1,1,0.9\n"
    )
    df = load_arda(csv)
    assert "cycle" in df.columns
    assert "bloc__race" in df.columns
    assert "bloc__gender" in df.columns
    assert "bloc__religion" in df.columns
    assert "turnout_indicator" in df.columns
    assert "vote_indicator" in df.columns


def test_load_gallup_passes_through_unchanged(tmp_path):
    # {wave} template keys cannot be resolved statically — df returned as-is
    csv = tmp_path / "voter_panel.csv"
    csv.write_text("caseid,race_2016,presvote_2016,weight_genpop_2016\n1,1,1,1.0\n")
    df = load_gallup(csv)
    assert "race_2016" in df.columns
    assert "presvote_2016" in df.columns
    assert "bloc__race" not in df.columns


def test_load_nep_derives_cycle_from_filename(tmp_path):
    csv = tmp_path / "nep_2020_exit_poll.csv"
    csv.write_text(
        "category,sub_category,sub_pct,dem_candidate,rep_candidate,dem_pct,rep_pct,n_total\n"
        "Race,White,67,Biden,Trump,41,57,3000\n"
    )
    df = load_nep(csv)
    assert df["cycle"].iloc[0] == 2020
    assert df["source"].iloc[0] == "NEP"
    assert "stratum" in df.columns
    assert "bloc" in df.columns
    assert "vote_share" in df.columns


def test_load_nep_bad_filename_raises(tmp_path):
    csv = tmp_path / "exit_poll.csv"
    csv.write_text("category,sub_category\nRace,White\n")
    with pytest.raises(ValueError, match="nep_"):
        load_nep(csv)


def test_load_pew_renames_uppercase_columns(tmp_path):
    csv = tmp_path / "npors.csv"
    csv.write_text("RACETHN,RELIG,GENDER,PARTY,WEIGHT\n1,1,1,1,1.0\n")
    df = load_pew(csv)
    assert df["cycle"].iloc[0] == 2024
    assert "bloc__race" in df.columns
    assert "bloc__religion" in df.columns
    assert "bloc__gender" in df.columns
    assert "weight" in df.columns
    assert "RACETHN" not in df.columns


def test_load_pew_labeled_supplement(tmp_path):
    # labeled parquet uses lowercase descriptive names
    csv = tmp_path / "npors_labeled.csv"
    csv.write_text("race_ethnicity,religion,gender,party,party_lean,weight\n1,1,1,1,1,1.0\n")
    df = load_pew(csv)
    assert df["cycle"].iloc[0] == 2024
    assert "bloc__race" in df.columns
    assert "bloc__religion" in df.columns
    assert "vote_proxy" in df.columns
    assert "vote_proxy_lean" in df.columns


def test_load_ces_renames_raw_columns(tmp_path):
    csv = tmp_path / "ces.csv"
    csv.write_text(
        "year,race_h,hispanic,religion,relig_bornagain,relig_church,"
        "gender,gender4,voted_pres_party,vv_turnout_gvm,weight_cumulative\n"
        "2020,1,0,1,0,3,2,2,Democratic,Voted,1.1\n"
    )
    df = load_ces(csv)
    assert "cycle" in df.columns
    assert "bloc__race" in df.columns
    assert "bloc__religion" in df.columns
    assert "bloc__gender" in df.columns
    assert "vote_indicator" in df.columns
    assert "turnout_indicator" in df.columns
    assert "year" not in df.columns


def test_load_ces_weight_collision_not_duplicated(tmp_path):
    # labeled parquet has both weight and weight_cumulative — no duplicate.
    csv = tmp_path / "ces_labeled.csv"
    csv.write_text(
        "year,race_h,gender,voted_pres_party,vv_turnout_gvm,weight,weight_cumulative\n"
        "2020,1,2,Democratic,Voted,0.9,1.1\n"
    )
    df = load_ces(csv)
    assert list(df.columns).count("weight") == 1
    assert "weight_cumulative" in df.columns


def test_load_ces_2024_renames_raw_columns(tmp_path):
    csv = tmp_path / "ces24.csv"
    csv.write_text(
        "race,hispanic,gender4,religpew,pew_bornagain,CC24_410,TS_g2024,commonweight\n"
        "1,0,2,1,0,1,Voted,1.0\n"
    )
    df = load_ces_2024(csv)
    assert df["cycle"].iloc[0] == 2024
    assert "bloc__race" in df.columns
    assert "bloc__religion" in df.columns
    assert "bloc__gender" in df.columns
    assert "vote_indicator" in df.columns
    assert "turnout_indicator" in df.columns
    assert "weight" in df.columns


def test_load_ces_2024_labeled_supplement(tmp_path):
    # labeled parquet uses descriptive names instead of raw CCES column names
    csv = tmp_path / "ces24_labeled.csv"
    csv.write_text(
        "race,hispanic,gender4,religion,born_again,pres_vote_2024,turnout_validated,weight\n"
        "1,0,2,1,0,1,Voted,1.0\n"
    )
    df = load_ces_2024(csv)
    assert df["cycle"].iloc[0] == 2024
    assert "bloc__religion" in df.columns
    assert "bloc__religion_evangelical_flag" in df.columns
    assert "vote_indicator" in df.columns
    assert "turnout_indicator" in df.columns
