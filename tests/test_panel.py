"""Five unit tests for validate_panel — one per structural invariant."""

from __future__ import annotations

import pandas as pd
import pytest

from electoral.data.panel import validate_panel

CONTEXT = "VoterPanelData"

# Minimal valid row used as a baseline for each test.
_VALID_ROW = {
    "cycle": pd.array([2020], dtype="Int64"),
    "bloc": pd.array(["white"], dtype="string"),
    "vote_share": pd.array([0.41], dtype="Float64"),
    "turnout": pd.array([0.71], dtype="Float64"),
}


def _df(**overrides) -> pd.DataFrame:
    data = {**_VALID_ROW, **overrides}
    return pd.DataFrame(data)


# ── Invariant 1: required columns present ────────────────────────────────────


def test_missing_required_column_raises():
    df = _df().drop(columns=["vote_share"])
    with pytest.raises(ValueError, match=r"VoterPanelData\.columns.*vote_share"):
        validate_panel(
            df, required_cols=["cycle", "bloc", "vote_share", "turnout"], context=CONTEXT
        )


# ── Invariant 2: no nulls in required columns ─────────────────────────────


def test_null_in_required_column_raises():
    df = _df(vote_share=pd.array([None], dtype="Float64"))
    with pytest.raises(ValueError, match=r"VoterPanelData\.vote_share.*null"):
        validate_panel(
            df, required_cols=["cycle", "bloc", "vote_share", "turnout"], context=CONTEXT
        )


# ── Invariant 3: cycle YYYY in [1900, 2100] ──────────────────────────────


def test_cycle_out_of_range_raises():
    df = _df(cycle=pd.array([1776], dtype="Int64"))
    with pytest.raises(ValueError, match=r"VoterPanelData\.cycle.*1776"):
        validate_panel(df, required_cols=["cycle", "bloc"], context=CONTEXT)


# ── Invariant 4: bloc IDs are lowercase snake_case ───────────────────────


def test_non_snake_case_bloc_raises():
    df = _df(bloc=pd.array(["AfricanAmerican"], dtype="string"))
    with pytest.raises(ValueError, match=r"VoterPanelData\.bloc.*AfricanAmerican"):
        validate_panel(df, required_cols=["cycle", "bloc"], context=CONTEXT)


# ── Invariant 5: share columns in [0, 1] ─────────────────────────────────


def test_vote_share_out_of_range_raises():
    df = _df(vote_share=pd.array([1.5], dtype="Float64"))
    with pytest.raises(ValueError, match=r"VoterPanelData\.vote_share.*\[0, 1\]"):
        validate_panel(df, required_cols=["cycle", "bloc", "vote_share"], context=CONTEXT)
