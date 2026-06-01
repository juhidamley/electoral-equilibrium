"""Panel loaders — generic CSV loader and source-specific survey wrappers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_STR_NA: list[str] = ["", "NA", "N/A", "NaN", "null"]

# Read numeric columns as object so non-numeric strings survive to NaN coercion.
_READ_DTYPE: dict[str, str] = {
    "cycle": "object",
    "bloc": "string",
    "vote_share": "object",
    "turnout": "object",
}

_COLUMN_MAPS_PATH = Path(__file__).parents[2] / "configs" / "column_maps.json"
_NEP_YEAR_RE = re.compile(r"nep_(\d{4})_")


def _column_maps() -> dict[str, Any]:
    with _COLUMN_MAPS_PATH.open() as fh:
        return json.load(fh)


def _build_rename(source_key: str) -> dict[str, str]:
    """Return {raw_col: canonical_col} for source_key, skipping _meta/_notes entries."""
    return {k: v for k, v in _column_maps()[source_key].items() if not k.startswith("_")}


def _read_any(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Read a .parquet or .csv file; format determined by extension.

    ``encoding`` is passed to read_csv only.  Use ``"latin-1"`` for files
    known to contain Windows-1252 bytes (e.g. VOTER Panel voter_panel.csv).
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(
        path,
        parse_dates=False,
        na_values=_STR_NA,
        keep_default_na=False,
        encoding=encoding,
    )


def _apply_map(df: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    """Rename columns, skipping three classes of problematic mappings:

    1. Source column absent from df — column simply not in this file.
    2. Destination already exists in df — would create a duplicate column
       (e.g. weight_cumulative → weight when weight already present).
    3. Many-to-one mapping — two or more source columns share the same
       destination (e.g. pres16 + pres20 → vote_indicator in GSS).  Renaming
       any of them would silently discard the others; leave all originals so
       the cleaning step can select the cycle-appropriate column explicitly.
    """
    existing = set(df.columns)
    # Count how many present source columns map to each destination.
    dest_count: dict[str, int] = {}
    for src, dst in rename.items():
        if src in existing:
            dest_count[dst] = dest_count.get(dst, 0) + 1

    applicable = {
        src: dst
        for src, dst in rename.items()
        if src in existing  # rule 1
        and dst not in existing  # rule 2
        and dest_count[dst] == 1  # rule 3
    }
    return df.rename(columns=applicable)


# ── Source-specific loaders ───────────────────────────────────────────────────
#
# Devplan name → actual source:
#   load_arda      → ANES CDF        (DECISIONS.md: "ARDA" was mis-labelled)
#   load_gss       → GSS
#   load_gallup    → VOTER_PANEL     (DECISIONS.md: "Gallup" was mis-labelled)
#   load_nep       → NEP exit polls  (data/surveys/cnn_ssrs_polls/)
#   load_pew       → NPORS           (Pew Research's National Public Opinion Reference Survey)
#   load_ces       → CES cumulative  (data/surveys/CES_2006_2024/)
#   load_ces_2024  → CES 2024        (data/surveys/CES_2024/)
#
# Labeled-subset supplements
# --------------------------
# column_maps.json maps RAW source column names (VCF codes, UPPERCASE NPORS vars,
# etc.).  The labeled .parquet files shipped with each survey already have
# descriptive column names that differ from the raw names.  These dicts cover
# the labeled-parquet naming convention so each loader works on both the raw
# file AND the labeled parquet without requiring a separate code path.

# ANES labeled parquet (anes_labeled_subset.parquet) uses descriptive names;
# VCF codes absent.
_ANES_LABELED: dict[str, str] = {
    "year": "cycle",
    "race_7cat": "bloc__race",
    "hispanic_origin": "bloc__race_hisp_flag",
    "religion": "bloc__religion",
    "religion_denomination": "bloc__religion_denomination",
    "church_attendance": "bloc__religion_attendance",
    "gender": "bloc__gender",
    "pres_vote_dem": "vote_indicator",
    "voted": "turnout_indicator",
}

# NPORS labeled parquet (npors_2024_labeled.parquet) uses lowercase descriptive
# names; UPPERCASE raw variable names absent.
_NPORS_LABELED: dict[str, str] = {
    "race_ethnicity": "bloc__race",
    "hispanic": "bloc__race_hisp_flag",
    "religion": "bloc__religion",
    "born_again": "bloc__religion_evangelical_flag",
    "religion_importance": "bloc__religion_importance",
    "prayer_freq": "bloc__religion_prayer",
    "gender": "bloc__gender",
    "party": "vote_proxy",
    "party_lean": "vote_proxy_lean",
}

# CES 2024 labeled parquet (ces_2024_labeled.parquet) was created with
# different column names than the raw CCES24 file the column_maps.json maps.
_CES_2024_LABELED: dict[str, str] = {
    "religion": "bloc__religion",
    "born_again": "bloc__religion_evangelical_flag",
    "church_attend": "bloc__religion_attendance",
    "pres_vote_2024": "vote_indicator",
    "turnout_validated": "turnout_indicator",
}


def load_arda(path: str | Path) -> pd.DataFrame:
    """Load an ANES CDF file (devplan name: ARDA).

    Applies the ANES column map (VCF codes → canonical names) for the raw
    anes_timeseries_cdf_csv_20260205.csv, and the _ANES_LABELED supplement
    for anes_labeled_subset.parquet (which uses descriptive column names
    instead of VCF codes). Accepts .parquet (primary) or .csv.
    Returns raw, uncleaned DataFrame — bloc codes are still numeric.
    """
    path = Path(path)
    rename = {**_build_rename("ANES"), **_ANES_LABELED}
    return _apply_map(_read_any(path), rename)


def load_gss(path: str | Path) -> pd.DataFrame:
    """Load a GSS labeled subset.

    Renames GSS variable names to canonical column names per column_maps.json.
    Accepts .parquet (primary) or .csv.

    Note: only wtssnrps is renamed to ``weight`` here.  The wtssnrps → wtssps
    → wtss composite-weight fallback chain (DECISIONS.md) is a cleaning step
    handled by electoral/data/cleaning.py.
    Returns raw, uncleaned DataFrame.
    """
    path = Path(path)
    return _apply_map(_read_any(path), _build_rename("GSS"))


def load_gallup(path: str | Path) -> pd.DataFrame:
    """Load a Democracy Fund VOTER Panel file (devplan name: Gallup).

    The VOTER_PANEL column map uses ``{wave}`` placeholder templates
    (e.g. ``race_{wave}``).  Actual column names in the file vary by wave
    (e.g. ``race_2016``, ``race_2020Nov``), so no rename can be applied
    statically — the DataFrame is returned as-is.  Wave-specific column
    parsing is handled by electoral/data/cleaning.py.

    Only .csv is produced for this source (DECISIONS.md: raw voter_panel.csv
    is the only format; no labeled subset built).  voter_panel.csv uses
    Windows-1252 encoding.
    Returns raw, uncleaned DataFrame.
    """
    path = Path(path)
    return _apply_map(_read_any(path, encoding="latin-1"), _build_rename("VOTER_PANEL"))


def load_nep(path: str | Path) -> pd.DataFrame:
    """Load a NEP exit-poll CSV (filename: nep_{year}_exit_poll.csv).

    Renames NEP columns to canonical names per column_maps.json, derives
    ``cycle`` from the filename (no cycle column exists in the CSV), and
    adds a literal ``source`` column with value ``"NEP"``.

    Percentage columns (sub_pct, dem_pct, rep_pct, other_pct) are left as
    raw integers; division by 100 is a cleaning step.
    Returns raw, uncleaned DataFrame.
    """
    path = Path(path)
    match = _NEP_YEAR_RE.search(path.name)
    if not match:
        raise ValueError(
            f"load_nep: cannot derive cycle year from filename {path.name!r}. "
            "Expected pattern: nep_{{year}}_exit_poll.csv"
        )
    cycle = int(match.group(1))
    df = _apply_map(_read_any(path), _build_rename("NEP"))
    df["cycle"] = cycle
    df["source"] = "NEP"
    return df


def load_pew(path: str | Path) -> pd.DataFrame:
    """Load an NPORS file (devplan name: Pew).

    Applies the NPORS column map (UPPERCASE variable names) for the raw
    NPORS_2024_for_public_release.sav, and the _NPORS_LABELED supplement
    for npors_2024_labeled.parquet (which uses lowercase descriptive names).
    Hardcodes ``cycle=2024`` (single cross-section, no cycle variable).
    Accepts .parquet (primary) or .csv.
    Returns raw, uncleaned DataFrame — bloc codes are still string labels.
    """
    path = Path(path)
    rename = {**_build_rename("NPORS"), **_NPORS_LABELED}
    df = _apply_map(_read_any(path), rename)
    df["cycle"] = 2024
    return df


# ── Generic panel loader ──────────────────────────────────────────────────────


def load_csv_panel(path: str | Path) -> pd.DataFrame:
    """Read a panel CSV and return a DataFrame with explicit column dtypes.

    Columns:
      cycle      — Int64 (nullable) election year
      bloc       — string demographic bloc identifier (snake_case)
      vote_share — Float64 (nullable) in [0, 1]
      turnout    — Float64 (nullable) in [0, 1]
    Any additional columns are passed through with pandas' default dtype inference.
    Non-numeric values in numeric columns become NaN; validation is the
    caller's responsibility via electoral/data/cleaning.py.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        dtype=_READ_DTYPE,
        parse_dates=False,
        na_values=_STR_NA,
        keep_default_na=False,
    )
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce").astype("Int64")
    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce").astype("Float64")
    df["turnout"] = pd.to_numeric(df["turnout"], errors="coerce").astype("Float64")
    return df


def load_ces(path: str | Path) -> pd.DataFrame:
    """Load a CES cumulative labeled subset (2006–2024).

    Renames CES column names to canonical names per column_maps.json.
    Accepts .parquet (primary) or .csv.

    Note: use ``race_h`` (any-part Hispanic) not ``race`` — the cumulative
    file maps race_h → bloc__race.  See DECISIONS.md for rationale.
    Returns raw, uncleaned DataFrame — bloc codes are still numeric.
    """
    path = Path(path)
    return _apply_map(_read_any(path), _build_rename("CES"))


def load_ces_2024(path: str | Path) -> pd.DataFrame:
    """Load the CES 2024 single-year file.

    Applies the CES_2024 column map (raw CCES24 names: CC24_410, TS_g2024,
    etc.) for the raw DTA, and the _CES_2024_LABELED supplement for
    ces_2024_labeled.parquet (which uses descriptive names: pres_vote_2024,
    turnout_validated, etc.).  Hardcodes ``cycle=2024``.
    Accepts .parquet (primary) or .csv.

    Note: vote_indicator maps to CC24_410 / pres_vote_2024.  Verify coding
    against CES_2024_GUIDE_vv.pdf before using in production (DECISIONS.md).
    Returns raw, uncleaned DataFrame — bloc codes are still numeric/string.
    """
    path = Path(path)
    rename = {**_build_rename("CES_2024"), **_CES_2024_LABELED}
    df = _apply_map(_read_any(path), rename)
    df["cycle"] = 2024
    return df
