"""Voter panel kernel — loads all survey sources, aggregates, cleans, and validates."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from electoral.artifacts import VoterPanelData
from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.data.cleaning import clean_raw_panel, impute_missing_cells, normalize_bloc
from electoral.data.loaders import load_arda, load_ces, load_gss, load_nep
from electoral.data.panel import validate_panel

log = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).parents[2] / "configs"
_LAYER_WEIGHTS_PATH = _CONFIGS_DIR / "layer_weights.json"

# ── Required columns for validate_panel ──────────────────────────────────────
_PANEL_REQUIRED = ["cycle", "bloc"]

# ── Source-specific label → canonical bloc remaps ─────────────────────────────
#
# Each dict maps the raw string value (as it appears in the labeled parquet)
# to the canonical bloc ID.  Applied before the shared clean_raw_panel pass.
# Values not present in these dicts produce NaN and are dropped by _agg_stratum.

_ANES_RACE: dict[str, str] = {
    "White non-Hispanic (1948-2012)": "white",
    "Black non-Hispanic (1948-2012)": "african_american",
    "Hispanic (1966-2012)": "latino",
    "Asian or Pacific Islander, non-Hispanic (1966-2012)": "asian",
    "Other or multiple races, non-Hispanic (1968-2012)": "other_race",
    "American Indian or Alaska Native non-Hispanic (1966-2012)": "other_race",
    "Non-white and non-black (1948-1964)": "other_race",
}
_ANES_RELIGION: dict[str, str] = {
    "Protestant": "protestant",
    "Catholic [Roman Catholic]": "catholic",
    "Jewish": "jewish",
    # "Other and none" merges secular and other_rel; assign other_rel (conservative)
    "Other and none (also includes DK preference)": "other_rel",
}
_ANES_GENDER: dict[str, str] = {
    "Female": "women",
    "Male": "men",
    "Other (2016)": "other_gender",
}

_GSS_RACE: dict[str, str] = {
    "white": "white",
    "black or african american": "african_american",
    "hispanic": "latino",
    "american indian or alaska native": "other_race",
    "some other race": "other_race",
    "asian indian": "asian",
    "chinese": "asian",
    "filipino": "asian",
    "japanese": "asian",
    "korean": "asian",
    "vietnamese": "asian",
    "other asian": "asian",
}
_GSS_RELIGION: dict[str, str] = {
    "protestant": "protestant",
    "catholic": "catholic",
    "none": "secular",
    "jewish": "jewish",
    "other": "other_rel",
    "christian": "protestant",
    "buddhism": "other_rel",
    "muslim/islam": "muslim",
    "orthodox-christian": "other_rel",
    "hinduism": "other_rel",
    "other eastern": "other_rel",
    "inter-nondenominational": "protestant",
}
_GSS_GENDER: dict[str, str] = {"Female": "women", "Male": "men"}

# Retrospective presidential-vote columns present in the GSS labeled subset
# and which candidate name maps to Democratic (1.0) or Republican (0.0).
_GSS_ELECTIONS: dict[int, tuple[str, dict[str, float]]] = {
    2016: ("pres16", {"clinton": 1.0, "trump": 0.0}),
    2020: ("pres20", {"biden": 1.0, "trump": 0.0}),
}

_CES_RACE: dict[str, str] = {
    "White": "white",
    "Hispanic": "latino",
    "Black": "african_american",
    "Asian": "asian",
    "Mixed": "other_race",
    "Other": "other_race",
    "Middle Eastern": "other_race",
    "Native American": "other_race",
    "Native Hawaiian / Pacific Islander": "other_race",
}
_CES_RELIGION: dict[str, str] = {
    "Protestant": "protestant",
    "Roman Catholic": "catholic",
    "Nothing in particular": "secular",
    "Something else": "other_rel",
    "Agnostic": "secular",
    "Atheist": "secular",
    "Jewish": "jewish",
    "Mormon": "other_rel",
    "Buddhist": "other_rel",
    "Muslim": "muslim",
    "Hindu": "other_rel",
    "Orthodox Christian": "other_rel",
    "Evangelical Protestant": "evangelical",
    "Mainline Protestant": "protestant",
    "Born Again": "evangelical",
}
_CES_GENDER: dict[str, str] = {
    "Female": "women",
    "Male": "men",
    "Non-binary": "other_gender",
    "Other": "other_gender",
}


# ── Shared aggregation helper ─────────────────────────────────────────────────


def _agg_stratum(
    df: pd.DataFrame,
    bloc_col: str,
    remap: dict[str, str],
    dem_series: pd.Series,
    cycle_col: str,
    weight_col: str | None,
    source: str,
) -> pd.DataFrame:
    """Aggregate individual survey responses to weighted bloc-level vote_share.

    dem_series must be 1.0 (Democratic vote), 0.0 (non-Democratic vote), or NaN (excluded).
    Rows with NaN dem or unmapped bloc are dropped before aggregation.
    """
    sub = pd.DataFrame(
        {
            "cycle": df[cycle_col].values,
            "bloc": df[bloc_col].map(remap).values,
            "dem": dem_series.values,
            "w": (
                pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).values
                if weight_col and weight_col in df.columns
                else 1.0
            ),
        },
        index=df.index,
    )
    sub = sub.dropna(subset=["bloc", "dem"])
    sub = sub[sub["dem"].isin([0.0, 1.0])]
    if sub.empty:
        return pd.DataFrame(columns=["cycle", "bloc", "vote_share", "se", "source"])

    def _stats(g: pd.DataFrame) -> pd.Series:
        w = g["w"]
        w_sum = w.sum()
        vs = (g["dem"] * w).sum() / w_sum
        # Kish (1965) effective sample size for weighted surveys
        n_eff = w_sum**2 / (w**2).sum()
        se = (vs * (1.0 - vs) / max(n_eff, 1.0)) ** 0.5
        return pd.Series({"vote_share": vs, "se": se})

    result = sub.groupby(["cycle", "bloc"]).apply(_stats, include_groups=False).reset_index()
    result["source"] = source
    return result[["cycle", "bloc", "vote_share", "se", "source"]]


# ── Source-specific extractors ────────────────────────────────────────────────


def _from_nep(paths: list[Path]) -> pd.DataFrame:
    """Load all NEP exit-poll CSVs, filter to Race/Religion/Gender strata,
    scale dem_pct from integer percentage to [0, 1], and keep only rows
    whose bloc maps to a canonical ID.

    Evangelical handling: NEP evangelical rows appear under a category like
    "white evangelical/born-again?" with sub_category="Yes"/"No" (not "evangelical").
    These rows are excluded by the "relig" filter because their category string
    never contains that token.  We detect them via an evangelical-specific pattern
    and remap the "Yes" sub_category to the canonical "evangelical" bloc before the
    standard normalize_bloc pass.  The "No" (non-evangelical) rows are discarded
    because they represent a heterogeneous residual group, not a clean bloc.
    """
    records: list[dict] = []
    for path in paths:
        df = load_nep(path)
        stratum_col = (
            df["stratum"].astype("string") if "stratum" in df.columns else pd.Series(dtype="string")
        )
        s = stratum_col.str.lower()

        is_race = s.str.contains(r"\brace\b", na=False)
        is_religion = s.str.contains("relig", na=False)
        # Match Gender/Sex while excluding cross-tab strata that also contain these tokens.
        is_gender = s.str.contains(r"gender|sex", regex=True, na=False) & ~s.str.contains(
            r"marital|education|white|race|income", regex=True, na=False
        )
        # Evangelical: category contains "evang" or "born" (covers "born-again" /
        # "born again") AND the sub_category row is "Yes" (the actual evangelical
        # respondents).  OCR noise in 2020 garbles "Yes" into a long string ending
        # in "Y e s" — match the spaced-character form as well.
        is_evang_cat = s.str.contains(r"evang|born", regex=True, na=False)
        bloc_lower = df["bloc"].astype(str).str.lower().str.strip()
        # Covers: "Yes", "yes", "Y e s" (OCR-spaced), and similar variants.
        is_evang_yes = is_evang_cat & bloc_lower.str.contains(
            r"\byes\b|y\s+e\s+s", regex=True, na=False
        )
        # Remap the "Yes" bloc to canonical "evangelical" before normalize_bloc runs.
        df = df.copy()
        df.loc[is_evang_yes, "bloc"] = "evangelical"

        df = df[is_race | is_religion | is_gender | is_evang_yes].copy()

        raw_vs = pd.to_numeric(df["vote_share"], errors="coerce")
        df["vote_share"] = raw_vs / 100.0

        for _, row in df.iterrows():
            try:
                canonical = normalize_bloc(str(row["bloc"]))
            except (ValueError, TypeError):
                log.debug(
                    "NEP %s: skipping unrecognized bloc %r",
                    path.name,
                    row.get("bloc"),
                )
                continue
            vs = row["vote_share"]
            if pd.isna(vs):
                continue
            # n_bloc for binomial SE: n_total_respondents × stratum_share%
            n_resp = pd.to_numeric(row.get("n_respondents"), errors="coerce")
            strat = pd.to_numeric(row.get("stratum_share"), errors="coerce")
            n_bloc = (
                float(n_resp * strat / 100.0)
                if pd.notna(n_resp) and pd.notna(strat)
                else float("nan")
            )
            records.append(
                {
                    "cycle": int(row["cycle"]),
                    "bloc": canonical,
                    "vote_share": float(vs),
                    "n_bloc": n_bloc,
                    "source": "NEP",
                }
            )

    if not records:
        return pd.DataFrame(columns=["cycle", "bloc", "vote_share", "se", "source"])

    result = pd.DataFrame(records)
    # NEP PDFs can produce duplicate rows for the same bloc (e.g. "White" and
    # "White voters" in the same table both normalize to "white").  Average them.
    n_raw = len(result)
    result = result.groupby(["cycle", "bloc", "source"], as_index=False).agg(
        vote_share=("vote_share", "mean"),
        n_bloc=("n_bloc", "mean"),
    )
    n_averaged = n_raw - len(result)
    if n_averaged:
        log.info(
            "NEP: averaged %d duplicate (cycle, bloc, source) row(s) — "
            "multiple raw labels normalised to the same canonical bloc",
            n_averaged,
        )
    # Compute binomial SE from pooled vote_share and n_bloc
    vs = result["vote_share"]
    n = result["n_bloc"].clip(lower=1.0)
    result["se"] = (vs * (1.0 - vs) / n) ** 0.5
    return result.drop(columns=["n_bloc"])[["cycle", "bloc", "vote_share", "se", "source"]]


def _from_anes(path: Path) -> pd.DataFrame:
    """ANES CDF labeled subset → bloc-level vote_share via weighted aggregation.

    vote_indicator coding: 1.0=Democrat, 2.0=Republican, 3.0=Other, 0.0=didn't vote.
    Only actual voters (vote_indicator ∈ {1, 2, 3}) enter the denominator.
    """
    df = load_arda(path)
    # Map vote_indicator to Dem=1.0 / Rep=0.0 / NaN (excluded)
    dem_flag = df["vote_indicator"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0})

    frames = [
        _agg_stratum(df, "bloc__race", _ANES_RACE, dem_flag, "cycle", "weight", "ANES"),
        _agg_stratum(df, "bloc__religion", _ANES_RELIGION, dem_flag, "cycle", "weight", "ANES"),
        _agg_stratum(df, "bloc__gender", _ANES_GENDER, dem_flag, "cycle", "weight", "ANES"),
    ]
    non_empty = [f for f in frames if not f.empty]
    return (
        pd.concat(non_empty, ignore_index=True)
        if non_empty
        else pd.DataFrame(columns=["cycle", "bloc", "vote_share", "source"])
    )


def _from_gss(path: Path) -> pd.DataFrame:
    """GSS labeled subset → bloc-level vote_share via retrospective presidential vote columns.

    Each election (2016, 2020) is handled separately because the GSS asks respondents
    which presidential candidate they supported, regardless of the survey wave year.
    """
    df = load_gss(path)
    frames = []

    for election_cycle, (vote_col, dem_map) in _GSS_ELECTIONS.items():
        if vote_col not in df.columns:
            log.debug("GSS: column %r not found, skipping cycle %d", vote_col, election_cycle)
            continue

        # Create a per-election sub-frame; cycle is the election year, not survey year.
        sub = df.copy()
        sub["cycle"] = election_cycle
        # astype(str) before lower() so numeric codes (int/float) don't silently
        # produce all-NaN via pandas StringMethods on non-string dtype.
        dem_flag = sub[vote_col].astype(str).str.lower().map(dem_map)

        for bloc_col, remap in [
            ("bloc__race", _GSS_RACE),
            ("bloc__religion", _GSS_RELIGION),
            ("bloc__gender", _GSS_GENDER),
        ]:
            frames.append(_agg_stratum(sub, bloc_col, remap, dem_flag, "cycle", "weight", "GSS"))

    non_empty = [f for f in frames if not f.empty]
    return (
        pd.concat(non_empty, ignore_index=True)
        if non_empty
        else pd.DataFrame(columns=["cycle", "bloc", "vote_share", "source"])
    )


def _from_ces(path: Path) -> pd.DataFrame:
    """CES cumulative labeled subset → bloc-level vote_share via weighted aggregation.

    Only presidential-election cycles (year divisible by 4) are retained.
    vote_indicator coding: "Democratic"=1.0, "Republican"=0.0, anything else=NaN.
    weight_cumulative is used (appropriate for multi-year analyses).
    """
    df = load_ces(path)
    cycle_num = pd.to_numeric(df["cycle"], errors="coerce")
    df = df[cycle_num.notna() & (cycle_num % 4 == 0)].copy()
    if df.empty:
        return pd.DataFrame(columns=["cycle", "bloc", "vote_share", "source"])

    # Normalise to Title Case before mapping so "democratic", "DEMOCRATIC",
    # "Democratic" all match.  Any other value (Other, Third Party, etc.) → NaN.
    vote = df["vote_indicator"]
    # 1.0 for Democratic vote; 0.0 for any other *reported* vote (Rep/Third Party/etc.).
    # Keep missing/NA as pd.NA so it is excluded from aggregation.
    dem_flag = pd.Series(pd.NA, index=df.index, dtype="Float64")
    mask = vote.notna()
    norm = vote[mask].astype(str).str.strip().str.title()
    dem_flag.loc[mask] = norm.eq("Democratic").astype(float)

    weight_col = "weight_cumulative" if "weight_cumulative" in df.columns else "weight"

    # Split CES Protestant into evangelical vs. mainline using relig_bornagain flag.
    # CES asks "Are you a born-again or evangelical Christian?" (Yes/No).
    # Protestant + born-again=Yes → "Evangelical Protestant" → canonical "evangelical".
    # Protestant + born-again=No/null → stays "Protestant" → canonical "protestant".
    # _CES_RELIGION already maps "Evangelical Protestant" → "evangelical".
    df = df.copy()
    # The labeled CES parquet exposes the born-again flag as
    # "bloc__religion_evangelical_flag" (renamed from raw "relig_bornagain").
    evang_flag_col = next(
        (c for c in df.columns if "evangelical_flag" in c or c == "relig_bornagain"),
        None,
    )
    if evang_flag_col and "bloc__religion" in df.columns:
        born_again_mask = (
            df["bloc__religion"].astype(str).str.strip() == "Protestant"
        ) & (
            df[evang_flag_col].astype(str).str.strip().str.title() == "Yes"
        )
        df.loc[born_again_mask, "bloc__religion"] = "Evangelical Protestant"
        log.info(
            "CES: split %d Protestant+born-again respondents → 'Evangelical Protestant'",
            int(born_again_mask.sum()),
        )

    frames = [
        _agg_stratum(df, "bloc__race", _CES_RACE, dem_flag, "cycle", weight_col, "CES"),
        _agg_stratum(df, "bloc__religion", _CES_RELIGION, dem_flag, "cycle", weight_col, "CES"),
        _agg_stratum(df, "bloc__gender", _CES_GENDER, dem_flag, "cycle", weight_col, "CES"),
    ]
    non_empty = [f for f in frames if not f.empty]
    return (
        pd.concat(non_empty, ignore_index=True)
        if non_empty
        else pd.DataFrame(columns=["cycle", "bloc", "vote_share", "source"])
    )


# ── Layer weights loader ──────────────────────────────────────────────────────


def _load_layer_weights() -> dict[str, float]:
    with _LAYER_WEIGHTS_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: float(raw[k]) for k in ("lambda_1", "lambda_2", "lambda_3")}


# ── Cross-source conflict resolution ─────────────────────────────────────────


def resolve_conflicts(panel: pd.DataFrame) -> pd.DataFrame:
    """Resolve cross-source vote_share conflicts via inverse-SE weighting.

    For (cycle, bloc) pairs present in exactly one source the row is returned
    unchanged.  For pairs in multiple sources, compute the inverse-SE weighted
    average:

        vote_share_merged = Σ(w_i · vs_i) / Σ(w_i)
        where w_i = 1 / se_i  if se_i > 0
              w_i = 1          otherwise  (equal-weight fallback)

    This is the rule documented in DECISIONS.md §Data Ingest.

    The merged row's ``source`` field is the sorted, "+"-joined names of the
    contributing sources (e.g. ``"ANES+CES+GSS+NEP"``).
    The ``se`` column is consumed internally and dropped from the output.
    """
    if panel.empty:
        return panel.drop(columns=["se"], errors="ignore")

    # Ensure se column exists; rows without SE fall back to equal weights
    if "se" not in panel.columns:
        panel = panel.copy()
        panel["se"] = float("nan")

    key_counts = panel.groupby(["cycle", "bloc"])["vote_share"].transform("count")
    single = panel[key_counts == 1]
    multi = panel[key_counts > 1].copy()

    if multi.empty:
        return panel.drop(columns=["se"], errors="ignore")

    # Log each conflict so operators know which blocs were resolved
    for (cycle, bloc), grp in multi.groupby(["cycle", "bloc"]):
        sources = sorted(grp["source"].dropna().unique())
        vs_by_src = {
            src: round(float(grp.loc[grp["source"] == src, "vote_share"].iloc[0]), 4)
            for src in sources
        }
        log.info(
            "resolve_conflicts: (cycle=%d, bloc=%r) — %d sources %s vote_shares=%s; "
            "applying inverse-SE weights",
            cycle,
            bloc,
            len(sources),
            sources,
            vs_by_src,
        )

    multi["w"] = multi["se"].apply(lambda se: 1.0 / se if pd.notna(se) and se > 0 else 1.0)

    def _merge_group(g: pd.DataFrame) -> pd.Series:
        w_sum = g["w"].sum()
        return pd.Series(
            {
                "vote_share": (g["vote_share"] * g["w"]).sum() / w_sum,
                "source": "+".join(sorted(g["source"].dropna().unique())),
            }
        )

    merged = (
        multi.groupby(["cycle", "bloc"]).apply(_merge_group, include_groups=False).reset_index()
    )

    cols = ["cycle", "bloc", "vote_share", "source"]
    result = pd.concat(
        [single[cols].copy(), merged[cols]],
        ignore_index=True,
    )
    return result.sort_values(["cycle", "bloc"]).reset_index(drop=True)


# ── Coverage diagnostics ─────────────────────────────────────────────────────


def _log_coverage(
    panel: pd.DataFrame,
    config: PipelineConfig,
    raw_frames: list[pd.DataFrame],
) -> None:
    """Log per-source-per-cycle row counts and flag missing (cycle, bloc) pairs.

    Logs two views:
    - *Raw* counts from each source frame before conflict resolution, so it
      is clear which individual sources contributed to each cycle.
    - *Resolved* counts in the final panel (composite labels like "ANES+CES").
    """
    from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

    all_blocs = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)
    pres_cycles = [c for c in sorted(panel["cycle"].dropna().unique()) if int(c) % 4 == 0]
    log.info(
        "=== Panel coverage: %d rows, %d presidential cycles ===", len(panel), len(pres_cycles)
    )

    # ── Raw source counts (before conflict resolution) ────────────────────────
    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        raw_pres = raw[raw["cycle"].isin(pres_cycles)]
        raw_pivot = raw_pres.groupby(["source", "cycle"]).size().unstack(fill_value=0)
        log.info("  Raw blocs per source per cycle (before conflict resolution):")
        for src in sorted(raw_pivot.index):
            counts = {str(int(c)): int(n) for c, n in raw_pivot.loc[src].items() if n > 0}
            log.info("    source=%-6s %s", src, counts)

    # ── Resolved counts (after conflict resolution) ───────────────────────────
    pres_panel = panel[panel["cycle"].isin(pres_cycles)]
    src_cycle = pres_panel.groupby(["cycle", "source"]).size().unstack(fill_value=0)
    log.info("  Resolved blocs per cycle (composite source labels after conflict resolution):")
    for cycle in sorted(src_cycle.index):
        row = src_cycle.loc[cycle]
        non_zero = {str(src): int(n) for src, n in row.items() if n > 0}
        log.info("    cycle=%d blocs=%d %s", cycle, row.sum(), non_zero)

    # Flag missing (cycle, bloc) pairs in presidential cycles
    missing = []
    for cycle in pres_cycles:
        present = set(pres_panel.loc[pres_panel["cycle"] == cycle, "bloc"].dropna())
        for bloc in all_blocs:
            if bloc not in present:
                missing.append((int(cycle), bloc))

    if missing:
        log.warning(
            "Missing (cycle, bloc) pairs across all sources (%d total): %s",
            len(missing),
            [(c, b) for c, b in sorted(missing)],
        )
    else:
        log.info("Coverage complete: all (cycle, bloc) pairs present in presidential cycles")


# ── Main kernel function ──────────────────────────────────────────────────────


def build_voter_panel(config: PipelineConfig) -> tuple[VoterPanelData, pd.DataFrame]:
    """Load all survey sources, aggregate, clean, validate, and return the voter panel.

    Sources (in priority order):
      1. NEP exit polls — primary source; already aggregated dem_pct per bloc per cycle.
      2. ANES CDF labeled subset — individual-level; aggregated to bloc-level via
         weighted mean of presidential vote_indicator.
      3. GSS labeled subset — individual-level; pres16/pres20 retrospective vote columns.
      4. CES cumulative labeled subset — individual-level; voted_pres_party column.
      5. VOTER Panel (Democracy Fund) — deferred: wave-specific column parsing not yet
         implemented.
      6. NPORS (Pew) — deferred: party proxy only, no direct vote_share.

    Returns
    -------
    tuple[VoterPanelData, pd.DataFrame]
        payload : frozen VoterPanelData artifact ready for downstream stages
        panel   : cleaned, validated panel DataFrame (cycle, bloc, vote_share, source)
                  Note: ``turnout`` column is absent — no current source exposes
                  validated turnout at the per-bloc-per-cycle aggregation level.
                  It will be added in a later sprint once VOTER Panel wave parsing
                  is implemented.
    """
    surveys = Path(config.data_path) / "surveys"
    frames: list[pd.DataFrame] = []

    # ── 1. NEP ────────────────────────────────────────────────────────────────
    nep_dir = surveys / "cnn_ssrs_polls"
    nep_files = sorted(nep_dir.glob("nep_*_exit_poll.csv")) if nep_dir.exists() else []
    if nep_files:
        df_nep = _from_nep(nep_files)
        frames.append(df_nep)
        log.info("NEP: %d rows from %d file(s)", len(df_nep), len(nep_files))
    else:
        log.warning("NEP: no exit-poll CSVs found in %s", nep_dir)

    # ── 2. ANES ───────────────────────────────────────────────────────────────
    anes_dir = surveys / "anes_timeseries_cdf_csv_20260205"
    anes_path = anes_dir / "anes_labeled_subset.parquet"
    if anes_path.exists():
        df_anes = _from_anes(anes_path)
        frames.append(df_anes)
        log.info("ANES: %d rows", len(df_anes))
    else:
        log.warning("ANES: labeled subset not found at %s", anes_path)

    # ── 3. GSS ────────────────────────────────────────────────────────────────
    gss_candidates = (
        list((surveys / "GSS_stata (1)").glob("gss_labeled_subset.parquet"))
        if (surveys / "GSS_stata (1)").exists()
        else []
    )
    if gss_candidates:
        df_gss = _from_gss(gss_candidates[0])
        frames.append(df_gss)
        log.info("GSS: %d rows", len(df_gss))
    else:
        log.warning("GSS: labeled subset not found under %s", surveys)

    # ── 4. CES ────────────────────────────────────────────────────────────────
    ces_path = surveys / "CES_2006_2024" / "ces_cumulative_labeled.parquet"
    if ces_path.exists():
        df_ces = _from_ces(ces_path)
        frames.append(df_ces)
        log.info("CES: %d rows", len(df_ces))
    else:
        log.warning("CES: labeled subset not found at %s", ces_path)

    # ── 5. VOTER Panel — deferred ─────────────────────────────────────────────
    voter_path = surveys / "VOTER Panel Data Files" / "voter_panel.csv"
    if voter_path.exists():
        log.info("VOTER Panel: found at %s — wave-specific column parsing deferred", voter_path)
    else:
        log.warning("VOTER Panel: not found at %s", voter_path)

    # ── 6. NPORS — deferred ───────────────────────────────────────────────────
    npors_dir = surveys / "NPORS-2024-Data-Release"
    if npors_dir.exists():
        log.info("NPORS: found — party proxy only, vote_share extraction deferred")

    if not frames:
        raise RuntimeError(
            "build_voter_panel: no survey data loaded. "
            f"Check config.data_path ({config.data_path!r})."
        )

    # ── Concatenate ───────────────────────────────────────────────────────────
    # Keep a reference to the per-source frames so _log_coverage can report
    # raw counts before conflict resolution merges them into composite labels.
    raw_frames = list(frames)
    panel = pd.concat(frames, ignore_index=True)
    log.info("Concatenated panel: %d rows before conflict resolution", len(panel))

    # Record all original source names before resolution merges them into
    # composite labels like "ANES+CES+GSS+NEP".
    source_names = sorted(panel["source"].dropna().unique())

    # ── Resolve cross-source conflicts ────────────────────────────────────────
    panel = resolve_conflicts(panel)
    log.info("After resolve_conflicts: %d rows", len(panel))

    # ── Clean ─────────────────────────────────────────────────────────────────
    panel = clean_raw_panel(panel)
    log.info("After clean_raw_panel: %d rows", len(panel))

    # ── Impute structurally absent cells ─────────────────────────────────────
    panel = impute_missing_cells(panel)
    log.info("After imputation: %d rows", len(panel))

    # ── Validate ──────────────────────────────────────────────────────────────
    validate_panel(panel, required_cols=_PANEL_REQUIRED, context="VoterPanelData")

    # ── Log per-source-per-cycle counts ──────────────────────────────────────
    _log_coverage(panel, config, raw_frames)

    # ── Derive VoterPanelData fields ──────────────────────────────────────────
    cycles = sorted(int(c) for c in panel["cycle"].dropna().unique())
    n_race = int(panel["bloc"].isin(CANONICAL_RACES).sum())
    n_religion = int(panel["bloc"].isin(CANONICAL_RELIGIONS).sum())
    n_gender = int(panel["bloc"].isin(CANONICAL_GENDERS).sum())
    # Use pre-resolution source names so composite labels ("ANES+CES") don't
    # inflate the list with merged entries.
    sources_used = "+".join(source_names) if source_names else None

    layer_weights = _load_layer_weights()

    payload = VoterPanelData(
        cycles=cycles,
        races=config.races,
        religions=config.religions,
        genders=config.genders,
        n_rows_race=n_race,
        n_rows_religion=n_religion,
        n_rows_gender=n_gender,
        layer_weights=layer_weights,
        source=sources_used,
    )
    payload.validate()

    return payload, panel
