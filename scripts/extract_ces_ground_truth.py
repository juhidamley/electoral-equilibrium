#!/usr/bin/env python3
"""Extract per-bloc year-over-year deltas from the CES cumulative file (2006-2024).

Ground-truth extractor for Phase 1, Step 1.3, Part A. Covers shocks the VOTER
panel (scripts/extract_panel_ground_truth.py) cannot bracket -- pre-2011,
post-Nov-2020, or in its 2012->2016 gap -- using CES_2006_2024's annual
REPEATED CROSS-SECTIONS. Uses the same pid7 -> dem_loyalty transform as the
panel so the two are numerically comparable, but computes two-INDEPENDENT-
-samples statistics (not a paired difference -- different respondents each
year) and tags every cell with a distinct, lower fidelity tier.

Read-only w.r.t. survey files. Writes only:
    data/ground_truth/ces_deltas.json

Usage:
    python scripts/extract_ces_ground_truth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

REPO_ROOT = Path(__file__).parents[1]
CES_PARQUET = Path(
    "/Volumes/JUHIDRIVE/electoralData/surveys/CES_2006_2024/ces_cumulative_labeled.parquet"
)
SHOCKS_JSON = REPO_ROOT / "configs" / "shocks.json"
PANEL_JSON = REPO_ROOT / "data" / "ground_truth" / "panel_deltas.json"
OUT_JSON = REPO_ROOT / "data" / "ground_truth" / "ces_deltas.json"

RACE_BLOCS = list(CANONICAL_RACES)
RELIGION_BLOCS = list(CANONICAL_RELIGIONS)
GENDER_BLOCS = list(CANONICAL_GENDERS)
ALL_BLOCS = RACE_BLOCS + RELIGION_BLOCS + GENDER_BLOCS
BLOC_STRATUM = (
    {b: "race" for b in RACE_BLOCS}
    | {b: "religion" for b in RELIGION_BLOCS}
    | {b: "gender" for b in GENDER_BLOCS}
)

CES_YEARS = [str(y) for y in range(2006, 2025)]

# CES's `race`/`religion`/`gender`/`gender4` are already string-labeled in the
# _labeled.parquet export (same Lucid/YouGov-family vendor coding as the VOTER
# panel -- confirmed by direct inspection, not assumed).
RACE_MAP = {
    "White": "white",
    "Black": "african_american",
    "Hispanic": "latino",
    "Asian": "asian",
    "Native American": "other_race",
    "Mixed": "other_race",
    "Other": "other_race",
    "Middle Eastern": "other_race",
}
RELIGION_DIRECT_MAP = {
    "Roman Catholic": "catholic",
    "Atheist": "secular",
    "Agnostic": "secular",
    "Nothing in particular": "secular",
    "Jewish": "jewish",
    "Muslim": "muslim",
    "Mormon": "other_rel",
    "Orthodox": "other_rel",
    "Buddhist": "other_rel",
    "Hindu": "other_rel",
    "Something else": "other_rel",
}
PID7_MAP = {
    "Strong Democrat": 1,
    "Not Very Strong Democrat": 2,
    "Lean Democrat": 3,
    "Independent": 4,
    "Lean Republican": 5,
    "Not Very Strong Republican": 6,
    "Strong Republican": 7,
}

SUPPRESS_N = 30
LOW_CONFIDENCE_N = 100
CI_Z = 1.96
FIDELITY_TIER = "ces_annual_cross_section"
FIDELITY_RANK = 3  # panel_A=1, panel_B=2, ces_annual_cross_section=3, exit_poll_cross_cycle=4 (parallel, not strictly ordered vs. 3 -- see ground_truth_layers.md)

FIDELITY_NOTE = (
    "CES year-over-year comparisons are REPEATED CROSS-SECTIONS, not a panel: "
    "the 'before' and 'after' respondents in any pair are different people. A "
    "measured shift here confounds (a) individual attitude change with (b) "
    "compositional change in who happened to respond that year, in a way the "
    "VOTER panel's within-person design does not. CES's annual fielding "
    "resolution (~2 months/year, all other months uncovered) is also far "
    "coarser than the panel's tightest 46-day window -- a CES year-pair can "
    "span anywhere from ~1 day to ~11 months of gap between fielding periods, "
    "and typically contains many more configured shocks than any panel "
    "bracket. This tier is always ranked below every panel Tier A/B cell."
)


def load_shocks() -> list[dict]:
    with open(SHOCKS_JSON, encoding="utf-8") as f:
        shocks = json.load(f)
    for s in shocks:
        s["_date"] = pd.Timestamp(s["date"])
    return shocks


def load_ces() -> pd.DataFrame:
    cols = [
        "year", "case_id", "weight", "pid7", "race", "religion",
        "relig_bornagain", "gender", "gender4", "starttime",
    ]
    df = pd.read_parquet(CES_PARQUET, columns=cols)
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce").dt.tz_localize(None)
    df["pid7_num"] = df["pid7"].map(PID7_MAP)
    return df


def compute_year_windows(df: pd.DataFrame) -> dict:
    windows = {}
    for y in CES_YEARS:
        sub = df[df["year"] == int(y)]
        windows[y] = {
            "start": sub["starttime"].min(),
            "end": sub["starttime"].max(),
            "n_total": int(len(sub)),
            "n_pid7_valid": int(sub["pid7_num"].notna().sum()),
        }
    return windows


def assign_shock_to_ces_years(shock_date: pd.Timestamp, windows: dict) -> dict | None:
    """Same straddle-aware logic as the panel's assign_shocks_to_brackets():
    a shock whose date falls inside a CES year's own fielding window is
    excluded from BOTH the before and after role for that year (the
    before/after search below naturally skips it, since its own end is not
    < shock_date and its own start is not > shock_date).
    """
    straddle_year = None
    for y in CES_YEARS:
        s, e = windows[y]["start"], windows[y]["end"]
        if pd.notna(s) and pd.notna(e) and s <= shock_date <= e:
            straddle_year = y
            break

    before_year = None
    for y in CES_YEARS:
        e = windows[y]["end"]
        if pd.notna(e) and e < shock_date:
            before_year = y  # keep overwriting -> ends up as the LATEST qualifying year
    after_year = None
    for y in CES_YEARS:
        s = windows[y]["start"]
        if pd.notna(s) and s > shock_date:
            after_year = y  # first (chronologically earliest) qualifying year
            break

    if before_year is None or after_year is None:
        return None
    return {"before_year": before_year, "after_year": after_year, "straddle_year": straddle_year}


def compute_blocs(df: pd.DataFrame, year: str) -> dict[str, pd.Series]:
    sub = df[df["year"] == int(year)]
    race_bloc = sub["race"].map(RACE_MAP)
    relig = sub["religion"]
    born = sub["relig_bornagain"]
    relig_bloc = pd.Series(np.nan, index=sub.index, dtype=object)
    is_prot = relig == "Protestant"
    relig_bloc.loc[is_prot & (born == "Yes")] = "evangelical"
    relig_bloc.loc[is_prot & (born != "Yes")] = "protestant"
    for k, v in RELIGION_DIRECT_MAP.items():
        relig_bloc.loc[relig == k] = v

    gender4 = sub["gender4"] if "gender4" in sub.columns else None
    gender_bloc = pd.Series(np.nan, index=sub.index, dtype=object)
    if gender4 is not None and gender4.notna().any():
        # 2021+ only: real Non-Binary/Other categories exist -- use them where present.
        gender_bloc.loc[gender4 == "Woman"] = "women"
        gender_bloc.loc[gender4 == "Man"] = "men"
        gender_bloc.loc[gender4 == "Non-Binary"] = "other_gender"
        gender_bloc.loc[gender4 == "Other"] = "other_gender"
        # respondents with gender4 null (not asked / pre-2021 rows mixed in -- shouldn't
        # happen within a single year, but fall back to binary gender just in case)
        fallback_mask = gender_bloc.isna() & sub["gender"].notna()
        gender_bloc.loc[fallback_mask & (sub["gender"] == "Male")] = "men"
        gender_bloc.loc[fallback_mask & (sub["gender"] == "Female")] = "women"
    else:
        gender_bloc.loc[sub["gender"] == "Male"] = "men"
        gender_bloc.loc[sub["gender"] == "Female"] = "women"

    masks = {}
    for bloc in RACE_BLOCS:
        masks[bloc] = (race_bloc == bloc).reindex(df.index, fill_value=False)
    for bloc in RELIGION_BLOCS:
        masks[bloc] = (relig_bloc == bloc).reindex(df.index, fill_value=False)
    for bloc in GENDER_BLOCS:
        masks[bloc] = (gender_bloc == bloc).reindex(df.index, fill_value=False)
    return masks


def dem_loyalty(pid7_num: pd.Series) -> pd.Series:
    return (7.0 - pid7_num) / 6.0


def two_sample_stats(before: pd.Series, after: pd.Series, w_before: pd.Series | None, w_after: pd.Series | None):
    """Two-INDEPENDENT-samples version of the panel's paired_stats(): different
    respondents in `before` vs `after`, so the delta's variance is the SUM of
    each sample's variance (not a paired difference's variance).
    """
    n_before, n_after = int(before.notna().sum()), int(after.notna().sum())
    out = {"n_before": n_before, "n_after": n_after, "n": min(n_before, n_after)}
    if n_before == 0 or n_after == 0:
        out.update({
            "raw_pid7_change": None, "measured_delta": None, "ci_low": None, "ci_high": None,
            "reason": f"n_before={n_before}, n_after={n_after} -- at least one side has zero respondents in this bloc",
        })
    else:
        loy_before, loy_after = dem_loyalty(before.dropna()), dem_loyalty(after.dropna())
        raw_change = float(after.mean() - before.mean())
        measured_delta = float(loy_after.mean() - loy_before.mean())
        if n_before >= 2 and n_after >= 2:
            se = float(np.sqrt(loy_before.var(ddof=1) / n_before + loy_after.var(ddof=1) / n_after))
            ci_low, ci_high = measured_delta - CI_Z * se, measured_delta + CI_Z * se
        else:
            ci_low, ci_high = None, None
        out.update({
            "raw_pid7_change": raw_change, "measured_delta": measured_delta,
            "ci_low": ci_low, "ci_high": ci_high, "reason": None,
        })

    if w_before is None or w_after is None:
        out["weighted"] = None
        return out
    wb_mask, wa_mask = w_before.notna() & (w_before > 0), w_after.notna() & (w_after > 0)
    nwb, nwa = int(wb_mask.sum()), int(wa_mask.sum())
    if nwb == 0 or nwa == 0:
        out["weighted"] = {
            "n_before": nwb, "n_after": nwa, "n": min(nwb, nwa),
            "raw_pid7_change": None, "measured_delta": None, "ci_low": None, "ci_high": None,
            "reason": f"n_before={nwb}, n_after={nwa} -- at least one side has zero weighted respondents",
        }
        return out
    bw, ww_b = before[wb_mask], w_before[wb_mask]
    aw, ww_a = after[wa_mask], w_after[wa_mask]
    loy_bw, loy_aw = dem_loyalty(bw), dem_loyalty(aw)
    wmean_b = float((loy_bw * ww_b).sum() / ww_b.sum())
    wmean_a = float((loy_aw * ww_a).sum() / ww_a.sum())
    w_measured_delta = wmean_a - wmean_b
    w_raw_change = float((aw * ww_a).sum() / ww_a.sum() - (bw * ww_b).sum() / ww_b.sum())
    if nwb >= 2 and nwa >= 2:
        n_eff_b = float((ww_b.sum() ** 2) / (ww_b**2).sum())
        n_eff_a = float((ww_a.sum() ** 2) / (ww_a**2).sum())
        wvar_b = float(((loy_bw - wmean_b) ** 2 * ww_b).sum() / ww_b.sum())
        wvar_a = float(((loy_aw - wmean_a) ** 2 * ww_a).sum() / ww_a.sum())
        se_w = np.sqrt(wvar_b / n_eff_b + wvar_a / n_eff_a) if n_eff_b > 0 and n_eff_a > 0 else None
        wci = (w_measured_delta - CI_Z * se_w, w_measured_delta + CI_Z * se_w) if se_w is not None else (None, None)
    else:
        wci = (None, None)
    out["weighted"] = {
        "n_before": nwb, "n_after": nwa, "n": min(nwb, nwa),
        "raw_pid7_change": w_raw_change, "measured_delta": w_measured_delta,
        "ci_low": wci[0], "ci_high": wci[1], "reason": None,
    }
    return out


def annotate_suppression(stat: dict) -> dict:
    n = stat["n"]
    if n == 0:
        stat["suppressed_flag"], stat["low_confidence_flag"], stat["suppression_reason"] = True, False, stat.get("reason")
    elif n < SUPPRESS_N:
        stat["suppressed_flag"], stat["low_confidence_flag"] = True, False
        stat["suppression_reason"] = f"n={n} below suppression threshold ({SUPPRESS_N}); numbers are real but unreliable"
    elif n < LOW_CONFIDENCE_N:
        stat["suppressed_flag"], stat["low_confidence_flag"], stat["suppression_reason"] = False, True, None
    else:
        stat["suppressed_flag"], stat["low_confidence_flag"], stat["suppression_reason"] = False, False, None
    w = stat.get("weighted")
    if w is not None:
        wn = w["n"]
        if wn == 0:
            w["suppressed_flag"], w["low_confidence_flag"] = True, False
        elif wn < SUPPRESS_N:
            w["suppressed_flag"], w["low_confidence_flag"] = True, False
        elif wn < LOW_CONFIDENCE_N:
            w["suppressed_flag"], w["low_confidence_flag"] = False, True
        else:
            w["suppressed_flag"], w["low_confidence_flag"] = False, False
    return stat


def main() -> None:
    print("Loading CES cumulative file (2006-2024)...")
    df = load_ces()
    print(f"  {len(df):,} respondent-year rows loaded")

    dup = df.groupby("case_id")["year"].nunique()
    n_repeat = int((dup > 1).sum())
    print(f"  Repeated-cross-section check: {n_repeat:,} of {len(dup):,} unique case_ids "
          f"appear in >1 year ({100*n_repeat/len(dup):.1f}%) -- confirms this is NOT a within-person panel.")

    windows = compute_year_windows(df)
    print("\nCES year fielding windows:")
    for y in CES_YEARS:
        w = windows[y]
        print(f"  {y}  {w['start']} -> {w['end']}  n={w['n_total']:,}  n_pid7_valid={w['n_pid7_valid']:,}")

    shocks = load_shocks()
    shocks_by_id = {s["id"]: s for s in shocks}
    with open(PANEL_JSON, encoding="utf-8") as f:
        panel_bracketed = set(json.load(f).keys())
    print(f"\n{len(panel_bracketed)} shocks already bracketed by the VOTER panel -- excluded from this CES layer.")

    candidates = [s for s in shocks if s["id"] not in panel_bracketed]
    assignments = {}
    for s in candidates:
        a = assign_shock_to_ces_years(s["_date"], windows)
        assignments[s["id"]] = a
    bracketed = {sid: a for sid, a in assignments.items() if a is not None}
    print(f"{len(bracketed)} of {len(candidates)} panel-unbracketed shocks get a CES year-pair.")

    window_groups: dict[tuple, list[str]] = {}
    for sid, a in bracketed.items():
        window_groups.setdefault((a["before_year"], a["after_year"]), []).append(sid)

    bloc_masks_cache: dict[str, dict] = {}

    def get_masks(year: str) -> dict:
        if year not in bloc_masks_cache:
            bloc_masks_cache[year] = compute_blocs(df, year)
        return bloc_masks_cache[year]

    output = {}
    for sid, a in bracketed.items():
        before_year, after_year = a["before_year"], a["after_year"]
        gap_days = (windows[after_year]["start"] - windows[before_year]["end"]).days
        cooccur = [s for s in window_groups[(before_year, after_year)] if s != sid]

        before_df = df[df["year"] == int(before_year)]
        after_df = df[df["year"] == int(after_year)]
        before_masks, after_masks = get_masks(before_year), get_masks(after_year)

        bloc_stats = {}
        for bloc in ALL_BLOCS:
            bm, am = before_masks[bloc].loc[before_df.index], after_masks[bloc].loc[after_df.index]
            b_pid7, a_pid7 = before_df.loc[bm, "pid7_num"], after_df.loc[am, "pid7_num"]
            b_pid7, a_pid7 = b_pid7.dropna(), a_pid7.dropna()
            b_w = before_df.loc[b_pid7.index, "weight"]
            a_w = after_df.loc[a_pid7.index, "weight"]
            stat = two_sample_stats(b_pid7, a_pid7, b_w, a_w)
            stat = annotate_suppression(stat)
            stat["stratum"] = BLOC_STRATUM[bloc]
            bloc_stats[bloc] = stat

        output[sid] = {
            "shock_id": sid,
            "shock_date": shocks_by_id[sid]["date"],
            "source": "CES_2006_2024",
            "fidelity_tier": FIDELITY_TIER,
            "fidelity_rank": FIDELITY_RANK,
            "fidelity_note": FIDELITY_NOTE,
            "single_shock_attributable": False,
            "attributability_reason": (
                "CES is a cross-sectional annual layer -- co-occurring configured shocks in this "
                f"year-pair: {', '.join(cooccur) if cooccur else 'none, but the annual window still spans far more time than any panel bracket'}. "
                "Never treat a CES cell as single-shock-attributable."
            ),
            "co_occurring_shocks_in_window": cooccur,
            "window": {
                "before_year": before_year,
                "after_year": after_year,
                "before_window": [str(windows[before_year]["start"]), str(windows[before_year]["end"])],
                "after_window": [str(windows[after_year]["start"]), str(windows[after_year]["end"])],
                "gap_days_between_waves": gap_days,
                "straddle_year": a["straddle_year"],
            },
            "n_before_total": windows[before_year]["n_total"],
            "n_after_total": windows[after_year]["n_total"],
            "weight_variable_used": "weight",
            "bloc": bloc_stats,
        }

    not_bracketed = sorted(sid for sid, a in assignments.items() if a is None)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}  ({len(output)} shocks)")
    print(f"Not CES-bracketable ({len(not_bracketed)}): {', '.join(not_bracketed)}")


if __name__ == "__main__":
    main()
