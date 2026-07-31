#!/usr/bin/env python3
"""Extract per-bloc party-loyalty deltas from the VOTER panel (2011-2020Nov).

Ground-truth extractor for Phase 1, Step 1.1. Turns the wave brackets identified
in the Step 0.3 investigation into measured per-(shock, bloc) Democratic-loyalty
deltas, with weighting, confidence intervals, small-cell suppression, and tiering.

Read-only w.r.t. survey files. Writes only:
    data/ground_truth/panel_deltas.json
    data/ground_truth/panel_deltas_summary.md

Usage:
    python scripts/extract_panel_ground_truth.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parents[1]))
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

# ── paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parents[1]
VOTER_PANEL_CSV = Path(
    "/Volumes/JUHIDRIVE/electoralData/surveys/VOTER Panel Data Files/voter_panel.csv"
)
SHOCKS_JSON = REPO_ROOT / "configs" / "shocks.json"
OUT_JSON = REPO_ROOT / "data" / "ground_truth" / "panel_deltas.json"
OUT_MD = REPO_ROOT / "data" / "ground_truth" / "panel_deltas_summary.md"

SEED = 42
np.random.seed(SEED)

# ── bloc definitions (reuse the canonical lists, not a local re-declaration) ──

RACE_BLOCS = list(CANONICAL_RACES)
RELIGION_BLOCS = list(CANONICAL_RELIGIONS)
GENDER_BLOCS = list(CANONICAL_GENDERS)
ALL_BLOCS = RACE_BLOCS + RELIGION_BLOCS + GENDER_BLOCS
BLOC_STRATUM = (
    {b: "race" for b in RACE_BLOCS}
    | {b: "religion" for b in RELIGION_BLOCS}
    | {b: "gender" for b in GENDER_BLOCS}
)

# panel race_{wave} codes -> canonical race bloc (Step 0.3 §1, confirmed via
# pandas.io.stata.StataReader.value_labels() on voter_panel.dta)
RACE_MAP = {
    1: "white",
    2: "african_american",
    3: "latino",
    4: "asian",
    5: "other_race",  # Native American
    6: "other_race",  # Mixed / Two or more races
    7: "other_race",  # Other
    8: "other_race",  # Middle Eastern
}
# panel gender_{wave} codes -> canonical gender bloc. NOTE: every wave 2011-2020
# is a strict Male/Female binary -- there is no nonbinary/other code at all, so
# "other_gender" is structurally unmeasurable in this panel (see §3 caveats).
GENDER_MAP = {1: "men", 2: "women"}

# panel religion_{wave} (Pew religpew) codes -> canonical religion bloc.
# Protestant (1) is split into evangelical vs. mainline protestant using the
# companion bornagain_{wave} self-identification flag (1 = born-again/evangelical).
RELIGPEW_DIRECT_MAP = {
    2: "catholic",
    9: "secular",  # Atheist
    10: "secular",  # Agnostic
    11: "secular",  # Nothing in particular
    5: "jewish",
    6: "muslim",
    3: "other_rel",  # Mormon
    4: "other_rel",  # Eastern/Greek Orthodox
    7: "other_rel",  # Buddhist
    8: "other_rel",  # Hindu
    12: "other_rel",  # Something else
}

# waves with real starttime/endtime timestamps in this export, in chronological
# order. 2011/2012 have no starttime/endtime columns in voter_panel.csv, so they
# cannot be dated precisely from this file and are excluded from the bracket
# algorithm (Step 0.3 already found the 2012->2016 gap unusable at ~4-5 years).
DATED_WAVES = ["2016", "2017", "2018", "2019Jan", "2019Nov", "2020Sep", "2020Nov"]
BRACKET_PAIRS = list(zip(DATED_WAVES[:-1], DATED_WAVES[1:]))

# sanity bounds for starttime/endtime parsing -- see load_panel()
PANEL_DATE_FLOOR = pd.Timestamp("2010-01-01")
PANEL_DATE_CEILING = pd.Timestamp("2025-12-31")

# demographic-variable backfill search order (oldest -> newest); a respondent's
# bloc for a given bracket is taken from the bracket's own "before" wave if
# available, else the nearest earlier wave (race/gender are treated as
# time-invariant; religion is refilled from the nearest wave at-or-before the
# bracket's "before" wave only, never from a later wave).
DEMO_WAVE_ORDER = ["2011", "2012", "2016", "2017", "2018", "2019Jan", "2019Nov", "2020Sep", "2020Nov"]

# small-cell handling
SUPPRESS_N = 30  # below this: still report the real numbers, but suppressed_flag=True
LOW_CONFIDENCE_N = 100  # below this (and >= SUPPRESS_N): low_confidence_flag=True
CI_Z = 1.96  # 95% normal-approximation CI on the paired within-person difference

RBG_SHOCK_ID = "ruth_bader_ginsburg_2020"
RBG_CUTOFF = pd.Timestamp("2020-09-18")  # RBG died this date; used as the
# respondent-level split point within the 2020Sep wave's own fielding window.

# ── Step 1.2: explicit, justified tier criteria ─────────────────────────────
#
# A bracket window is classified by three measured quantities -- gap width,
# co-occurring configured-shock count, and both-waves sample size -- applied
# as a decision tree, in this priority order (crowding and sample size are
# checked first because no amount of gap tightness rescues a window that
# fails either):
#
#   1. n_both_waves_total < TIER_MIN_N            -> Tier C (insufficient N)
#   2. n_cooccurring      >= TIER_C_MIN_COOCCURRING -> Tier C (too crowded)
#   3. gap_days           >  TIER_MAX_GAP_DAYS     -> Tier C (window too wide)
#   4. gap_days           <= TIER_A_MAX_GAP_DAYS   -> Tier A (tight window)
#   5. otherwise                                   -> Tier B (usable, wider)
#
# Justification for each threshold (all four are load-bearing, not
# decorative -- see the "does this reproduce the prior assignment" check in
# main()):
#
# TIER_A_MAX_GAP_DAYS = 60. The panel's 6 actual gaps are 46, 187, 196, 249,
# 251, 319 days -- a >4x jump from the shortest (46d) to the next-shortest
# (187d). 60 sits in that empirical gap, not at a round number chosen to hit
# a target count: it captures the one window (2020Sep->2020Nov, 46 days)
# where the before/after waves closely bookend the shock(s), while every
# other window gets ~6-10.5 months for ambient, non-shock political drift to
# accumulate between waves.
#
# TIER_C_MIN_COOCCURRING = 3. Co-occurring *other configured shocks* sharing
# a window are 1, 1, 2, 2, 2 across five windows, and 4 in the sixth
# (2017->2018: charlottesville_2017, daca_rescission_2017,
# las_vegas_shooting_2017, metoo_2017, parkland_shooting_2018). 4 is a clear
# outlier, not a marginal difference -- a shift over that window could
# plausibly be "about" any of five qualitatively different events (a
# white-nationalist rally, a DACA reversal, a mass shooting, a #MeToo-adjacent
# news cycle, another mass shooting) or none of them specifically. The
# threshold sits at the natural break between "a small, nameable set of
# co-occurring events" (<=2) and "too many to meaningfully single any one
# out" (>=3, i.e. 4+ total shocks sharing a window).
#
# TIER_MAX_GAP_DAYS = 330 (~11 months). The widest gap actually usable in
# this data is 319 days (2019Jan->2019Nov); 330 sets the ceiling just above
# that so nothing currently viable is cut, while stating a principled outer
# bound for any future bracket: a window approaching or exceeding a full year
# risks encompassing an entire distinct news/election cycle of its own,
# diluting any specific-shock signal past the point of usability. This is
# honestly a data-anchored buffer, not an independently-derived constant --
# stated as such rather than dressed up as more principled than it is.
#
# TIER_MIN_N = 1,000 both-waves respondents. Chosen so per-bloc analysis
# stays plausible after a bracket's N gets split across 15 blocs -- it's also
# 10x the SUPPRESS_N=30 floor and 10x the LOW_CONFIDENCE_N=100 threshold used
# per-bloc, i.e. big enough that the panel's 2-3 largest blocs (white,
# secular, catholic) could plausibly clear LOW_CONFIDENCE_N even in a
# maximally-thin bracket. All 6 real windows have 3,952-5,905 both-waves
# respondents and clear this easily -- it is non-binding today, but documents
# a real floor for any future panel data added to this extractor with a much
# smaller recontact sample.
TIER_A_MAX_GAP_DAYS = 60
TIER_MAX_GAP_DAYS = 330
TIER_C_MIN_COOCCURRING = 3
TIER_MIN_N = 1000

# "solely_attributable" (a shock's delta may be attributed to that shock ALONE,
# with zero other configured shocks sharing its window) requires literally 0
# co-occurring shocks. Kept as a separate constant from TIER_C_MIN_COOCCURRING
# so the two thresholds can never silently drift together.
SOLELY_ATTRIBUTABLE_MAX_COOCCURRING = 0


def classify_tier(gap_days: int, n_cooccurring: int, n_both_waves: int) -> tuple[str, str]:
    """Apply the Step 1.2 decision tree. Returns (tier, rationale)."""
    if n_both_waves < TIER_MIN_N:
        return "C", f"n_both_waves={n_both_waves:,} < {TIER_MIN_N:,} minimum -- insufficient sample to trust any bloc-level estimate from this window"
    if n_cooccurring >= TIER_C_MIN_COOCCURRING:
        return "C", f"{n_cooccurring} other configured shocks share this window (>= {TIER_C_MIN_COOCCURRING} threshold) -- too crowded for any single-shock attribution to be defensible"
    if gap_days > TIER_MAX_GAP_DAYS:
        return "C", f"gap_days={gap_days} > {TIER_MAX_GAP_DAYS} -- window too wide, ambient political drift dominates any shock-specific signal"
    if gap_days <= TIER_A_MAX_GAP_DAYS:
        return "A", f"gap_days={gap_days} <= {TIER_A_MAX_GAP_DAYS}, {n_cooccurring} co-occurring shock(s) (< {TIER_C_MIN_COOCCURRING}), n_both_waves={n_both_waves:,} >= {TIER_MIN_N:,} -- tight window, usable"
    return "B", f"gap_days={gap_days} (<= {TIER_MAX_GAP_DAYS}), {n_cooccurring} co-occurring shock(s) (< {TIER_C_MIN_COOCCURRING}), n_both_waves={n_both_waves:,} >= {TIER_MIN_N:,} -- wider window, usable with mandatory co-occurrence disclosure"


# ── dominance judgment, per window (qualitative, documented, NOT tier-gating) ─
#
# Which shock is plausibly the dominant national story in a shared window is
# an editorial judgment call, not something computable from the panel or from
# configs/shocks.json. It is recorded here for disclosure alongside every
# affected shock's entry, but deliberately does NOT change tier membership or
# any attributability flag -- promoting it to a hard filter would smuggle an
# unverified qualitative claim in as if it were a measured quantity.
DOMINANT_SHOCK_BY_WINDOW = {
    ("2020Sep", "2020Nov"): {
        "dominant": "election_2020",
        "rationale": (
            "A presidential election is the highest-salience, highest-turnout-relevant "
            "event category in US politics. ruth_bader_ginsburg_2020 (Sep 18) and "
            "trump_covid_diagnosis_2020 (Oct 2) were both major, but were largely covered "
            "and processed through an electoral lens in the 4-6 weeks before voting -- "
            "RBG's vacancy became a Supreme Court campaign issue; Trump's diagnosis was "
            "framed as an October-surprise election story."
        ),
    },
    ("2016", "2017"): {
        "dominant": "travel_ban_2017",
        "rationale": (
            "The travel ban (Jan 27, 2017) triggered immediate mass protests, airport "
            "demonstrations, and court injunctions -- a sharp, acute partisan flashpoint "
            "covered as the first major controversy of the new administration. "
            "paris_climate_withdrawal_2017 (Jun 1, 2017) was significant but a more "
            "discrete policy announcement without comparable acute mobilization."
        ),
    },
    ("2017", "2018"): {
        "dominant": None,
        "rationale": (
            "Genuinely a toss-up among charlottesville_2017 (Unite the Right rally and "
            "its national political fallout), las_vegas_shooting_2017 (deadliest mass "
            "shooting in modern US history at the time), and parkland_shooting_2018 "
            "(catalyzed the March for Our Lives movement) -- no single event in this "
            "5-shock window is clearly dominant over the others, which is itself part of "
            "why this window is classified Tier C rather than rescued by a dominance call."
        ),
    },
    ("2018", "2019Jan"): {
        "dominant": "kavanaugh_2018",
        "rationale": (
            "The Kavanaugh confirmation fight (Ford testimony Sep 27, confirmed Oct 6) "
            "dominated sustained national news coverage for weeks with intense "
            "partisan mobilization on both sides -- arguably the single most polarizing "
            "domestic political event of 2018. family_separation_2018 (implemented "
            "~Apr-Jun 2018) had already peaked in media attention months before this "
            "window's after-wave; pittsburgh_synagogue_shooting_2018 (Oct 27) was "
            "devastating but a shorter news cycle."
        ),
    },
    ("2019Jan", "2019Nov"): {
        "dominant": "ukraine_impeachment_2019",
        "rationale": (
            "The impeachment inquiry (launched Sep 24, 2019) was a sustained, months-long, "
            "historically significant constitutional process directly implicating the "
            "president, driving nightly coverage through year-end. el_paso_shooting_2019 "
            "(Aug 3, 2019) was a horrific mass shooting with an intense but shorter cycle."
        ),
    },
    ("2019Nov", "2020Sep"): {
        "dominant": "blm_george_floyd_2020",
        "rationale": (
            "George Floyd's murder (May 25, 2020) triggered the largest sustained protest "
            "movement in modern US history, with weeks of nationwide unrest and continued "
            "salience through the summer. This is a closer call than the other windows -- "
            "covid_pandemic_2020 (as dated here, Jul 24 2020, a case-resurgence data point "
            "rather than the initial March declaration) and daca_scotus_2020 (Jun 18 SCOTUS "
            "ruling) were both also highly salient across this window -- but the Floyd/BLM "
            "mobilization was the more singular, acute event."
        ),
    },
}

# ── vote-choice (vote-intention) variables, checked against 2 or more waves ──
#
# pid7 is a sticky identity scale; vote CHOICE is the more direct behavioral
# analogue of "which party a bloc currently supports." The panel carries two
# families of vote-choice item that are coded consistently within their family
# and span exactly one bracket's before/after wave pair each:
#   - presvote_{wave}: "who would you vote for / did you vote for, President"
#     -- consistently 1=Republican nominee, 2=Democratic nominee across
#     2019Jan/2019Nov/2020Sep/2020Nov (all about the same 2020 Trump-Biden
#     matchup: prospective in 2019Jan/2019Nov/2020Sep, actual/recalled in
#     2020Nov).
#   - housevote_{wave}: "US House generic ballot" -- consistently
#     1=Republican, 2=Democrat across 2017/2018/2019Jan (all around the 2018
#     midterm cycle).
# No comparable item exists in BOTH waves of the 2016->2017 bracket
# (presvote_2017 and housevote_2016 do not exist in this file), so that
# bracket has no vote-intention comparison -- documented, not guessed.
VOTE_VAR_CONFIG = {
    ("2017", "2018"): {
        "prefix": "housevote",
        "dem_code": 2,
        "rep_code": 1,
        "construct": "US House generic-ballot vote intention (around the 2018 midterm cycle)",
    },
    ("2018", "2019Jan"): {
        "prefix": "housevote",
        "dem_code": 2,
        "rep_code": 1,
        "construct": "US House generic-ballot vote intention (around the 2018 midterm cycle)",
    },
    ("2019Jan", "2019Nov"): {
        "prefix": "presvote",
        "dem_code": 2,
        "rep_code": 1,
        "construct": "2020 presidential matchup vote intention (prospective, Trump vs. Democratic nominee)",
    },
    ("2019Nov", "2020Sep"): {
        "prefix": "presvote",
        "dem_code": 2,
        "rep_code": 1,
        "construct": "2020 presidential matchup vote intention (prospective, Trump vs. Biden)",
    },
    ("2020Sep", "2020Nov"): {
        "prefix": "presvote",
        "dem_code": 2,
        "rep_code": 1,
        "construct": "2020 presidential vote: prospective intention (Sep) -> actual/recalled vote (Nov)",
    },
}


# ── loading ─────────────────────────────────────────────────────────────────


def load_shocks() -> list[dict]:
    with open(SHOCKS_JSON, encoding="utf-8") as f:
        shocks = json.load(f)
    for s in shocks:
        s["_date"] = pd.Timestamp(s["date"])
    return shocks


def load_panel() -> pd.DataFrame:
    all_demo_waves = set(DEMO_WAVE_ORDER)
    wanted_cols = set()
    for w in DATED_WAVES:
        wanted_cols.add(f"starttime_{w}")
        wanted_cols.add(f"endtime_{w}")
        wanted_cols.add(f"pid7_{w}")
        wanted_cols.add(f"weight_panel_{w}")
        wanted_cols.add(f"weight_genpop_{w}")
    for w in all_demo_waves:
        for prefix in ("race", "gender", "religion", "bornagain"):
            wanted_cols.add(f"{prefix}_{w}")
    for (before_wave, after_wave), cfg in VOTE_VAR_CONFIG.items():
        wanted_cols.add(f"{cfg['prefix']}_{before_wave}")
        wanted_cols.add(f"{cfg['prefix']}_{after_wave}")

    df = pd.read_csv(
        VOTER_PANEL_CSV,
        encoding="latin-1",
        low_memory=False,
        usecols=lambda c: c in wanted_cols,
    )
    for w in DATED_WAVES:
        for col in (f"starttime_{w}", f"endtime_{w}"):
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                # Data-quality fix, not a guess: 287 respondents in endtime_2017
                # carry the literal string "31 Dec 69", parsed by dateutil as
                # 2069-12-31 (a two-digit-year sentinel/placeholder, not a real
                # interview end time -- the field study ran Jul 2017, and no
                # other wave/column has any date outside 2010-2025). Any parsed
                # timestamp outside the panel's real operating range
                # (2010-01-01..2025-12-31) is treated as missing.
                out_of_range = (parsed < PANEL_DATE_FLOOR) | (parsed > PANEL_DATE_CEILING)
                if out_of_range.any():
                    print(
                        f"  [data-quality] {col}: {int(out_of_range.sum())} value(s) outside "
                        f"{PANEL_DATE_FLOOR.date()}..{PANEL_DATE_CEILING.date()} treated as missing "
                        f"(e.g. {df.loc[out_of_range, col].iloc[0]!r} -> {parsed[out_of_range].iloc[0]})"
                    )
                parsed = parsed.mask(out_of_range)
                df[col] = parsed
    return df


def resolve_weight_column(df: pd.DataFrame, after_wave: str) -> tuple[str | None, str | None]:
    """Prefer the panel recontact weight; fall back to the general-population
    weight where the panel weight does not exist for this wave (2016/2017 have
    only weight_genpop_{wave} -- confirmed by direct column inspection, not a
    naming-convention guess). Returns (column_name, kind) or (None, None).
    """
    panel_col = f"weight_panel_{after_wave}"
    genpop_col = f"weight_genpop_{after_wave}"
    if panel_col in df.columns:
        return panel_col, "panel_recontact"
    if genpop_col in df.columns:
        return genpop_col, "genpop_fallback"
    return None, None


def compute_wave_windows(df: pd.DataFrame) -> dict:
    windows = {}
    for w in DATED_WAVES:
        sc, ec = f"starttime_{w}", f"endtime_{w}"
        pcol = f"pid7_{w}"
        panel_col = f"weight_panel_{w}"
        genpop_col = f"weight_genpop_{w}"
        windows[w] = {
            "start": df[sc].min() if sc in df.columns else None,
            "end": df[ec].max() if ec in df.columns else None,
            "n_pid7_valid": int(df[pcol].between(1, 7).sum()) if pcol in df.columns else None,
            "has_weight_panel": panel_col in df.columns,
            "has_weight_genpop": genpop_col in df.columns,
        }
    return windows


# ── bracket assignment ───────────────────────────────────────────────────────


def assign_shocks_to_brackets(shocks: list[dict], windows: dict) -> dict:
    """For every shock in configs/shocks.json, determine (a) whether its date
    falls inside a wave's own fielding window (a "straddle", handled specially),
    (b) falls cleanly in the gap between two consecutive dated waves (a normal
    bracket), or (c) falls outside the panel's dated range entirely (2016 start
    to 2020Nov end) and is not bracketable from this file.
    """
    assignments = {}
    for s in shocks:
        d = s["_date"]
        sid = s["id"]

        straddle_wave = None
        for w in DATED_WAVES:
            start, end = windows[w]["start"], windows[w]["end"]
            if start is not None and end is not None and start <= d <= end:
                straddle_wave = w
                break
        if straddle_wave is not None:
            idx = DATED_WAVES.index(straddle_wave)
            if idx + 1 < len(DATED_WAVES):
                assignments[sid] = {
                    "before_wave": straddle_wave,
                    "after_wave": DATED_WAVES[idx + 1],
                    "straddles_before_wave": True,
                }
            continue

        placed = False
        for a, b in BRACKET_PAIRS:
            end_a, start_b = windows[a]["end"], windows[b]["start"]
            if end_a is not None and start_b is not None and end_a <= d <= start_b:
                assignments[sid] = {"before_wave": a, "after_wave": b, "straddles_before_wave": False}
                placed = True
                break
        if not placed:
            assignments[sid] = None  # not bracketable from this file
    return assignments


def bracket_key(before_wave: str, after_wave: str, straddle_cutoff: pd.Timestamp | None) -> tuple:
    return (before_wave, after_wave, straddle_cutoff)


# ── demographic bloc assignment ─────────────────────────────────────────────


def backfill_series(df: pd.DataFrame, prefix: str, before_wave: str) -> pd.Series:
    """Fill each respondent's value for `prefix` starting from `before_wave`
    (the bracket's own before-wave), falling back to progressively earlier
    waves for respondents missing a value at `before_wave`. Never looks forward
    past `before_wave`.
    """
    if before_wave in DEMO_WAVE_ORDER:
        idx = DEMO_WAVE_ORDER.index(before_wave)
    else:
        idx = len(DEMO_WAVE_ORDER) - 1
    candidates = [DEMO_WAVE_ORDER[i] for i in range(idx, -1, -1)]

    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for w in candidates:
        col = f"{prefix}_{w}"
        if col not in df.columns:
            continue
        mask = result.isna() & df[col].notna()
        result.loc[mask] = df.loc[mask, col]
    return result


def religion_bloc_from(relig: pd.Series, bornagain: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=relig.index, dtype=object)
    is_protestant = relig == 1
    out.loc[is_protestant & (bornagain == 1)] = "evangelical"
    out.loc[is_protestant & (bornagain != 1)] = "protestant"
    for code, bloc in RELIGPEW_DIRECT_MAP.items():
        out.loc[relig == code] = bloc
    return out


def compute_blocs_for_bracket(df: pd.DataFrame, before_wave: str) -> dict[str, pd.Series]:
    race_raw = backfill_series(df, "race", before_wave)
    gender_raw = backfill_series(df, "gender", before_wave)
    relig_raw = backfill_series(df, "religion", before_wave)
    born_raw = backfill_series(df, "bornagain", before_wave)

    race_bloc = race_raw.map(RACE_MAP)
    gender_bloc = gender_raw.map(GENDER_MAP)
    relig_bloc = religion_bloc_from(relig_raw, born_raw)

    bloc_of = pd.Series(np.nan, index=df.index, dtype=object)
    per_bloc_mask = {}
    for bloc in RACE_BLOCS:
        per_bloc_mask[bloc] = race_bloc == bloc
    for bloc in RELIGION_BLOCS:
        per_bloc_mask[bloc] = relig_bloc == bloc
    for bloc in GENDER_BLOCS:
        # other_gender is never assignable: the panel's gender variable is a
        # strict Male/Female binary in every wave (2011-2020).
        per_bloc_mask[bloc] = gender_bloc == bloc if bloc != "other_gender" else pd.Series(False, index=df.index)
    return per_bloc_mask


# ── pid7 -> Democratic-loyalty transform ────────────────────────────────────
#
# pid7 is an ordinal 7-point scale: 1=Strong Democrat ... 4=Independent ...
# 7=Strong Republican. We define Democratic-loyalty share as a linear rescaling
# of this ordinal scale onto [0, 1]:
#
#     dem_loyalty(pid7) = (7 - pid7) / 6
#
# so pid7=1 (Strong Dem) -> 1.0, pid7=7 (Strong Rep) -> 0.0, pid7=4 -> 0.5.
# The measured delta for a (shock, bloc) is then:
#
#     measured_delta = mean(dem_loyalty_after) - mean(dem_loyalty_before)
#                     = -(mean(pid7_after) - mean(pid7_before)) / 6
#
# ASSUMPTION, stated explicitly: this treats each 1-point move on the 7-point
# ordinal identification scale as exactly 1/6 of the maximum possible loyalty
# range. It is a transparent linear rescaling of an ordinal self-identification
# item, NOT an empirically-calibrated pid7-to-vote-share elasticity (no such
# elasticity model is applied here; building one would require an external
# vote-choice model and was out of scope for this extractor). The raw,
# untransformed mean-pid7 change is always reported alongside the transformed
# value so this assumption is never silently baked into the only number saved.


def dem_loyalty(pid7: pd.Series) -> pd.Series:
    return (7.0 - pid7) / 6.0


def paired_stats(before: pd.Series, after: pd.Series, weight: pd.Series | None):
    """Paired within-person stats for one bloc's before/after pid7 values.
    Returns a dict with n, raw_pid7_change, measured_delta, ci_low, ci_high
    (unweighted), plus the same weighted if `weight` is not None.
    """
    n = int(before.notna().sum())
    out = {"n": n}
    if n == 0:
        out.update(
            {
                "raw_pid7_change": None,
                "measured_delta": None,
                "ci_low": None,
                "ci_high": None,
                "reason": "n=0: no respondents in this bloc for this bracket",
            }
        )
    else:
        raw_change = float(after.mean() - before.mean())
        per_resp_delta = -(after - before) / 6.0
        measured_delta = float(per_resp_delta.mean())
        if n >= 2:
            se = float(per_resp_delta.std(ddof=1)) / np.sqrt(n)
            ci_low, ci_high = measured_delta - CI_Z * se, measured_delta + CI_Z * se
        else:
            ci_low, ci_high = None, None
        out.update(
            {
                "raw_pid7_change": raw_change,
                "measured_delta": measured_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "reason": None,
            }
        )

    if weight is None:
        out["weighted"] = None
        return out

    w = weight.astype(float)
    wmask = w.notna() & (w > 0)
    nw = int(wmask.sum())
    if nw == 0:
        out["weighted"] = {
            "n": 0,
            "raw_pid7_change": None,
            "measured_delta": None,
            "ci_low": None,
            "ci_high": None,
            "reason": "n=0: no respondents with a valid weight in this bloc/bracket",
        }
        return out

    wb, wa, ww = before[wmask], after[wmask], w[wmask]
    wsum = ww.sum()
    wmean_before = float((wb * ww).sum() / wsum)
    wmean_after = float((wa * ww).sum() / wsum)
    w_raw_change = wmean_after - wmean_before
    w_measured_delta = -(w_raw_change) / 6.0

    per_resp_delta_w = -(wa - wb) / 6.0
    wmean_delta = float((per_resp_delta_w * ww).sum() / wsum)
    if nw >= 2:
        # weighted variance + Kish effective-N for the SE of a weighted mean
        wvar = float(((per_resp_delta_w - wmean_delta) ** 2 * ww).sum() / wsum)
        n_eff = float((ww.sum() ** 2) / (ww**2).sum())
        se_w = np.sqrt(wvar / n_eff) if n_eff > 0 else None
        if se_w is not None:
            wci_low, wci_high = wmean_delta - CI_Z * se_w, wmean_delta + CI_Z * se_w
        else:
            wci_low, wci_high = None, None
    else:
        wci_low, wci_high = None, None

    out["weighted"] = {
        "n": nw,
        "raw_pid7_change": w_raw_change,
        "measured_delta": w_measured_delta,
        "ci_low": wci_low,
        "ci_high": wci_high,
        "reason": None,
    }
    return out


def annotate_suppression(stat: dict) -> dict:
    n = stat["n"]
    if n == 0:
        stat["suppressed_flag"] = True
        stat["low_confidence_flag"] = False
        stat["suppression_reason"] = stat.get("reason")
    elif n < SUPPRESS_N:
        stat["suppressed_flag"] = True
        stat["low_confidence_flag"] = False
        stat["suppression_reason"] = f"n={n} below suppression threshold ({SUPPRESS_N}); numbers are real but unreliable"
    elif n < LOW_CONFIDENCE_N:
        stat["suppressed_flag"] = False
        stat["low_confidence_flag"] = True
        stat["suppression_reason"] = None
    else:
        stat["suppressed_flag"] = False
        stat["low_confidence_flag"] = False
        stat["suppression_reason"] = None
    if stat.get("weighted") is not None:
        wn = stat["weighted"]["n"]
        if wn == 0:
            stat["weighted"]["suppressed_flag"] = True
            stat["weighted"]["low_confidence_flag"] = False
        elif wn < SUPPRESS_N:
            stat["weighted"]["suppressed_flag"] = True
            stat["weighted"]["low_confidence_flag"] = False
        elif wn < LOW_CONFIDENCE_N:
            stat["weighted"]["suppressed_flag"] = False
            stat["weighted"]["low_confidence_flag"] = True
        else:
            stat["weighted"]["suppressed_flag"] = False
            stat["weighted"]["low_confidence_flag"] = False
    return stat


def paired_stats_share(before_share: pd.Series, after_share: pd.Series, weight: pd.Series | None):
    """Same paired-difference logic as paired_stats(), but for a variable
    already on a [0,1] two-party Democratic-vote-share scale (no /6 rescaling
    -- there is nothing to transform, `measured_delta` IS the raw share
    change). Used for the vote-intention comparison (Step 1.1 loose end #3).
    """
    n = int(before_share.notna().sum())
    out = {"n": n}
    if n == 0:
        out.update(
            {
                "measured_delta": None,
                "ci_low": None,
                "ci_high": None,
                "reason": "n=0: no respondents in this bloc had a valid Democratic/Republican choice in both waves",
            }
        )
    else:
        per_resp_delta = after_share - before_share
        measured_delta = float(per_resp_delta.mean())
        if n >= 2:
            se = float(per_resp_delta.std(ddof=1)) / np.sqrt(n)
            ci_low, ci_high = measured_delta - CI_Z * se, measured_delta + CI_Z * se
        else:
            ci_low, ci_high = None, None
        out.update({"measured_delta": measured_delta, "ci_low": ci_low, "ci_high": ci_high, "reason": None})

    if weight is None:
        out["weighted"] = None
        return out

    w = weight.astype(float)
    wmask = w.notna() & (w > 0)
    nw = int(wmask.sum())
    if nw == 0:
        out["weighted"] = {
            "n": 0,
            "measured_delta": None,
            "ci_low": None,
            "ci_high": None,
            "reason": "n=0: no respondents with a valid weight in this bloc/bracket",
        }
        return out

    bs, as_, ww = before_share[wmask], after_share[wmask], w[wmask]
    wsum = ww.sum()
    per_resp_delta_w = as_ - bs
    wmean_delta = float((per_resp_delta_w * ww).sum() / wsum)
    if nw >= 2:
        wvar = float(((per_resp_delta_w - wmean_delta) ** 2 * ww).sum() / wsum)
        n_eff = float((ww.sum() ** 2) / (ww**2).sum())
        se_w = np.sqrt(wvar / n_eff) if n_eff > 0 else None
        if se_w is not None:
            wci_low, wci_high = wmean_delta - CI_Z * se_w, wmean_delta + CI_Z * se_w
        else:
            wci_low, wci_high = None, None
    else:
        wci_low, wci_high = None, None

    out["weighted"] = {
        "n": nw,
        "measured_delta": wmean_delta,
        "ci_low": wci_low,
        "ci_high": wci_high,
        "reason": None,
    }
    return out


def compute_vote_stats_for_bracket(
    df: pd.DataFrame, before_wave: str, after_wave: str, straddle_cutoff: pd.Timestamp | None
) -> dict:
    """Mirrors compute_bracket_bloc_stats() but for the vote-choice variable
    configured for this (before_wave, after_wave) pair, if any exists.
    """
    cfg = VOTE_VAR_CONFIG.get((before_wave, after_wave))
    if cfg is None:
        return {
            "available": False,
            "reason": (
                f"no comparable vote-choice variable exists in both {before_wave} and "
                f"{after_wave} (checked presvote_* and housevote_*)"
            ),
        }

    before_col = f"{cfg['prefix']}_{before_wave}"
    after_col = f"{cfg['prefix']}_{after_wave}"
    dem, rep = cfg["dem_code"], cfg["rep_code"]

    before_valid = df[before_col].isin([dem, rep])
    if straddle_cutoff is not None:
        st_col = f"starttime_{before_wave}"
        before_valid = before_valid & (df[st_col] < straddle_cutoff)
    after_valid = df[after_col].isin([dem, rep])
    both = before_valid & after_valid

    bloc_masks = compute_blocs_for_bracket(df, before_wave)
    weight_col, weight_kind = resolve_weight_column(df, after_wave)
    has_weight = weight_col is not None
    weight_series = df[weight_col] if has_weight else None

    before_share_all = (df[before_col] == dem).astype(float)
    after_share_all = (df[after_col] == dem).astype(float)

    results = {}
    for bloc in ALL_BLOCS:
        sub_mask = both & bloc_masks[bloc]
        before_vals = before_share_all.loc[sub_mask]
        after_vals = after_share_all.loc[sub_mask]
        w_vals = weight_series.loc[sub_mask] if has_weight else None
        stat = paired_stats_share(before_vals, after_vals, w_vals)
        stat = annotate_suppression(stat)
        stat["stratum"] = BLOC_STRATUM[bloc]
        results[bloc] = stat

    return {
        "available": True,
        "variable_before": before_col,
        "variable_after": after_col,
        "construct": cfg["construct"],
        "dem_code": dem,
        "rep_code": rep,
        "n_both_waves_total": int(both.sum()),
        "n_before_wave_total": int(before_valid.sum()),
        "n_after_wave_total": int(after_valid.sum()),
        "weight_variable_used": weight_col,
        "weight_kind": weight_kind,
        "blocs": results,
    }


def compute_trust_flags(stat: dict, single_shock_attributable: bool) -> dict:
    """Four-filter trust assessment for one (shock, bloc) cell -- works for
    both the pid7 `bloc` structure and the `vote_intention.blocs` structure
    (identical n/measured_delta/ci_low/ci_high/suppressed_flag/
    low_confidence_flag/weighted schema).

    - sign_stable: unweighted and weighted measured_delta agree in sign
      (0 counts as its own sign; a cell where one side is exactly 0 and the
      other is not is NOT sign-stable -- there is nothing to confirm).
    - significant: the unweighted 95% CI excludes zero.
    - not_suppressed: neither suppressed_flag nor low_confidence_flag is set.
    - single_shock_attributable: passed in from the shock-level flag.

    trustworthy_for_direction = all four. magnitude_usable = all but
    sign_stable (a cell can have a well-estimated, significant magnitude even
    if weighting flips which direction that magnitude points).
    """

    def sign(x):
        if x is None:
            return None
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    unweighted_delta = stat.get("measured_delta")
    weighted = stat.get("weighted")
    weighted_delta = weighted.get("measured_delta") if weighted else None
    weighted_n = weighted.get("n", 0) if weighted else 0

    if unweighted_delta is None or weighted is None or weighted_n == 0 or weighted_delta is None:
        sign_stable = False
        sign_stable_reason = "weighted estimate unavailable (n=0 or not computable) -- cannot confirm sign stability"
    else:
        su, sw = sign(unweighted_delta), sign(weighted_delta)
        sign_stable = bool(su == sw)
        sign_stable_reason = None if sign_stable else f"unweighted sign={su:+d}, weighted sign={sw:+d}"

    ci_low, ci_high = stat.get("ci_low"), stat.get("ci_high")
    if ci_low is None or ci_high is None:
        significant = False
        significant_reason = "CI not computable (n<2)"
    else:
        # bool(...) here matters: ci_low/ci_high can be numpy.float64, whose
        # comparison operators return numpy.bool_, which json.dump(default=str)
        # silently serializes as the STRING "True"/"False" instead of a real
        # JSON boolean if not coerced back to a plain Python bool first.
        significant = bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
        significant_reason = None if significant else "95% CI includes zero"

    is_suppressed = bool(stat.get("suppressed_flag"))
    is_low_conf = bool(stat.get("low_confidence_flag"))
    not_suppressed = (not is_suppressed) and (not is_low_conf)
    not_suppressed_reason = None
    if not not_suppressed:
        reasons = []
        if is_suppressed:
            reasons.append(stat.get("suppression_reason") or "suppressed")
        if is_low_conf:
            reasons.append(f"n={stat.get('n')} below low-confidence threshold ({LOW_CONFIDENCE_N})")
        not_suppressed_reason = "; ".join(reasons)

    trustworthy_for_direction = bool(sign_stable and significant and not_suppressed and single_shock_attributable)
    magnitude_usable = bool(significant and not_suppressed and single_shock_attributable)

    return {
        "sign_stable": sign_stable,
        "sign_stable_reason": sign_stable_reason,
        "significant": significant,
        "significant_reason": significant_reason,
        "not_suppressed": not_suppressed,
        "not_suppressed_reason": not_suppressed_reason,
        "single_shock_attributable": single_shock_attributable,
        "trustworthy_for_direction": trustworthy_for_direction,
        "magnitude_usable": magnitude_usable,
    }


# ── bracket-level computation cache ─────────────────────────────────────────


def compute_bracket_bloc_stats(df: pd.DataFrame, before_wave: str, after_wave: str, straddle_cutoff: pd.Timestamp | None) -> dict:
    pid7_before_col = f"pid7_{before_wave}"
    pid7_after_col = f"pid7_{after_wave}"

    before_valid = df[pid7_before_col].between(1, 7)
    if straddle_cutoff is not None:
        st_col = f"starttime_{before_wave}"
        before_valid = before_valid & (df[st_col] < straddle_cutoff)
    after_valid = df[pid7_after_col].between(1, 7)
    both = before_valid & after_valid

    bloc_masks = compute_blocs_for_bracket(df, before_wave)

    weight_col, weight_kind = resolve_weight_column(df, after_wave)
    has_weight = weight_col is not None
    weight_series = df[weight_col] if has_weight else None

    results = {}
    for bloc in ALL_BLOCS:
        sub_mask = both & bloc_masks[bloc]
        before_vals = df.loc[sub_mask, pid7_before_col]
        after_vals = df.loc[sub_mask, pid7_after_col]
        w_vals = weight_series.loc[sub_mask] if has_weight else None
        stat = paired_stats(before_vals, after_vals, w_vals)
        stat = annotate_suppression(stat)
        stat["stratum"] = BLOC_STRATUM[bloc]
        results[bloc] = stat

    return {
        "n_both_waves_total": int(both.sum()),
        "n_before_wave_total": int(before_valid.sum()),
        "n_after_wave_total": int(after_valid.sum()),
        "weight_variable_used": weight_col,
        "weight_kind": weight_kind,
        "blocs": results,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading VOTER panel (encoding=latin-1)...")
    df = load_panel()
    print(f"  {len(df):,} respondent rows, {len(df.columns)} columns loaded")

    windows = compute_wave_windows(df)
    print("\nWave fielding windows (from actual starttime/endtime):")
    for w in DATED_WAVES:
        win = windows[w]
        print(
            f"  {w:8s} {win['start']} -> {win['end']}  n_pid7_valid={win['n_pid7_valid']}  "
            f"weight_panel={'yes' if win['has_weight_panel'] else 'no'}  "
            f"weight_genpop={'yes' if win['has_weight_genpop'] else 'no'}"
        )

    shocks = load_shocks()
    shocks_by_id = {s["id"]: s for s in shocks}
    assignments = assign_shocks_to_brackets(shocks, windows)

    bracketed = {sid: a for sid, a in assignments.items() if a is not None}
    print(f"\n{len(bracketed)} of {len(shocks)} shocks fall inside a dated VOTER-panel wave bracket.")

    # group shocks sharing the exact same (before_wave, after_wave, straddle) window
    window_groups: dict[tuple, list[str]] = {}
    for sid, a in bracketed.items():
        key = (a["before_wave"], a["after_wave"], a["straddles_before_wave"])
        window_groups.setdefault(key, []).append(sid)

    # RBG gets its own straddle-adjusted before-window; it still shares the
    # (2020Sep, 2020Nov) AFTER-wave with election_2020 / trump_covid_diagnosis_2020,
    # so for "co-occurring events" purposes group by (before_wave, after_wave)
    # regardless of the straddle flag.
    coshock_groups: dict[tuple, list[str]] = {}
    for sid, a in bracketed.items():
        key = (a["before_wave"], a["after_wave"])
        coshock_groups.setdefault(key, []).append(sid)

    # bracket-level stats cache, keyed by (before_wave, after_wave, straddle_cutoff)
    stats_cache: dict[tuple, dict] = {}
    vote_stats_cache: dict[tuple, dict] = {}

    def get_bracket_stats(before_wave: str, after_wave: str, straddle: bool) -> dict:
        cutoff = RBG_CUTOFF if straddle else None
        key = (before_wave, after_wave, cutoff)
        if key not in stats_cache:
            stats_cache[key] = compute_bracket_bloc_stats(df, before_wave, after_wave, cutoff)
        return stats_cache[key]

    def get_vote_stats(before_wave: str, after_wave: str, straddle: bool) -> dict:
        cutoff = RBG_CUTOFF if straddle else None
        key = (before_wave, after_wave, cutoff)
        if key not in vote_stats_cache:
            vote_stats_cache[key] = compute_vote_stats_for_bracket(df, before_wave, after_wave, cutoff)
        return vote_stats_cache[key]

    output = {}
    for sid, a in bracketed.items():
        before_wave, after_wave, straddle = a["before_wave"], a["after_wave"], a["straddles_before_wave"]
        gap_days = (windows[after_wave]["start"] - windows[before_wave]["end"]).days

        cooccur = [s for s in coshock_groups[(before_wave, after_wave)] if s != sid]
        n_cooccur = len(cooccur)

        bstats = get_bracket_stats(before_wave, after_wave, straddle)
        vstats = get_vote_stats(before_wave, after_wave, straddle)

        # n_both_waves_total is per-shock (RBG's straddle split gives it a
        # smaller N than election_2020/trump_covid_diagnosis_2020 despite
        # sharing the same window), so tier is classified per-shock too, not
        # per-window -- in practice this never actually splits a window
        # across tiers in the current data, but the RBG straddle case is
        # exactly the scenario where it *could*.
        tier, tier_rationale = classify_tier(gap_days, n_cooccur, bstats["n_both_waves_total"])

        # Backward-compatible with Step 1.1's trust-flag pipeline: Tier A/B
        # ("usable") is numerically identical to the old ad hoc
        # `n_cooccur < 3` rule -- Tier C is exactly the old "not attributable"
        # 5-shock set, now promoted from a flag to a full tier.
        attributable = tier in ("A", "B")
        attributability_reason = (
            None
            if attributable
            else tier_rationale
        )

        solely_attributable = n_cooccur <= SOLELY_ATTRIBUTABLE_MAX_COOCCURRING
        attribution_confidence = {"A": "high", "B": "moderate", "C": "none"}[tier]

        window_key = (before_wave, after_wave)
        dominance = DOMINANT_SHOCK_BY_WINDOW.get(window_key, {"dominant": None, "rationale": "not assessed"})
        is_dominant_in_window = dominance["dominant"] == sid

        # attach the 4-filter trust assessment to every pid7 bloc cell and
        # (where available) every vote-intention bloc cell. bstats/vstats are
        # cached per (before_wave, after_wave, straddle) and attributability
        # is identical for every shock sharing a window, so mutating in place
        # here is safe and idempotent across shocks that share a bracket.
        for bloc in ALL_BLOCS:
            bstats["blocs"][bloc]["trust"] = compute_trust_flags(bstats["blocs"][bloc], attributable)
        if vstats.get("available"):
            for bloc in ALL_BLOCS:
                vstats["blocs"][bloc]["trust"] = compute_trust_flags(vstats["blocs"][bloc], attributable)

        entry = {
            "shock_id": sid,
            "shock_date": shocks_by_id[sid]["date"],
            "tier": tier,
            "tier_rationale": tier_rationale,
            "attribution_confidence": attribution_confidence,
            "single_shock_attributable": attributable,
            "attributability_reason": attributability_reason,
            "solely_attributable": solely_attributable,
            "solely_attributable_reason": (
                None
                if solely_attributable
                else f"{n_cooccur} other configured shock(s) share this exact window -- a delta can never be attributed to this shock alone, regardless of tier"
            ),
            "dominant_shock_in_window": dominance["dominant"],
            "is_dominant_in_window": is_dominant_in_window,
            "dominance_rationale": dominance["rationale"],
            "bracket": {
                "before_wave": before_wave,
                "after_wave": after_wave,
                "before_window": [str(windows[before_wave]["start"]), str(windows[before_wave]["end"])],
                "after_window": [str(windows[after_wave]["start"]), str(windows[after_wave]["end"])],
                "gap_days_between_waves": gap_days,
                "straddles_before_wave": straddle,
                "special_handling": (
                    f"respondent-level date split: only {before_wave} respondents interviewed "
                    f"before {RBG_CUTOFF.date()} are counted as 'before' (RBG died {RBG_CUTOFF.date()}, "
                    f"inside this wave's own fielding window)"
                    if straddle
                    else None
                ),
            },
            "co_occurring_shocks_in_window": cooccur,
            "n_before_wave_total": bstats["n_before_wave_total"],
            "n_after_wave_total": bstats["n_after_wave_total"],
            "n_both_waves_total": bstats["n_both_waves_total"],
            "weight_variable_used": bstats["weight_variable_used"],
            "weight_kind": bstats["weight_kind"],
            "bloc": bstats["blocs"],
            "vote_intention": vstats,
        }
        output[sid] = entry

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}")

    write_markdown(output, windows, assignments, shocks_by_id, window_groups)
    print(f"Wrote {OUT_MD}")


def write_markdown(output: dict, windows: dict, assignments: dict, shocks_by_id: dict, window_groups: dict) -> None:
    lines = []
    a = lines.append
    a("# Panel Ground-Truth Deltas — VOTER Panel Extraction\n")
    a(
        "Phase 1, Step 1.1. Extracted from `VOTER Panel Data Files/voter_panel.csv` "
        "(true 9-wave panel, 12,517 respondents recruited 2011, last wave "
        "Nov-Dec 2020). Every number below is computed directly from the panel "
        "file — no placeholders, no estimates. Script: "
        "`scripts/extract_panel_ground_truth.py`.\n"
    )

    a("## Wave fielding windows (actual starttime/endtime, not wave labels)\n")
    a("| Wave | Fielding start | Fielding end | N (valid pid7) | `weight_panel` available |")
    a("|---|---|---|---:|---|")
    for w in DATED_WAVES:
        win = windows[w]
        a(f"| {w} | {win['start']} | {win['end']} | {win['n_pid7_valid']:,} | {'yes' if win['has_weight_panel'] else 'no'} |")
    a(
        "\n2011 and 2012 waves have no `starttime`/`endtime` columns in this export "
        "and are excluded from the bracket algorithm entirely (Step 0.3 already "
        "found the resulting 2012→2016 gap, ~4-5 years, too wide to attribute to "
        "any single shock). The panel ends after the 2020Nov wave — nothing from "
        "2021 onward is bracketable from this file.\n"
    )

    a("## Bloc-mapping documentation\n")
    a(
        "**Race** — `race_{wave}` (Stata value labels from `voter_panel.dta`): "
        "1=White→`white`, 2=Black→`african_american`, 3=Hispanic→`latino`, "
        "4=Asian→`asian`, 5=Native American, 6=Mixed/Two-or-more, 7=Other, "
        "8=Middle Eastern → all four collapse to `other_race`.\n"
    )
    a(
        "**Religion** — `religion_{wave}` (Pew religpew scale) + `bornagain_{wave}` "
        "(self-identified born-again/evangelical flag): "
        "religpew=1 (Protestant) AND bornagain=1 → `evangelical`; "
        "religpew=1 AND bornagain≠1 → `protestant`; "
        "religpew=2 (Roman Catholic) → `catholic`; "
        "religpew∈{9,10,11} (Atheist/Agnostic/Nothing in particular) → `secular`; "
        "religpew=5 (Jewish) → `jewish`; religpew=6 (Muslim) → `muslim`; "
        "religpew∈{3,4,7,8,12} (Mormon/Orthodox/Buddhist/Hindu/Something else) → `other_rel`.\n"
    )
    a(
        "**Gender** — `gender_{wave}`: 1=Male→`men`, 2=Female→`women`. "
        "**`other_gender` is structurally unmeasurable**: every wave 2011-2020 "
        "uses a strict Male/Female binary with no nonbinary/other code at all — "
        "this is a hard survey-side gap, not a processing choice. Every "
        "`other_gender` cell below is suppressed with `n=0`.\n"
    )
    a(
        "**Demographic backfill rule**: for a given bracket, each respondent's "
        "race/gender/religion is taken from that bracket's own before-wave if "
        "answered there, else the nearest earlier wave (searched in chronological "
        "order back to the 2011 baseline). This assumes race/gender are "
        "time-invariant (standard) and that religion is reasonably stable over "
        "the ~1-year gaps between waves (less certain, but the alternative — "
        "dropping anyone who skipped the demographic question in the before-wave "
        "— would shrink N further for no clear benefit).\n"
    )

    a("## pid7 → Democratic-loyalty transform\n")
    a(
        "pid7 is 1=Strong Democrat … 4=Independent … 7=Strong Republican. "
        "Democratic-loyalty share is defined as `dem_loyalty(pid7) = (7 - pid7) / 6`, "
        "so pid7=1 → 1.0, pid7=7 → 0.0, pid7=4 → 0.5. The reported "
        "`measured_delta` for a (shock, bloc) is "
        "`mean(dem_loyalty_after) - mean(dem_loyalty_before) = -(mean(pid7_after) - mean(pid7_before)) / 6`.\n"
    )
    a(
        "**This is a transparent linear rescaling of the ordinal pid7 item, "
        "NOT an empirically-calibrated pid7-to-vote-share elasticity.** It "
        "assumes a 1-point move on the 7-point scale is exactly 1/6 of the "
        "maximum possible loyalty range. No external vote-choice model was used "
        "to calibrate this mapping — building one was out of scope here. "
        "`raw_pid7_change` (the untransformed mean-pid7 difference, positive = "
        "shift toward Republican) is always reported alongside `measured_delta` "
        "(positive = shift toward Democratic) so the transform assumption is "
        "never the only number preserved.\n"
    )

    a("## Weighting\n")
    a(
        "Weight variable, per bracket: `weight_panel_{after_wave}` (the after-wave's "
        "panel recontact weight) where it exists, falling back to "
        "`weight_genpop_{after_wave}` (the after-wave's general-population weight) "
        "where it does not — confirmed by direct column inspection, not a naming "
        "guess. `weight_panel_2016` and `weight_panel_2017` do not exist anywhere "
        "in this file (only `weight_genpop_2016`/`weight_genpop_2017` do), so the "
        "2016→2017 bracket (`travel_ban_2017`, `paris_climate_withdrawal_2017`) "
        "uses the genpop fallback — every other bracket has a real panel-recontact "
        "weight. Every cell's `weight_kind` field (`panel_recontact` or "
        "`genpop_fallback`) makes this explicit; see "
        "`data/ground_truth/step1_1_loose_ends.md` for the full audit and how "
        "much the weighted vs. unweighted deltas actually differ. Both a weighted "
        "and an unweighted delta are always reported when computable so the "
        "effect of weighting is visible, not hidden.\n"
    )

    a("## Vote-intention vs. pid7\n")
    a(
        "In addition to pid7, the panel carries a real vote-CHOICE item "
        "(`presvote_{wave}` for the 2020 presidential matchup, `housevote_{wave}` "
        "for the US House generic ballot) spanning 5 of the 6 bracket windows — "
        "not the 2016→2017 one, where neither `presvote_2017` nor `housevote_2016` "
        "exists. Each shock's `vote_intention` field carries the same per-bloc "
        "measured_delta/ci/weighted structure as `bloc`, already on a native "
        "[0,1] two-party Democratic-vote-share scale (no /6 rescaling needed). "
        "Full comparison and the answer to \"are vote-intention shifts bigger than "
        "pid7 shifts\" is in `data/ground_truth/step1_1_loose_ends.md` §3.\n"
    )

    a("## Small-cell handling\n")
    a(
        f"- `n = 0` → both `measured_delta` and `raw_pid7_change` are `null`, "
        f"with a stated `reason`.\n"
        f"- `0 < n < {SUPPRESS_N}` → the real computed numbers are still "
        f"reported (never a placeholder), but `suppressed_flag = true` with a "
        f"`suppression_reason`. Treat these as directional only — this is "
        f"exactly the Muslim-bloc situation flagged in Step 0.3 (n=8-23 per "
        f"bracket).\n"
        f"- `{SUPPRESS_N} <= n < {LOW_CONFIDENCE_N}` → `low_confidence_flag = true`, "
        f"not suppressed.\n"
        f"- `n >= {LOW_CONFIDENCE_N}` → no flag.\n"
        f"- 95% CIs use a paired-difference normal approximation "
        f"(`mean ± 1.96 * SE`, SE from the per-respondent before/after "
        f"difference); weighted CIs use a Kish effective-N approximation for "
        f"the weighted SE. CIs are `null` when `n < 2`.\n"
    )

    a("## Tier criteria (Step 1.2 — explicit, justified)\n")
    a(
        "Every bracket window is classified by a decision tree over three "
        "measured quantities, checked in this order (crowding and sample size "
        "are checked before gap width because no amount of gap tightness "
        "rescues a window that fails either):\n"
    )
    a(
        f"1. `n_both_waves_total < {TIER_MIN_N:,}` → **Tier C** (insufficient sample)\n"
        f"2. `n_cooccurring >= {TIER_C_MIN_COOCCURRING}` → **Tier C** (too crowded)\n"
        f"3. `gap_days > {TIER_MAX_GAP_DAYS}` → **Tier C** (window too wide)\n"
        f"4. `gap_days <= {TIER_A_MAX_GAP_DAYS}` → **Tier A** (tight window)\n"
        f"5. otherwise → **Tier B** (usable, wider window)\n"
    )
    a("**Threshold justifications** (none are round numbers picked to hit a target count):\n")
    a(
        f"- **`{TIER_A_MAX_GAP_DAYS}`-day Tier A gap ceiling**: the panel's 6 actual "
        f"gaps are 46, 187, 196, 249, 251, 319 days — a >4x jump from the shortest "
        f"(46d) to the next-shortest (187d). {TIER_A_MAX_GAP_DAYS} sits in that "
        f"empirical gap: it isolates the one window where before/after waves "
        f"closely bookend the shock(s), from every other window's 6-10.5 months "
        f"of room for ambient political drift between waves.\n"
    )
    a(
        f"- **`{TIER_C_MIN_COOCCURRING}`-shock crowding ceiling**: co-occurring "
        f"*other configured shocks* per window are 1, 1, 2, 2, 2 across five "
        f"windows, and 4 in the sixth (2017→2018). 4 is a clear outlier, not a "
        f"marginal difference — a shift over that window could plausibly be "
        f"\"about\" any of 5 qualitatively different events, or none of them "
        f"specifically. The threshold sits at the natural break between \"a "
        f"small, nameable set of co-occurring events\" (≤2) and \"too many to "
        f"single any one out\" (≥3).\n"
    )
    a(
        f"- **`{TIER_MAX_GAP_DAYS}`-day Tier B ceiling (~11 months)**: the widest "
        f"gap actually usable in this data is 319 days; {TIER_MAX_GAP_DAYS} sets "
        f"the ceiling just above that so nothing currently viable is cut, while "
        f"stating a principled outer bound for any future bracket — a window "
        f"approaching or exceeding a full year risks encompassing an entire "
        f"distinct news/election cycle of its own. **This one is honestly a "
        f"data-anchored buffer, not an independently-derived constant** — stated "
        f"as such rather than dressed up as more principled than it is.\n"
    )
    a(
        f"- **`{TIER_MIN_N:,}`-respondent minimum N**: 10x the per-bloc "
        f"`SUPPRESS_N`={SUPPRESS_N} floor and 10x the per-bloc "
        f"`LOW_CONFIDENCE_N`={LOW_CONFIDENCE_N} threshold — big enough that the "
        f"panel's 2-3 largest blocs could plausibly still clear "
        f"`LOW_CONFIDENCE_N` even in a maximally-thin bracket. All 6 real windows "
        f"have 3,952-5,905 both-waves respondents and clear this easily — "
        f"non-binding today, but a real floor for any future panel data added to "
        f"this extractor.\n"
    )
    a(
        "**Does the rule reproduce the prior (Step 1.1) A/B assignment, or "
        "reclassify anything?** Tiers A and B are unchanged from Step 1.1 — the "
        "3 Tier A and 10 Tier B shocks are identical. **The 5 shocks previously "
        "flagged `single_shock_attributable: false` within Tier B (the crowded "
        "2017→2018 window) are now reclassified as a distinct Tier C** rather "
        "than a flagged-but-nominally-Tier-B entry: `charlottesville_2017`, "
        "`daca_rescission_2017`, `las_vegas_shooting_2017`, `metoo_2017`, "
        "`parkland_shooting_2018`. This is a labeling promotion, not a change in "
        "which shocks are usable — those 5 were already excluded from the "
        "Step 1.1 trust-flag pipeline (`single_shock_attributable`), so the "
        "19-cell `trustworthy_for_direction` and 24-cell `magnitude_usable` "
        "counts from `data/ground_truth/trustworthy_subset.md` are unchanged by "
        "this reclassification.\n"
    )
    a(
        "**`solely_attributable`** is a separate, stricter flag: true only if "
        "ZERO other configured shocks share the window "
        f"(`n_cooccurring <= {SOLELY_ATTRIBUTABLE_MAX_COOCCURRING}`). Every "
        "bracketed window has at least 1 co-occurring shock — **`solely_attributable` "
        "is `false` for all 18 shocks, with no exceptions, including every Tier A "
        "shock.** This is the honest answer to \"can a delta be attributed to this "
        "shock ALONE\": no. Use `attribution_confidence` "
        "(`high`/`moderate`/`none`, from tier) for a graded read, and always read "
        "`co_occurring_shocks_in_window` before reporting any single-shock claim.\n"
    )
    a(
        "**Dominance** (`dominant_shock_in_window`, `dominance_rationale`) is a "
        "documented editorial judgment about which co-occurring shock is "
        "plausibly the most nationally salient in a shared window — recorded for "
        "disclosure, and deliberately **not used to gate tier or any "
        "attributability flag**, since promoting a qualitative call to a hard "
        "filter would overclaim precision it doesn't have.\n"
    )

    tier_a = sorted([sid for sid, e in output.items() if e["tier"] == "A"])
    tier_b = sorted([sid for sid, e in output.items() if e["tier"] == "B"])
    tier_c = sorted([sid for sid, e in output.items() if e["tier"] == "C"])

    a(f"### Tier A ({len(tier_a)} shocks) — high confidence, tight window\n")
    for sid in tier_a:
        e = output[sid]
        co = e["co_occurring_shocks_in_window"]
        special = e["bracket"]["special_handling"]
        dom = e["dominant_shock_in_window"]
        dom_note = f"dominant (per judgment call): `{dom}`" if dom else "dominance: no clear call"
        a(f"- **`{sid}`** ({e['shock_date']}) — window {e['bracket']['before_wave']}→{e['bracket']['after_wave']}, "
          f"gap {e['bracket']['gap_days_between_waves']}d, n_both_waves={e['n_both_waves_total']:,}"
          + (f", co-occurring: {', '.join(co)}" if co else ", no co-occurring shocks in this exact window")
          + f", {dom_note}"
          + (f". **{special}**" if special else ""))
    a(
        "\nAll three Tier A shocks share the identical 46-day 2020Sep→2020Nov "
        "window. Even Tier A cannot attribute a shift to a single shock without "
        "further argument — `solely_attributable` is `false` for all three.\n"
    )

    a(f"\n### Tier B ({len(tier_b)} shocks) — moderate confidence, wider window, mandatory co-occurrence disclosure\n")
    for key, sids in sorted(window_groups.items()):
        b_sids = [s for s in sids if output[s]["tier"] == "B"]
        if not b_sids:
            continue
        before_wave, after_wave, _ = key
        dom = output[b_sids[0]]["dominant_shock_in_window"]
        a(f"\n**Window {before_wave}→{after_wave}** ({output[b_sids[0]]['bracket']['gap_days_between_waves']}d gap, "
          f"n_both_waves={output[b_sids[0]]['n_both_waves_total']:,}) — {len(b_sids)} shock(s) share this window"
          + (f", plausibly dominant: `{dom}`" if dom else "") + ":")
        for sid in sorted(b_sids):
            e = output[sid]
            others = [s for s in b_sids if s != sid]
            is_dom = " (judged dominant)" if e["is_dominant_in_window"] else ""
            a(f"  - `{sid}`{is_dom} ({e['shock_date']})" + (f" — co-occurring: {', '.join(others)}" if others else " — sole shock in this window"))
        a(f"\n  *Dominance rationale*: {output[b_sids[0]]['dominance_rationale']}")

    a(f"\n### Tier C ({len(tier_c)} shocks) — unusable, no attribution defensible\n")
    a(
        "These shocks' numbers are still computed and reported in the JSON "
        "(never dropped — Step 1.1's \"never emit without a loud flag\" "
        "principle holds), but `attribution_confidence: none` and "
        "`single_shock_attributable: false`: do not use them to score a model "
        "against any specific shock. Read them, if at all, only as a "
        "description of the shared window as a whole.\n"
    )
    for key, sids in sorted(window_groups.items()):
        c_sids = [s for s in sids if output[s]["tier"] == "C"]
        if not c_sids:
            continue
        before_wave, after_wave, _ = key
        a(f"\n**Window {before_wave}→{after_wave}** ({output[c_sids[0]]['bracket']['gap_days_between_waves']}d gap, "
          f"n_both_waves={output[c_sids[0]]['n_both_waves_total']:,}) — {len(c_sids)} shocks share this window:")
        for sid in sorted(c_sids):
            e = output[sid]
            others = [s for s in c_sids if s != sid]
            a(f"  - `{sid}` ({e['shock_date']}) — co-occurring: {', '.join(others)}")
        a(f"\n  *Why unusable*: {output[c_sids[0]]['tier_rationale']}")
        a(f"\n  *Dominance rationale*: {output[c_sids[0]]['dominance_rationale']}")

    a("\n## Not bracketable\n")
    not_bracketed = sorted([sid for sid, v in assignments.items() if v is None])
    a(
        f"{len(not_bracketed)} of {len(assignments)} shocks in `configs/shocks.json` "
        f"do not fall inside any dated VOTER-panel bracket (before 2016, after "
        f"2020Nov, or in the 2012→2016 gap):\n"
    )
    for sid in not_bracketed:
        a(f"- `{sid}` ({shocks_by_id[sid]['date']})")

    a("\n## Per-shock, per-bloc results\n")
    for sid in sorted(output.keys()):
        e = output[sid]
        a(f"\n### `{sid}` — Tier {e['tier']} ({e['shock_date']}) — "
          f"attribution_confidence: **{e['attribution_confidence']}**, "
          f"solely_attributable: **{e['solely_attributable']}**\n")
        a(f"Bracket: {e['bracket']['before_wave']} ({e['bracket']['before_window'][0]} to {e['bracket']['before_window'][1]}) "
          f"→ {e['bracket']['after_wave']} ({e['bracket']['after_window'][0]} to {e['bracket']['after_window'][1]}), "
          f"gap {e['bracket']['gap_days_between_waves']} days. "
          f"n_before={e['n_before_wave_total']:,}, n_after={e['n_after_wave_total']:,}, "
          f"n_both={e['n_both_waves_total']:,}. Weight: `{e['weight_variable_used']}` ({e['weight_kind']}).")
        a(f"\n**Tier rationale**: {e['tier_rationale']}")
        if e["bracket"]["special_handling"]:
            a(f"\n**Special handling**: {e['bracket']['special_handling']}")
        if e["co_occurring_shocks_in_window"]:
            a(f"\n**Co-occurring shocks in this exact window**: {', '.join(e['co_occurring_shocks_in_window'])}")
        a(f"\n**Solely attributable to `{sid}` alone**: {e['solely_attributable']}" + (f" — {e['solely_attributable_reason']}" if e["solely_attributable_reason"] else ""))
        if e["dominant_shock_in_window"]:
            dom_self = " (this shock)" if e["is_dominant_in_window"] else f" (not this shock — `{sid}` is a co-occurring event, not the judged-dominant one)"
            a(f"\n**Plausibly dominant shock in this window**: `{e['dominant_shock_in_window']}`{dom_self}")
        if e["attributability_reason"]:
            a(f"\n**Not single-shock attributable**: {e['attributability_reason']}")
        a("\n| Bloc | n | raw Δpid7 | measured_delta | 95% CI | flag | weighted n | weighted measured_delta |")
        a("|---|---:|---:|---:|---|---|---:|---:|")
        for stratum_name, bloc_list in (("Race", RACE_BLOCS), ("Religion", RELIGION_BLOCS), ("Gender", GENDER_BLOCS)):
            for bloc in bloc_list:
                b = e["bloc"][bloc]
                flag = []
                if b["suppressed_flag"]:
                    flag.append("SUPPRESSED")
                if b["low_confidence_flag"]:
                    flag.append("low-N")
                flag_str = ",".join(flag) if flag else ""
                ci = f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]" if b["ci_low"] is not None else "n/a"
                raw = f"{b['raw_pid7_change']:+.4f}" if b["raw_pid7_change"] is not None else "null"
                md = f"{b['measured_delta']:+.4f}" if b["measured_delta"] is not None else "null"
                w = b.get("weighted")
                if w is not None and w.get("n", 0) > 0:
                    wn = w["n"]
                    wmd = f"{w['measured_delta']:+.4f}" if w["measured_delta"] is not None else "null"
                else:
                    wn = 0
                    wmd = "null"
                a(f"| {bloc} | {b['n']} | {raw} | {md} | {ci} | {flag_str} | {wn} | {wmd} |")

        vi = e["vote_intention"]
        if vi["available"]:
            a(f"\n**Vote intention** (`{vi['variable_before']}` → `{vi['variable_after']}`, "
              f"{vi['construct']}, n_both={vi['n_both_waves_total']:,}, weight `{vi['weight_variable_used']}`):\n")
            a("\n| Bloc | n | measured_delta (vote share) | 95% CI | flag |")
            a("|---|---:|---:|---|---|")
            for stratum_name, bloc_list in (("Race", RACE_BLOCS), ("Religion", RELIGION_BLOCS), ("Gender", GENDER_BLOCS)):
                for bloc in bloc_list:
                    b = vi["blocs"][bloc]
                    flag = []
                    if b["suppressed_flag"]:
                        flag.append("SUPPRESSED")
                    if b["low_confidence_flag"]:
                        flag.append("low-N")
                    flag_str = ",".join(flag) if flag else ""
                    ci = f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]" if b["ci_low"] is not None else "n/a"
                    md = f"{b['measured_delta']:+.4f}" if b["measured_delta"] is not None else "null"
                    a(f"| {bloc} | {b['n']} | {md} | {ci} | {flag_str} |")
        else:
            a(f"\n**Vote intention**: not available — {vi['reason']}")

    a("\n## Caveats\n")
    a(
        "1. **18 vs. 10 bracketed shocks, resolved**: the algorithm finds "
        f"**{sum(1 for v in assignments.values() if v is not None)} shocks** fall "
        "inside a dated bracket (the Step 0.3 prose's '10' was an inconsistent "
        "rounded headline, not a distinct methodological result — see "
        "`data/ground_truth/step1_1_loose_ends.md` §1 for the full reconciliation). "
        f"Of those 18, **{len(tier_a)+len(tier_b)} are Tier A/B** (`single_shock_attributable: true`) "
        f"and **{len(tier_c)} are Tier C** (`charlottesville_2017`, `daca_rescission_2017`, "
        "`las_vegas_shooting_2017`, `metoo_2017`, `parkland_shooting_2018` — see the "
        "Tier criteria section above). All 18 are still in this file with real "
        "computed numbers regardless of tier — tier and attributability are flags, "
        "not filters that delete data.\n"
    )
    a(
        "2. **Tier B deltas are joint-window deltas, not single-shock-attributed "
        "ones.** A delta computed for a shock sharing its window with 1-2 other "
        "events reflects the change over the whole window, not that shock in "
        "isolation. Treat co-occurring-shock lists above as a required caveat on "
        "every Tier B number, not decoration.\n"
    )
    a(
        "3. **Tier A shares one window across all three shocks, and "
        "`solely_attributable` is `false` for every one of the 18 bracketed "
        "shocks, Tier A included.** election_2020, trump_covid_diagnosis_2020, "
        "and ruth_bader_ginsburg_2020 all draw their 'after' measurement from the "
        "same 2020Nov wave. Even with RBG's before-window respondent-split, the "
        "three deltas are not independent measurements — a difference between "
        "them reflects the combined effect of whichever of the three events "
        "(plus ordinary campaign-season movement) happened between the relevant "
        "before-point and Nov 2020. No bracket in this panel has zero "
        "co-occurring configured shocks, so no shock in this file — regardless "
        "of tier — may be presented as \"this delta is caused by this shock "
        "alone.\" `attribution_confidence` (`high`/`moderate`/`none`) is the "
        "graded signal; `solely_attributable` is the honest strict answer, and "
        "it is `false` everywhere.\n"
    )
    a(
        "4. **The pid7→loyalty transform is a stated assumption, not a "
        "calibration.** See the transform section above; `raw_pid7_change` is "
        "always available for anyone who wants to apply a different mapping.\n"
    )
    a(
        "5. **Muslim-bloc (and to a lesser extent Asian- and Jewish-bloc) "
        "numbers are frequently suppressed** (`n < 30`) per Step 0.3's finding "
        "of 8-23 Muslim respondents per bracket. The real computed numbers are "
        "still in the JSON for anyone who wants them, flagged loudly — they are "
        "never silently dropped, but should not be trusted as point estimates.\n"
    )
    a(
        "6. **`other_gender` is suppressed everywhere** (`n=0` in every bracket) "
        "because the panel's gender variable is a strict binary in every wave — "
        "this cannot be fixed by reprocessing; it is a genuine gap in the source "
        "survey.\n"
    )
    a(
        "7. **Demographic backfill assumes stability across waves.** Race and "
        "gender are standard to backfill this way; religion backfill across "
        "gaps as wide as ~9 months (or, for a handful of respondents missing "
        "the before-wave question, back to the 2011 baseline) is a weaker "
        "assumption and could misclassify respondents who changed religious "
        "identification between waves.\n"
    )
    a(
        "8. **Weighting**: `weight_panel_{after_wave}` restricts the weighted "
        "estimate to a smaller recontact-weighted subsample than the full "
        "both-waves N (see the Weighting section) — a smaller weighted N is "
        "expected and correct, not a bug.\n"
    )

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
