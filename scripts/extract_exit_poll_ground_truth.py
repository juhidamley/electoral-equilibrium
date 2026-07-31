#!/usr/bin/env python3
"""Extract per-bloc two-party vote share from NEP/CNN national exit polls.

Ground-truth extractor for Phase 1, Step 1.3, Part B. surveys/cnn_ssrs_polls/
holds four scraped exit-poll crosstabs (2004, 2016, 2020, 2024 -- NOT 2012,
which is absent from this folder) as (category, sub_category, sub_pct,
dem_pct, rep_pct) rows scraped from CNN election-result pages. The scrape has
real corruption (autoplay-video-ad text like "Audio Live TV" injected into
some category/sub_category labels, some rows duplicated, some sub_pct blank)
-- this script hand-maps each year's clean rows to the 15 canonical blocs
after direct manual inspection of every category in every file (see the
per-year ROW_MAP tables below, each row commented with what was found and any
exclusion reason). No auto-parsing of the corrupted category text was
attempted; that would silently propagate scrape errors.

Read-only w.r.t. survey files. Writes only:
    data/ground_truth/exit_poll_deltas.json

Usage:
    python scripts/extract_exit_poll_ground_truth.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

REPO_ROOT = Path(__file__).parents[1]
EXIT_POLL_DIR = Path("/Volumes/JUHIDRIVE/electoralData/surveys/cnn_ssrs_polls")
OUT_JSON = REPO_ROOT / "data" / "ground_truth" / "exit_poll_deltas.json"

RACE_BLOCS = list(CANONICAL_RACES)
RELIGION_BLOCS = list(CANONICAL_RELIGIONS)
GENDER_BLOCS = list(CANONICAL_GENDERS)
ALL_BLOCS = RACE_BLOCS + RELIGION_BLOCS + GENDER_BLOCS

FIDELITY_TIER = "exit_poll_cross_cycle"
FIDELITY_RANK = 4  # parallel to (not strictly below/above) ces_annual_cross_section=3 -- see ground_truth_layers.md
CI_Z = 1.96

# ── per-year row extraction, built from direct manual inspection of every ──
# category/sub_category in each nep_{year}_exit_poll.csv. Each entry:
#   bloc: (dem_pct, rep_pct, sub_pct_or_None, source_note)
# sub_pct is used only to approximate a subgroup N (=n_total * sub_pct/100)
# for a binomial CI; when sub_pct is missing/blank in the scrape, n and CI
# are left null (never guessed).

YEAR_DATA = {
    "2004": {
        "n_total": 13660,
        "candidates": {"dem": "Kerry", "rep": "Bush"},
        "rows": {
            "men": (44, 55, 46, "vote by gender / Male"),
            "women": (51, 48, 54, "vote by gender / Female"),
            "white": (41, 58, 77, "vote by race / White"),
            "african_american": (88, 11, 11, "vote by race / African-American"),
            "latino": (53, 44, 8, "vote by race / Latino"),
            "asian": (56, 44, 2, "vote by race / Asian"),
            "other_race": (54, 40, 2, "vote by race / Other"),
            "protestant": (40, 59, 54, "vote by religion / Protestant -- NOTE: combined Protestant+evangelical, not separable this year"),
            "catholic": (47, 52, 27, "vote by religion / Catholic"),
            "jewish": (74, 25, 3, "vote by religion / Jewish"),
            "other_rel": (74, 23, 7, "vote by religion / Other"),
            "secular": (67, 31, 10, "vote by religion / None"),
        },
        "unavailable": {
            "evangelical": "not separable from mainline protestant in this year's table (no white-evangelical-only row usable without confounding by race)",
            "muslim": "not present as a distinct row in the scraped table (sample too small to report)",
            "other_gender": "exit polls do not ask gender identity beyond male/female",
        },
    },
    "2016": {
        "n_total": 24558,
        "candidates": {"dem": "Clinton", "rep": "Trump"},
        "rows": {
            # gender rows were found under a corrupted category label ('national
            # president') that also contains age-bracket rows interleaved --
            # only the exact 'male'/'female' sub_category rows were used.
            "men": (41, 52, 47, "[corrupted category 'national president'] / male"),
            "women": (54, 41, 53, "[corrupted category 'national president'] / female"),
            "white": (37, 57, 71, "race / white"),
            "african_american": (89, 8, 12, "race / black"),
            # No clean overall 'Latino' row exists in this scrape -- only
            # gender-split subgroups. Derived as a weighted average of the two
            # (weights = their own reported sub_pct, which together are the
            # full latino population), NOT an estimate: both inputs are real
            # reported CNN figures.
            "latino": ("DERIVE_FROM", [(63, 32, 5, "race and gender / latino men"), (69, 25, 6, "race and gender / latino women")]),
            "protestant": (39, 56, 52, "religion / protestant (2nd occurrence, full-sample total) -- NOTE: combined Protestant+evangelical"),
            "catholic": (46, 50, 23, "religion / catholic"),
            "jewish": (71, 23, 3, "religion / jewish"),
        },
        "unavailable": {
            "asian": "no Asian row present anywhere in the scraped 2016 table (present in the original CNN exit poll but lost in this scrape)",
            "other_race": "no clean 'other race' row; the only aggregate available ('non-white', 74/21) double-counts black+latino+asian and cannot be used as a residual without an asian figure",
            "other_rel": "the scrape has a 'mormon' row (1%, 28/56) and an 'other christian' row with BLANK sub_pct -- cannot weight-average them without both subgroup sizes, so left unmapped rather than guessed",
            "secular": "the only candidate row ('n/a n/a n/a', 7%, 58/32) is ambiguous -- does not match the ~15% 'no religion' share reported in the actual 2016 NEP poll, so excluded as unreliable rather than assumed to mean 'secular'",
            "evangelical": "not separable from mainline protestant in this year's table",
            "muslim": "not present in the scraped table",
            "other_gender": "exit polls do not ask gender identity beyond male/female",
        },
    },
    "2020": {
        "n_total": 15590,
        "candidates": {"dem": "Biden", "rep": "Trump"},
        "rows": {
            # Gender rows found under a corrupted, disclaimer-prefixed category
            # label ("Subgroups indicated with an n/a ... Gender").
            "men": (45, 53, 48, "[corrupted category, disclaimer-prefixed 'Gender'] / Male"),
            "women": (57, 42, 52, "[corrupted category, disclaimer-prefixed 'Gender'] / Female"),
            "white": (41, 58, 67, "Race / White"),
            "african_american": (87, 12, 13, "Race / Black"),
            "latino": (65, 32, 13, "Race / Latino"),
            "asian": (61, 34, 4, "Race / Asian"),
            "other_race": (55, 41, 4, "Race / Other racial/ethnic groups"),
            "protestant": (39, 60, 43, "Religion / Protestant/Other Christian -- NOTE: combined Protestant+evangelical"),
            "secular": (65, 31, 22, "Religion / 'No religious affiliAautidoino Live TV' (corrupted label, cleaned to 'No religious affiliation' by context)"),
            "other_rel": (69, 29, 8, "Religion / Other religious affiliation"),
        },
        "unavailable": {
            "catholic": "2020's scraped table merges Catholic AND Jewish into one combined row ('Catholic Jewish', 25%, 52/47) -- cannot be disaggregated from this source; left null rather than assigning the combined figure to either bloc",
            "jewish": "same reason as catholic -- merged in the 2020 scrape",
            "evangelical": "not separable from mainline protestant in this year's table",
            "muslim": "not present in the scraped table",
            "other_gender": "exit polls do not ask gender identity beyond male/female",
        },
    },
    "2024": {
        "n_total": 22966,
        "candidates": {"dem": "Harris", "rep": "Trump"},
        "rows": {
            "men": (43, 55, 47, "Gender / Male"),
            "women": (53, 45, 53, "Gender / Female"),
            "white": (42, 57, 71, "Race / White"),
            "african_american": (86, 13, 11, "Race / Black"),
            "latino": (51, 46, 11, "Race / Latino"),
            "asian": (55, 40, 3, "Race / Asian"),
            # finer 6-category race breakdown available this year (with Native
            # American separate) -- weighted-average Native American + 'other
            # racial/e' into other_race using their own reported sub_pct.
            "other_race": ("DERIVE_FROM", [(31, 68, 1, "Race / Native American"), (44, 52, 2, "Race / Oth racial/e")]),
            "protestant": (36, 63, 43, "Religion / Protestant/Other Christian -- NOTE: combined Protestant+evangelical"),
            "catholic": (39, 59, None, "Religion / Catholic (sub_pct blank in scrape)"),
            "jewish": (78, 22, None, "Religion / Jewish (sub_pct blank in scrape)"),
            "other_rel": (61, 34, 10, "Religion / Other religious affiliation"),
            "secular": (71, 27, 24, "Religion / 'N relig affili' (corrupted label, cleaned to 'No religious affiliation' by context)"),
        },
        "unavailable": {
            "evangelical": "not separable from mainline protestant in this year's table",
            "muslim": "not present in the scraped table",
            "other_gender": "exit polls do not ask gender identity beyond male/female",
        },
    },
}

YEARS = ["2004", "2016", "2020", "2024"]

# Election shocks these years correspond to, and the adjacent-cycle pairs to
# compute cross-cycle deltas for.
ELECTION_SHOCK_BY_YEAR = {
    "2016": "trump_election_upset_2016",
    "2020": "election_2020",
    "2024": "election_2024",
}
CYCLE_PAIRS = [("2004", "2016"), ("2016", "2020"), ("2020", "2024")]


def resolve_bloc_row(year_data: dict, bloc: str):
    """Returns (dem_pct, rep_pct, n_or_None, note) or None if unavailable."""
    rows = year_data["rows"]
    if bloc not in rows:
        return None
    entry = rows[bloc]
    if entry[0] == "DERIVE_FROM":
        parts = entry[1]
        total_w = sum(p[2] for p in parts)
        dem = sum(p[0] * p[2] for p in parts) / total_w
        rep = sum(p[1] * p[2] for p in parts) / total_w
        note = "derived as sub_pct-weighted average of: " + "; ".join(f"{p[3]} ({p[2]}%, dem={p[0]}/rep={p[1]})" for p in parts)
        n = int(round(year_data["n_total"] * total_w / 100.0))
        return (dem, rep, n, note)
    dem_pct, rep_pct, sub_pct, note = entry
    n = int(round(year_data["n_total"] * sub_pct / 100.0)) if sub_pct is not None else None
    return (dem_pct, rep_pct, n, note)


def two_party_share_stats(dem_pct: float, rep_pct: float, n: int | None):
    """dem_pct/rep_pct are percentages (0-100) of the two named candidates
    among the subgroup; converts to a [0,1] two-party Democratic share and an
    approximate binomial CI using n = n_total * sub_pct (when available).
    CAVEAT (stated in the output, not hidden): this binomial approximation
    ignores exit-poll design effects (clustering by precinct) which real NEP
    margins of error account for -- true SEs are almost certainly wider than
    this formula produces. Used only as a rough, clearly-labeled indicator,
    not a precise interval.
    """
    total = dem_pct + rep_pct
    if total <= 0:
        return {"n": None, "dem_two_party_share": None, "ci_low": None, "ci_high": None, "reason": "dem_pct+rep_pct <= 0"}
    share = dem_pct / total
    out = {"dem_two_party_share": share, "raw_dem_pct": dem_pct, "raw_rep_pct": rep_pct, "n": n}
    if n is None or n <= 0:
        out["ci_low"], out["ci_high"] = None, None
        out["reason"] = "sub_pct not reported in scraped table -- subgroup N (and therefore a CI) is not computable"
    else:
        se = math.sqrt(share * (1 - share) / n)
        out["ci_low"], out["ci_high"] = max(0.0, share - CI_Z * se), min(1.0, share + CI_Z * se)
        out["reason"] = None
    return out


def build_bloc_cell(year: str, bloc: str) -> dict:
    yd = YEAR_DATA[year]
    resolved = resolve_bloc_row(yd, bloc)
    if resolved is None:
        reason = yd["unavailable"].get(bloc, "no row identified in the scraped table for this year")
        return {
            "available": False, "reason": reason,
            "n": None, "dem_two_party_share": None, "raw_dem_pct": None, "raw_rep_pct": None,
            "ci_low": None, "ci_high": None, "source_note": None,
        }
    dem_pct, rep_pct, n, note = resolved
    stats = two_party_share_stats(dem_pct, rep_pct, n)
    stats["available"] = True
    stats["source_note"] = note
    return stats


def main() -> None:
    print("Building per-bloc two-party vote share from NEP/CNN exit poll crosstabs...")
    print(f"Years available: {YEARS} (2012 not present in {EXIT_POLL_DIR})")

    per_year_blocs = {y: {b: build_bloc_cell(y, b) for b in ALL_BLOCS} for y in YEARS}

    for y in YEARS:
        avail = [b for b in ALL_BLOCS if per_year_blocs[y][b]["available"]]
        miss = [b for b in ALL_BLOCS if not per_year_blocs[y][b]["available"]]
        print(f"  {y}: {len(avail)}/{len(ALL_BLOCS)} blocs available. Missing: {miss}")

    output = {"years": per_year_blocs, "cycle_deltas": {}}

    for before_y, after_y in CYCLE_PAIRS:
        key = f"{before_y}_to_{after_y}"
        shock_id = ELECTION_SHOCK_BY_YEAR.get(after_y)
        bloc_deltas = {}
        for bloc in ALL_BLOCS:
            b, a = per_year_blocs[before_y][bloc], per_year_blocs[after_y][bloc]
            if not (b["available"] and a["available"]):
                reasons = []
                if not b["available"]:
                    reasons.append(f"{before_y}: {b['reason']}")
                if not a["available"]:
                    reasons.append(f"{after_y}: {a['reason']}")
                bloc_deltas[bloc] = {
                    "available": False, "reason": "; ".join(reasons),
                    "measured_delta": None, "ci_low": None, "ci_high": None,
                }
                continue
            delta = a["dem_two_party_share"] - b["dem_two_party_share"]
            # approximate two-independent-samples CI on the share difference,
            # using the same binomial-approximation caveat as above
            ci = None
            if b["n"] and a["n"]:
                se_b = math.sqrt(b["dem_two_party_share"] * (1 - b["dem_two_party_share"]) / b["n"])
                se_a = math.sqrt(a["dem_two_party_share"] * (1 - a["dem_two_party_share"]) / a["n"])
                se = math.sqrt(se_b**2 + se_a**2)
                ci = (delta - CI_Z * se, delta + CI_Z * se)
            bloc_deltas[bloc] = {
                "available": True,
                "before_share": b["dem_two_party_share"], "after_share": a["dem_two_party_share"],
                "measured_delta": delta,
                "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
                "n_before": b["n"], "n_after": a["n"],
            }
        output["cycle_deltas"][key] = {
            "before_year": before_y, "after_year": after_y,
            "associated_shock_id": shock_id,
            "fidelity_tier": FIDELITY_TIER,
            "fidelity_rank": FIDELITY_RANK,
            "gap_years": int(after_y) - int(before_y),
            "single_shock_attributable": False,
            "attributability_reason": (
                f"Cross-cycle exit-poll comparison spans {int(after_y)-int(before_y)} years between "
                f"elections -- captures the FULL cycle (campaign, primaries, all intervening news), not "
                "any single configured shock. Even the associated election shock is only one of very "
                "many events in this window. Never treat as single-shock-attributable."
            ),
            "bloc": bloc_deltas,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
