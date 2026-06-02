#!/usr/bin/env python3
"""
Audit the cleaned voter panel: coverage, summary statistics, and outliers.

Usage
-----
    python scripts/audit_panel.py [--config configs/base.json] [--out rawdata/audit_report.txt]

Output
------
    (i)   Coverage matrix — bloc × cycle, showing which cells have data.
    (ii)  Vote-share summary statistics per bloc (N, mean, std, min, Q25, Q75, max).
    (iii) Outliers — rows with vote_share < 0.05 or > 0.95.

The report is printed to stdout and written to --out.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── project root on sys.path so the script is runnable from any directory ─────
sys.path.insert(0, str(Path(__file__).parents[1]))

from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.kernels.data import build_voter_panel

# ── constants ─────────────────────────────────────────────────────────────────

_ALL_BLOCS: list[str] = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)
_STRATA: dict[str, str] = (
    {b: "race" for b in CANONICAL_RACES}
    | {b: "religion" for b in CANONICAL_RELIGIONS}
    | {b: "gender" for b in CANONICAL_GENDERS}
)
# Short abbreviations used in table columns — must differ across all three strata.
_STRAT_ABBREV: dict[str, str] = {"race": "Ra", "religion": "Re", "gender": "G"}

_OUTLIER_LO = 0.05
_OUTLIER_HI = 0.95
_LINE_WIDTH = 78


# ── formatting helpers ────────────────────────────────────────────────────────


def _rule(char: str = "=") -> str:
    return char * _LINE_WIDTH


def _section(title: str) -> str:
    return f"\n{_rule()}\n{title}\n{_rule()}\n"


def _subsection(title: str) -> str:
    return f"\n{_rule('-')}\n{title}\n{_rule('-')}\n"


# ── (i) coverage matrix ───────────────────────────────────────────────────────


def _coverage_matrix(panel: pd.DataFrame) -> str:
    cycles = sorted(int(c) for c in panel["cycle"].dropna().unique())
    present: set[tuple[int, str]] = set(zip(panel["cycle"].astype(int), panel["bloc"]))

    col_w = 5  # width per cycle column
    lbl_w = 20  # bloc label width

    # Header: cycle years
    header = f"{'Bloc':<{lbl_w}}  Strat" + "".join(f"{c:>{col_w}}" for c in cycles) + "  Coverage"
    sep = _rule("-")

    rows = [header, sep]

    total_cells = 0
    filled_cells = 0

    for bloc in _ALL_BLOCS:
        stratum = _STRAT_ABBREV.get(_STRATA.get(bloc, "?"), "?")
        cells = ["   ●" if (c, bloc) in present else "   ·" for c in cycles]
        n = sum(1 for c in cycles if (c, bloc) in present)
        total_cells += len(cycles)
        filled_cells += n
        pct = f"{n}/{len(cycles)}"
        rows.append(f"{bloc:<{lbl_w}}  {stratum}    {''.join(cells)}  {pct}")

    rows.append(sep)

    # Footer: blocs-per-cycle totals
    footer = f"{'N blocs present':<{lbl_w}}  " + " " * 4
    for c in cycles:
        n_c = sum(1 for b in _ALL_BLOCS if (c, b) in present)
        footer += f"{n_c:>{col_w}}"
    rows.append(footer)

    rows.append("")
    rows.append(
        f"Total cells filled: {filled_cells} / {total_cells} "
        f"({100 * filled_cells / total_cells:.1f}%)"
    )
    rows.append("Key: ● = data present  · = missing  " "Strat: Ra=race  Re=religion  G=gender")

    return "\n".join(rows)


# ── (ii) summary statistics ───────────────────────────────────────────────────


def _summary_statistics(panel: pd.DataFrame) -> str:
    agg = (
        panel.groupby("bloc")["vote_share"]
        .agg(
            N="count",
            Mean="mean",
            Std="std",
            Min="min",
            Q25=lambda x: x.quantile(0.25),
            Q75=lambda x: x.quantile(0.75),
            Max="max",
        )
        .round(4)
    )

    hdr = (
        f"{'Bloc':<22}  Strat  {'N':>4}  {'Mean':>6}  {'Std':>6}"
        f"  {'Min':>6}  {'Q25':>6}  {'Q75':>6}  {'Max':>6}"
    )
    sep = _rule("-")
    rows = [hdr, sep]

    for bloc in _ALL_BLOCS:
        stratum = _STRATA.get(bloc, "?")
        if bloc not in agg.index:
            rows.append(
                f"{bloc:<22}  {stratum:<6}  {'0':>4}  {'--':>6}  {'--':>6}"
                f"  {'--':>6}  {'--':>6}  {'--':>6}  {'--':>6}  ← NO DATA"
            )
            continue
        s = agg.loc[bloc]
        std_str = f"{s['Std']:>6.4f}" if pd.notna(s["Std"]) else "  n/a"
        rows.append(
            f"{bloc:<22}  {stratum:<6}  {int(s['N']):>4}  {s['Mean']:>6.4f}"
            f"  {std_str}  {s['Min']:>6.4f}  {s['Q25']:>6.4f}"
            f"  {s['Q75']:>6.4f}  {s['Max']:>6.4f}"
        )

    return "\n".join(rows)


# ── (iii) outliers ────────────────────────────────────────────────────────────


def _outliers(panel: pd.DataFrame) -> str:
    mask = (panel["vote_share"] < _OUTLIER_LO) | (panel["vote_share"] > _OUTLIER_HI)
    out = panel[mask].sort_values(["vote_share", "cycle"]).copy()

    if out.empty:
        return f"No outliers found — all vote_share values in " f"[{_OUTLIER_LO}, {_OUTLIER_HI}].\n"

    hdr = f"{'Cycle':>5}  {'Bloc':<22}  Strat  {'vote_share':>10}" f"  {'Flag':<5}  Source"
    sep = _rule("-")
    rows = [
        f"Threshold: vote_share < {_OUTLIER_LO}  OR  vote_share > {_OUTLIER_HI}",
        f"Found {len(out)} outlier row(s).\n",
        hdr,
        sep,
    ]

    for _, row in out.iterrows():
        flag = "HIGH" if float(row["vote_share"]) > _OUTLIER_HI else "LOW "
        stratum = _STRATA.get(str(row["bloc"]), "?")
        note = _interpret_outlier(row)
        rows.append(
            f"{int(row['cycle']):>5}  {str(row['bloc']):<22}  {stratum:<6}"
            f"  {float(row['vote_share']):>10.4f}  {flag}   {str(row.get('source','?'))}  {note}"
        )

    rows.append(sep)
    rows.append(
        "\nNote: early cycles (1948–1968) have very small ANES subgroup samples — "
        "100% or 0% rates\n"
        "      are expected artefacts, not data errors. High african_american rates\n"
        "      post-1964 reflect genuine bloc loyalty."
    )
    return "\n".join(rows)


def _interpret_outlier(row: pd.Series) -> str:
    """Return a short diagnostic note for a flagged outlier."""
    cycle = int(row["cycle"])
    bloc = str(row["bloc"])
    vs = float(row["vote_share"])
    if vs >= 1.0:
        return "← n likely <10; all sampled respondents voted Dem"
    if vs <= 0.0:
        return "← n likely <10; all sampled respondents voted Rep"
    if bloc == "african_american" and vs > 0.90:
        return "← genuine bloc loyalty (expected)"
    if bloc == "other_gender" and cycle == 2016:
        return "← ANES 2016 n=11 only; unreliable"
    return ""


# ── report assembly ───────────────────────────────────────────────────────────


def build_report(panel: pd.DataFrame, config_path: str) -> str:
    n_cycles = panel["cycle"].nunique()
    n_blocs = panel["bloc"].nunique()
    n_rows = len(panel)
    sources = sorted({s for raw in panel["source"].dropna() for s in str(raw).split("+")})

    lines: list[str] = [
        _rule(),
        "VOTER PANEL AUDIT REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Config    : {config_path}",
        f"Panel     : {n_rows} rows  ·  {n_cycles} cycles  ·  {n_blocs} blocs",
        f"Sources   : {', '.join(sources)}",
        _rule(),
    ]

    lines.append(_section("(i) COVERAGE MATRIX   ● = data  · = missing"))
    lines.append(_coverage_matrix(panel))

    lines.append(_section("(ii) VOTE-SHARE SUMMARY STATISTICS PER BLOC"))
    lines.append(_summary_statistics(panel))

    lines.append(_section(f"(iii) OUTLIERS   vote_share < {_OUTLIER_LO}  OR  > {_OUTLIER_HI}"))
    lines.append(_outliers(panel))

    return "\n".join(lines) + "\n"


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--out", default="rawdata/audit_report.txt")
    args = parser.parse_args()

    # Suppress kernel INFO logging so only the report reaches stdout
    logging.getLogger("electoral").setLevel(logging.WARNING)

    config = PipelineConfig.from_json(args.config)
    _, panel = build_voter_panel(config)

    report = build_report(panel, args.config)

    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
