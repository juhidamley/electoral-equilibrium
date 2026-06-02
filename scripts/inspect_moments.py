#!/usr/bin/env python3
"""Inspect estimated μ and Σ from the voter panel.

Usage
-----
    python scripts/inspect_moments.py \\
        [--config configs/base.json] \\
        [--party democrat|republican] \\
        [--threshold 0.05] \\
        [--out rawdata/moment_inspection.md]

Loads the voter panel via build_voter_panel(), runs estimate_moments(), then:
  (i)   Prints bloc-level mean vote shares alongside NEP/Espinosa benchmarks.
  (ii)  Flags discrepancies that exceed --threshold (default 5 pp) for the
        supervision check-in.
  (iii) Prints the 5×5 Σ covariance matrix with eigenvalue diagnostics.
  (iv)  Saves a full Markdown report to --out.

Benchmark sources
-----------------
Race/gender benchmarks: National Election Pool (NEP) exit-poll averages across
  competitive cycles (2000–2024), as cross-referenced in ESPINOSA.md.
Religion benchmarks: Pew Research Center NPORS + CES cumulative averages.
Electorate share reference: CLAUDE.md §Demographic architecture (canonical).

Flag thresholds
---------------
  WARN  : |estimated − benchmark| ∈ (threshold, 2×threshold)
  ALERT : |estimated − benchmark| > 2×threshold
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.kernels.data import build_voter_panel
from electoral.models.ml_baseline import (
    MomentEstimates,
    estimate_moments,
    ground_truth_winning_cycles,
)

# ── Espinosa / NEP / Pew benchmark vote shares ───────────────────────────────
# Democrat two-party vote share per bloc, averaged over competitive cycles
# (2000–2024) unless noted.  Sources documented in ESPINOSA.md.
#
# Race: NEP exit-poll averages (Espinosa cross-ref ESPINOSA.md §Q1.1–Q1.3).
#   african_american: NEP 2016=89%, 2020=87%, 2024=86% → mean ≈ 0.87
#   white: NEP 2016=37%, 2020=41%, 2024=39% → mean ≈ 0.39  (ESPINOSA.md §Q1.2)
#   latino: NEP 2016=66%, 2020=63%, 2024=58% → mean ≈ 0.62
#   asian: AAPI Data 2016=65%, 2020=72%, 2024=60% → mean ≈ 0.66
#   other_race: NEP average ≈ 0.58
#
# Religion: Pew NPORS + CES cumulative (ESPINOSA.md §Q2.4, §Q6.1).
#   evangelical: exit-poll avg 2004–2024 ≈ 0.23  (ESPINOSA.md: "near 24%")
#   catholic: NEP avg ≈ 0.52
#   protestant: NEP avg (mainline) ≈ 0.46
#   secular: NEP/Pew avg ≈ 0.68
#   jewish: exit-poll avg ≈ 0.72
#   muslim: exit-poll avg (limited data) ≈ 0.80
#   other_rel: Pew avg ≈ 0.58
#
# Gender: NEP averages 2000–2024.
#   women: avg ≈ 0.54
#   men: avg ≈ 0.43
#   other_gender: limited cycles, avg ≈ 0.78

_DEM_BENCHMARKS: dict[str, float] = {
    # Race
    "african_american": 0.87,
    "latino": 0.62,
    "asian": 0.66,
    "white": 0.39,
    "other_race": 0.58,
    # Religion
    "evangelical": 0.23,
    "catholic": 0.52,
    "protestant": 0.46,
    "secular": 0.68,
    "jewish": 0.72,
    "muslim": 0.80,
    "other_rel": 0.58,
    # Gender
    "women": 0.54,
    "men": 0.43,
    "other_gender": 0.78,
}

# Republican benchmarks = 1 − Democrat benchmark (two-party vote share).
_REP_BENCHMARKS: dict[str, float] = {k: 1.0 - v for k, v in _DEM_BENCHMARKS.items()}

# Canonical electorate share per bloc (CLAUDE.md §Demographic architecture).
# Used to spot-check that the panel's implied coverage is plausible.
_ELECTORATE_SHARE: dict[str, float] = {
    "african_american": 0.12,
    "latino": 0.11,
    "asian": 0.05,
    "white": 0.62,
    "other_race": 0.10,
    "evangelical": 0.24,
    "catholic": 0.21,
    "protestant": 0.13,
    "secular": 0.26,
    "jewish": 0.02,
    "muslim": 0.01,
    "other_rel": 0.13,
    "women": 0.52,
    "men": 0.47,
    "other_gender": 0.01,
}

_LINE = "=" * 78
_DASH = "-" * 78


def _flag(delta: float, threshold: float) -> str:
    ad = abs(delta)
    if ad > 2 * threshold:
        return "ALERT"
    if ad > threshold:
        return "WARN "
    return "ok   "


def _mu_table(moments: MomentEstimates, party: str, threshold: float) -> str:
    benchmarks = _DEM_BENCHMARKS if party == "democrat" else _REP_BENCHMARKS
    strata = (
        [("RACE", CANONICAL_RACES, moments.mu_race)]
        + [("RELIGION", CANONICAL_RELIGIONS, moments.mu_religion)]
        + [("GENDER", CANONICAL_GENDERS, moments.mu_gender)]
    )

    hdr = (
        f"{'Bloc':<22}  {'Stratum':<9}  {'Estimated':>9}  {'Benchmark':>9}"
        f"  {'Delta':>7}  {'Flag'}"
    )
    rows = [hdr, _DASH]
    flags: list[tuple[str, str, float, float, float]] = []

    for stratum_name, blocs, mu_dict in strata:
        rows.append(f"  — {stratum_name} —")
        for bloc in blocs:
            est = mu_dict.get(bloc, float("nan"))
            ref = benchmarks.get(bloc, float("nan"))
            if np.isnan(est):
                rows.append(f"  {bloc:<22}  {stratum_name:<9}  {'n/a':>9}  {ref:>9.4f}  {'n/a':>7}  MISSING")
                continue
            delta = est - ref
            flag = _flag(delta, threshold)
            sign = "+" if delta >= 0 else ""
            rows.append(
                f"  {bloc:<22}  {stratum_name:<9}  {est:>9.4f}  {ref:>9.4f}"
                f"  {sign}{delta:>6.4f}  {flag}"
            )
            if flag != "ok   ":
                flags.append((flag.strip(), bloc, est, ref, delta))

    rows.append(_DASH)
    return "\n".join(rows), flags


def _sigma_block(moments: MomentEstimates) -> str:
    blocs = moments.race_blocs
    S = moments.Sigma
    eigs = np.linalg.eigvalsh(S)
    cond = float(eigs.max() / max(eigs.min(), 1e-12))

    lbl_w = 20
    col_w = 12
    header = f"  {'':>{lbl_w}}" + "".join(f"{b:>{col_w}}" for b in blocs)
    rows = [header, "  " + _DASH]
    for i, bi in enumerate(blocs):
        row_str = f"  {bi:<{lbl_w}}"
        for j in range(len(blocs)):
            row_str += f"  {S[i, j]:>10.6f}"
        rows.append(row_str)
    rows.append("  " + _DASH)
    rows.append(f"  Eigenvalues : {np.array2string(eigs, precision=6, floatmode='fixed')}")
    rows.append(f"  Condition # : {cond:.2f}  ({'well-conditioned' if cond < 100 else 'ILL-CONDITIONED — check LedoitWolf shrinkage'})")
    rows.append(f"  Min eigenval: {eigs.min():.2e}  ({'PSD ok' if eigs.min() >= 0 else 'NOT PSD — psd_repair fired'})")
    return "\n".join(rows)


def _summary_flags(flags: list[tuple[str, str, float, float, float]], threshold: float) -> str:
    if not flags:
        return f"  All blocs within ±{threshold:.0%} of benchmark. No discrepancies to report."
    rows = [f"  {'Flag':<6}  {'Bloc':<22}  {'Estimated':>9}  {'Benchmark':>9}  {'Delta':>8}  Action"]
    rows.append("  " + _DASH)
    for flag, bloc, est, ref, delta in sorted(flags, key=lambda x: -abs(x[4])):
        action = "verify data sources" if flag == "ALERT" else "review coverage"
        sign = "+" if delta >= 0 else ""
        rows.append(
            f"  {flag:<6}  {bloc:<22}  {est:>9.4f}  {ref:>9.4f}  {sign}{delta:>7.4f}  {action}"
        )
    return "\n".join(rows)


def build_report(
    moments: MomentEstimates,
    panel_rows: int,
    party: str,
    config_path: str,
    threshold: float,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mu_table_str, flags = _mu_table(moments, party, threshold)

    lines: list[str] = [
        "# Moment Inspection Report — Electoral Equilibrium",
        "",
        f"Generated : {now}",
        f"Config    : {config_path}",
        f"Party     : {party}",
        f"Panel     : {panel_rows} rows",
        f"Threshold : ±{threshold:.0%} (WARN) / ±{2*threshold:.0%} (ALERT)",
        "",
        _LINE,
        "## (i) μ^(P) — Bloc Mean Vote Shares vs. Espinosa / NEP Benchmarks",
        _LINE,
        "",
        f"Party: {party.upper()}   |   Winning cycles: {moments.winning_cycles}",
        "",
        mu_table_str,
        "",
        "Benchmark sources:",
        "  Race    — National Election Pool (NEP) exit-poll averages 2000–2024",
        "            (ESPINOSA.md §Q1.1–Q1.3 cross-reference)",
        "  Religion— Pew NPORS + CES cumulative averages",
        "            (Espinosa: evangelical ≈ 24% of electorate, Dem share ≈ 23%)",
        "  Gender  — NEP exit-poll averages 2000–2024",
        "",
        _LINE,
        "## (ii) Discrepancy Flags — Supervision Check-In Items",
        _LINE,
        "",
        _summary_flags(flags, threshold),
        "",
        _LINE,
        "## (iii) Σ — 5×5 Race-Bloc Covariance Matrix",
        _LINE,
        "",
        "Estimated over ALL available cycles (not winning-cycles only).",
        "ddof=1 (unbiased); psd_repair applied if min eigenvalue < 0.",
        "",
        _sigma_block(moments),
        "",
        _LINE,
        "## (iv) Notes for Prof. Espinosa",
        _LINE,
        "",
        "Key questions raised by this inspection (see ESPINOSA.md for full context):",
        "",
        "  Q1.1  ANES 2020 has fully null weight column → equal-weighted; white Dem",
        "        estimate may be ~3 pp above NEP gold standard.",
        "  Q1.2  Systematic ANES–NEP gap for white voters (~4-5 pp); merged panel",
        "        moderates but does not eliminate it.",
        "  Q1.3  CES 2024 african_american 81.9% vs NEP 86%; merged ≈ 83.8%.",
        "  Q2.2  λ₁/λ₂/λ₃ are placeholder (50/30/20) — not empirically calibrated.",
        "  Q2.3  V_eq = 0.535 hardcoded; not yet derived from winning-cycle average.",
        "",
        "Winning cycles used for μ estimation:",
        f"  {moments.winning_cycles}",
        "",
        "Any ALERT-flagged blocs above should be discussed at the next check-in.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--party", choices=["democrat", "republican"], default=None,
                        help="Override party from config.")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Discrepancy threshold in vote-share units (default 0.05 = 5 pp).")
    parser.add_argument("--derive-from-panel", action="store_true",
                        help="Derive winning cycles from the panel rather than using the "
                             "certified election results table. Not recommended for real data "
                             "(produces four known misclassifications).")
    parser.add_argument("--out", default="rawdata/moment_inspection.md")
    args = parser.parse_args()

    logging.getLogger("electoral").setLevel(logging.WARNING)

    config = PipelineConfig.from_json(args.config)
    party: str = args.party or config.party

    _, panel = build_voter_panel(config)

    if args.derive_from_panel:
        winning = None  # panel-derived (legacy behaviour)
    else:
        winning = ground_truth_winning_cycles(party)  # type: ignore[arg-type]

    moments = estimate_moments(panel, party, winning_cycles=winning)  # type: ignore[arg-type]

    report = build_report(moments, len(panel), party, args.config, args.threshold)

    print(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
