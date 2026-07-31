"""Score model shock-response predictions against the three ground-truth layers.

Ground-truth layers (Phase 1, Step 1.1-1.3), each a JSON file under
data/ground_truth/, built by scripts/extract_{panel,ces,exit_poll}_ground_truth.py:

  panel_deltas.json      -- VOTER panel, true within-person deltas. Tier A/B
                             (usable) or C (excluded). Per-cell trust flags:
                             trustworthy_for_direction, magnitude_usable.
  ces_deltas.json         -- CES annual cross-sections. Always lower fidelity
                             than the panel; never single-shock-attributable.
  exit_poll_deltas.json   -- NEP/CNN cross-cycle exit polls, election shocks
                             only. Always cross-cycle; never single-shock-
                             attributable.

CORE PRINCIPLE (carried over from the ground-truth build): CROSS-CHECK, DON'T
MERGE. A shock can have ground truth from more than one layer (e.g.
election_2020 has both a panel cell and an exit-poll cycle-delta). Scores are
always reported PER SOURCE, never pooled into one blended number -- pooling
would silently average a true within-person measurement with a confounded
cross-cycle one.

Predictions are scored in the schema electoral/artifacts.py's ShockResponseData
already uses: deltas_race / deltas_religion / deltas_gender dicts of
bloc_id -> float, keyed by shock_id, each carrying a "party" field. Positive
values in a "democrat"-party prediction mean a shift toward Democratic support
-- the same sign convention as ground truth's dem_loyalty-based measured_delta.
A "republican"-party prediction is sign-flipped before scoring (see
normalize_prediction()) since a positive Republican-perspective delta is a
NEGATIVE Democratic-loyalty change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

BLOC_TO_STRATUM: dict[str, str] = (
    {b: "race" for b in CANONICAL_RACES}
    | {b: "religion" for b in CANONICAL_RELIGIONS}
    | {b: "gender" for b in CANONICAL_GENDERS}
)
STRATUM_TO_PRED_KEY = {"race": "deltas_race", "religion": "deltas_religion", "gender": "deltas_gender"}


@dataclass(frozen=True)
class GroundTruthCell:
    """One (shock, bloc) ground-truth observation, normalized from whichever
    source produced it. `measured_delta` is always on the same dem_loyalty
    scale as a "democrat"-party ShockResponseData prediction.
    """

    source: str  # "panel" | "ces" | "exit_poll"
    shock_id: str
    bloc: str
    measured_delta: float
    fidelity_tier: str
    fidelity_rank: int
    eligible_for_direction: bool
    eligible_for_magnitude: bool
    ineligibility_reason: str | None = None
    window_id: str | None = None  # e.g. "2020Sep->2020Nov" (panel) -- lets callers
    # tell "N cells" apart from "N independent temporal windows"; several
    # shocks can and do share one window (see docs/ground_truth_layers.md),
    # so cell count alone overstates independence.


@dataclass
class CellScore:
    cell: GroundTruthCell
    predicted_delta: float
    abs_error: float
    sign_match: bool | None  # None if either side is exactly 0 (no sign to compare)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shock_id": self.cell.shock_id,
            "bloc": self.cell.bloc,
            "fidelity_tier": self.cell.fidelity_tier,
            "predicted_delta": self.predicted_delta,
            "measured_delta": self.cell.measured_delta,
            "abs_error": self.abs_error,
            "sign_match": self.sign_match,
            "eligible_for_direction": self.cell.eligible_for_direction,
            "eligible_for_magnitude": self.cell.eligible_for_magnitude,
        }


@dataclass
class RefusedCell:
    shock_id: str
    bloc: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"shock_id": self.shock_id, "bloc": self.bloc, "reason": self.reason}


@dataclass
class DirectionScore:
    """Scored ONLY on cells with eligible_for_direction=True (panel
    trustworthy_for_direction cells -- CES and exit-poll are never eligible,
    by construction of normalize_ces()/normalize_exit_poll()). This is
    HARD REQUIREMENT #3: report is not usable without reading `caveat`.
    """

    n_scored: int
    n_correct: int
    accuracy: float | None
    n_shocks: int
    n_windows: int | None  # distinct temporal brackets -- may be << n_shocks (several shocks can share one window)
    bloc_distribution: dict[str, int]
    top2_bloc_share: float | None  # fraction of scored cells from the 2 most common blocs
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_scored": self.n_scored, "n_correct": self.n_correct, "accuracy": self.accuracy,
            "n_shocks": self.n_shocks, "n_windows": self.n_windows, "bloc_distribution": self.bloc_distribution,
            "top2_bloc_share": self.top2_bloc_share, "caveat": self.caveat,
        }


@dataclass
class MagnitudeScore:
    """Scored on cells with eligible_for_magnitude=True. HARD REQUIREMENT #2:
    `calibration_ratio` = mean(|predicted|) / mean(|measured|) is the primary
    number for "is the model overstating magnitude" -- >1 means the model's
    predicted deltas are, on average, larger than what's actually measured.
    Real panel-measured deltas run roughly 0.0002-0.03 in magnitude
    (data/ground_truth/panel_deltas_summary.md); a model still calibrated to
    the old theoretical +-0.15 DELTA_BINS range would show a calibration_ratio
    on the order of 5-10x here. As of Step 2.1 (DECISIONS.md), the decode
    table itself is rescaled to +-0.03 -- a model/corpus still trained against
    the OLD scale (i.e. before the corpus regeneration step) would still show
    this same overstatement until that regeneration lands.
    `median_cellwise_ratio` is a robustness
    companion that excludes near-zero-measured cells (division blowup guard),
    not a replacement for the aggregate ratio.
    """

    n_scored: int
    mean_abs_error: float | None
    calibration_ratio: float | None
    median_cellwise_ratio: float | None
    n_cellwise_ratio_excluded_near_zero: int
    per_stratum_mae: dict[str, float]
    global_pearson_correlation: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_scored": self.n_scored, "mean_abs_error": self.mean_abs_error,
            "calibration_ratio": self.calibration_ratio,
            "median_cellwise_ratio": self.median_cellwise_ratio,
            "n_cellwise_ratio_excluded_near_zero": self.n_cellwise_ratio_excluded_near_zero,
            "per_stratum_mae": self.per_stratum_mae,
            "global_pearson_correlation": self.global_pearson_correlation,
        }


@dataclass
class ShockRankCorrelation:
    shock_id: str
    n_blocs: int
    spearman_rho: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"shock_id": self.shock_id, "n_blocs": self.n_blocs, "spearman_rho": self.spearman_rho}


@dataclass
class RankCorrelationScore:
    """HARD REQUIREMENT #4. Per shock (>=5 magnitude-usable scored blocs):
    Spearman correlation between |predicted_delta| and |measured_delta|
    across that shock's blocs -- does the model rank which blocs moved MOST
    correctly, independent of whether it gets any individual sign or
    magnitude right? Reported per-shock AND as a summary; never blended into
    the direction or magnitude scores (HARD REQUIREMENT #1).
    """

    per_shock: list[ShockRankCorrelation]
    n_shocks_eligible: int
    n_shocks_total_with_any_magnitude_cells: int
    mean_spearman_rho: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_shock": [s.to_dict() for s in self.per_shock],
            "n_shocks_eligible": self.n_shocks_eligible,
            "n_shocks_total_with_any_magnitude_cells": self.n_shocks_total_with_any_magnitude_cells,
            "mean_spearman_rho": self.mean_spearman_rho,
        }


@dataclass
class SourceReport:
    """One source's (panel | ces | exit_poll) full report. HARD REQUIREMENT
    #6: never construct a report that mixes cells from more than one source
    -- see score_all_sources(), which always calls score_source() once per
    source and never merges the inputs.
    """

    source: str
    fidelity_rank: int | None
    n_ground_truth_cells: int
    n_refused_no_prediction: int
    n_refused_ineligible: int
    direction: DirectionScore | None
    magnitude: MagnitudeScore | None
    rank_correlation: RankCorrelationScore
    refused_cells: list[RefusedCell] = field(default_factory=list)
    scored_cells: list[CellScore] = field(default_factory=list)

    @property
    def n_refused_total(self) -> int:
        return self.n_refused_no_prediction + self.n_refused_ineligible

    @property
    def n_scored_total(self) -> int:
        return len(self.scored_cells)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fidelity_rank": self.fidelity_rank,
            "n_ground_truth_cells": self.n_ground_truth_cells,
            "n_scored_total": self.n_scored_total,
            "n_refused_total": self.n_refused_total,
            "n_refused_no_prediction": self.n_refused_no_prediction,
            "n_refused_ineligible": self.n_refused_ineligible,
            "direction": self.direction.to_dict() if self.direction else None,
            "magnitude": self.magnitude.to_dict() if self.magnitude else None,
            "rank_correlation": self.rank_correlation.to_dict(),
            "refused_cells": [c.to_dict() for c in self.refused_cells],
            "scored_cells": [c.to_dict() for c in self.scored_cells],
        }


# ── normalization: each source's native JSON -> list[GroundTruthCell] ──────


def normalize_panel(panel_json: dict[str, Any]) -> list[GroundTruthCell]:
    """panel_deltas.json -> GroundTruthCell list. Eligibility comes directly
    from each bloc cell's `trust` sub-object (Step 1.1-1.2), which already
    encodes sign-stability, statistical significance, suppression, and
    single-shock-attributability -- see data/ground_truth/trustworthy_subset.md.
    """
    cells = []
    for shock_id, entry in panel_json.items():
        tier = entry.get("tier")
        bracket = entry.get("bracket", {})
        window_id = f"{bracket.get('before_wave')}->{bracket.get('after_wave')}" if bracket else None
        for bloc, b in entry.get("bloc", {}).items():
            measured = b.get("measured_delta")
            if measured is None:
                continue
            trust = b.get("trust", {})
            cells.append(
                GroundTruthCell(
                    source="panel",
                    shock_id=shock_id,
                    bloc=bloc,
                    measured_delta=float(measured),
                    fidelity_tier=f"panel_{tier}",
                    fidelity_rank=1 if tier == "A" else (2 if tier == "B" else 99),
                    eligible_for_direction=bool(trust.get("trustworthy_for_direction", False)),
                    eligible_for_magnitude=bool(trust.get("magnitude_usable", False)),
                    ineligibility_reason=None
                    if (trust.get("trustworthy_for_direction") or trust.get("magnitude_usable"))
                    else (b.get("suppression_reason") or "not sign-stable/significant -- see trust sub-object"),
                    window_id=window_id,
                )
            )
    return cells


def normalize_ces(ces_json: dict[str, Any]) -> list[GroundTruthCell]:
    """ces_deltas.json -> GroundTruthCell list. CES has no trust sub-object
    (it was never eligible for the panel's sign-stability/significance
    pipeline -- there's no "weighted vs unweighted panel-recontact" pair to
    check sign-stability against). Eligibility here is suppression-only:
    not suppressed and not low-confidence => usable for magnitude; CES is
    NEVER eligible for direction scoring (single_shock_attributable is
    always false, by design -- see ground_truth_layers.md).
    """
    cells = []
    for shock_id, entry in ces_json.items():
        tier = entry.get("fidelity_tier", "ces_annual_cross_section")
        rank = entry.get("fidelity_rank", 3)
        for bloc, b in entry.get("bloc", {}).items():
            measured = b.get("measured_delta")
            if measured is None:
                continue
            eligible_mag = not b.get("suppressed_flag", True) and not b.get("low_confidence_flag", False)
            cells.append(
                GroundTruthCell(
                    source="ces",
                    shock_id=shock_id,
                    bloc=bloc,
                    measured_delta=float(measured),
                    fidelity_tier=tier,
                    fidelity_rank=rank,
                    eligible_for_direction=False,
                    eligible_for_magnitude=eligible_mag,
                    ineligibility_reason=None if eligible_mag else (b.get("suppression_reason") or "suppressed/low-confidence"),
                )
            )
    return cells


def normalize_exit_poll(exit_poll_json: dict[str, Any]) -> list[GroundTruthCell]:
    """exit_poll_deltas.json -> GroundTruthCell list, from `cycle_deltas`
    only (the `years` block is single-year snapshots, not deltas -- nothing
    to score a shock's predicted CHANGE against). Never eligible for
    direction scoring, same reasoning as CES: cross-cycle windows bundle an
    entire campaign's worth of events, never attributable to one shock.
    """
    cells = []
    for _key, cd in exit_poll_json.get("cycle_deltas", {}).items():
        shock_id = cd.get("associated_shock_id")
        if not shock_id:
            continue
        tier = cd.get("fidelity_tier", "exit_poll_cross_cycle")
        rank = cd.get("fidelity_rank", 4)
        for bloc, b in cd.get("bloc", {}).items():
            if not b.get("available"):
                continue
            measured = b.get("measured_delta")
            if measured is None:
                continue
            has_ci = b.get("ci_low") is not None
            cells.append(
                GroundTruthCell(
                    source="exit_poll",
                    shock_id=shock_id,
                    bloc=bloc,
                    measured_delta=float(measured),
                    fidelity_tier=tier,
                    fidelity_rank=rank,
                    eligible_for_direction=False,
                    eligible_for_magnitude=True,
                    ineligibility_reason=None if has_ci else "no CI computable (missing sub_pct in scraped table) -- magnitude usable, uncertainty unknown",
                )
            )
    return cells


# ── prediction normalization ────────────────────────────────────────────────


def normalize_prediction(pred_entry: dict[str, Any], bloc: str) -> float | None:
    """Extract the predicted delta for one bloc from a ShockResponseData-shaped
    dict (either raw or {"data": {...}}-wrapped), applying the party sign-flip.
    Returns None if the bloc or the whole prediction is missing.
    """
    payload = pred_entry.get("data", pred_entry) if isinstance(pred_entry, dict) else None
    if payload is None:
        return None
    stratum = BLOC_TO_STRATUM.get(bloc)
    if stratum is None:
        return None
    key = STRATUM_TO_PRED_KEY[stratum]
    deltas = payload.get(key, {})
    if bloc not in deltas:
        return None
    value = float(deltas[bloc])
    party = payload.get("party", "democrat")
    if party == "republican":
        value = -value  # a Republican-perspective gain is a Democratic-loyalty loss
    elif party != "democrat":
        return None  # unrecognized party -- do not guess a sign
    return value


# ── scoring ──────────────────────────────────────────────────────────────────


def _sign(x: float, epsilon: float = 1e-9) -> int:
    if x > epsilon:
        return 1
    if x < -epsilon:
        return -1
    return 0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def _rank(values: list[float]) -> list[float]:
    """1-indexed ranks, average rank assigned on ties (the standard
    convention for Spearman's rho)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


MIN_BLOCS_FOR_RANK_CORRELATION = 5  # HARD REQUIREMENT #4


def _compute_direction_score(dir_scores: list[CellScore]) -> DirectionScore | None:
    if not dir_scores:
        return None
    n_correct = sum(1 for s in dir_scores if s.sign_match)
    n_shocks = len({s.cell.shock_id for s in dir_scores})
    window_ids = {s.cell.window_id for s in dir_scores if s.cell.window_id}
    n_windows = len(window_ids) if window_ids else None
    bloc_counts: dict[str, int] = {}
    for s in dir_scores:
        bloc_counts[s.cell.bloc] = bloc_counts.get(s.cell.bloc, 0) + 1
    top2 = sorted(bloc_counts.values(), reverse=True)[:2]
    top2_share = sum(top2) / len(dir_scores)
    top2_blocs = sorted(bloc_counts, key=lambda b: -bloc_counts[b])[:2]

    window_clause = (
        f" Those {n_shocks} shocks trace back to only {n_windows} independent temporal "
        f"window(s) -- several shocks share a before/after bracket, so this is NOT "
        f"{n_shocks} independent observations of model performance, only {n_windows}."
        if n_windows is not None else ""
    )
    caveat = (
        f"DIRECTION SCORE CAVEAT (read before citing this number): scored on only "
        f"{len(dir_scores)} cell(s) across {n_shocks} distinct shock(s) -- direction "
        f"scoring is restricted to panel cells flagged trustworthy_for_direction, which "
        f"is a small, non-random subset (see data/ground_truth/trustworthy_subset.md)."
        f"{window_clause} "
        f"{top2_share:.0%} of scored cells come from just 2 blocs ({', '.join(top2_blocs)}). "
        f"This accuracy number describes model performance on that narrow, "
        f"bloc-concentrated slice -- it does NOT generalize to blocs or shocks outside it, "
        f"and should never be reported without this caveat attached."
    )

    return DirectionScore(
        n_scored=len(dir_scores), n_correct=n_correct,
        accuracy=n_correct / len(dir_scores),
        n_shocks=n_shocks, n_windows=n_windows, bloc_distribution=bloc_counts,
        top2_bloc_share=top2_share, caveat=caveat,
    )


def _compute_magnitude_score(mag_scores: list[CellScore]) -> MagnitudeScore | None:
    if not mag_scores:
        return None
    mae = sum(s.abs_error for s in mag_scores) / len(mag_scores)

    mean_abs_pred = sum(abs(s.predicted_delta) for s in mag_scores) / len(mag_scores)
    mean_abs_true = sum(abs(s.cell.measured_delta) for s in mag_scores) / len(mag_scores)
    calibration_ratio = (mean_abs_pred / mean_abs_true) if mean_abs_true > 0 else None

    near_zero_epsilon = 1e-4  # ~0.01 percentage points; below this a cell-wise ratio is numerically unstable
    cellwise_ratios = []
    n_excluded = 0
    for s in mag_scores:
        if abs(s.cell.measured_delta) < near_zero_epsilon:
            n_excluded += 1
            continue
        cellwise_ratios.append(abs(s.predicted_delta) / abs(s.cell.measured_delta))
    median_ratio = None
    if cellwise_ratios:
        sr = sorted(cellwise_ratios)
        mid = len(sr) // 2
        median_ratio = sr[mid] if len(sr) % 2 else (sr[mid - 1] + sr[mid]) / 2

    strata: dict[str, list[float]] = {}
    for s in mag_scores:
        strata.setdefault(BLOC_TO_STRATUM[s.cell.bloc], []).append(s.abs_error)
    per_stratum_mae = {k: sum(v) / len(v) for k, v in strata.items()}

    correlation = None
    if len(mag_scores) >= 3:
        correlation = _pearson([s.predicted_delta for s in mag_scores], [s.cell.measured_delta for s in mag_scores])

    return MagnitudeScore(
        n_scored=len(mag_scores), mean_abs_error=mae,
        calibration_ratio=calibration_ratio, median_cellwise_ratio=median_ratio,
        n_cellwise_ratio_excluded_near_zero=n_excluded,
        per_stratum_mae=per_stratum_mae, global_pearson_correlation=correlation,
    )


def _compute_rank_correlation(mag_scores: list[CellScore]) -> RankCorrelationScore:
    by_shock: dict[str, list[CellScore]] = {}
    for s in mag_scores:
        by_shock.setdefault(s.cell.shock_id, []).append(s)

    per_shock = []
    for shock_id, cells in sorted(by_shock.items()):
        if len(cells) < MIN_BLOCS_FOR_RANK_CORRELATION:
            continue
        preds = [abs(c.predicted_delta) for c in cells]
        trues = [abs(c.cell.measured_delta) for c in cells]
        rho = _spearman(preds, trues)
        per_shock.append(ShockRankCorrelation(shock_id=shock_id, n_blocs=len(cells), spearman_rho=rho))

    valid_rhos = [s.spearman_rho for s in per_shock if s.spearman_rho is not None]
    mean_rho = sum(valid_rhos) / len(valid_rhos) if valid_rhos else None

    return RankCorrelationScore(
        per_shock=per_shock, n_shocks_eligible=len(per_shock),
        n_shocks_total_with_any_magnitude_cells=len(by_shock), mean_spearman_rho=mean_rho,
    )


def score_source(cells: list[GroundTruthCell], predictions: dict[str, Any]) -> SourceReport:
    """Score exactly ONE source's ground-truth cells against `predictions`.
    HARD REQUIREMENT #5: every ground-truth cell is accounted for as either
    scored (direction and/or magnitude) or refused, with a stated reason --
    nothing is silently dropped from the counts.
    """
    if not cells:
        return SourceReport(
            source="unknown", fidelity_rank=None, n_ground_truth_cells=0,
            n_refused_no_prediction=0, n_refused_ineligible=0,
            direction=None, magnitude=None,
            rank_correlation=RankCorrelationScore([], 0, 0, None),
        )

    source = cells[0].source
    fidelity_rank = cells[0].fidelity_rank
    n_refused_no_prediction = 0
    n_refused_ineligible = 0
    refused_cells: list[RefusedCell] = []
    scored_cells: list[CellScore] = []

    for cell in cells:
        pred_entry = predictions.get(cell.shock_id)
        pred_delta = normalize_prediction(pred_entry, cell.bloc) if pred_entry is not None else None
        if pred_delta is None:
            n_refused_no_prediction += 1
            reason = "no prediction for this shock_id" if pred_entry is None else "prediction present but missing this bloc, or unrecognized party"
            refused_cells.append(RefusedCell(cell.shock_id, cell.bloc, reason))
            continue
        if not (cell.eligible_for_direction or cell.eligible_for_magnitude):
            n_refused_ineligible += 1
            refused_cells.append(RefusedCell(
                cell.shock_id, cell.bloc,
                cell.ineligibility_reason or "ground-truth cell ineligible for both direction and magnitude scoring",
            ))
            continue

        sign_match = None
        if cell.eligible_for_direction:
            sign_match = _sign(pred_delta) == _sign(cell.measured_delta)
        abs_error = abs(pred_delta - cell.measured_delta)
        scored_cells.append(CellScore(cell=cell, predicted_delta=pred_delta, abs_error=abs_error, sign_match=sign_match))

    dir_scores = [s for s in scored_cells if s.cell.eligible_for_direction]
    mag_scores = [s for s in scored_cells if s.cell.eligible_for_magnitude]

    return SourceReport(
        source=source, fidelity_rank=fidelity_rank, n_ground_truth_cells=len(cells),
        n_refused_no_prediction=n_refused_no_prediction, n_refused_ineligible=n_refused_ineligible,
        direction=_compute_direction_score(dir_scores),
        magnitude=_compute_magnitude_score(mag_scores),
        rank_correlation=_compute_rank_correlation(mag_scores),
        refused_cells=refused_cells, scored_cells=scored_cells,
    )


def score_all_sources(
    predictions: dict[str, Any],
    panel_json: dict[str, Any] | None = None,
    ces_json: dict[str, Any] | None = None,
    exit_poll_json: dict[str, Any] | None = None,
) -> dict[str, SourceReport]:
    """Top-level entry point. Returns {"panel": SourceReport, "ces": SourceReport,
    "exit_poll": SourceReport} for whichever ground-truth files were supplied.

    HARD REQUIREMENT #1 (no combined score) and #6 (never pool layers): each
    SourceReport is computed independently, from that source's cells only,
    and nothing in this module ever averages or otherwise combines two
    SourceReports into one number. Read them side by side, not together.
    """
    reports = {}
    if panel_json is not None:
        reports["panel"] = score_source(normalize_panel(panel_json), predictions)
    if ces_json is not None:
        reports["ces"] = score_source(normalize_ces(ces_json), predictions)
    if exit_poll_json is not None:
        reports["exit_poll"] = score_source(normalize_exit_poll(exit_poll_json), predictions)
    return reports
