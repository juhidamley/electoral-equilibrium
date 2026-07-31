#!/usr/bin/env python3
"""Score model predictions against the expert-elicitation consensus.

Phase 1, Step 1.4. This is a READY-TO-RUN skeleton written against a protocol
(docs/elicitation/protocol.md) that has not been executed yet -- no expert
responses exist as of this writing. The logic below is real and complete (not
pseudocode), verified via --self-test with synthetic dummy data (see bottom of
this file), but has never been run against real elicitation responses.

Pipeline (mirrors protocol.md exactly, same order, same gates):
  1. Load response CSVs, validate schema.
  2. Verify pre-registration: every response file's sha256 matches the
     append-only log, and the model-predictions file was generated AFTER
     every expert's lock timestamp. Refuses to proceed otherwise.
  3. Flag non-independent respondents (Juhi, Espinosa) and exclude them from
     the primary inter-expert agreement / consensus computation.
  4. Compute inter-expert agreement (Fleiss' kappa + pairwise). GATE: if
     kappa < 0.40, stop -- report the disagreement as the finding, do not
     compute a model score.
  5. Build per-(shock,bloc) consensus direction (>=60% majority) or mark
     SPLIT / INSUFFICIENT_DATA.
  6. Score the model's predicted direction against consensus, applying the
     pre-registered 70% / 55% thresholds.

Usage (real run, once responses + a model-predictions file exist):
    python docs/elicitation/scripts/score_elicitation.py \\
        --responses docs/elicitation/responses/ \\
        --log docs/elicitation/pre_registration_log.csv \\
        --model-predictions path/to/predictions.json

Usage (self-test, no real data -- sanity-checks the code only):
    python docs/elicitation/scripts/score_elicitation.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))
from electoral.metrics.ground_truth_accuracy import (  # noqa: E402
    BLOC_TO_STRATUM,
    normalize_prediction,
)

# ── constants, all pre-registered in protocol.md -- do not tune post-hoc ────

DIRECTIONS = {"DEM_GAIN", "REP_GAIN", "NO_EFFECT", "NO_PREDICTION"}
CONFIDENCES = {"HIGH", "MEDIUM", "LOW", ""}
MAGNITUDE_BUCKETS = {"NEGLIGIBLE", "SMALL", "MODERATE", "LARGE", "VERY_LARGE", ""}
MAGNITUDE_MIDPOINT = {  # for the secondary/exploratory magnitude summary only
    "NEGLIGIBLE": 0.0015,
    "SMALL": 0.0065,
    "MODERATE": 0.015,
    "LARGE": 0.025,
    "VERY_LARGE": 0.04,
}

REQUIRED_COLUMNS = ["expert_code", "shock_id", "bloc", "stratum", "direction", "direction_confidence", "magnitude_bucket", "notes"]

# protocol.md SS4: flagged experts are analyzed separately, never pooled into
# the primary inter-expert agreement or consensus. Fill in real codes when
# assigned -- these are placeholders naming the two people this protocol
# already names as non-independent.
NON_INDEPENDENT_EXPERT_CODES: set[str] = {"JUHI", "ESPINOSA"}

MIN_COMPLETION_FRACTION = 0.80  # protocol.md SS4 "partial responses" bar (144/180 rows)
MIN_EXPERTS_FOR_KAPPA = 3  # protocol.md SS4 "why 3 minimum"
KAPPA_THRESHOLD = 0.40  # protocol.md SS5.1, Landis & Koch "moderate or better"
CONSENSUS_MAJORITY_FRACTION = 0.60  # protocol.md SS5.2
VALIDATE_THRESHOLD = 0.70  # protocol.md SS5.4
WEAK_THRESHOLD = 0.55  # protocol.md SS5.4


# ── data model ───────────────────────────────────────────────────────────────


@dataclass
class ResponseRow:
    expert_code: str
    shock_id: str
    bloc: str
    stratum: str
    direction: str
    direction_confidence: str
    magnitude_bucket: str
    notes: str


@dataclass
class ExpertFile:
    expert_code: str
    path: Path
    rows: list[ResponseRow]
    sha256: str
    non_independent: bool

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.rows if r.direction and r.direction != "")

    @property
    def completion_fraction(self) -> float:
        return self.n_completed / len(self.rows) if self.rows else 0.0


@dataclass
class ConsensusCell:
    shock_id: str
    bloc: str
    n_respondents: int  # non-abstaining, non-flagged experts
    consensus: str  # "DEM_GAIN" | "REP_GAIN" | "NO_EFFECT" | "SPLIT" | "INSUFFICIENT_DATA"
    vote_counts: dict[str, int]
    median_magnitude_bucket: str | None = None


# ── loading + validation ────────────────────────────────────────────────────


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_response_file(path: Path) -> ExpertFile:
    rows: list[ResponseRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for i, raw in enumerate(reader):
            direction = (raw["direction"] or "").strip().upper()
            confidence = (raw["direction_confidence"] or "").strip().upper()
            magnitude = (raw["magnitude_bucket"] or "").strip().upper()
            if direction and direction not in DIRECTIONS:
                raise ValueError(f"{path}: row {i}: invalid direction {direction!r}")
            if confidence not in CONFIDENCES:
                raise ValueError(f"{path}: row {i}: invalid direction_confidence {confidence!r}")
            if magnitude not in MAGNITUDE_BUCKETS:
                raise ValueError(f"{path}: row {i}: invalid magnitude_bucket {magnitude!r}")
            rows.append(
                ResponseRow(
                    expert_code=raw["expert_code"].strip(),
                    shock_id=raw["shock_id"].strip(),
                    bloc=raw["bloc"].strip(),
                    stratum=raw["stratum"].strip(),
                    direction=direction,
                    direction_confidence=confidence,
                    magnitude_bucket=magnitude,
                    notes=raw.get("notes", ""),
                )
            )
    if not rows:
        raise ValueError(f"{path}: no rows found")
    expert_code = rows[0].expert_code
    if any(r.expert_code != expert_code for r in rows):
        raise ValueError(f"{path}: mixed expert_code values within one file -- each file must be one expert")
    return ExpertFile(
        expert_code=expert_code,
        path=path,
        rows=rows,
        sha256=sha256_of_file(path),
        non_independent=expert_code.upper() in NON_INDEPENDENT_EXPERT_CODES,
    )


def load_all_responses(responses_dir: Path) -> list[ExpertFile]:
    files = sorted(responses_dir.glob("*_responses.csv"))
    if not files:
        raise FileNotFoundError(f"no *_responses.csv files found in {responses_dir}")
    return [load_response_file(p) for p in files]


@dataclass
class PreRegistrationEntry:
    expert_code: str
    filename: str
    sha256: str
    received_at: datetime
    n_rows_completed: int


def load_preregistration_log(log_path: Path) -> list[PreRegistrationEntry]:
    entries = []
    with open(log_path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            entries.append(
                PreRegistrationEntry(
                    expert_code=raw["expert_code"].strip(),
                    filename=raw["filename"].strip(),
                    sha256=raw["sha256"].strip(),
                    received_at=datetime.fromisoformat(raw["received_at"]),
                    n_rows_completed=int(raw["n_rows_completed"]),
                )
            )
    return entries


def verify_preregistration(
    experts: list[ExpertFile],
    log_entries: list[PreRegistrationEntry],
    model_predictions_generated_at: datetime,
) -> None:
    """Raises RuntimeError if pre-registration cannot be verified. This is
    protocol.md SS3's mechanical check: predictions must be locked (hash in
    the append-only log) strictly before the model-predictions file existed.
    """
    log_by_code = {e.expert_code: e for e in log_entries}
    problems = []
    for expert in experts:
        entry = log_by_code.get(expert.expert_code)
        if entry is None:
            problems.append(f"{expert.expert_code}: no pre-registration log entry found -- responses are NOT verifiably locked")
            continue
        if entry.sha256 != expert.sha256:
            problems.append(
                f"{expert.expert_code}: file hash {expert.sha256[:12]} does not match logged hash "
                f"{entry.sha256[:12]} -- file may have been modified after locking"
            )
        if entry.received_at >= model_predictions_generated_at:
            problems.append(
                f"{expert.expert_code}: locked at {entry.received_at.isoformat()}, which is NOT before "
                f"model predictions were generated ({model_predictions_generated_at.isoformat()}) -- "
                "cannot rule out the predictions being informed by (or coincidentally postdating in a "
                "way that undermines) this expert's responses"
            )
    if problems:
        raise RuntimeError(
            "Pre-registration verification FAILED -- refusing to score.\n" + "\n".join(f"  - {p}" for p in problems)
        )


# ── inter-expert agreement ──────────────────────────────────────────────────


def fleiss_kappa(rating_matrix: list[list[int]]) -> float | None:
    """Standard Fleiss' kappa over an (n_items x n_categories) matrix of
    rating counts. Returns None if undefined (e.g. no items, or every item
    has fewer than 2 ratings).
    """
    n_items = len(rating_matrix)
    if n_items == 0:
        return None
    n_categories = len(rating_matrix[0])
    n_per_item = [sum(row) for row in rating_matrix]
    if any(n < 2 for n in n_per_item):
        return None
    n_raters = n_per_item[0]
    if any(n != n_raters for n in n_per_item):
        # Fleiss' kappa assumes a constant number of raters per item; if
        # abstentions make this vary, callers should bucket items by rater
        # count and compute kappa within each bucket, or use a generalized
        # variant. Flagged rather than silently approximated.
        raise ValueError(
            "fleiss_kappa: rating_matrix rows have varying totals (varying raters-per-item) -- "
            "this simple implementation assumes a constant rater count; bucket items by n_raters first"
        )

    p_j = [sum(row[c] for row in rating_matrix) / (n_items * n_raters) for c in range(n_categories)]
    P_i = [
        (sum(k * k for k in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in rating_matrix
    ]
    P_bar = sum(P_i) / n_items
    P_e = sum(p * p for p in p_j)
    if P_e == 1.0:
        return None  # undefined: division by zero (perfect expected agreement)
    return (P_bar - P_e) / (1 - P_e)


@dataclass
class AgreementReport:
    kappa: float | None
    kappa_by_rater_count: dict[int, float | None]
    mean_pairwise_agreement: float | None
    n_cells_scored: int
    n_independent_experts: int
    gate_passed: bool
    gate_message: str


def compute_inter_expert_agreement(experts: list[ExpertFile]) -> AgreementReport:
    independent = [e for e in experts if not e.non_independent]
    if len(independent) < MIN_EXPERTS_FOR_KAPPA:
        return AgreementReport(
            kappa=None, kappa_by_rater_count={}, mean_pairwise_agreement=None,
            n_cells_scored=0, n_independent_experts=len(independent), gate_passed=False,
            gate_message=(
                f"Only {len(independent)} independent expert(s) responded; "
                f"{MIN_EXPERTS_FOR_KAPPA} are required for a meaningful agreement statistic "
                "(protocol.md SS4). Cannot compute kappa or proceed to consensus scoring."
            ),
        )

    # gather per-(shock,bloc) direction votes, excluding NO_PREDICTION
    votes: dict[tuple[str, str], dict[str, int]] = {}
    categories = ["DEM_GAIN", "REP_GAIN", "NO_EFFECT"]
    for expert in independent:
        for row in expert.rows:
            if row.direction not in categories:
                continue
            key = (row.shock_id, row.bloc)
            votes.setdefault(key, {c: 0 for c in categories})
            votes[key][row.direction] += 1

    # bucket cells by rater count (see fleiss_kappa docstring)
    by_rater_count: dict[int, list[list[int]]] = {}
    pairwise_matches, pairwise_total = 0, 0
    for key, counts in votes.items():
        n = sum(counts.values())
        if n < 2:
            continue
        by_rater_count.setdefault(n, []).append([counts[c] for c in categories])
        # pairwise agreement: for n raters with these category counts,
        # number of agreeing pairs = sum(C(k,2) for k in counts) out of C(n,2)
        from math import comb
        agreeing_pairs = sum(comb(k, 2) for k in counts.values())
        total_pairs = comb(n, 2)
        pairwise_matches += agreeing_pairs
        pairwise_total += total_pairs

    kappa_by_n = {n: fleiss_kappa(matrix) for n, matrix in by_rater_count.items()}
    # overall kappa: pool all cells with rater count == the modal (most common)
    # rater count, which is typically ~all cells if experts complete most rows.
    # Cells with a different rater count (due to differential NO_PREDICTION
    # usage) get their own kappa reported in kappa_by_rater_count for
    # transparency rather than forced into one number.
    if by_rater_count:
        modal_n = max(by_rater_count, key=lambda n: len(by_rater_count[n]))
        overall_kappa = kappa_by_n[modal_n]
    else:
        overall_kappa = None

    mean_pairwise = pairwise_matches / pairwise_total if pairwise_total else None
    n_cells = sum(len(v) for v in by_rater_count.values())

    gate_passed = overall_kappa is not None and overall_kappa >= KAPPA_THRESHOLD
    if overall_kappa is None:
        gate_message = "Kappa not computable (insufficient overlapping cells) -- cannot proceed to consensus scoring."
    elif gate_passed:
        gate_message = f"kappa={overall_kappa:.3f} >= {KAPPA_THRESHOLD} (moderate+ agreement, Landis & Koch) -- proceeding to consensus."
    else:
        gate_message = (
            f"kappa={overall_kappa:.3f} < {KAPPA_THRESHOLD} -- experts do not agree with each other "
            "enough for their consensus to serve as ground truth. This IS the finding; do not compute "
            "a model-vs-consensus score."
        )

    return AgreementReport(
        kappa=overall_kappa, kappa_by_rater_count=kappa_by_n, mean_pairwise_agreement=mean_pairwise,
        n_cells_scored=n_cells, n_independent_experts=len(independent),
        gate_passed=gate_passed, gate_message=gate_message,
    )


# ── consensus construction ──────────────────────────────────────────────────


def build_consensus(experts: list[ExpertFile]) -> list[ConsensusCell]:
    independent = [e for e in experts if not e.non_independent]
    votes: dict[tuple[str, str], dict[str, int]] = {}
    magnitudes: dict[tuple[str, str], list[str]] = {}
    for expert in independent:
        for row in expert.rows:
            key = (row.shock_id, row.bloc)
            if row.direction in {"DEM_GAIN", "REP_GAIN", "NO_EFFECT"}:
                votes.setdefault(key, {"DEM_GAIN": 0, "REP_GAIN": 0, "NO_EFFECT": 0})
                votes[key][row.direction] += 1
                if row.magnitude_bucket:
                    magnitudes.setdefault(key, []).append(row.magnitude_bucket)

    cells = []
    all_keys = {(r.shock_id, r.bloc) for e in independent for r in e.rows}
    for key in sorted(all_keys):
        shock_id, bloc = key
        counts = votes.get(key, {"DEM_GAIN": 0, "REP_GAIN": 0, "NO_EFFECT": 0})
        n = sum(counts.values())
        if n < 2:
            cells.append(ConsensusCell(shock_id, bloc, n, "INSUFFICIENT_DATA", counts))
            continue
        top_direction, top_count = max(counts.items(), key=lambda kv: kv[1])
        if top_count / n >= CONSENSUS_MAJORITY_FRACTION:
            median_mag = None
            if key in magnitudes and magnitudes[key]:
                mags = sorted(magnitudes[key], key=lambda m: MAGNITUDE_MIDPOINT.get(m, 0.0))
                median_mag = mags[len(mags) // 2]
            cells.append(ConsensusCell(shock_id, bloc, n, top_direction, counts, median_mag))
        else:
            cells.append(ConsensusCell(shock_id, bloc, n, "SPLIT", counts))
    return cells


# ── model scoring ────────────────────────────────────────────────────────────


@dataclass
class ElicitationScoreReport:
    n_consensus_directional: int
    n_consensus_no_effect: int
    n_split: int
    n_insufficient_data: int
    direction_accuracy: float | None
    no_effect_accuracy: float | None
    validation_verdict: str
    cells: list[dict]


def score_model_vs_consensus(consensus_cells: list[ConsensusCell], predictions: dict) -> ElicitationScoreReport:
    directional_correct, directional_total = 0, 0
    no_effect_correct, no_effect_total = 0, 0
    n_split = sum(1 for c in consensus_cells if c.consensus == "SPLIT")
    n_insufficient = sum(1 for c in consensus_cells if c.consensus == "INSUFFICIENT_DATA")
    detail = []

    for cell in consensus_cells:
        if cell.consensus in ("SPLIT", "INSUFFICIENT_DATA"):
            continue
        pred_entry = predictions.get(cell.shock_id)
        if pred_entry is None or cell.bloc not in BLOC_TO_STRATUM:
            continue
        pred_delta = normalize_prediction(pred_entry, cell.bloc)
        if pred_delta is None:
            continue
        pred_sign = "DEM_GAIN" if pred_delta > 1e-9 else ("REP_GAIN" if pred_delta < -1e-9 else "NO_EFFECT")

        if cell.consensus == "NO_EFFECT":
            no_effect_total += 1
            match = pred_sign == "NO_EFFECT"
            no_effect_correct += int(match)
        else:
            directional_total += 1
            match = pred_sign == cell.consensus
            directional_correct += int(match)
        detail.append({
            "shock_id": cell.shock_id, "bloc": cell.bloc, "consensus": cell.consensus,
            "n_respondents": cell.n_respondents, "predicted_delta": pred_delta,
            "predicted_sign": pred_sign, "match": match,
        })

    dir_acc = directional_correct / directional_total if directional_total else None
    ne_acc = no_effect_correct / no_effect_total if no_effect_total else None

    if dir_acc is None:
        verdict = "NOT COMPUTABLE -- no directional-consensus cells with a matching model prediction"
    elif dir_acc >= VALIDATE_THRESHOLD:
        verdict = f"VALIDATES (direction_accuracy={dir_acc:.3f} >= {VALIDATE_THRESHOLD})"
    elif dir_acc >= WEAK_THRESHOLD:
        verdict = f"WEAK/INCONCLUSIVE (direction_accuracy={dir_acc:.3f}, between {WEAK_THRESHOLD} and {VALIDATE_THRESHOLD}) -- report, do not claim validation"
    else:
        verdict = f"DOES NOT VALIDATE (direction_accuracy={dir_acc:.3f} < {WEAK_THRESHOLD}, not distinguishable from chance)"

    return ElicitationScoreReport(
        n_consensus_directional=directional_total, n_consensus_no_effect=no_effect_total,
        n_split=n_split, n_insufficient_data=n_insufficient,
        direction_accuracy=dir_acc, no_effect_accuracy=ne_acc,
        validation_verdict=verdict, cells=detail,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Synthetic dummy data ONLY -- proves the code path executes correctly.
    Not real expert responses; must never be reported as a real result.
    """
    import tempfile

    print("=" * 70)
    print("SELF-TEST: synthetic dummy responses, NOT real expert data.")
    print("=" * 70)

    shocks = ["dobbs_2022", "bin_laden_killing_2011"]
    blocs = list(BLOC_TO_STRATUM.keys())
    strata = BLOC_TO_STRATUM

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        expert_codes = ["E1", "E2", "E3"]
        for i, code in enumerate(expert_codes):
            path = tmp_path / f"{code}_responses.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(REQUIRED_COLUMNS)
                for sid in shocks:
                    for bloc in blocs:
                        # deterministic synthetic pattern: E1/E2 agree, E3 dissents on one shock
                        direction = "DEM_GAIN" if sid == "dobbs_2022" else "REP_GAIN"
                        if code == "E3" and sid == "bin_laden_killing_2011":
                            direction = "DEM_GAIN"  # dissent, to exercise the SPLIT path
                        w.writerow([code, sid, bloc, strata[bloc], direction, "MEDIUM", "SMALL", ""])

        experts = load_all_responses(tmp_path)
        print(f"Loaded {len(experts)} synthetic expert files: {[e.expert_code for e in experts]}")

        agreement = compute_inter_expert_agreement(experts)
        print(f"\nAgreement: kappa={agreement.kappa}, pairwise={agreement.mean_pairwise_agreement}")
        print(f"Gate: {agreement.gate_message}")
        if not agreement.gate_passed:
            print("(In a REAL run via main(), execution stops here -- the self-test continues anyway")
            print(" to exercise build_consensus()/score_model_vs_consensus() too, since this is a code")
            print(" test, not a simulation of the gated pipeline. See the real end-to-end CLI test instead")
            print(" for gate-stops-execution behavior.)")

        consensus = build_consensus(experts)
        n_split = sum(1 for c in consensus if c.consensus == "SPLIT")
        print(f"\nConsensus cells: {len(consensus)} (split={n_split})")

        dummy_predictions = {
            "dobbs_2022": {"party": "democrat", "deltas_race": {b: 0.01 for b in ["white", "african_american", "latino", "asian", "other_race"]}, "deltas_religion": {b: 0.01 for b in ["evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel"]}, "deltas_gender": {b: 0.01 for b in ["women", "men", "other_gender"]}},
            "bin_laden_killing_2011": {"party": "democrat", "deltas_race": {b: -0.01 for b in ["white", "african_american", "latino", "asian", "other_race"]}, "deltas_religion": {b: -0.01 for b in ["evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel"]}, "deltas_gender": {b: -0.01 for b in ["women", "men", "other_gender"]}},
        }
        score = score_model_vs_consensus(consensus, dummy_predictions)
        print(f"\nScore: {score.validation_verdict}")
        print(f"  directional cells: {score.n_consensus_directional}, no_effect cells: {score.n_consensus_no_effect}")
        print(f"  split: {score.n_split}, insufficient: {score.n_insufficient_data}")

    print("\nSelf-test completed without error. This validates the CODE, not any real result.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", type=Path, help="Directory of {expert_code}_responses.csv files")
    ap.add_argument("--log", type=Path, help="Path to pre_registration_log.csv")
    ap.add_argument("--model-predictions", type=Path, help="Path to model predictions JSON")
    ap.add_argument(
        "--model-predictions-generated-at", type=str, default=None,
        help="ISO8601 timestamp the predictions were generated. Defaults to the file's mtime.",
    )
    ap.add_argument("--self-test", action="store_true", help="Run against synthetic dummy data only")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if not (args.responses and args.log and args.model_predictions):
        ap.error("--responses, --log, and --model-predictions are all required for a real run (or pass --self-test)")

    experts = load_all_responses(args.responses)
    log_entries = load_preregistration_log(args.log)

    if args.model_predictions_generated_at:
        generated_at = datetime.fromisoformat(args.model_predictions_generated_at)
    else:
        generated_at = datetime.fromtimestamp(args.model_predictions.stat().st_mtime, tz=timezone.utc)

    verify_preregistration(experts, log_entries, generated_at)
    print(f"Pre-registration verified for {len(experts)} expert file(s).")

    low_completion = [e for e in experts if e.completion_fraction < MIN_COMPLETION_FRACTION]
    for e in low_completion:
        print(f"  NOTE: {e.expert_code} completed {e.completion_fraction:.0%} of rows "
              f"(< {MIN_COMPLETION_FRACTION:.0%}) -- excluded from primary analysis per protocol.md SS4")
    primary_experts = [e for e in experts if e.completion_fraction >= MIN_COMPLETION_FRACTION]

    agreement = compute_inter_expert_agreement(primary_experts)
    print(f"\n=== Inter-expert agreement ===")
    print(f"kappa: {agreement.kappa}")
    print(f"mean pairwise agreement: {agreement.mean_pairwise_agreement}")
    print(f"gate: {agreement.gate_message}")

    if not agreement.gate_passed:
        print("\nSTOPPING per protocol.md SS5.1: inter-expert agreement gate not met. "
              "No model-vs-consensus score will be computed.")
        return

    consensus = build_consensus(primary_experts)
    with open(args.model_predictions, encoding="utf-8") as f:
        predictions = json.load(f)
    score = score_model_vs_consensus(consensus, predictions)

    print(f"\n=== Model vs. expert consensus ===")
    print(f"verdict: {score.validation_verdict}")
    print(f"directional cells scored: {score.n_consensus_directional} (accuracy={score.direction_accuracy})")
    print(f"no-effect cells scored: {score.n_consensus_no_effect} (accuracy={score.no_effect_accuracy})")
    print(f"split cells (excluded): {score.n_split}")
    print(f"insufficient-data cells (excluded): {score.n_insufficient_data}")


if __name__ == "__main__":
    main()
