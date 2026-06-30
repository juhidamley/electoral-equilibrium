#!/usr/bin/env python3
"""Event-level partisan-symmetry test for the fine-tuned shock→delta model.

Runs LOCALLY on Hopper (loads the adapter; NO Modal, NO network). Measures whether
the LLM's DELTA predictions treat the two parties symmetrically — independent of the
(separately-fixed) baseline μ. For each of ~10 MATCHED event pairs (the same shock
framed once for the Democratic candidate and once for the Republican candidate), it
runs the seeded constrained `predict_delta_bins` for the matching party and compares
the two 15-bloc delta vectors.

Each party's deltas are in that party's OWN "loyalty toward this party" coordinates,
so a symmetric model should move each party by the same amount for mirror events:
    asymmetry = mean(Dem-framing deltas) − mean(Rep-framing deltas)
    asymmetry ≈ 0  → symmetric
    asymmetry > 0  → Democratic framing came out more positive (Dem-favoring)
    asymmetry < 0  → Republican framing came out more positive (Rep-favoring)
Reporting is direction-NEUTRAL: the two directions are treated identically.

Usage (on Hopper, GPU node):
    ADAPTER_PATH=models/mistral-r16-v2 python scripts/symmetry_test_events.py

Writes per-pair deltas + aggregate metrics to data/validation/event_symmetry.json.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

# Repo root on path so `import electoral` resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from electoral.core.types import (  # noqa: E402
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)

SEED = 42
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "models/mistral-r16-v2")
BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.3")
OUT_PATH = Path("data/validation/event_symmetry.json")

ALL_BLOCS: list[str] = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)

# ── Interpretation thresholds (tunable) ───────────────────────────────────────
SYMMETRIC_THRESH = 0.01   # |aggregate asymmetry| below this → symmetric
MILD_THRESH = 0.03        # between SYMMETRIC and this → mild lean; above → notable lean
MAG_PARITY_MIN = 0.5      # weaker framing must be ≥ this fraction of the stronger one
SIGN_EPS = 1e-9           # deadzone for calling a mean's direction

# ── Matched event pairs (EDIT HERE) ───────────────────────────────────────────
# Each pair is the SAME shock, framed for each party with the candidate named so the
# `party` field and the text agree. `expected_sign` is the direction the shock should
# move the AFFECTED party's own loyalty: -1 = should HURT, +1 = should HELP.
EVENT_PAIRS: list[dict] = [
    {"type": "financial_scandal", "expected_sign": -1,
     "dem": "A major financial scandal engulfs the Democratic presidential candidate",
     "rep": "A major financial scandal engulfs the Republican presidential candidate"},
    {"type": "major_endorsement", "expected_sign": +1,
     "dem": "A widely admired former president endorses the Democratic presidential candidate",
     "rep": "A widely admired former president endorses the Republican presidential candidate"},
    {"type": "enthusiastic_rally", "expected_sign": +1,
     "dem": "A record-breaking, highly enthusiastic rally energizes the Democratic candidate's base",
     "rep": "A record-breaking, highly enthusiastic rally energizes the Republican candidate's base"},
    {"type": "debate_loss", "expected_sign": -1,
     "dem": "The Democratic presidential candidate performs poorly and is widely seen as losing the debate",
     "rep": "The Republican presidential candidate performs poorly and is widely seen as losing the debate"},
    {"type": "popular_policy", "expected_sign": +1,
     "dem": "The Democratic presidential candidate proposes a widely popular middle-class tax cut",
     "rep": "The Republican presidential candidate proposes a widely popular middle-class tax cut"},
    {"type": "damaging_gaffe", "expected_sign": -1,
     "dem": "The Democratic presidential candidate makes a damaging on-camera gaffe",
     "rep": "The Republican presidential candidate makes a damaging on-camera gaffe"},
    {"type": "criminal_indictment", "expected_sign": -1,
     "dem": "The Democratic presidential candidate is criminally indicted",
     "rep": "The Republican presidential candidate is criminally indicted"},
    {"type": "strong_jobs_news", "expected_sign": +1,
     "dem": "A strong national jobs report boosts the Democratic candidate's economic message",
     "rep": "A strong national jobs report boosts the Republican candidate's economic message"},
    {"type": "extremist_remarks", "expected_sign": -1,
     "dem": "Extremist remarks by the Democratic presidential candidate alienate moderate voters",
     "rep": "Extremist remarks by the Republican presidential candidate alienate moderate voters"},
    {"type": "unifying_convention", "expected_sign": +1,
     "dem": "The Democratic presidential candidate unifies the party at a highly successful convention",
     "rep": "The Republican presidential candidate unifies the party at a highly successful convention"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _deltas_from_bins(bins: dict[str, str]) -> dict[str, float]:
    """Map the 15 bloc → bin tokens to their numeric midpoints."""
    return {b: float(BIN_MIDPOINTS[bins[b]]) for b in ALL_BLOCS}


def _mean(deltas: dict[str, float]) -> float:
    return float(sum(deltas.values()) / len(deltas))


def _sign_parity(dem_mean: float, rep_mean: float, expected: int) -> dict:
    """Both framings should move in the expected direction with comparable magnitude."""
    dem_dir_ok = (dem_mean * expected) > SIGN_EPS
    rep_dir_ok = (rep_mean * expected) > SIGN_EPS
    a, b = abs(dem_mean), abs(rep_mean)
    ratio = (min(a, b) / max(a, b)) if max(a, b) > 0 else 1.0
    return {
        "dem_dir_ok": dem_dir_ok,
        "rep_dir_ok": rep_dir_ok,
        "mag_ratio": round(ratio, 4),
        "ok": bool(dem_dir_ok and rep_dir_ok and ratio >= MAG_PARITY_MIN),
    }


def _interpret(agg: float) -> tuple[str, str]:
    """Return (verdict, favored_framing) — direction-neutral."""
    if abs(agg) < SYMMETRIC_THRESH:
        return "symmetric", "neutral"
    favored = "democratic" if agg > 0 else "republican"
    return ("mild_lean" if abs(agg) <= MILD_THRESH else "notable_lean"), favored


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    try:
        import torch

        from electoral.llm.inference import load_model, predict_delta_bins
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: cannot import inference deps ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2

    if not Path(ADAPTER_PATH).exists():
        print(f"ERROR: adapter not found at {ADAPTER_PATH} (set ADAPTER_PATH).", file=sys.stderr)
        return 2

    print(f"Event symmetry test — seed={SEED}, adapter={ADAPTER_PATH} (no Modal)\n"
          f"Loading model once…")
    torch.manual_seed(SEED)
    model, tokenizer = load_model(ADAPTER_PATH, BASE_MODEL)

    def predict(text: str, party: str) -> tuple[dict[str, str], dict[str, float], float]:
        torch.manual_seed(SEED)
        bins = predict_delta_bins(text, party, model, tokenizer, use_constrained=True, seed=SEED)
        deltas = _deltas_from_bins(bins)
        return bins, deltas, _mean(deltas)

    pair_results: list[dict] = []
    asymmetries: list[float] = []

    for pair in EVENT_PAIRS:
        d_bins, d_deltas, d_mean = predict(pair["dem"], "democrat")
        r_bins, r_deltas, r_mean = predict(pair["rep"], "republican")
        asym = d_mean - r_mean
        parity = _sign_parity(d_mean, r_mean, pair["expected_sign"])
        asymmetries.append(asym)
        pair_results.append({
            "type": pair["type"],
            "expected_sign": pair["expected_sign"],
            "dem": {"text": pair["dem"], "bins": d_bins, "deltas": d_deltas, "mean": d_mean},
            "rep": {"text": pair["rep"], "bins": r_bins, "deltas": r_deltas, "mean": r_mean},
            "asymmetry": asym,
            "sign_parity": parity,
        })

    agg_mean = float(statistics.fmean(asymmetries))
    agg_std = float(statistics.pstdev(asymmetries)) if len(asymmetries) > 1 else 0.0
    verdict, favored = _interpret(agg_mean)
    n_parity_fail = sum(1 for r in pair_results if not r["sign_parity"]["ok"])

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'event type':<22}{'Dem mean':>11}{'Rep mean':>11}{'asymmetry':>12}{'parity OK?':>12}")
    print("-" * 68)
    for r in pair_results:
        print(f"{r['type']:<22}{r['dem']['mean']:>11.4f}{r['rep']['mean']:>11.4f}"
              f"{r['asymmetry']:>+12.4f}{('yes' if r['sign_parity']['ok'] else 'NO'):>12}")
    print("-" * 68)

    # ── Direction-neutral aggregate verdict ───────────────────────────────────
    print(f"\nAggregate asymmetry = {agg_mean:+.4f}  (std {agg_std:.4f}, n={len(asymmetries)})")
    print(f"  |asymmetry| = {abs(agg_mean):.4f}  →  VERDICT: {verdict.upper().replace('_', ' ')}")
    if verdict == "symmetric":
        print("  Interpretation: deltas are partisan-symmetric (|asymmetry| < "
              f"{SYMMETRIC_THRESH}).")
    else:
        print(f"  Interpretation: {verdict.replace('_', ' ')} toward the {favored.upper()} "
              f"framing (positive=Dem-favoring, negative=Rep-favoring, reported neutrally).")
        if verdict == "notable_lean":
            print("  >>> FLAG FOR INVESTIGATION: |asymmetry| exceeds "
                  f"{MILD_THRESH}.")
    print(f"  sign-parity failures: {n_parity_fail}/{len(pair_results)}"
          + (" — review flagged pairs above." if n_parity_fail else ""))

    # ── Persist for the paper ─────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "seed": SEED,
        "adapter_path": ADAPTER_PATH,
        "base_model": BASE_MODEL,
        "n_pairs": len(pair_results),
        "thresholds": {"symmetric": SYMMETRIC_THRESH, "mild": MILD_THRESH,
                       "mag_parity_min": MAG_PARITY_MIN},
        "pairs": pair_results,
        "aggregate": {
            "mean_asymmetry": agg_mean,
            "std_asymmetry": agg_std,
            "abs_mean_asymmetry": abs(agg_mean),
            "verdict": verdict,
            "favored_framing": favored,
            "n_sign_parity_fail": n_parity_fail,
        },
    }, indent=2))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
