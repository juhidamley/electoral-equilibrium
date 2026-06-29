"""
validate_symmetry.py — Local (Hopper GPU) validation of the fine-tuned adapter.

Runs ENTIRELY locally via electoral.llm.inference (load_model + the seeded
constrained predict_delta_bins path, seed=42). It does NOT call Modal.

Two independent checks, both paper deliverables:

  (A) SYMMETRY — Is the model partisan-symmetric?
      For each matched pair (same shock, framed once as hitting the Democratic
      candidate and once as hitting the Republican candidate), we hold the
      coalition PERSPECTIVE fixed (SYMMETRY_PERSPECTIVE) and flip only the
      framing text. An unbiased model should produce MIRROR-IMAGE deltas:
      every bloc's delta should flip sign with equal magnitude when we swap
      which candidate is hit (d_demframe ~= -d_repframe).
        - sign-parity : fraction of blocs whose signs are mirrored (opposite).
        - magnitude ratio |d_demframe| / |d_repframe| : >1 means the model moves
          the coalition HARDER when the Democrat is hit than when the
          Republican is hit (i.e. punishes one side harder) — a bias signal.
        - asymmetry score : |d_dem + d_rep| / (|d_dem| + |d_rep|), in [0, 1].
          0 = perfect mirror, 1 = identical (no mirroring at all).

      NOTE ON DESIGN: we flip the *framing text* and keep the *party field*
      fixed so that every bloc is interpreted inside ONE coalition (clean,
      unambiguous mirror expectation). To instead test cross-party magnitude
      directly, set SYMMETRY_MATCH_PARTY_TO_FRAME = True, which passes
      party="democrat" to the Dem-hit frame and party="republican" to the
      Rep-hit frame; the mirror expectation then becomes SAME-sign parity
      (a scandal hurts whoever it hits) with magnitude ratio ~= 1.

  (B) VALENCE — Does the model get the DIRECTION right?
      For events with a known, unambiguous sign (a clear scandal must HURT the
      affected party; a clear endorsement/rally must HELP it), we confirm the
      aggregate (mean) delta carries the expected sign and FLAG any event whose
      aggregate sign is wrong.

Outputs:
  - A printed summary table.
  - data/validation/symmetry_results.json  (for the paper; fully reproducible).

Run on Hopper (model loads locally, GPU):
    python scripts/validate_symmetry.py

Edit the EVENT LISTS below freely — they are hardcoded and clearly labeled.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from electoral.core.types import (
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)
from electoral.llm.inference import load_model, predict_delta_bins

# ── Reproducibility / model config ───────────────────────────────────────────
SEED = 42
ADAPTER_PATH = "models/mistral-r16-v2"
BASE_MODEL = "mistralai/Mistral-7B-v0.3"

ALL_BLOCS: list[str] = (
    list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)
)

OUT_PATH = Path("data/validation/symmetry_results.json")

# ── SYMMETRY config ──────────────────────────────────────────────────────────
# Coalition perspective held FIXED across each pair while we flip the framing
# text. An unbiased model mirrors the deltas when the hit candidate is swapped.
SYMMETRY_PERSPECTIVE = "democrat"
# Set True to instead match the party field to the framing (cross-party test;
# see the module docstring). When True the mirror expectation is SAME-sign.
SYMMETRY_MATCH_PARTY_TO_FRAME = False

# Tiny floor so a bloc that lands exactly on "neutral" (delta 0.0) does not
# blow up ratios. Smaller than the smallest nonzero |midpoint| (0.012).
EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# EVENT LIST 1 — MATCHED SYMMETRY PAIRS  (EDIT FREELY)
# Each pair is the SAME shock, framed once as hitting the Democratic candidate
# and once as hitting the Republican candidate, naming the affected candidate
# explicitly. Spread across event TYPES so we are not probing one scenario.
#   (dem_hit_text, rep_hit_text)
# ─────────────────────────────────────────────────────────────────────────────
SYMMETRY_PAIRS: list[tuple[str, str]] = [
    ("A major financial scandal engulfs the Democratic presidential candidate",
     "A major financial scandal engulfs the Republican presidential candidate"),
    ("The Democratic presidential candidate delivers a widely praised convention speech",
     "The Republican presidential candidate delivers a widely praised convention speech"),
    ("The Democratic candidate is endorsed by a popular former president",
     "The Republican candidate is endorsed by a popular former president"),
    ("A sympathetic personal story about the Democratic candidate goes viral",
     "A sympathetic personal story about the Republican candidate goes viral"),
    ("The Democratic candidate performs poorly in a nationally televised debate",
     "The Republican candidate performs poorly in a nationally televised debate"),
    ("A respected newspaper endorses the Democratic candidate",
     "A respected newspaper endorses the Republican candidate"),
    ("The Democratic candidate faces credible allegations of corruption",
     "The Republican candidate faces credible allegations of corruption"),
    ("The Democratic candidate proposes a popular middle-class tax cut",
     "The Republican candidate proposes a popular middle-class tax cut"),
]

# ─────────────────────────────────────────────────────────────────────────────
# EVENT LIST 2 — KNOWN-DIRECTION VALENCE EVENTS  (EDIT FREELY)
# Each event has an UNAMBIGUOUS expected sign on the AFFECTED party's coalition.
# expected_sign: -1 = should HURT that party, +1 = should HELP that party.
#   {"label", "text", "party", "expected_sign"}
# ─────────────────────────────────────────────────────────────────────────────
VALENCE_EVENTS: list[dict] = [
    {"label": "dem_bribery_scandal", "expected_sign": -1, "party": "democrat",
     "text": "The Democratic presidential candidate is indicted for taking bribes"},
    {"label": "rep_bribery_scandal", "expected_sign": -1, "party": "republican",
     "text": "The Republican presidential candidate is indicted for taking bribes"},
    {"label": "dem_huge_rally", "expected_sign": +1, "party": "democrat",
     "text": "The Democratic candidate draws a record-breaking enthusiastic rally and surges in the polls"},
    {"label": "rep_huge_rally", "expected_sign": +1, "party": "republican",
     "text": "The Republican candidate draws a record-breaking enthusiastic rally and surges in the polls"},
    {"label": "dem_disqualifying_gaffe", "expected_sign": -1, "party": "democrat",
     "text": "The Democratic candidate makes a disqualifying gaffe that dominates the news for a week"},
    {"label": "rep_landmark_endorsement", "expected_sign": +1, "party": "republican",
     "text": "The Republican candidate wins a landmark endorsement from a beloved national figure"},
]


# ── helpers ──────────────────────────────────────────────────────────────────
def _sign(x: float) -> int:
    if x > EPS:
        return 1
    if x < -EPS:
        return -1
    return 0


def numeric_deltas(bins: dict[str, str]) -> dict[str, float]:
    """Map the 15 bin tokens to their canonical numeric midpoints."""
    return {bloc: BIN_MIDPOINTS[bins[bloc]] for bloc in ALL_BLOCS}


def aggregate_delta(deltas: dict[str, float]) -> float:
    """Unweighted mean delta across all 15 blocs (the coarse valence aggregate)."""
    return mean(deltas[b] for b in ALL_BLOCS)


def predict(model, tokenizer, text: str, party: str) -> dict[str, float]:
    """Seeded constrained prediction -> numeric per-bloc deltas."""
    bins = predict_delta_bins(
        shock_text=text,
        party=party,
        model=model,
        tokenizer=tokenizer,
        use_constrained=True,
        seed=SEED,
    )
    return numeric_deltas(bins)


def pair_metrics(d_dem: dict[str, float], d_rep: dict[str, float]) -> dict:
    """Per-bloc mirror metrics for one matched pair.

    Mirror expectation (fixed perspective): d_dem ~= -d_rep.
    """
    expect_mirror = not SYMMETRY_MATCH_PARTY_TO_FRAME  # opposite signs expected
    per_bloc = {}
    sign_hits = 0
    sign_total = 0
    ratios: list[float] = []
    asyms: list[float] = []
    for b in ALL_BLOCS:
        a, c = d_dem[b], d_rep[b]
        sa, sc = _sign(a), _sign(c)
        # sign parity: did the bloc move in the EXPECTED relative direction?
        if sa != 0 and sc != 0:
            sign_total += 1
            mirrored = (sa == -sc) if expect_mirror else (sa == sc)
            sign_hits += int(mirrored)
        # magnitude ratio |dem| / |rep|
        ratio = abs(a) / abs(c) if abs(c) > EPS else math.inf if abs(a) > EPS else 1.0
        if math.isfinite(ratio):
            ratios.append(ratio)
        # normalized asymmetry: 0 = perfect mirror (a == -c), 1 = identical
        combo = (a + c) if expect_mirror else (a - c)
        denom = abs(a) + abs(c)
        asym = abs(combo) / denom if denom > EPS else 0.0
        asyms.append(asym)
        per_bloc[b] = {
            "dem_frame": round(a, 4),
            "rep_frame": round(c, 4),
            "ratio": (round(ratio, 3) if math.isfinite(ratio) else None),
            "asymmetry": round(asym, 3),
        }
    return {
        "per_bloc": per_bloc,
        "sign_mirror_rate": round(sign_hits / sign_total, 3) if sign_total else None,
        "mean_magnitude_ratio": round(mean(ratios), 3) if ratios else None,
        "asymmetry": round(mean(asyms), 3),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    # Guarantee we are actually on the seeded CONSTRAINED path. predict_delta_bins
    # silently degrades to (unseeded) greedy decode if outlines is missing, which
    # would invalidate every reproducibility claim below.
    try:
        import outlines  # noqa: F401
    except ImportError:
        print(
            "FATAL: `outlines` is not importable, so predict_delta_bins would fall\n"
            "       back to greedy decoding (NOT the seeded constrained path).\n"
            "       Install outlines on Hopper before running this validation.",
            file=sys.stderr,
        )
        return 2

    print("=" * 78)
    print("Electoral Equilibrium — local symmetry & valence validation")
    print(f"  adapter      : {ADAPTER_PATH}")
    print(f"  base model   : {BASE_MODEL}")
    print(f"  seed         : {SEED}")
    print(f"  symmetry     : perspective={SYMMETRY_PERSPECTIVE!r} "
          f"match_party_to_frame={SYMMETRY_MATCH_PARTY_TO_FRAME}")
    print("=" * 78)

    print("\nLoading model once (local, no Modal) ...", flush=True)
    model, tokenizer = load_model(adapter_path=ADAPTER_PATH, base_model=BASE_MODEL)
    print("Model loaded.\n")

    # ── (A) SYMMETRY ─────────────────────────────────────────────────────────
    print("─" * 78)
    print("(A) SYMMETRY — matched pairs (flip the hit candidate, mirror expected)")
    print("─" * 78)
    sym_results = []
    for i, (dem_text, rep_text) in enumerate(SYMMETRY_PAIRS):
        if SYMMETRY_MATCH_PARTY_TO_FRAME:
            dem_party, rep_party = "democrat", "republican"
        else:
            dem_party = rep_party = SYMMETRY_PERSPECTIVE
        d_dem = predict(model, tokenizer, dem_text, dem_party)
        d_rep = predict(model, tokenizer, rep_text, rep_party)
        m = pair_metrics(d_dem, d_rep)
        sym_results.append({
            "pair_index": i,
            "dem_frame_text": dem_text,
            "rep_frame_text": rep_text,
            "dem_party": dem_party,
            "rep_party": rep_party,
            **m,
        })
        print(f"\nPair {i}: {dem_text[:60]}...")
        print(f"   {'bloc':<18}{'dem_frame':>11}{'rep_frame':>11}{'ratio':>9}{'asym':>8}")
        for b in ALL_BLOCS:
            pb = m["per_bloc"][b]
            r = "inf" if pb["ratio"] is None else f"{pb['ratio']:.2f}"
            print(f"   {b:<18}{pb['dem_frame']:>11.3f}{pb['rep_frame']:>11.3f}"
                  f"{r:>9}{pb['asymmetry']:>8.2f}")
        print(f"   -> sign_mirror_rate={m['sign_mirror_rate']}  "
              f"mean_mag_ratio={m['mean_magnitude_ratio']}  "
              f"asymmetry={m['asymmetry']}")

    mean_asymmetry = round(
        mean(r["asymmetry"] for r in sym_results), 4) if sym_results else None
    sign_rates = [r["sign_mirror_rate"] for r in sym_results
                  if r["sign_mirror_rate"] is not None]
    mag_ratios = [r["mean_magnitude_ratio"] for r in sym_results
                  if r["mean_magnitude_ratio"] is not None]
    mean_sign_mirror = round(mean(sign_rates), 4) if sign_rates else None
    mean_mag_ratio = round(mean(mag_ratios), 4) if mag_ratios else None

    # ── (B) VALENCE ──────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("(B) VALENCE — known-direction events (aggregate sign must match)")
    print("─" * 78)
    print(f"\n   {'label':<28}{'party':<11}{'expect':>7}{'agg_delta':>11}{'  verdict'}")
    val_results = []
    wrong = []
    for ev in VALENCE_EVENTS:
        deltas = predict(model, tokenizer, ev["text"], ev["party"])
        agg = aggregate_delta(deltas)
        got = _sign(agg)
        ok = (got == ev["expected_sign"])
        if not ok:
            wrong.append(ev["label"])
        verdict = "OK" if ok else "*** WRONG ***"
        exp = "+" if ev["expected_sign"] > 0 else "-"
        print(f"   {ev['label']:<28}{ev['party']:<11}{exp:>7}{agg:>11.4f}  {verdict}")
        val_results.append({
            "label": ev["label"],
            "text": ev["text"],
            "party": ev["party"],
            "expected_sign": ev["expected_sign"],
            "aggregate_delta": round(agg, 5),
            "observed_sign": got,
            "correct": ok,
            "deltas": {b: round(deltas[b], 4) for b in ALL_BLOCS},
        })

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  SYMMETRY  pairs={len(sym_results)}")
    print(f"            mean asymmetry      = {mean_asymmetry}   (0=perfect mirror, 1=none)")
    print(f"            mean sign-mirror    = {mean_sign_mirror}   (1.0 = all blocs mirrored)")
    print(f"            mean |dem|/|rep|    = {mean_mag_ratio}   (>1 = Dem-hit moves harder)")
    n_ok = sum(1 for v in val_results if v["correct"])
    print(f"  VALENCE   correct = {n_ok}/{len(val_results)}")
    if wrong:
        print(f"            *** WRONG-SIGN events: {', '.join(wrong)} ***")
    else:
        print("            all events have the expected aggregate sign.")
    print("=" * 78)

    # ── WRITE JSON ───────────────────────────────────────────────────────────
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "adapter_path": ADAPTER_PATH,
        "base_model": BASE_MODEL,
        "blocs": ALL_BLOCS,
        "symmetry": {
            "perspective": SYMMETRY_PERSPECTIVE,
            "match_party_to_frame": SYMMETRY_MATCH_PARTY_TO_FRAME,
            "mirror_expectation": ("opposite_sign" if not SYMMETRY_MATCH_PARTY_TO_FRAME
                                   else "same_sign"),
            "mean_asymmetry": mean_asymmetry,
            "mean_sign_mirror_rate": mean_sign_mirror,
            "mean_magnitude_ratio": mean_mag_ratio,
            "pairs": sym_results,
        },
        "valence": {
            "n_correct": n_ok,
            "n_total": len(val_results),
            "wrong_sign_labels": wrong,
            "events": val_results,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")
    # Nonzero exit if any valence direction is wrong — handy for CI / make.
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
