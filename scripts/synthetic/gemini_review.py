"""Stage 2 — synthetic record review via Gemini.

Reads the candidates file from stage 1, sends each record to Gemini for a
political-plausibility review, and triages:

  APPROVE      -> written to reviewed_approved.jsonl (ready to merge)
  REVISE       -> Gemini supplies corrected bins; written to reviewed_approved.jsonl
                  AND logged to revisions.jsonl so the change is auditable
  HUMAN_REVIEW -> written to human_review_queue.jsonl for the manual Opus pass

Gemini checks four things:
  1. Direction sanity   — does each bloc move the politically expected way?
  2. Internal coherence  — do delta_eff and the per-bloc bins agree in sign?
  3. Cross-bloc logic    — are opposed blocs (evangelical vs secular) consistent?
  4. Magnitude realism   — bin sizes checked against real panel numbers (Step
                            5.1: median 0.0029 / mean 0.0046 / p90 0.0112 /
                            max-ever 0.0286), not vibes; multiple blocs at
                            strong/moderate in one record is a red flag.

IMPORTANT (designed-in caveat): Gemini agreeing does not make a label TRUE —
models share blind spots. So beyond Gemini's own flags, this script ALSO routes
a random spot-check sample (default 5%) to the human queue regardless of verdict,
so the manual pass always sees some "approved" records too.

Usage:
    python scripts/synthetic/gemini_review.py \
        --candidates data/finetune/synthetic_step5_5_20260809.jsonl \
        --approved   data/finetune/synthetic_step5_5_20260809_approved.jsonl \
        --queue      data/finetune/synthetic_step5_5_20260809_human_queue.jsonl \
        --revisions  data/finetune/synthetic_step5_5_20260809_revisions.jsonl \
        --spotcheck-frac 0.05

Env: GEMINI_API_KEY or GOOGLE_API_KEY (read from .env via python-dotenv).

HELD-OUT ENFORCEMENT: stage 1 (generate_deepseek.py) already checks every record
before it's written. This stage re-checks anyway (belt and suspenders -- it's the
last stage before a record is considered "ready to merge," and the check is nearly
free compared to the Gemini API call already being made per record). See that
script's module docstring for why this is a generative-collision check, not a
lookup-table leak.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from electoral.data.held_out import HeldOutShockError, assert_not_held_out

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gemini_review")

BIN_TOKENS = {
    "strong_neg",
    "mod_neg",
    "mild_neg",
    "slight_neg",
    "neutral",
    "slight_pos",
    "mild_pos",
    "mod_pos",
    "strong_pos",
}
GEMINI_MODEL = "gemini-2.5-flash"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY / GOOGLE_API_KEY not set (check .env)")
    return genai.Client(api_key=key)


REVIEW_SYSTEM = (
    "You are a careful reviewer of synthetic electoral training data. For each "
    "record you receive a political shock event and predicted per-bloc vote-share "
    "delta bins. Judge whether the labels are politically plausible. Reason about "
    "each demographic bloc independently. Be willing to say a record is wrong: the "
    "ideological flavor of an event does NOT determine which party it helps. "
    "Respond ONLY with a JSON object, no markdown."
)


def _review_prompt(rec: dict[str, Any]) -> str:
    meta = rec.get("_seed_meta", {})
    return f"""Review this synthetic training record for political plausibility.

EVENT: "{rec['description']}"
PARTY MODELED: {rec['party']}
TAXONOMY HINT (expected net effect): {meta.get('expected_effect','?')}
MOBILIZATION DYNAMIC (stated by the seed, not inferred from tone): {meta.get('mobilization','?')}
MOBILIZED / BENEFITING PARTY: {meta.get('benefiting_party','NOT DECLARED')}
  For a "mobilizing" event, the declared benefiting party determines NET DIRECTION FIRST:
  delta_eff must be positive when that party is modeled and negative otherwise. This OVERRIDES
  surface sentiment/tone. Negative tone commonly reflects backlash turnout and therefore a
  GAIN for the benefiting party. Tone may affect only intensity/concentration across blocs.
  A mobilizing record without a declared beneficiary or an explicit beneficiary rationale is
  REVISE, never APPROVE. Mobilizing events may warrant more concentrated movement, but that
  never excuses several blocs landing on "strong" simultaneously (see check 4).

PREDICTED LABELS:
delta_bins_race: {json.dumps(rec['delta_bins_race'])}
delta_bins_religion: {json.dumps(rec['delta_bins_religion'])}
delta_bins_gender: {json.dumps(rec['delta_bins_gender'])}
delta_eff: {rec['delta_eff']}

Valid bins (ordered): strong_neg, mod_neg, mild_neg, slight_neg, neutral, slight_pos, mild_pos, mod_pos, strong_pos.

REAL-WORLD MAGNITUDE, measured from a decade of voter-panel data (n=234
non-suppressed bloc/cycle cells): median per-bloc shift 0.0029, mean 0.0046,
90th percentile 0.0112, largest single-bloc shift EVER measured 0.0286. A
"strong" bin means roughly a 3-point move — the top of the observed range, not
a typical reaction to a notable event. Judge magnitude against these numbers,
not against how dramatic the event sounds.

BIN-DISTRIBUTION TARGET across a record's 15 bloc-level bins — MEASURED from
the panel's own empirical bin-magnitude distribution (not intuition; see
scripts/synthetic/check_panel_target_calibration.py): 34% neutral, 31% slight,
28% mild, 6% moderate, 1% strong. Use this, not just the panel stats above, to
judge whether THIS record's overall spread looks plausible.

Check:
1. Direction — does each bloc move the way real political behavior predicts?
2. Coherence — does delta_eff agree in sign with the weighted bloc movement?
3. Cross-bloc — are opposed blocs (e.g. evangelical vs secular) plausibly inverse?
4. Magnitude — are bin sizes realistic against the panel numbers and
   bin-distribution target above? A record with several blocs at "strong" or
   "moderate" simultaneously should almost always be flagged REVISE or
   HUMAN_REVIEW — real events that move many blocs strongly at once do not
   appear in the panel data.
5. Mobilizing-only — name the declared beneficiary and confirm that delta_eff's
   sign matches it. If beneficiary is missing, rationale is missing, or the
   sign conflicts, return REVISE (or HUMAN_REVIEW if no defensible correction).

Respond with a JSON object:
{{
  "verdict": "APPROVE" | "REVISE" | "HUMAN_REVIEW",
  "reasoning": "one or two sentences",
  "beneficiary_rationale": "required non-empty statement for mobilizing records; otherwise null",
  "corrected": null OR {{ "delta_bins_race": {{...}}, "delta_bins_religion": {{...}}, "delta_bins_gender": {{...}}, "delta_eff": <float> }}
}}

Use REVISE only when you are confident of the corrected labels. Use HUMAN_REVIEW
when the event is genuinely ambiguous or you are uncertain. Provide "corrected"
ONLY for REVISE."""


def _parse_obj(text: str) -> dict[str, Any] | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").replace("json", "", 1).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(t[s : e + 1])
    except json.JSONDecodeError:
        return None


def _mobilizing_review_ok(rec: dict[str, Any], verdict_obj: dict[str, Any]) -> tuple[bool, str]:
    """Require the reviewer to document beneficiary-first direction for mobilizing records."""
    meta = rec.get("_seed_meta", {})
    if meta.get("mobilization") != "mobilizing":
        return True, ""
    beneficiary = meta.get("benefiting_party")
    rationale = verdict_obj.get("beneficiary_rationale")
    if beneficiary not in {"democrat", "republican"}:
        return False, "missing valid declared beneficiary"
    if not isinstance(rationale, str) or not rationale.strip():
        return False, "missing beneficiary rationale"
    candidate = verdict_obj.get("corrected") or rec
    try:
        delta_eff = float(candidate["delta_eff"])
    except (TypeError, ValueError, KeyError):
        return False, "missing numeric delta_eff for beneficiary check"
    expected_positive = rec.get("party") == beneficiary
    if (expected_positive and delta_eff <= 0) or (not expected_positive and delta_eff >= 0):
        return False, "net sign contradicts declared beneficiary"
    return True, ""


def _apply_correction(rec: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and apply a Gemini-suggested correction. Returns None if invalid."""
    out = dict(rec)
    for field in ("delta_bins_race", "delta_bins_religion", "delta_bins_gender"):
        if field not in corrected:
            return None
        d = corrected[field]
        if not isinstance(d, dict) or any(v not in BIN_TOKENS for v in d.values()):
            return None
        out[field] = d
    try:
        de = float(corrected["delta_eff"])
    except (TypeError, ValueError, KeyError):
        return None
    # Rescaled Step 5.1 (see DECISIONS.md), matching generate_deepseek.py's
    # _validate() and electoral.core.types.DELTA_BINS' tiled range.
    if not (-0.0375 <= de <= 0.0375):
        return None
    out["delta_eff"] = de
    return out


def review(
    candidates: Path,
    approved: Path,
    queue: Path,
    revisions: Path,
    spotcheck_frac: float,
    seed: int,
) -> None:
    _load_env()
    model = _client()
    import random

    rng = random.Random(seed)

    recs = [json.loads(line) for line in candidates.read_text().splitlines() if line.strip()]
    log.info("reviewing %d candidates", len(recs))

    n_app = n_rev = n_hum = n_spot = n_err = held_out_blocked = 0
    for p in (approved, queue, revisions):
        p.parent.mkdir(parents=True, exist_ok=True)

    f_app = approved.open("w")
    f_q = queue.open("w")
    f_rev = revisions.open("w")

    try:
        for i, rec in enumerate(recs):
            # Belt-and-suspenders re-check (see module docstring). Dropped
            # entirely -- not routed to the human queue -- a held-out
            # collision isn't "needs a human opinion," it's "must not exist
            # in any output file at all." Checked before spending the Gemini
            # call, not after.
            try:
                assert_not_held_out(
                    rec.get("shock_id", ""), context=f"gemini_review.py input (record {i})"
                )
            except HeldOutShockError as exc:
                held_out_blocked += 1
                log.error("HELD-OUT COLLISION -- record dropped, NOT written anywhere: %s", exc)
                continue

            try:
                resp = model.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=REVIEW_SYSTEM + "\n\n" + _review_prompt(rec),
                )
                verdict_obj = _parse_obj(resp.text)
            except Exception as exc:
                log.warning("record %d review error: %s — routing to human queue", i, exc)
                verdict_obj = None
                n_err += 1

            if verdict_obj is None:
                rec["_review"] = {
                    "verdict": "HUMAN_REVIEW",
                    "reasoning": "review failed/unparseable",
                }
                f_q.write(json.dumps(rec) + "\n")
                n_hum += 1
                continue

            verdict = verdict_obj.get("verdict", "HUMAN_REVIEW")
            reasoning = verdict_obj.get("reasoning", "")
            mobilizing_ok, mobilizing_reason = _mobilizing_review_ok(rec, verdict_obj)
            if not mobilizing_ok:
                verdict = "REVISE"
                reasoning = f"{reasoning} (mobilizing gate: {mobilizing_reason})".strip()
            rec["_review"] = {"verdict": verdict, "reasoning": reasoning}
            if rec.get("_seed_meta", {}).get("mobilization") == "mobilizing":
                rec["_review"]["beneficiary_rationale"] = verdict_obj.get("beneficiary_rationale")
            if not mobilizing_ok:
                # A missing beneficiary rationale or contradictory sign is explicitly
                # REVISE, even when Gemini supplied no usable correction. Keep it out
                # of approved/revised output and send it to the manual queue as REVISE.
                f_q.write(json.dumps(rec) + "\n")
                n_hum += 1
                continue

            if verdict == "APPROVE":
                # designed-in spot check: even approved records sometimes go to human queue
                if rng.random() < spotcheck_frac:
                    rec["_review"]["spotcheck"] = True
                    f_q.write(json.dumps(rec) + "\n")
                    n_spot += 1
                else:
                    f_app.write(json.dumps(rec) + "\n")
                    n_app += 1
            elif verdict == "REVISE":
                corrected = _apply_correction(rec, verdict_obj.get("corrected") or {})
                if corrected is None:
                    rec["_review"]["verdict"] = "HUMAN_REVIEW"
                    rec["_review"]["reasoning"] += " (revision invalid, escalated)"
                    f_q.write(json.dumps(rec) + "\n")
                    n_hum += 1
                else:
                    corrected["_review"] = rec["_review"]
                    f_rev.write(json.dumps({"before": rec, "after": corrected}) + "\n")
                    f_app.write(json.dumps(corrected) + "\n")
                    n_rev += 1
            else:  # HUMAN_REVIEW
                f_q.write(json.dumps(rec) + "\n")
                n_hum += 1

            if (i + 1) % 25 == 0:
                log.info(
                    "…%d/%d  approve=%d revise=%d human=%d spot=%d",
                    i + 1,
                    len(recs),
                    n_app,
                    n_rev,
                    n_hum,
                    n_spot,
                )
            time.sleep(0.2)  # gentle rate limiting
    finally:
        f_app.close()
        f_q.close()
        f_rev.close()

    log.info(
        "DONE — approved=%d revised=%d human_queue=%d (incl %d spot-checks) errors=%d held_out_blocked=%d",
        n_app,
        n_rev,
        n_hum + n_spot,
        n_spot,
        n_err,
        held_out_blocked,
    )
    log.info("ready to merge: %s", approved)
    log.info("manual Opus/human pass needed on: %s", queue)
    if held_out_blocked:
        log.error(
            "held_out_blocked=%d > 0 -- %d record(s) were dropped for colliding with a held-out "
            "shock id. Investigate before treating this corpus as clean.",
            held_out_blocked,
            held_out_blocked,
        )


def main() -> None:
    # No hardcoded static defaults, same reasoning as generate_deepseek.py:
    # data/finetune/reviewed_approved.jsonl / human_review_queue.jsonl /
    # revisions.jsonl are all existing legacy files (Hard Requirement 2).
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--approved", required=True, type=Path)
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--revisions", required=True, type=Path)
    p.add_argument("--spotcheck-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for out_path in (args.approved, args.queue, args.revisions):
        if out_path.exists():
            raise SystemExit(
                f"{out_path} already exists -- refusing to overwrite (Hard Requirement 2). "
                f"Pick a fresh, distinctly-named path."
            )
    review(
        args.candidates,
        args.approved,
        args.queue,
        args.revisions,
        args.spotcheck_frac,
        args.seed,
    )


if __name__ == "__main__":
    main()
