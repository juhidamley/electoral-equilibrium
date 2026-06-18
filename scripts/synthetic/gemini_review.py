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
  4. Magnitude realism   — slight nudge vs realignment, matched to bin sizes?

IMPORTANT (designed-in caveat): Gemini agreeing does not make a label TRUE —
models share blind spots. So beyond Gemini's own flags, this script ALSO routes
a random spot-check sample (default 5%) to the human queue regardless of verdict,
so the manual pass always sees some "approved" records too.

Usage:
    python scripts/synthetic/gemini_review.py \
        --candidates data/finetune/candidates.jsonl \
        --approved   data/finetune/reviewed_approved.jsonl \
        --queue      data/finetune/human_review_queue.jsonl \
        --revisions  data/finetune/revisions.jsonl \
        --spotcheck-frac 0.05

Env: GEMINI_API_KEY or GOOGLE_API_KEY (read from .env via python-dotenv).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gemini_review")

BIN_TOKENS = {
    "strong_neg", "mod_neg", "mild_neg", "slight_neg", "neutral",
    "slight_pos", "mild_pos", "mod_pos", "strong_pos",
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

PREDICTED LABELS:
delta_bins_race: {json.dumps(rec['delta_bins_race'])}
delta_bins_religion: {json.dumps(rec['delta_bins_religion'])}
delta_bins_gender: {json.dumps(rec['delta_bins_gender'])}
delta_eff: {rec['delta_eff']}

Valid bins (ordered): strong_neg, mod_neg, mild_neg, slight_neg, neutral, slight_pos, mild_pos, mod_pos, strong_pos.

Check:
1. Direction — does each bloc move the way real political behavior predicts?
2. Coherence — does delta_eff agree in sign with the weighted bloc movement?
3. Cross-bloc — are opposed blocs (e.g. evangelical vs secular) plausibly inverse?
4. Magnitude — are bin sizes realistic (most shocks are slight/mild, not strong)?

Respond with a JSON object:
{{
  "verdict": "APPROVE" | "REVISE" | "HUMAN_REVIEW",
  "reasoning": "one or two sentences",
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
    if not (-0.15 <= de <= 0.15):
        return None
    out["delta_eff"] = de
    return out


def review(
    candidates: Path, approved: Path, queue: Path, revisions: Path,
    spotcheck_frac: float, seed: int,
) -> None:
    _load_env()
    model = _client()
    import random
    rng = random.Random(seed)

    recs = [json.loads(line) for line in candidates.read_text().splitlines() if line.strip()]
    log.info("reviewing %d candidates", len(recs))

    n_app = n_rev = n_hum = n_spot = n_err = 0
    for p in (approved, queue, revisions):
        p.parent.mkdir(parents=True, exist_ok=True)

    f_app = approved.open("w")
    f_q = queue.open("w")
    f_rev = revisions.open("w")

    try:
        for i, rec in enumerate(recs):
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
                rec["_review"] = {"verdict": "HUMAN_REVIEW", "reasoning": "review failed/unparseable"}
                f_q.write(json.dumps(rec) + "\n")
                n_hum += 1
                continue

            verdict = verdict_obj.get("verdict", "HUMAN_REVIEW")
            reasoning = verdict_obj.get("reasoning", "")
            rec["_review"] = {"verdict": verdict, "reasoning": reasoning}

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
                log.info("…%d/%d  approve=%d revise=%d human=%d spot=%d", i + 1, len(recs), n_app, n_rev, n_hum, n_spot)
            time.sleep(0.2)  # gentle rate limiting
    finally:
        f_app.close()
        f_q.close()
        f_rev.close()

    log.info(
        "DONE — approved=%d revised=%d human_queue=%d (incl %d spot-checks) errors=%d",
        n_app, n_rev, n_hum + n_spot, n_spot, n_err,
    )
    log.info("ready to merge: %s", approved)
    log.info("manual Opus/human pass needed on: %s", queue)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/finetune/candidates.jsonl")
    p.add_argument("--approved", default="data/finetune/reviewed_approved.jsonl")
    p.add_argument("--queue", default="data/finetune/human_review_queue.jsonl")
    p.add_argument("--revisions", default="data/finetune/revisions.jsonl")
    p.add_argument("--spotcheck-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    review(
        Path(args.candidates), Path(args.approved), Path(args.queue),
        Path(args.revisions), args.spotcheck_frac, args.seed,
    )


if __name__ == "__main__":
    main()
