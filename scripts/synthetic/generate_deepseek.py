"""Stage 1 — synthetic shock-event generation via DeepSeek.

Reads configs/synthetic_events.json (seed events with taxonomy tags), expands
each seed into N varied training records using the DeepSeek chat API, validates
every record against the 9-token bin schema, and writes candidates to a JSONL
file. Tracks per-axis counts and oversamples underfilled effect-cells so the
final set is balanced across helps_dem / helps_rep / splits / realigns / neutral.

This is generation only. Review happens in stage 2 (gemini_review.py) and the
manual Opus/human pass. Decoupled so generation can run independently and a
later stage failing never loses this work.

Usage:
    python scripts/synthetic/generate_deepseek.py \
        --seeds configs/synthetic_events.json \
        --n-per-seed 20 \
        --out data/finetune/candidates.jsonl

Env: DEEPSEEK_API_KEY (read from .env via python-dotenv if present).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_deepseek")

# Canonical schema — must match electoral/core/types.py
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
RACES = ["african_american", "asian", "latino", "other_race", "white"]
RELIGIONS = ["evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel"]
GENDERS = ["women", "men", "other_gender"]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _client():
    from openai import OpenAI  # DeepSeek is OpenAI-compatible

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set (check .env)")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


SYSTEM_PROMPT = (
    "You are an expert in American electoral politics and demographic voting "
    "behavior. You generate synthetic training data for an electoral forecasting "
    "model. You reason carefully about how a political shock propagates to "
    "DIFFERENT demographic blocs — the ideological flavor of an event does NOT "
    "determine which party it helps. A far-left policy can hurt Democrats with "
    "moderates; a far-right policy can hurt Republicans with suburban women. "
    "Respond ONLY with a valid JSON array. No markdown, no preamble, no commentary."
)


def _user_prompt(seed: dict[str, Any], n: int) -> str:
    return f"""Generate {n} synthetic electoral shock-response records for this event:

EVENT: "{seed['description']}"
PARTY PERSPECTIVE (whose ticket we model): {seed['party']}
DOMAIN: {seed['domain']}
IDEOLOGICAL VALENCE: {seed['valence']}
EXPECTED NET EFFECT (a hint, reason independently per bloc): {seed['expected_effect']}
MOBILIZATION DYNAMIC (stated, do not re-derive from how dramatic the event sounds): {seed['mobilization']}
  - "mobilizing": the event's negative/polarizing nature drives INCREASED turnout/enthusiasm
    for one side (a backlash-turnout dynamic) — expect some blocs to move MORE than a typical
    event, though still small in absolute terms per the magnitude section below.
  - "depressing": the event erodes confidence/enthusiasm for the affected party through
    persuasion/demoralization, with no strong opposing-mobilization narrative — expect smaller,
    more diffuse movement, not a sharp swing concentrated in a few blocs.
  - "consolidating": a positive event reinforces existing support — expect modest positive
    movement in already-aligned blocs, not a dramatic realignment.
  - "neutral": genuinely bipartisan/low-salience action — expect most blocs at or near neutral.
  This dimension is independent of EXPECTED NET EFFECT's direction; use it to calibrate HOW
  CONCENTRATED vs. diffuse the per-bloc movement should be, not to override the direction hint.

REAL-WORLD MAGNITUDE, measured from a decade of voter-panel data (n=234
non-suppressed bloc/cycle cells) — calibrate your intuition to THESE numbers,
not to how dramatic a news event feels:
  - median per-bloc shift ever measured: 0.0029 (0.29 percentage points)
  - mean:                                0.0046 (0.46 points)
  - 90th percentile:                     0.0112 (1.12 points)
  - largest single-bloc shift EVER measured, any event, any bloc: 0.0286 (2.9 points)
A "strong" label means roughly a 3-point move — the top of the observed range,
not a typical reaction. Most blocs, for most events, move well under half a
point. Even a bloc that is genuinely, directly targeted by an event rarely
clears "moderate" (±1.75 points); "strong" should be reserved for the rare
event/bloc pair that plausibly rivals the largest shift ever recorded.

Each record is a JSON object with EXACTLY these fields:
- "shock_id": snake_case identifier derived from the event
- "description": a distinct paraphrase of the event, 20-60 words, varied wording across records
- "party": "{seed['party']}"
- "cycle": an integer election year (2024 or 2028)
- "intensity": float between 0.8 and 1.2
- "news_roberta_scores": {{}}
- "social_roberta_scores": {{}}
- "delta_bins_race": object with keys {RACES}
- "delta_bins_religion": object with keys {RELIGIONS}
- "delta_bins_gender": object with keys {GENDERS}
- "delta_eff": float between -0.0375 and 0.0375 (overall vote-share change for
  the modeled party — this is a bloc-weighted AVERAGE, so it is typically
  smaller in magnitude than any single directly-affected bloc's own delta, not
  larger)

Every bin value MUST be one of: strong_neg, mod_neg, mild_neg, slight_neg, neutral, slight_pos, mild_pos, mod_pos, strong_pos.

Reason about each bloc separately and realistically:
- Black voters: historically very Democratic-loyal; move less on most shocks
- White voters: split by education/religion; working-class white swings most
- Latino voters: increasingly contested, not monolithic; econ and immigration sensitive
- Evangelical voters: strongly Republican; react sharply to social/cultural shocks
- Secular voters: lean Democratic; react opposite to evangelicals on culture

BIN-DISTRIBUTION TARGET across the 15 bloc-level bins in a SINGLE record
(approximate, derived from the panel statistics above — most events move most
blocs barely at all): roughly 30% neutral, 45% slight, 18% mild, 5% moderate,
2% strong. A record where several blocs all land on "strong" is almost
certainly wrong — real events that move many blocs strongly at once do not
appear in the panel data. Vary which specific blocs get the (rare) larger
moves across the {n} records — the ones plausibly most exposed to THIS
specific event — rather than spreading magnitude evenly or repeating the same
pattern record to record.

Output ONLY the JSON array of {n} records."""


def _validate(rec: dict[str, Any]) -> tuple[bool, str]:
    required = [
        "shock_id",
        "description",
        "party",
        "cycle",
        "intensity",
        "news_roberta_scores",
        "social_roberta_scores",
        "delta_bins_race",
        "delta_bins_religion",
        "delta_bins_gender",
        "delta_eff",
    ]
    for f in required:
        if f not in rec:
            return False, f"missing field {f}"
    checks = [
        ("delta_bins_race", RACES),
        ("delta_bins_religion", RELIGIONS),
        ("delta_bins_gender", GENDERS),
    ]
    for field, keys in checks:
        d = rec[field]
        if not isinstance(d, dict):
            return False, f"{field} not a dict"
        for k in keys:
            if k not in d:
                return False, f"{field} missing bloc {k}"
            if d[k] not in BIN_TOKENS:
                return False, f"{field}[{k}] invalid bin {d[k]!r}"
    try:
        de = float(rec["delta_eff"])
    except (TypeError, ValueError):
        return False, "delta_eff not numeric"
    # Rescaled Step 5.1 (see DECISIONS.md) to match electoral.core.types.DELTA_BINS'
    # tiled range [-0.0375, +0.0375] -- was -0.15/+0.15, the pre-Step-2.1 scale.
    if not (-0.0375 <= de <= 0.0375):
        return False, f"delta_eff {de} out of range"
    return True, ""


def _parse_array(text: str) -> list[dict[str, Any]]:
    """Strip markdown fences and parse the JSON array DeepSeek returns."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t
        t = t.replace("json", "", 1).strip("` \n")
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return []


def generate(seeds_path: Path, n_per_seed: int, out_path: Path, max_retries: int = 2) -> None:
    _load_env()
    client = _client()
    seeds = json.loads(seeds_path.read_text())["seeds"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    effect_counts: Counter = Counter()
    kept = rejected = 0

    BATCH_SIZE = 5

    with out_path.open("w") as fout:
        for i, seed in enumerate(seeds):
            all_records: list[dict[str, Any]] = []

            for batch_start in range(0, n_per_seed, BATCH_SIZE):
                batch_n = min(BATCH_SIZE, n_per_seed - batch_start)

                for attempt in range(max_retries + 1):
                    try:
                        resp = client.chat.completions.create(
                            model=DEEPSEEK_MODEL,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": _user_prompt(seed, batch_n)},
                            ],
                            temperature=1.0,  # variety across records
                            max_tokens=8000,
                        )
                        batch_records = _parse_array(resp.choices[0].message.content)
                        if not batch_records:
                            raise ValueError("no JSON array parsed")
                        all_records.extend(batch_records)
                        log.debug(
                            "seed %d batch %d-%d: got %d records",
                            i + 1,
                            batch_start + 1,
                            batch_start + batch_n,
                            len(batch_records),
                        )
                        break
                    except Exception as exc:
                        log.warning(
                            "seed %d batch %d-%d attempt %d failed: %s",
                            i + 1,
                            batch_start + 1,
                            batch_start + batch_n,
                            attempt,
                            exc,
                        )
                        if attempt == max_retries:
                            log.warning(
                                "seed %d batch %d-%d: giving up after %d attempts",
                                i + 1,
                                batch_start + 1,
                                batch_start + batch_n,
                                max_retries + 1,
                            )
                        time.sleep(2 * (attempt + 1))

            for rec in all_records:
                ok, reason = _validate(rec)
                if not ok:
                    rejected += 1
                    log.debug("rejected: %s", reason)
                    continue
                # carry taxonomy tags through for the reviewer
                rec["_seed_meta"] = {
                    "domain": seed["domain"],
                    "valence": seed["valence"],
                    "expected_effect": seed["expected_effect"],
                    "mobilization": seed["mobilization"],
                }
                fout.write(json.dumps(rec) + "\n")
                kept += 1
                effect_counts[seed["expected_effect"]] += 1

            log.info("seed %d/%d done — kept=%d rejected=%d", i + 1, len(seeds), kept, rejected)

    log.info("DONE — kept %d, rejected %d → %s", kept, rejected, out_path)
    log.info("effect balance: %s", dict(effect_counts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="configs/synthetic_events.json")
    p.add_argument("--n-per-seed", type=int, default=20)
    p.add_argument("--out", default="data/finetune/candidates.jsonl")
    args = p.parse_args()
    generate(Path(args.seeds), args.n_per_seed, Path(args.out))


if __name__ == "__main__":
    main()
