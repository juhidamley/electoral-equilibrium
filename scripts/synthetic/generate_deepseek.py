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
        --out data/finetune/synthetic_step5_5_20260809.jsonl

Env: DEEPSEEK_API_KEY (read from .env via python-dotenv if present).

HELD-OUT ENFORCEMENT (Phase 5, Step 5.5): every seed in configs/synthetic_events.json
is a fully synthetic archetype (see its `seeds` array) with no reference to any real
shock_id from configs/shocks.json -- confirmed by direct search before this step ran.
There is therefore no DIRECT leak vector here the way there was in
scripts/build_grounded_v2.py's sentiment-injection path (which pulled real per-shock
data and attached it to synthetic records). The live risk this script guards against
instead is DeepSeek spontaneously inventing a shock_id/description that collides with
a real held-out event by name -- a softer, generative-model risk rather than a lookup
bug, but still checked on every record, not just at the end: see the
assert_not_held_out() call in the write loop below, and re-checked independently by
scripts/check_held_out_leakage.py after the full run.

PROVENANCE (Hard Requirement 6): every written record carries a `_provenance` block
(run_id, scale, step, generated_at, generator, model) so no future session can mistake
this output for a legacy (pre-rescale or leaked-grounded-v2) corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from electoral.data.held_out import HeldOutShockError, assert_not_held_out

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_deepseek")

SCALE_TAG = "delta_scale_0.03"  # this run's decode-table scale, for provenance
STEP_TAG = "phase5_step5.5"

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

BIN-DISTRIBUTION TARGET across the 15 bloc-level bins in a SINGLE record —
MEASURED from the panel's own empirical bin-magnitude distribution (not
intuition; see scripts/synthetic/check_panel_target_calibration.py), most
events move most blocs barely at all: 34% neutral, 31% slight, 28% mild,
6% moderate, 1% strong. A record where several blocs all land on "strong" is
almost certainly wrong — real events that move many blocs strongly at once do
not appear in the panel data. Vary which specific blocs get the (rare) larger
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


def generate(
    seeds_path: Path,
    n_per_seed: int,
    out_path: Path,
    max_retries: int = 2,
    max_seeds: int | None = None,
    seed_indices: list[int] | None = None,
) -> None:
    _load_env()
    client = _client()
    all_seeds = json.loads(seeds_path.read_text())["seeds"]
    if seed_indices is not None:
        # Arbitrary selection (e.g. spanning domains), not just the first N --
        # --max-seeds always pulls a contiguous prefix, which silently biases a
        # pilot toward whatever domain happens to sort first in the seed file
        # (Step 5.5 pilot 1: all 3 selected seeds were domain=candidate,
        # mobilization=consolidating, the calmest archetype in the set).
        seeds = [all_seeds[i] for i in seed_indices]
        log.info(
            "--seed-indices %s: using %d explicitly selected seed(s), spanning whatever "
            "domains/mobilization types the caller chose",
            seed_indices,
            len(seeds),
        )
    elif max_seeds is not None:
        seeds = all_seeds[:max_seeds]
        log.info(
            "--max-seeds %d: using first %d of the full seed set (contiguous prefix -- "
            "prefer --seed-indices for a domain-diverse sample)",
            max_seeds,
            len(seeds),
        )
    else:
        seeds = all_seeds

    run_id = f"synth_{STEP_TAG}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    log.info("run_id=%s scale=%s -> %s", run_id, SCALE_TAG, out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    effect_counts: Counter = Counter()
    kept = rejected = held_out_blocked = 0
    total_prompt_tokens = total_completion_tokens = 0

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
                        if getattr(resp, "usage", None) is not None:
                            total_prompt_tokens += resp.usage.prompt_tokens
                            total_completion_tokens += resp.usage.completion_tokens
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

                # HARD REQUIREMENT 1: live, per-record, before-write check --
                # not a post-hoc scan. See module docstring for why this is a
                # generative-collision check, not a lookup-table leak (the
                # seeds have no real shock_id references). A single collision
                # drops that ONE record and is logged loudly; it does not
                # abort the run (Hard Requirement 3/4 spend and Requirement 1
                # both point the same way: a caught collision should be
                # removed, not used to throw away already-spent API budget on
                # the rest of a batch).
                try:
                    assert_not_held_out(
                        rec["shock_id"],
                        context=f"generate_deepseek.py output (seed {i + 1}/{len(seeds)})",
                    )
                except HeldOutShockError as exc:
                    held_out_blocked += 1
                    log.error("HELD-OUT COLLISION -- record dropped, NOT written: %s", exc)
                    continue

                # carry taxonomy tags through for the reviewer
                rec["_seed_meta"] = {
                    "domain": seed["domain"],
                    "valence": seed["valence"],
                    "expected_effect": seed["expected_effect"],
                    "mobilization": seed["mobilization"],
                }
                rec["_provenance"] = {
                    "run_id": run_id,
                    "scale": SCALE_TAG,
                    "step": STEP_TAG,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "generator": "scripts/synthetic/generate_deepseek.py",
                    "model": DEEPSEEK_MODEL,
                }
                fout.write(json.dumps(rec) + "\n")
                kept += 1
                effect_counts[seed["expected_effect"]] += 1

            log.info(
                "seed %d/%d done — kept=%d rejected=%d held_out_blocked=%d",
                i + 1,
                len(seeds),
                kept,
                rejected,
                held_out_blocked,
            )

    log.info(
        "DONE — kept %d, rejected %d, held_out_blocked %d → %s",
        kept,
        rejected,
        held_out_blocked,
        out_path,
    )
    log.info("effect balance: %s", dict(effect_counts))
    log.info(
        "token usage — prompt=%d completion=%d (DeepSeek billed usage, not a char-count estimate)",
        total_prompt_tokens,
        total_completion_tokens,
    )
    if held_out_blocked:
        log.error(
            "held_out_blocked=%d > 0 -- %d record(s) were dropped for colliding with a held-out "
            "shock id. Investigate before treating this corpus as clean; do not assume this is "
            "fine just because the run completed.",
            held_out_blocked,
            held_out_blocked,
        )


def main() -> None:
    # NOTE: no hardcoded static default for --out. data/finetune/candidates.jsonl
    # (the old default) already holds ~1.1MB of pre-rescale legacy output --
    # Hard Requirement 2 is "do NOT overwrite or merge into any existing corpus
    # file," so a default that silently pointed at that path was itself the risk
    # this step is supposed to eliminate. Require --out explicitly instead.
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="configs/synthetic_events.json")
    p.add_argument("--n-per-seed", type=int, default=20)
    p.add_argument("--out", required=True, help="Must be a fresh path -- see module docstring")
    p.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Pilot mode: only use the first N seeds (contiguous prefix -- spend control, biased toward whichever domain sorts first)",
    )
    p.add_argument(
        "--seed-indices",
        type=str,
        default=None,
        help="Pilot mode: comma-separated 0-based seed indices, e.g. '0,12,28,33,42,46,54,60' "
        "-- for a domain-diverse sample instead of a contiguous prefix. Takes precedence over --max-seeds.",
    )
    args = p.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(
            f"{out_path} already exists -- refusing to overwrite (Hard Requirement 2). "
            f"Pick a fresh, distinctly-named path."
        )
    seed_indices = [int(x) for x in args.seed_indices.split(",")] if args.seed_indices else None
    generate(
        Path(args.seeds),
        args.n_per_seed,
        out_path,
        max_seeds=args.max_seeds,
        seed_indices=seed_indices,
    )


if __name__ == "__main__":
    main()
