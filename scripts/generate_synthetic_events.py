#!/usr/bin/env python3
"""Generate synthetic fine-tuning examples via the DeepSeek API.

Usage:
    python scripts/generate_synthetic_events.py \
        --event-list configs/synthetic_events.json \
        --n-per-event 20 \
        --output data/finetune/synthetic_deepseek.jsonl

Requires:
    pip install openai
    export DEEPSEEK_API_KEY=sk-...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ── Canonical vocabulary ──────────────────────────────────────────────────────

VALID_BINS = frozenset([
    "strong_neg", "mod_neg", "mild_neg", "slight_neg", "neutral",
    "slight_pos", "mild_pos", "mod_pos", "strong_pos",
])

RACE_KEYS = ("african_american", "asian", "latino", "other_race", "white")
RELIGION_KEYS = ("evangelical", "catholic", "protestant", "secular",
                 "jewish", "muslim", "other_rel")
GENDER_KEYS = ("women", "men", "other_gender")

REQUIRED_TOP_KEYS = {
    "shock_id", "description", "party", "cycle", "intensity",
    "news_roberta_scores", "social_roberta_scores",
    "delta_bins_race", "delta_bins_religion", "delta_bins_gender",
    "delta_eff",
}

# ── Prompt builders ───────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert in American electoral politics and demographic voting "
    "behavior. Generate synthetic training data for an electoral forecasting "
    "model. Respond ONLY with valid JSON, no markdown, no preamble."
)


def _user_prompt(event: dict, n: int) -> str:
    party = event["party"]
    cycle = event["cycle"]
    desc = event["description"]
    return (
        f'Generate {n} synthetic electoral shock response records for this event: "{desc}"\n\n'
        f"Party: {party}\n"
        f"Cycle: {cycle}\n\n"
        "For each record output a JSON object with these exact fields:\n"
        "- shock_id: snake_case event identifier\n"
        f'- description: varied paraphrase of the event (20-60 words)\n'
        f'- party: "{party}"\n'
        f"- cycle: {cycle}\n"
        "- intensity: float between 0.8 and 1.2\n"
        "- news_roberta_scores: {}\n"
        "- social_roberta_scores: {}\n"
        "- delta_bins_race: dict with keys african_american, asian, latino, "
        "other_race, white. Values must be one of: strong_neg, mod_neg, "
        "mild_neg, slight_neg, neutral, slight_pos, mild_pos, mod_pos, strong_pos\n"
        "- delta_bins_religion: dict with keys evangelical, catholic, protestant, "
        "secular, jewish, muslim, other_rel. Same values.\n"
        "- delta_bins_gender: dict with keys women, men, other_gender. Same values.\n"
        "- delta_eff: float between -0.15 and 0.15 representing overall vote share change\n\n"
        f"Output a JSON array of {n} records. Apply genuine political domain knowledge "
        "about how this event would affect each demographic bloc."
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_record(rec: dict) -> list[str]:
    """Return list of error strings; empty list means valid."""
    errors: list[str] = []

    missing = REQUIRED_TOP_KEYS - set(rec.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
        return errors  # can't check sub-fields if top-level is broken

    for strat, keys in (
        ("delta_bins_race", RACE_KEYS),
        ("delta_bins_religion", RELIGION_KEYS),
        ("delta_bins_gender", GENDER_KEYS),
    ):
        bins = rec.get(strat)
        if not isinstance(bins, dict):
            errors.append(f"{strat} is not a dict")
            continue
        missing_keys = set(keys) - set(bins.keys())
        if missing_keys:
            errors.append(f"{strat} missing keys: {sorted(missing_keys)}")
        for k, v in bins.items():
            if v not in VALID_BINS:
                errors.append(f"{strat}[{k!r}] invalid bin: {v!r}")

    delta_eff = rec.get("delta_eff")
    if not isinstance(delta_eff, (int, float)):
        errors.append(f"delta_eff is not numeric: {delta_eff!r}")
    elif not (-0.15 <= float(delta_eff) <= 0.15):
        errors.append(f"delta_eff out of range: {delta_eff}")

    intensity = rec.get("intensity")
    if not isinstance(intensity, (int, float)):
        errors.append(f"intensity is not numeric: {intensity!r}")
    elif not (0.8 <= float(intensity) <= 1.2):
        errors.append(f"intensity out of range: {intensity}")

    return errors


# ── API call ──────────────────────────────────────────────────────────────────

def _call_deepseek(client, event: dict, n: int, retries: int = 3) -> list[dict]:
    """Call DeepSeek and return parsed list of records. Raises on unrecoverable failure."""
    prompt = _user_prompt(event, n)
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=8192,
            )
            content = response.choices[0].message.content.strip()
            # Strip accidental markdown fences
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )
            records = json.loads(content)
            if not isinstance(records, list):
                raise ValueError(f"Expected JSON array, got {type(records).__name__}")
            return records
        except json.JSONDecodeError as exc:
            log.warning("attempt %d/%d: JSON parse error — %s", attempt, retries, exc)
        except Exception as exc:
            log.warning("attempt %d/%d: API error — %s", attempt, retries, exc)
        if attempt < retries:
            time.sleep(2 ** attempt)  # exponential back-off: 2s, 4s
    raise RuntimeError(
        f"DeepSeek call failed after {retries} attempts for event: {event['shock_id']!r}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate synthetic fine-tuning examples via DeepSeek API"
    )
    parser.add_argument(
        "--event-list",
        default="configs/synthetic_events.json",
        help="JSON file with list of event objects (default: configs/synthetic_events.json)",
    )
    parser.add_argument(
        "--n-per-event",
        type=int,
        default=20,
        help="Synthetic records to generate per event (default: 20)",
    )
    parser.add_argument(
        "--output",
        default="data/finetune/synthetic_deepseek.jsonl",
        help="Output JSONL path (default: data/finetune/synthetic_deepseek.jsonl)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.error("DEEPSEEK_API_KEY environment variable not set")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package not installed — run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    event_path = Path(args.event_list)
    if not event_path.exists():
        log.error("event list not found: %s", event_path)
        sys.exit(1)
    events: list[dict] = json.loads(event_path.read_text(encoding="utf-8"))
    log.info("loaded %d events from %s", len(events), event_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_rejected = 0

    with out_path.open("a", encoding="utf-8") as fh:
        for event in events:
            shock_id = event.get("shock_id", "unknown")
            log.info("processing event: %s", shock_id)

            try:
                records = _call_deepseek(client, event, args.n_per_event)
            except RuntimeError as exc:
                log.error("skipping event %r: %s", shock_id, exc)
                continue

            accepted = 0
            rejected = 0
            for rec in records:
                if not isinstance(rec, dict):
                    log.warning("  skipping non-dict record: %r", rec)
                    rejected += 1
                    continue
                errors = _validate_record(rec)
                if errors:
                    log.warning(
                        "  rejected record for %r: %s",
                        shock_id,
                        "; ".join(errors),
                    )
                    rejected += 1
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                accepted += 1

            log.info(
                "  event %r: %d accepted, %d rejected (of %d returned)",
                shock_id,
                accepted,
                rejected,
                len(records),
            )
            total_generated += accepted
            total_rejected += rejected

    log.info(
        "done — %d records written to %s (%d rejected total)",
        total_generated,
        out_path,
        total_rejected,
    )


if __name__ == "__main__":
    main()
