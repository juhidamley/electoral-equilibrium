#!/usr/bin/env python3
"""clean_with_llm.py — four-step cleaning pipeline for sampled social media posts.

Step 1 — Off-topic detection (Gemini 2.0 Flash, skipped in --dry-run):
    Batch 50 posts per prompt. Drops posts not about the shock event.
    Rate-limited to 15 RPM (4 s sleep between requests). Exponential
    backoff on 429 errors.

Step 2 — Spam filter (deterministic):
    Drops pure retweets (RT @...), emoji-only posts, bare URLs, and
    posts under 10 chars after stripping URLs and @mentions.

Step 3 — Text normalisation (deterministic):
    Expands SCOTUS/POTUS/FLOTUS/VPOTUS/MAGA. Strips URLs and @mentions.
    Normalises hashtags (#word → word). Collapses whitespace.
    Preserves original_text in payload before modification.

Step 4 — Cross-platform deduplication (deterministic):
    Keyed on SHA-1(post_id + text[:50]). Keeps first occurrence.

Output schema: same JSONL envelope (schema_version 1.0, stage="clean")
with additional payload fields: clean_stage, original_text.
Dropped posts are not written; discard rates are logged to the
archives README after each run.

Design note: Steps 2-4 are fully deterministic. Step 1 uses Gemini
2.0 Flash as a binary classification gate only — it does not produce
delta bins or feed the optimizer, so its non-determinism does not
propagate into the seed contract. See DECISIONS.md §Cleaning model.

Usage
-----
All shocks:
    python scripts/clean_with_llm.py

One shock (dry-run, steps 2-4 only):
    python scripts/clean_with_llm.py --shock-id dobbs_2022 --dry-run --max-posts 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
SHOCKS_PATH = REPO_ROOT / "configs" / "shocks.json"

DEFAULT_INPUT_DIR = Path("/Volumes/JUHIDRIVE/electoralData/sampled")
DEFAULT_OUTPUT_DIR = Path("/Volumes/JUHIDRIVE/electoralData/cleaned")
ARCHIVES_README = Path("/Volumes/JUHIDRIVE/electoralData/archives/README.md")

# ── Gemini config ──────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_SLEEP = 4.0  # seconds between requests (60 / 15 RPM = 4 s)
GEMINI_BATCH = 50  # posts per prompt
GEMINI_MAX_RETRIES = 5
GEMINI_BACKOFF_BASE = 2.0  # wait = base ** attempt (seconds)

# ── Abbreviation map (Step 3) ─────────────────────────────────────────────────

_ABBREV: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bSCOTUS\b"), "Supreme Court"),
    (re.compile(r"\bPOTUS\b"), "President"),
    (re.compile(r"\bFLOTUS\b"), "First Lady"),
    (re.compile(r"\bVPOTUS\b"), "Vice President"),
    (re.compile(r"\bMAGA\b"), "Make America Great Again"),
]

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_MULTI_WS_RE = re.compile(r"\s{2,}")
_RT_RE = re.compile(r"^RT\s+@\w+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    r"^[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F9FF\s]+$",
    re.UNICODE,
)


# ── Shock registry ─────────────────────────────────────────────────────────────


def load_shocks() -> dict[str, dict]:
    return {s["id"]: s for s in json.loads(SHOCKS_PATH.read_text())}


# ── Step 1: Off-topic detection (Gemini 2.0 Flash) ────────────────────────────


def _gemini_client():
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed — run: pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY (or GEMINI) in your environment "
            "or .env file before running without --dry-run."
        )
    return genai.Client(api_key=api_key)


def _call_with_backoff(client, prompt: str) -> str:
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = GEMINI_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Gemini 429 — waiting %.0f s (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    GEMINI_MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini call failed after {GEMINI_MAX_RETRIES} retries")


def _parse_yes_no(response_text: str, n: int) -> list[bool]:
    """Parse '1 yes / 2 no / ...' Gemini reply into a keep-flag list."""
    if not response_text or not response_text.strip():
        logger.warning("_parse_yes_no: empty Gemini response — keeping all %d posts", n)
        return [True] * n
    keep = [True] * n
    for line in response_text.strip().splitlines():
        m = re.match(r"^\s*(\d+)[.:\s)]+(\w+)", line.strip().lower())
        if m:
            idx, answer = int(m.group(1)) - 1, m.group(2)
            if 0 <= idx < n:
                keep[idx] = answer.startswith("y")
    return keep


def filter_off_topic(
    posts: list[dict],
    shock_description: str,
    client,
    dry_run: bool,
) -> tuple[list[dict], int]:
    """Return (kept_posts, n_dropped). Skips Gemini call when dry_run=True."""
    if dry_run or not posts:
        return posts, 0

    kept: list[dict] = []
    dropped = 0

    for batch_start in range(0, len(posts), GEMINI_BATCH):
        batch = posts[batch_start : batch_start + GEMINI_BATCH]
        numbered = "\n".join(f"{i + 1}. {p['payload']['text'][:300]}" for i, p in enumerate(batch))
        prompt = (
            f"For each numbered post below, reply with just the number and yes or no — "
            f"is this post about {shock_description}? "
            f"Posts that mention the topic only incidentally count as no.\n\n"
            f"{numbered}"
        )
        response = _call_with_backoff(client, prompt)
        keep_flags = _parse_yes_no(response, len(batch))

        for post, keep in zip(batch, keep_flags):
            if keep:
                kept.append(post)
            else:
                dropped += 1

        time.sleep(GEMINI_SLEEP)

    return kept, dropped


# ── Step 2: Spam filter (deterministic) ───────────────────────────────────────


def _stripped_len(text: str) -> int:
    return len(_MENTION_RE.sub("", _URL_RE.sub("", text)).strip())


def _is_spam(text: str) -> bool:
    t = text.strip()
    if _RT_RE.match(t):
        return True
    if _EMOJI_RE.match(t):
        return True
    if re.fullmatch(r"https?://\S+", t):  # bare URL, no commentary
        return True
    if _stripped_len(t) < 10:
        return True
    return False


def filter_spam(posts: list[dict]) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for p in posts:
        if _is_spam(p["payload"]["text"]):
            dropped += 1
        else:
            kept.append(p)
    return kept, dropped


# ── Step 3: Text normalisation (deterministic) ────────────────────────────────


def normalise(text: str) -> str:
    for pattern, replacement in _ABBREV:
        text = pattern.sub(replacement, text)
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


def apply_normalisation(posts: list[dict]) -> list[dict]:
    for p in posts:
        original = p["payload"]["text"]
        p["payload"]["original_text"] = original
        p["payload"]["text"] = normalise(original)
        p["payload"]["clean_stage"] = "cleaned"
    return posts


# ── Step 4: Deduplication (deterministic) ─────────────────────────────────────


def deduplicate(posts: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    kept, dropped = [], 0
    for p in posts:
        payload = p["payload"]
        uid = str(payload.get("id") or payload.get("post_id") or "")
        key = hashlib.sha1(f"{uid}{payload.get('text', '')[:50]}".encode()).hexdigest()
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            kept.append(p)
    return kept, dropped


# ── Output ─────────────────────────────────────────────────────────────────────


def write_cleaned(posts: list[dict], output_path: Path, seed: int = 42) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc).isoformat()
    with open(output_path, "w", encoding="utf-8") as f:
        for p in posts:
            envelope = {
                "schema_version": "1.0",
                "collected_at": now,
                "stage": "clean",
                "seed": seed,
                "payload": p["payload"],
            }
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")


def append_readme_log(stats: list[dict], readme_path: Path, dry_run: bool) -> None:
    if not readme_path.exists():
        logger.warning("archives README not found at %s — skipping discard log", readme_path)
        return
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = [
        f"\n## clean_with_llm.py — {ts}" + (" (dry-run)" if dry_run else ""),
        "",
        "| shock_id | archive_id | input | off_topic ▼ | spam ▼ | dedup ▼ | output |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        rows.append(
            f"| {s['shock_id']} | {s['archive_id']} | {s['n_input']} "
            f"| {s['off_topic_dropped']} | {s['spam_dropped']} "
            f"| {s['dedup_dropped']} | {s['n_output']} |"
        )
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    logger.info("Discard rates appended to %s", readme_path)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Four-step LLM cleaning pipeline")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--shock-id", default=None, help="Process only this shock slug (default: all)")
    p.add_argument(
        "--dry-run", action="store_true", help="Skip Gemini off-topic filter (steps 2-4 only)"
    )
    p.add_argument(
        "--max-posts", type=int, default=None, help="Cap total posts processed (for testing)"
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    shocks_by_id = load_shocks()

    if not args.input_dir.exists():
        logger.error("Input directory not found: %s", args.input_dir)
        return

    # Group files by (shock_id, archive_id)
    # Expected layout: {input_dir}/{shock_id}/{archive_id}.jsonl
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for jsonl in sorted(args.input_dir.rglob("*.jsonl")):
        shock_id = jsonl.parent.name
        archive_id = jsonl.stem
        if args.shock_id and shock_id != args.shock_id:
            continue
        groups[(shock_id, archive_id)].append(jsonl)

    if not groups:
        logger.error("No JSONL files found under %s", args.input_dir)
        return

    client = None
    if not args.dry_run:
        client = _gemini_client()
        logger.info("Gemini client ready (model=%s, %.0f s/request)", GEMINI_MODEL, GEMINI_SLEEP)
    else:
        logger.info("--dry-run: Gemini off-topic filter skipped")

    all_stats: list[dict] = []
    total_processed = 0

    for (shock_id, archive_id), paths in sorted(groups.items()):
        shock = shocks_by_id.get(shock_id, {})
        shock_desc = shock.get("description", shock_id)

        posts: list[dict] = []
        for path in paths:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        posts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if args.max_posts and total_processed + len(posts) >= args.max_posts:
                posts = posts[: args.max_posts - total_processed]
                break

        if not posts:
            continue

        n_input = len(posts)
        logger.info("[%s / %s] %d posts loaded", shock_id, archive_id, n_input)

        # Step 1 — off-topic (Gemini)
        posts, off_topic_dropped = filter_off_topic(posts, shock_desc, client, args.dry_run)
        logger.info("  step 1 off-topic : -%d → %d kept", off_topic_dropped, len(posts))

        # Step 2 — spam
        posts, spam_dropped = filter_spam(posts)
        logger.info("  step 2 spam      : -%d → %d kept", spam_dropped, len(posts))

        # Step 3 — normalise (in-place, adds original_text and clean_stage)
        posts = apply_normalisation(posts)

        # Step 4 — dedup
        posts, dedup_dropped = deduplicate(posts)
        logger.info("  step 4 dedup     : -%d → %d kept", dedup_dropped, len(posts))

        output_path = args.output_dir / shock_id / f"{archive_id}.jsonl"
        write_cleaned(posts, output_path)
        logger.info("  → %s", output_path)

        all_stats.append(
            {
                "shock_id": shock_id,
                "archive_id": archive_id,
                "n_input": n_input,
                "off_topic_dropped": off_topic_dropped,
                "spam_dropped": spam_dropped,
                "dedup_dropped": dedup_dropped,
                "n_output": len(posts),
            }
        )

        total_processed += n_input
        if args.max_posts and total_processed >= args.max_posts:
            logger.info("--max-posts %d reached, stopping", args.max_posts)
            break

    if all_stats:
        append_readme_log(all_stats, ARCHIVES_README, dry_run=args.dry_run)
        total_in = sum(s["n_input"] for s in all_stats)
        total_out = sum(s["n_output"] for s in all_stats)
        logger.info(
            "Done — %d → %d posts (%.1f%% kept)",
            total_in,
            total_out,
            100 * total_out / total_in if total_in else 0,
        )


if __name__ == "__main__":
    main()
