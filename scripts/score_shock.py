"""Score a single shock's merged posts with RoBERTa and write per-post scores.

Usage
-----
    python scripts/score_shock.py \
        --shock-id metoo_2017 \
        --merged-dir /scratch/.../merged \
        --output-dir /scratch/.../scored \
        --cache-dir  /scratch/.../embeddings_cache \
        --batch-size 64

Input
-----
{merged_dir}/{shock_id}/posts.jsonl
  Each line is a JSON record. Supports two formats:
  - Canonical envelope: {"schema_version":"1.0","payload":{"text":"...","post_id":"...",...}}
  - Flat:               {"text":"...","id":"...","lang":"...",...}

Output
------
{output_dir}/{shock_id}/scored.jsonl
  All original fields preserved. Top-level "roberta_score" key added (float ∈ [-1, 1]).

Cache
-----
The embedding cache is partitioned per shock to avoid concurrent write conflicts between
array tasks: {cache_dir}/{shock_id}/cache.parquet

Resume
------
With --resume: skip if {output_dir}/{shock_id}/scored.jsonl already exists.
Without --resume (default): overwrite any existing output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
log = logging.getLogger("score_shock")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoBERTa per-post scorer for a single shock.")
    p.add_argument("--shock-id", required=True, help="Shock identifier (e.g. metoo_2017).")
    p.add_argument(
        "--merged-dir", required=True, help="Root dir containing {shock_id}/posts.jsonl."
    )
    p.add_argument(
        "--output-dir", required=True, help="Root dir for {shock_id}/scored.jsonl output."
    )
    p.add_argument(
        "--cache-dir", required=True, help="Root dir for embedding cache (partitioned per shock)."
    )
    p.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    p.add_argument("--resume", action="store_true", help="Skip if output already exists.")
    p.add_argument("--device", default=None, help="Force device: cpu | cuda | mps.")
    return p.parse_args()


def _extract_text_and_id(record: dict) -> tuple[str, str | None]:
    """Return (text, post_id) from either envelope or flat format."""
    if "payload" in record:
        payload = record["payload"]
        text = payload.get("text", "")
        pid = payload.get("post_id") or payload.get("id")
    else:
        text = record.get("text", "")
        pid = record.get("post_id") or record.get("id")
    return str(text), (str(pid) if pid else None)


def main() -> None:
    args = _parse_args()

    shock_id: str = args.shock_id
    input_path = Path(args.merged_dir) / shock_id / "posts.jsonl"
    output_path = Path(args.output_dir) / shock_id / "scored.jsonl"
    cache_path = Path(args.cache_dir) / shock_id  # per-shock partition

    if args.resume and output_path.exists():
        log.info("--resume: output exists at %s — skipping.", output_path)
        return

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        sys.exit(1)

    # ── Load posts ────────────────────────────────────────────────────────────
    t_load = time.perf_counter()
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSON at line %d: %s", lineno, exc)

    n_total = len(records)
    log.info(
        "shock=%s  loaded %d posts from %s  (%.1fs)",
        shock_id,
        n_total,
        input_path,
        time.perf_counter() - t_load,
    )

    if n_total == 0:
        log.warning("No posts to score for shock=%s — writing empty output.", shock_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return

    # ── Extract texts and IDs ─────────────────────────────────────────────────
    texts: list[str] = []
    post_ids: list[str | None] = []
    for rec in records:
        text, pid = _extract_text_and_id(rec)
        texts.append(text)
        post_ids.append(pid)

    # ── Score ─────────────────────────────────────────────────────────────────
    sys.path.insert(0, str(_REPO_ROOT))
    try:
        from electoral.nlp.scorer import RoBERTaScorer
    except ImportError as exc:
        log.error("Cannot import electoral package from %s: %s", _REPO_ROOT, exc)
        sys.exit(1)

    cache_path.mkdir(parents=True, exist_ok=True)
    scorer = RoBERTaScorer(
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=cache_path,
    )

    t_score = time.perf_counter()
    scores: list[float] = scorer.score_texts(texts, post_ids=post_ids)
    elapsed = time.perf_counter() - t_score

    hit_rate = scorer._cache.hit_rate
    log.info(
        "shock=%s  scored %d posts in %.1fs (%.0f posts/s)  cache_hit_rate=%.1f%%",
        shock_id,
        n_total,
        elapsed,
        n_total / elapsed if elapsed > 0 else 0,
        100 * hit_rate,
    )

    # ── Write output ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_write = time.perf_counter()
    with open(output_path, "w", encoding="utf-8") as fh:
        for record, score in zip(records, scores):
            record["roberta_score"] = round(float(score), 6)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(
        "shock=%s  wrote %d scored records to %s  (%.1fs)",
        shock_id,
        n_total,
        output_path,
        time.perf_counter() - t_write,
    )
    log.info(
        "shock=%s  DONE  total_posts=%d  cache_hit_rate=%.1f%%  score_time=%.1fs",
        shock_id,
        n_total,
        100 * hit_rate,
        elapsed,
    )


if __name__ == "__main__":
    main()
