#!/usr/bin/env python3
"""discord_find_servers.py — single-pass political server filter for Discord dataset.zst.

Streams the zstd-compressed tar archive through Python's tarfile module in
streaming mode (``r|``), so the full archive is never decompressed to disk.
For each server file (``./GUILD_ID.json``), reads up to 64 KB, parses the
first JSON message, and checks all string values against a political/identity
keyword list.

Output TSV columns:
    guild_id  channel_name  keywords_matched

Usage
-----
    python scripts/discord_find_servers.py \\
        --dataset /scratch/JDamley28@cmc.edu/electoralData/archives/discord/dataset.zst \\
        --output  /scratch/JDamley28@cmc.edu/electoralData/archives/discord/target_servers.txt

    # Custom zstd binary path:
    python scripts/discord_find_servers.py \\
        --dataset dataset.zst --output target_servers.txt \\
        --zstd ~/.conda/envs/electoral/bin/zstd
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tarfile
from pathlib import Path

# ── Political / identity keyword filter ───────────────────────────────────────

KEYWORDS: frozenset[str] = frozenset(
    [
        "politics",
        "conservative",
        "liberal",
        "christian",
        "catholic",
        "muslim",
        "jewish",
        "black",
        "latino",
        "asian",
        "election",
        "trump",
        "biden",
        "maga",
        "progressive",
        "religion",
        "faith",
        "evangelical",
        "baptist",
    ]
)

# Read this many bytes from each server file — enough for the first message
# without buffering an entire large file into memory.
_READ_CHUNK = 65_536  # 64 KB

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("discord_find_servers")


# ── JSON helpers ──────────────────────────────────────────────────────────────


def _first_message(raw: bytes) -> dict | None:
    """Parse the first JSON object from a raw byte chunk.

    Handles two layouts found in Discord archives:
    - NDJSON: one JSON object per line
    - JSON array: ``[{...}, {...}, ...]``
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None

    # NDJSON path: try each non-empty, non-bracket line
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj[0]
        except json.JSONDecodeError:
            pass

    # JSON array path: scan for the first complete {...} block
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return None


def _collect_strings(obj: object) -> list[str]:
    """Recursively collect all string *values* from a JSON object."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for v in obj.values():
            out.extend(_collect_strings(v))
        return out
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_collect_strings(item))
        return out
    return []


def _matched_keywords(msg: dict) -> list[str]:
    """Return sorted keywords found (case-insensitive) in any string field."""
    haystack = " ".join(_collect_strings(msg)).lower()
    return sorted(kw for kw in KEYWORDS if kw in haystack)


def _channel_name(msg: dict, fallback: str) -> str:
    """Best-effort extraction of a human-readable channel or server name."""
    for field in ("channel_name", "channel", "guild_name", "server_name", "name"):
        val = msg.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return fallback


# ── Core logic ────────────────────────────────────────────────────────────────


def find_servers(dataset: Path, output: Path, zstd_bin: str = "zstd") -> None:
    """Stream dataset.zst and write matched servers to output TSV.

    One pass through the archive — no temp files, no full decompression.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    log.info("dataset  : %s", dataset)
    log.info("output   : %s", output)
    log.info("keywords : %d", len(KEYWORDS))

    proc = subprocess.Popen(
        [zstd_bin, "-d", "-c", str(dataset)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    scanned = matched = 0

    with open(output, "w", encoding="utf-8") as fh:
        fh.write("guild_id\tchannel_name\tkeywords_matched\n")

        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue

                    guild_id = Path(member.name).stem
                    scanned += 1

                    if scanned % 5_000 == 0:
                        log.info("scanned=%d  matched=%d", scanned, matched)

                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue

                    raw = fobj.read(_READ_CHUNK)
                    msg = _first_message(raw)
                    if msg is None:
                        log.debug("no message parsed: guild=%s", guild_id)
                        continue

                    hits = _matched_keywords(msg)
                    if not hits:
                        continue

                    name = _channel_name(msg, guild_id)
                    fh.write(f"{guild_id}\t{name}\t{','.join(hits)}\n")
                    matched += 1
                    log.debug("match guild=%s name=%r kw=%s", guild_id, name, hits)

        except tarfile.TarError as exc:
            log.error("tar stream error: %s", exc)
        finally:
            proc.stdout.close()
            proc.wait()

    log.info(
        "done — scanned=%d  matched=%d (%.1f%%)  output=%s",
        scanned,
        matched,
        100 * matched / scanned if scanned else 0.0,
        output,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream Discord dataset.zst and filter for political servers."
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="zstd-compressed tar archive (dataset.zst)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output TSV: guild_id, channel_name, keywords_matched",
    )
    p.add_argument(
        "--zstd",
        default="~/.conda/envs/electoral/bin/zstd",
        metavar="PATH",
        help="Path to zstd binary (default: ~/.conda/envs/electoral/bin/zstd)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    find_servers(
        dataset=args.dataset,
        output=args.output,
        zstd_bin=str(Path(args.zstd).expanduser()),
    )


if __name__ == "__main__":
    main()
