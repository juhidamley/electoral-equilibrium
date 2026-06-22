"""verify_artifacts: compare frozen paper-baseline artifacts against a fresh run.

Usage:
    python scripts/verify_artifacts.py [frozen_dir] fresh_dir

Defaults:
    frozen_dir = artifacts/paper_baseline/
    fresh_dir  = (required positional arg)

Exit codes:
    0 — all compared fields match within tolerance
    1 — one or more mismatches, missing files, or non-finite values in frozen

Ignore-set discipline:
    _IGNORE_FIELDS is intentionally empty. Any field that differs is a finding.
    Add an entry here only with a documented reason — not as a convenience
    escape hatch. The envelope-level 'metadata' key is unconditionally skipped
    (it may contain a creation timestamp); everything in 'data' is fair game.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("verify_artifacts")

# ── Ignore-set discipline ─────────────────────────────────────────────────────
# Intentionally empty. Every data-payload field is compared. To exclude a field,
# add its name here with a one-line comment explaining why it's non-deterministic.
# Candidates if needed in the future:
#   "run_key" — only if paper_baseline.json uses a UUID key instead of a fixed string.
_IGNORE_FIELDS: frozenset[str] = frozenset()

# Float tolerance: seeded runs can differ in the last bit across BLAS versions.
# Log loudly if the gap exceeds 1e-6 — that's a real non-determinism finding.
_FLOAT_ATOL: float = 1e-9
_FLOAT_WARN_THRESHOLD: float = 1e-6

# ── Known static artifact filenames ──────────────────────────────────────────
# These exist in every run regardless of shock. Per-shock files are discovered
# dynamically from the filesystem (see _discover_shock_files).
_STATIC_FILES: list[str] = [
    "voter_panel.json",
    "baseline_portfolio.json",
    "sentiment_data.json",
    "llm_finetune.json",
    "shock_response.json",   # legacy single-shock name
    "equilibrium.json",      # legacy single-shock name
    "optimization.json",
    "simulation.json",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _walk_nonfinite(obj: Any, path: str = "") -> Iterator[str]:
    """Yield dotted-path=value strings for every non-finite float in obj."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            yield f"{path}={obj!r}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_nonfinite(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_nonfinite(v, f"{path}[{i}]")


def _compare_values(
    v1: Any, v2: Any, path: str, atol: float = _FLOAT_ATOL
) -> list[str]:
    """Recursively compare v1 (frozen) vs v2 (fresh). Return diff descriptions."""
    diffs: list[str] = []

    # Allow int/float interop — both become floats for numeric comparison.
    both_numeric = isinstance(v1, (int, float)) and isinstance(v2, (int, float))
    if not both_numeric and type(v1) != type(v2):
        diffs.append(
            f"{path}: type mismatch {type(v1).__name__} vs {type(v2).__name__} "
            f"— frozen={v1!r}  fresh={v2!r}"
        )
        return diffs

    if both_numeric:
        f1, f2 = float(v1), float(v2)
        if not math.isfinite(f1) or not math.isfinite(f2):
            if f1 != f2:
                diffs.append(
                    f"{path}: non-finite mismatch — frozen={f1!r}  fresh={f2!r}"
                )
        elif abs(f1 - f2) > atol:
            delta = abs(f1 - f2)
            diffs.append(
                f"{path}: numeric diff {delta:.3e} (atol={atol:.0e}) "
                f"— frozen={f1!r}  fresh={f2!r}"
            )
            if delta > _FLOAT_WARN_THRESHOLD:
                log.warning(
                    "LARGE DIFF at %s: frozen=%r fresh=%r (delta=%.3e) — "
                    "non-determinism finding, investigate before tagging",
                    path, f1, f2, delta,
                )

    elif isinstance(v1, str):
        if v1 != v2:
            diffs.append(f"{path}: string mismatch — frozen={v1!r}  fresh={v2!r}")

    elif isinstance(v1, bool):
        if v1 != v2:
            diffs.append(f"{path}: bool mismatch — frozen={v1!r}  fresh={v2!r}")

    elif isinstance(v1, dict):
        all_keys = set(v1) | set(v2)
        for k in sorted(all_keys):
            sub = f"{path}.{k}"
            if k not in v1:
                diffs.append(f"{sub}: present in fresh only (value={v2[k]!r})")
            elif k not in v2:
                diffs.append(f"{sub}: present in frozen only (value={v1[k]!r})")
            else:
                diffs.extend(_compare_values(v1[k], v2[k], sub, atol))

    elif isinstance(v1, list):
        if len(v1) != len(v2):
            diffs.append(
                f"{path}: list length mismatch — frozen={len(v1)}  fresh={len(v2)}"
            )
        else:
            for i, (a, b) in enumerate(zip(v1, v2)):
                diffs.extend(_compare_values(a, b, f"{path}[{i}]", atol))

    else:
        if v1 != v2:
            diffs.append(f"{path}: mismatch — frozen={v1!r}  fresh={v2!r}")

    return diffs


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("JSON parse error in %s: %s", path, exc)
        return None


def _compare_artifact(
    name: str,
    frozen_path: Path,
    fresh_path: Path,
    atol: float = _FLOAT_ATOL,
) -> tuple[str, list[str]]:
    """Compare one artifact. Returns (status, list_of_diff_strings)."""
    frozen = _load_json(frozen_path)
    fresh = _load_json(fresh_path)

    if frozen is None and fresh is None:
        return "BOTH_MISSING", [
            f"Neither {frozen_path} nor {fresh_path} exist — "
            "run the pipeline first to produce both artifact sets"
        ]
    if frozen is None:
        return "MISSING_FROZEN", [f"Frozen artifact not found: {frozen_path}"]
    if fresh is None:
        return "MISSING_FRESH", [f"Fresh artifact not found: {fresh_path}"]

    diffs: list[str] = []

    # Envelope-level sanity: stage must match, run_key must match.
    # metadata is skipped unconditionally (may hold creation timestamp).
    if frozen.get("stage") != fresh.get("stage"):
        diffs.append(
            f"envelope.stage mismatch: frozen={frozen.get('stage')!r}  "
            f"fresh={fresh.get('stage')!r}"
        )
    if frozen.get("run_key") != fresh.get("run_key"):
        diffs.append(
            f"envelope.run_key mismatch: frozen={frozen.get('run_key')!r}  "
            f"fresh={fresh.get('run_key')!r}"
        )

    frozen_data: dict = frozen.get("data", {})
    fresh_data: dict = fresh.get("data", {})

    # Non-finite values in the frozen artifact are a serialization bug.
    nf_hits = list(_walk_nonfinite(frozen_data, "frozen.data"))
    if nf_hits:
        for hit in nf_hits:
            log.warning("Non-finite in FROZEN %s: %s", name, hit)
        diffs.extend(f"non-finite in frozen: {h}" for h in nf_hits)

    # Field-by-field data comparison, respecting ignore-set.
    for field, fv in frozen_data.items():
        if field in _IGNORE_FIELDS:
            continue
        if field not in fresh_data:
            diffs.append(f"data.{field}: present in frozen only")
            continue
        diffs.extend(_compare_values(fv, fresh_data[field], f"data.{field}", atol))

    for field in fresh_data:
        if field not in frozen_data and field not in _IGNORE_FIELDS:
            diffs.append(
                f"data.{field}: present in fresh only (value={fresh_data[field]!r})"
            )

    return ("FAIL", diffs) if diffs else ("PASS", [])


# ── Dynamic per-shock file discovery ─────────────────────────────────────────

def _discover_shock_files(frozen_dir: Path, fresh_dir: Path) -> list[str]:
    """Find per-shock filenames (shock_{id}.json, equilibrium_{id}.json) in either dir."""
    found: set[str] = set()
    for d in (frozen_dir, fresh_dir):
        if not d.exists():
            continue
        for f in d.glob("shock_*.json"):
            if f.name != "shock_response.json":
                found.add(f.name)
        for f in d.glob("equilibrium_*.json"):
            if f.name != "equilibrium.json":
                found.add(f.name)
    return sorted(found)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare frozen paper-baseline artifacts against a fresh run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "frozen_dir",
        nargs="?",
        default="artifacts/paper_baseline",
        help="Directory of frozen (ground-truth) artifacts (default: artifacts/paper_baseline/)",
    )
    parser.add_argument(
        "fresh_dir",
        help="Directory of freshly produced artifacts to verify",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=_FLOAT_ATOL,
        metavar="TOL",
        help=f"Float absolute tolerance for numeric fields (default: {_FLOAT_ATOL:.0e})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print field-by-field breakdown even for PASS artifacts",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    frozen_dir = Path(args.frozen_dir)
    fresh_dir = Path(args.fresh_dir)

    dynamic_files = _discover_shock_files(frozen_dir, fresh_dir)
    all_files = _STATIC_FILES + dynamic_files

    print(f"\nverify_artifacts")
    print(f"  frozen : {frozen_dir.resolve()}")
    print(f"  fresh  : {fresh_dir.resolve()}")
    print(f"  atol   : {args.atol:.0e}")
    print(f"  ignore : {_IGNORE_FIELDS or 'frozenset()  (empty — all fields compared)'}")
    if dynamic_files:
        print(f"  dynamic: {', '.join(dynamic_files)}")
    print(f"{'─' * 72}")
    print(f"  {'Artifact':<44}  {'Status'}")
    print(f"{'─' * 72}")

    results: list[tuple[str, str, list[str]]] = []
    for name in all_files:
        status, diffs = _compare_artifact(
            name,
            frozen_dir / name,
            fresh_dir / name,
            atol=args.atol,
        )
        results.append((name, status, diffs))

        icon = "PASS" if status == "PASS" else status
        marker = "  " if status == "PASS" else "! "
        print(f"{marker} {name:<44}  {icon}")
        if diffs and (status != "PASS" or args.verbose):
            for d in diffs:
                print(f"      {d}")

    print(f"{'─' * 72}")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_total = len(results)
    any_fail = n_pass < n_total
    verdict = "PASS" if not any_fail else "FAIL"
    print(f"  Overall verdict: {verdict}  ({n_pass}/{n_total} artifacts match)")
    print()

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
