#!/usr/bin/env python3
"""Scan existing training corpora for held-out shock leakage.

Phase 1, Step 1.6. Read-only w.r.t. the corpora it scans -- this script never
modifies data/finetune/*.jsonl. It exists to answer one question honestly:
does anything currently in the training corpora reference a shock now frozen
in configs/held_out_shocks.json, either DIRECTLY (a real held-out shock_id
used as a training record) or INDIRECTLY (a held-out shock's sentiment
injected into a synthetic record via the thematic-grounding mechanism in
scripts/build_grounded_v2.py)?

Per Step 1.6's brief: if existing corpora leak, report which and how many;
do NOT fix them silently. This script only reports.

Detection covers every provenance field found in data/finetune/*.jsonl by
direct inspection (see docs/held_out_leakage_investigation in the commit
history / task notes for how these were found):
  - top-level `shock_id`            -- direct use of a real shock as a record
  - `_grounding.grounded_from`      -- v1 grounding provenance (rare, mostly null)
  - `_grounding_v2.source_shock`    -- v2 grounding: real shock sentiment injected
                                        into a synthetic record's social_roberta_scores
  - `_grounding.real_signal`        -- checked but is a description string, not an
                                        id, in every sample seen; included defensively
                                        in case a future writer puts an id there

A record's top-level `shock_id` is only meaningful as a "direct leak" signal
in files where that field holds REAL shock ids (grounded_aligned.jsonl, and
any future file following that convention) -- in train/eval/synthetic*/
candidates/reviewed_approved/human_review_queue, `shock_id` holds SYNTHETIC
descriptive slugs (e.g. "mass_legalization_program") that do not collide with
real ids by construction of the two different naming conventions. This script
does not assume that distinction -- it checks every file's shock_id values
against the real held-out id set either way, so a slug that happened to
collide would still be caught.

Usage:
    python scripts/check_held_out_leakage.py
    python scripts/check_held_out_leakage.py --out data/leakage_reports/2026-07-31.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))
from electoral.data.held_out import held_out_shock_ids  # noqa: E402

FINETUNE_DIR = REPO_ROOT / "data" / "finetune"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "leakage_reports"


def find_corpus_files() -> list[Path]:
    """Every *.jsonl under data/finetune/ -- deliberately a glob, not a
    hardcoded list, so a new corpus file added later gets scanned too."""
    return sorted(FINETUNE_DIR.glob("*.jsonl"))


def _extract_grounding_shock_refs(record: dict) -> list[tuple[str, str]]:
    """Returns [(field_path, shock_id), ...] for every real-shock reference
    found in a record's grounding provenance fields, direct or indirect.
    """
    refs = []
    top_shock_id = record.get("shock_id")
    if isinstance(top_shock_id, str) and top_shock_id:
        refs.append(("shock_id", top_shock_id))

    g = record.get("_grounding")
    if isinstance(g, dict):
        gf = g.get("grounded_from")
        if isinstance(gf, str) and gf:
            refs.append(("_grounding.grounded_from", gf))
        rs = g.get("real_signal")
        if isinstance(rs, str) and rs:
            refs.append(("_grounding.real_signal", rs))  # defensive; usually a description, not an id

    g2 = record.get("_grounding_v2")
    if isinstance(g2, dict):
        ss = g2.get("source_shock")
        if isinstance(ss, str) and ss:
            refs.append(("_grounding_v2.source_shock", ss))

    return refs


def scan_file(path: Path, held_out: frozenset[str]) -> dict:
    n_records = 0
    n_parse_errors = 0
    direct_leaks: list[dict] = []
    indirect_leaks: list[dict] = []
    field_paths_seen: set[str] = set()
    historical_exclusion_note = None

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_parse_errors += 1
                continue
            n_records += 1
            if not isinstance(record, dict):
                continue

            # informational only: does this record's OWN claimed exclusion list
            # (v1 grounding provenance) match today's frozen set? A mismatch here
            # is exactly how the blm_george_floyd_2020/kavanaugh_2018 regression
            # between grounding-script v1 and v2 was found.
            g = record.get("_grounding")
            if isinstance(g, dict) and isinstance(g.get("held_out_excluded"), list) and historical_exclusion_note is None:
                claimed = set(g["held_out_excluded"])
                dropped = sorted(claimed - held_out)  # excluded back then, NOT excluded now (fine, config only grows)
                added_since = sorted(held_out - claimed)  # excluded now, wasn't back then (expected, config grew)
                historical_exclusion_note = {
                    "field": "_grounding.held_out_excluded",
                    "claimed_excluded_at_write_time": sorted(claimed),
                    "in_current_config_too": sorted(claimed & held_out),
                    "in_claimed_but_not_current_config": dropped,
                }

            for field_path, shock_id in _extract_grounding_shock_refs(record):
                field_paths_seen.add(field_path)
                if shock_id in held_out:
                    entry = {
                        "line": line_no,
                        "record_shock_id": record.get("shock_id"),
                        "field": field_path,
                        "held_out_shock": shock_id,
                    }
                    if field_path == "shock_id":
                        direct_leaks.append(entry)
                    else:
                        indirect_leaks.append(entry)

    leaked_ids = sorted({e["held_out_shock"] for e in direct_leaks + indirect_leaks})
    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "n_records": n_records,
        "n_parse_errors": n_parse_errors,
        "fields_inspected": sorted(field_paths_seen),
        "n_direct_leaks": len(direct_leaks),
        "n_indirect_leaks": len(indirect_leaks),
        "leaked_held_out_ids": leaked_ids,
        "direct_leak_detail": direct_leaks,
        "indirect_leak_detail": indirect_leaks,
        "historical_exclusion_note": historical_exclusion_note,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="Path to write the full JSON report (default: data/leakage_reports/<UTC date>.json)")
    args = ap.parse_args()

    held_out = held_out_shock_ids()
    files = find_corpus_files()
    print(f"Held-out set: {len(held_out)} shocks (configs/held_out_shocks.json)")
    print(f"Scanning {len(files)} corpus file(s) under {FINETUNE_DIR.relative_to(REPO_ROOT)}/\n")

    results = [scan_file(p, held_out) for p in files]

    total_direct = sum(r["n_direct_leaks"] for r in results)
    total_indirect = sum(r["n_indirect_leaks"] for r in results)
    all_leaked_ids: set[str] = set()
    for r in results:
        all_leaked_ids.update(r["leaked_held_out_ids"])

    print(f"{'FILE':45s} {'RECORDS':>8s} {'DIRECT':>8s} {'INDIRECT':>9s}  LEAKED HELD-OUT IDS")
    print("-" * 130)
    for r in results:
        flag = " <-- LEAK" if (r["n_direct_leaks"] or r["n_indirect_leaks"]) else ""
        print(f"{r['file']:45s} {r['n_records']:8d} {r['n_direct_leaks']:8d} {r['n_indirect_leaks']:9d}  "
              f"{', '.join(r['leaked_held_out_ids'])}{flag}")

    print()
    print("=" * 70)
    if total_direct == 0 and total_indirect == 0:
        print("RESULT: no held-out shock leakage detected in current corpora.")
    else:
        print(f"RESULT: LEAKAGE DETECTED. {total_direct} direct + {total_indirect} indirect "
              f"leaked record(s), spanning {len(all_leaked_ids)} distinct held-out shock(s):")
        print(f"  {sorted(all_leaked_ids)}")
        print()
        print("This is NOT being fixed by this script. Per configs/held_out_shocks.json's")
        print("_meta.scope, this is a disclosed, known defect in existing corpora, to be")
        print("corrected in a future corpus rebuild -- not silently patched here.")
    print("=" * 70)

    # Cross-file regression check: did any shock that a v1-provenance file
    # claims it excluded show up as a LEAK (direct or indirect) in a
    # DIFFERENT file? That's the specific failure mode already found by hand
    # while building this: blm_george_floyd_2020 and kavanaugh_2018 were
    # excluded by name in synthetic_grounded.jsonl's v1 provenance, then
    # injected anyway in synthetic_grounded_v2.jsonl/train_grounded_v2.jsonl
    # /eval_grounded_v2.jsonl once the grounding script was rewritten.
    v1_claimed_excluded: set[str] = set()
    indirect_leaked_ids: set[str] = set()
    for r in results:
        note = r["historical_exclusion_note"]
        if note:
            v1_claimed_excluded.update(note["claimed_excluded_at_write_time"])
            print(f"\n[{r['file']}] carries a v1 `_grounding.held_out_excluded` provenance field.")
            print(f"  Claimed excluded at write time: {note['claimed_excluded_at_write_time']}")
        indirect_leaked_ids.update(e["held_out_shock"] for e in r["indirect_leak_detail"])

    if v1_claimed_excluded:
        # Precise regression = a shock v1's OWN provenance claims it protected
        # against injection, that nonetheless shows up as an INDIRECT
        # (injection-mechanism) leak somewhere -- NOT the broader "leaks via
        # any mechanism at all" set, which would also sweep in shocks that
        # only leak via the unrelated grounded_aligned.jsonl direct-id file
        # and overstate the specific "grounding script regressed" claim.
        regressed = sorted(v1_claimed_excluded & indirect_leaked_ids)
        other_leak_mechanism = sorted((v1_claimed_excluded & all_leaked_ids) - indirect_leaked_ids)
        if regressed:
            print(f"\n*** REGRESSION (same mechanism): {regressed} were explicitly excluded from "
                  f"sentiment injection in synthetic_grounded.jsonl's own v1 provenance metadata, "
                  f"but leak via that exact injection mechanism into a later file anyway (see "
                  f"synthetic_grounded_v2.jsonl / train_grounded_v2.jsonl / eval_grounded_v2.jsonl "
                  f"above). Direct evidence the held-out discipline already lapsed once, between "
                  f"grounding-script versions, before this freeze. ***")
        if other_leak_mechanism:
            print(f"\nNOTE (different mechanism, not a 'regression' in the injection sense): "
                  f"{other_leak_mechanism} were also v1-excluded from injection and do leak "
                  f"elsewhere in the corpus, but via grounded_aligned.jsonl's unrelated direct-shock-id "
                  f"mechanism, not via the injection path v1 was protecting against.")

    out_path = args.out or (DEFAULT_OUT_DIR / f"{datetime.now(timezone.utc).date().isoformat()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "held_out_set_size": len(held_out),
        "held_out_ids": sorted(held_out),
        "files_scanned": len(files),
        "total_direct_leaks": total_direct,
        "total_indirect_leaks": total_indirect,
        "distinct_leaked_held_out_ids": sorted(all_leaked_ids),
        "per_file": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path.relative_to(REPO_ROOT)}")

    if total_direct or total_indirect:
        sys.exit(1)  # nonzero exit so this can gate CI once corpora are cleaned


if __name__ == "__main__":
    main()
