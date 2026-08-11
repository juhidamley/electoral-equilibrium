"""Post-generation bin-distribution audit (Phase 5, Step 5.4).

Measures a generated synthetic corpus's empirical bin-magnitude distribution
against the target in configs/synthetic_events.json and reports PASS/WARN/FAIL,
overall and broken down per-bloc and per-shock. See
electoral/metrics/synthetic_distribution.py for the measurement logic and the
tolerance rationale.

THIS IS AN AUDIT, NOT A GATE (deliberately, for now): it does not reject a
batch or trigger regeneration. A hard gate could loop forever if the teacher
model structurally can't hit the target -- we don't yet know what the teacher
actually produces, so gating on it before knowing that risks a stuck pipeline
over a target that itself might need adjusting (see the panel-vs-target
sanity check in scripts/synthetic/check_panel_target_calibration.py -- it
originally found the target was partly intuition-derived and has since been
corrected to the panel-measured 34/31/28/6/1; re-run that script if you
suspect the two have drifted apart again). Once a few real batches have been
run through this audit and the numbers it reports are trusted, promoting
specific checks (the rare-bin overstatement check most plausibly) to a hard
gate is a reasonable next step -- not this one.

Usage:
    python scripts/synthetic/audit_distribution.py --corpus data/finetune/candidates.jsonl
    python scripts/synthetic/audit_distribution.py --corpus data/finetune/candidates.jsonl --target configs/synthetic_events.json
    python scripts/synthetic/audit_distribution.py --corpus data/finetune/candidates.jsonl --json-out /tmp/audit.json
    python scripts/synthetic/audit_distribution.py --corpus batch1.jsonl batch2.jsonl  # multiple files, e.g. a
        # generation run interrupted mid-batch and resumed into a second file -- audited together as one corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from electoral.metrics.synthetic_distribution import (  # noqa: E402
    DEFAULT_TARGET_CONFIG_PATH,
    audit_corpus,
    format_report,
    load_target,
)


def _load_corpus(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({e})")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        nargs="+",
        help="Generated corpus JSONL (one record per line). Multiple paths are audited together as one corpus.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_CONFIG_PATH,
        type=Path,
        help="Path to synthetic_events.json (reads _meta.bin_distribution_target)",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="Optional: also write the full report as JSON"
    )
    args = parser.parse_args()

    records: list[dict] = []
    for corpus_path in args.corpus:
        records.extend(_load_corpus(corpus_path))
    if not records:
        raise SystemExit(f"{args.corpus}: no records found")

    target = load_target(args.target)
    report = audit_corpus(records, target=target)

    corpus_desc = ", ".join(str(p) for p in args.corpus)
    print(
        f"Corpus: {corpus_desc}  ({len(records)} records, {report.overall.n} bloc-level bin observations)"
    )
    print(f"Target: {target}  (source: {args.target})")
    print()
    print(format_report(report))
    print()
    print(f"WORST VERDICT ACROSS ALL BREAKDOWNS: {report.worst_verdict}")

    if args.json_out:
        payload = {
            "corpus": [str(p) for p in args.corpus],
            "n_records": len(records),
            "target": target,
            "overall": asdict(report.overall),
            "per_bloc": {k: asdict(v) for k, v in report.per_bloc.items()},
            "per_shock": {k: asdict(v) for k, v in report.per_shock.items()},
            "worst_verdict": report.worst_verdict,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nFull report written to {args.json_out}")

    # Exit code reflects the audit outcome for CI/script chaining, but this
    # is still an audit, not a gate: a non-zero exit here should prompt a
    # human to look at the report, not trigger automatic regeneration.
    sys.exit({"PASS": 0, "WARN": 1, "FAIL": 2}[report.worst_verdict])


if __name__ == "__main__":
    main()
