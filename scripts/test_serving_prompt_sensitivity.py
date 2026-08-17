#!/usr/bin/env python3
"""Test whether serving infers a beneficiary absent explicit prompt framing.

This inference-only diagnostic sends four variants of one event through the
production ``ShockEstimator.estimate`` path.  It loads one local PEFT adapter
once, performs four deterministic constrained generations, and reports both
the model's scalar ``delta_eff`` and the mean of its 15 bin midpoints.  It
writes only the requested JSON artifact and its Markdown twin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from electoral.core.types import (
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)

PROMPT_PATH = "ShockEstimator.estimate -> format_prompt(event_dict) -> [INST]...[/INST]"
BARE = "Trump ends TPS, calling for the deportation of 300,000 immigrants"
STATED = BARE + (
    " The move triggers national backlash; the controversy energizes the Republican"
    " base and boosts conservative turnout, while galvanizing Latino and"
    " Democratic-leaning voters to mobilize in opposition."
)
FALLBACK_CONFIRMED = "FALLBACK-CONFIRMED"
INCONCLUSIVE = "INCONCLUSIVE"

STRATA = (
    ("race", tuple(CANONICAL_RACES)),
    ("religion", tuple(CANONICAL_RELIGIONS)),
    ("gender", tuple(CANONICAL_GENDERS)),
)
BLOCS = tuple(bloc for _, blocs in STRATA for bloc in blocs)
RUN_SPECS = (
    (1, "BARE", "democrat", BARE),
    (2, "BARE", "republican", BARE),
    (3, "STATED", "democrat", STATED),
    (4, "STATED", "republican", STATED),
)


def sign(value: float) -> str:
    """Return a three-way sign without coercing zero."""
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def verdict(net: dict[int, float]) -> dict[str, Any]:
    """Evaluate the two-clause fallback pattern from run-numbered net effects."""
    signs = {run: sign(net[run]) for run in range(1, 5)}
    bare_nonzero = signs[1] != "zero" and signs[2] != "zero"
    stated_nonzero = signs[3] != "zero" and signs[4] != "zero"
    bare_clause = bare_nonzero and signs[1] == signs[2]
    stated_clause = stated_nonzero and signs[3] != signs[4]

    failures: list[str] = []
    if not bare_clause:
        zero_runs = [str(run) for run in (1, 2) if signs[run] == "zero"]
        if zero_runs:
            failures.append(
                "bare clause failed because zero net occurred in run(s) " + ", ".join(zero_runs)
            )
        else:
            failures.append("bare clause failed because runs 1 and 2 have different signs")
    if not stated_clause:
        zero_runs = [str(run) for run in (3, 4) if signs[run] == "zero"]
        if zero_runs:
            failures.append(
                "stated clause failed because zero net occurred in run(s) " + ", ".join(zero_runs)
            )
        else:
            failures.append("stated clause failed because runs 3 and 4 have the same sign")

    if bare_clause and stated_clause:
        result = FALLBACK_CONFIRMED
        reason = (
            "both clauses passed: bare runs have the same nonzero sign, and stated "
            "runs have opposite nonzero signs"
        )
    else:
        result = INCONCLUSIVE
        reason = "; ".join(failures)
    return {
        "verdict": result,
        "reason": reason,
        "both_bare_negative": net[1] < 0 and net[2] < 0,
    }


def cross_check(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-run the verdict on the bin-midpoint proxy as an independent basis.

    The primary verdict uses the model's own scalar ``delta_eff``, which is what
    production serves.  ``run_confound_probe.py`` instead defines net direction as
    the unweighted mean of the 15 bin midpoints.  Computing both makes the two
    artifacts comparable, and a disagreement is itself a finding: it means the
    scalar and the per-bloc vector it is supposed to summarise point opposite
    ways, and the diagnosis is not decisive on that run.
    """
    proxy = verdict({row["run"]: row["mean_bin_midpoint"] for row in runs})
    primary_signs = {row["run"]: row["sign"] for row in runs}
    proxy_signs = {row["run"]: sign(row["mean_bin_midpoint"]) for row in runs}
    disagreeing = [run for run in sorted(primary_signs) if primary_signs[run] != proxy_signs[run]]
    return {
        "basis": "unweighted mean of 15 canonical bin midpoints",
        "verdict": proxy["verdict"],
        "reason": proxy["reason"],
        "per_run_sign_disagreement": disagreeing,
        "bases_agree": not disagreeing
        and proxy["verdict"] == verdict({row["run"]: row["delta_eff"] for row in runs})["verdict"],
    }


def build_event(run: int, party: str, description: str) -> dict[str, Any]:
    """Build the exact serving event shape for one diagnostic run."""
    return {
        "shock_id": f"run-{run}",
        "cycle": 2025,
        "party": party,
        "description": description,
        "news_roberta_scores": {},
        "social_roberta_scores": {},
    }


def mean_bin_midpoint(bins: dict[str, str]) -> float:
    """Return the unweighted mean midpoint after validating all canonical blocs."""
    missing = [bloc for bloc in BLOCS if bloc not in bins]
    invalid = {
        bloc: bins[bloc] for bloc in BLOCS if bloc in bins and bins[bloc] not in BIN_MIDPOINTS
    }
    if missing or invalid:
        raise ValueError(
            f"prediction must contain valid bins for all blocs; "
            f"missing={missing}, invalid={invalid}"
        )
    return sum(BIN_MIDPOINTS[bins[bloc]] for bloc in BLOCS) / len(BLOCS)


def response_bins(response: Any) -> dict[str, str]:
    """Flatten the serving response's three independent bin strata."""
    bins = {
        **dict(response.delta_bins_race),
        **dict(response.delta_bins_religion),
        **dict(response.delta_bins_gender),
    }
    mean_bin_midpoint(bins)
    return {bloc: bins[bloc] for bloc in BLOCS}


def build_runs() -> list[dict[str, Any]]:
    """Construct all four ordered run records without model I/O."""
    return [
        {
            "run": run,
            "framing": framing,
            "party": party,
            "event": build_event(run, party, description),
        }
        for run, framing, party, description in RUN_SPECS
    ]


def run_generations(
    adapter_path: Path,
    base_model: str,
    seed: int,
    show_prompt: bool,
) -> list[dict[str, Any]]:
    """Load one production estimator and run four independently seeded generations."""
    from electoral.llm.inference import ShockEstimator

    runs = build_runs()
    if show_prompt:
        from electoral.llm.trainer import format_prompt

        print("=== RUN 1 CONSTRUCTED PROMPT ===")
        print(format_prompt(runs[0]["event"]))
        print("=== END RUN 1 CONSTRUCTED PROMPT ===")

    estimator = ShockEstimator(adapter_path=str(adapter_path), base_model=base_model)
    import torch

    for row in runs:
        torch.manual_seed(seed)
        response = estimator.estimate(row["event"], intensity=1.0)
        bins = response_bins(response)
        row["delta_eff"] = float(response.delta_eff)
        row["sign"] = sign(row["delta_eff"])
        row["mean_bin_midpoint"] = mean_bin_midpoint(bins)
        row["bins"] = bins
    return runs


def markdown(
    runs: list[dict[str, Any]],
    result: dict[str, Any],
    cross: dict[str, Any] | None = None,
) -> str:
    """Render the human-readable report shared by stdout and the Markdown artifact."""
    lines = [
        "# Serving prompt sensitivity diagnostic",
        "",
        f"**Prompt path:** `{PROMPT_PATH}`",
        "",
        "## Run summary",
        "",
        "| Run | Framing | Party | Net delta_eff | Sign |",
        "|---:|---|---|---:|---|",
    ]
    for row in runs:
        lines.append(
            f"| {row['run']} | {row['framing']} | {row['party']} | "
            f"{row['delta_eff']:+.8f} | {row['sign']} |"
        )

    lines.extend(
        [
            "",
            "### Mean bin midpoints",
            "",
            "Unweighted mean of the 15 canonical bin midpoints, reported alongside "
            "the model's own `delta_eff`:",
            "",
            "| Run | mean_bin_midpoint | delta_eff |",
            "|---:|---:|---:|",
        ]
    )
    for row in runs:
        lines.append(
            f"| {row['run']} | {row['mean_bin_midpoint']:+.8f} | " f"{row['delta_eff']:+.8f} |"
        )

    lines.extend(
        [
            "",
            "## Per-bloc vectors",
            "",
            "| Stratum | Bloc | 1 BARE/dem | 2 BARE/rep | 3 STATED/dem | 4 STATED/rep |",
            "|---|---|---|---|---|---|",
        ]
    )
    for stratum, blocs in STRATA:
        for bloc in blocs:
            values = " | ".join(row["bins"][bloc] for row in runs)
            lines.append(f"| {stratum} | `{bloc}` | {values} |")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{result['verdict']}**",
            "",
            result["reason"],
            "",
            f"- `both_bare_negative`: `{str(result['both_bare_negative']).lower()}`",
        ]
    )
    if cross is not None:
        lines.extend(
            [
                "",
                "### Cross-check on the bin-midpoint basis",
                "",
                f"Primary verdict above uses the model's own `delta_eff`. Recomputed on "
                f"the {cross['basis']} (the basis `run_confound_probe.py` uses):",
                "",
                f"- cross-check verdict: **{cross['verdict']}**",
                f"- bases agree: `{str(cross['bases_agree']).lower()}`",
                f"- runs whose sign differs between bases: "
                f"`{cross['per_run_sign_disagreement'] or 'none'}`",
                "",
                "A disagreement means the scalar and the per-bloc vector it summarises "
                "point opposite ways; treat the diagnosis as not decisive on those runs.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    """Exercise every verdict branch without importing model dependencies."""
    confirmed = verdict({1: -0.1, 2: -0.2, 3: 0.1, 4: -0.1})
    bare_split = verdict({1: -0.1, 2: 0.2, 3: 0.1, 4: -0.1})
    stated_same = verdict({1: -0.1, 2: -0.2, 3: 0.1, 4: 0.2})
    zero_net = verdict({1: 0.0, 2: -0.2, 3: 0.1, 4: -0.1})
    if confirmed["verdict"] != FALLBACK_CONFIRMED or not confirmed["both_bare_negative"]:
        raise AssertionError(f"self-test failed for confirmed fixture: {confirmed}")
    if bare_split["verdict"] != INCONCLUSIVE or "different signs" not in bare_split["reason"]:
        raise AssertionError(f"self-test failed for bare-split fixture: {bare_split}")
    if stated_same["verdict"] != INCONCLUSIVE or "same sign" not in stated_same["reason"]:
        raise AssertionError(f"self-test failed for stated-same fixture: {stated_same}")
    if zero_net["verdict"] != INCONCLUSIVE or "zero net" not in zero_net["reason"]:
        raise AssertionError(f"self-test failed for zero fixture: {zero_net}")

    # Render the full report from stub bins so a formatting or serialization bug
    # surfaces here rather than after four expensive GPU generations on Hopper.
    stub_tokens = ("mild_neg", "slight_neg", "mod_pos", "mod_neg")
    runs = build_runs()
    for row, token in zip(runs, stub_tokens):
        bins = {bloc: token for bloc in BLOCS}
        row["delta_eff"] = BIN_MIDPOINTS[token]
        row["sign"] = sign(row["delta_eff"])
        row["mean_bin_midpoint"] = mean_bin_midpoint(bins)
        row["bins"] = bins
    stub_result = verdict({row["run"]: row["delta_eff"] for row in runs})
    stub_cross = cross_check(runs)
    report = markdown(runs, stub_result, stub_cross)
    json.dumps({"runs": runs, "verdict": stub_result, "cross_check": stub_cross})
    if stub_result["verdict"] != FALLBACK_CONFIRMED or not stub_cross["bases_agree"]:
        raise AssertionError(f"self-test failed for report fixture: {stub_result}, {stub_cross}")
    for bloc in BLOCS:
        if f"`{bloc}`" not in report:
            raise AssertionError(f"report is missing bloc row: {bloc}")
    print("SELF_TEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=False, help="PEFT adapter directory")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/serving_prompt_sensitivity.json"),
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the full constructed prompt for run 1",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if not args.adapter:
        parser.error("--adapter is required unless --self-test is supplied")
    adapter_path = Path(args.adapter)
    if not (adapter_path.is_dir() and (adapter_path / "adapter_config.json").is_file()):
        raise FileNotFoundError(
            f"adapter is not a local PEFT directory with adapter_config.json: {adapter_path}"
        )

    runs = run_generations(adapter_path, args.base_model, args.seed, args.show_prompt)
    net = {row["run"]: row["delta_eff"] for row in runs}
    result = verdict(net)
    cross = cross_check(runs)
    payload = {
        "prompt_path": PROMPT_PATH,
        "adapter": args.adapter,
        "base_model": args.base_model,
        "seed": args.seed,
        "event_texts": {"bare": BARE, "stated": STATED},
        "mean_bin_midpoint_definition": ("unweighted mean of 15 canonical bin midpoints"),
        "runs": runs,
        "verdict": result,
        "cross_check": cross,
    }
    report = markdown(runs, result, cross)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"out={args.out} markdown={args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
