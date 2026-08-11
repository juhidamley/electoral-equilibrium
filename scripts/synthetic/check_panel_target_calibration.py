"""Sanity-check the configured bin-distribution target against the panel it's
supposedly grounded in (Phase 5, Step 5.4, item 5).

configs/synthetic_events.json._meta.bin_distribution_target is asserted to
come "from the panel, not from intuition." This script recomputes the
panel's own empirical bin-magnitude distribution directly from
data/ground_truth/panel_deltas.json and reports the two side by side, so
that claim is checked rather than trusted. It originally caught the target
NOT matching (an intuition-derived 30/45/18/5/2 vs. the panel's actual
34/31/28/6/1) -- the target has since been corrected to match; see the
"History" note printed at the end of main() for that story. Keep re-running
this after any future change to either side; nothing else guards their
agreement (see that same note).

METHODOLOGY NOTE (worth reading if the numbers below ever look off): the
obvious-seeming approach -- take the built voter panel (electoral.kernels
.data.build_voter_panel), sort each bloc's vote_share by election cycle, and
diff consecutive cycles -- reproduces the right N (234) but numbers roughly
15-20x too large (median ~0.06, not ~0.003), because a full election cycle's
vote-share difference bakes in the ENTIRE 4-year (or, pre-1965, sometimes
40-year-gap) partisan shift, not one shock's marginal contribution, and
pre-1965 minority-bloc cells are extremely noisy (tiny ANES samples). The
grounding text's "n=234 non-suppressed bloc/cycle cells" describes a
DIFFERENT, shock-scoped dataset: the unweighted, top-level `measured_delta`
in data/ground_truth/panel_deltas.json (before/after windows immediately
around each of 18 real historical shocks, not cycle-to-cycle drift),
filtered to `suppressed_flag == False`. That combination is the only one
that reproduces all four grounding numbers (median 0.0029, mean 0.0046,
p90 0.0112, max 0.0286) simultaneously and exactly -- confirmed by grid
search over {raw, weighted} x {no filter, magnitude_usable-only,
not_suppressed-only, both} before writing this script. Do not "simplify"
this to the cycle-to-cycle panel diff; it measures a different, much larger
quantity.

Usage:
    python scripts/synthetic/check_panel_target_calibration.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from electoral.core.types import MAGNITUDE_CATEGORIES, delta_to_bin, bin_to_magnitude_category

PANEL_DELTAS_PATH = Path("data/ground_truth/panel_deltas.json")
TARGET_CONFIG_PATH = Path("configs/synthetic_events.json")


def load_panel_magnitude_sample() -> list[float]:
    """|measured_delta| across all (shock, bloc) cells, excluding suppressed
    ones. See the module docstring for why this specific combination.
    """
    data = json.loads(PANEL_DELTAS_PATH.read_text())
    values: list[float] = []
    for shock in data.values():
        for bloc_info in shock.get("bloc", {}).values():
            if bloc_info.get("suppressed_flag", False):
                continue
            md = bloc_info.get("measured_delta")
            if md is not None:
                values.append(abs(md))
    return values


def summarize(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    n = len(s)

    def _quantile(q: float) -> float:
        idx = q * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return {
        "n": n,
        "median": _quantile(0.5),
        "mean": sum(s) / n,
        "p90": _quantile(0.9),
        "max": s[-1],
    }


def magnitude_distribution(abs_deltas: list[float]) -> dict[str, float]:
    cats = Counter(bin_to_magnitude_category(delta_to_bin(d)) for d in abs_deltas)
    n = len(abs_deltas)
    return {cat: 100.0 * cats.get(cat, 0) / n for cat in MAGNITUDE_CATEGORIES}


def main() -> None:
    values = load_panel_magnitude_sample()
    stats = summarize(values)

    print("=== Reproducing the magnitude_grounding numbers ===")
    print(
        f"n={stats['n']}  median={stats['median']:.4f}  mean={stats['mean']:.4f}  "
        f"p90={stats['p90']:.4f}  max={stats['max']:.4f}"
    )
    print("configs/synthetic_events.json._meta.magnitude_grounding claims:")
    print("n=234  median=0.0029  mean=0.0046  p90=0.0112  max=0.0286")
    match = (
        stats["n"] == 234
        and abs(stats["median"] - 0.0029) < 1e-4
        and abs(stats["mean"] - 0.0046) < 1e-4
        and abs(stats["p90"] - 0.0112) < 1e-4
        and abs(stats["max"] - 0.0286) < 1e-4
    )
    print(f"EXACT MATCH: {match}")
    if not match:
        print(
            "!! The grounding text's numbers no longer match this reproduction. "
            "Either data/ground_truth/panel_deltas.json changed (re-ran the "
            "extraction with new panel waves / different suppression rules) or "
            "the grounding text is now stale. Investigate before trusting either."
        )

    print()
    print("=== Panel's own empirical bin-magnitude distribution vs the configured target ===")
    empirical = magnitude_distribution(values)
    target = json.loads(TARGET_CONFIG_PATH.read_text())["_meta"]["bin_distribution_target"]
    target = {k: v for k, v in target.items() if not k.startswith("_")}

    header = (
        f"{'category':10s} {'panel (empirical)':>18s} {'configured target':>18s} {'deviation':>10s}"
    )
    print(header)
    max_abs_dev = 0.0
    for cat in MAGNITUDE_CATEGORIES:
        emp_pct = empirical[cat]
        tgt_pct = target[cat] * 100
        dev = emp_pct - tgt_pct
        max_abs_dev = max(max_abs_dev, abs(dev))
        print(f"{cat:10s} {emp_pct:17.1f}% {tgt_pct:17.1f}% {dev:+9.1f}pp")

    print()
    if max_abs_dev <= 5.0:
        print(
            f"Target tracks the panel closely (largest single-bin deviation: {max_abs_dev:.1f}pp)."
        )
    else:
        print(
            f"Target and panel disagree by up to {max_abs_dev:.1f}pp on at least one bin. "
            "This is a finding to act on, not just note -- see below."
        )
    print(
        "\nHistory: this script originally found the target was partly intuition-derived "
        "(30/45/18/5/2) -- neutral/moderate/strong were within ~4pp of the panel, but "
        "'slight' was overstated and 'mild' understated by ~10-14pp, because the panel's "
        "own slight/mild split is much closer to even than 45/18. The target was corrected "
        "to the panel-measured distribution (34/31/28/6/1, rounded from the exact per-bin "
        "counts above) in every location it's stated -- configs/synthetic_events.json's "
        "magnitude_grounding prose and bin_distribution_target object, and the numeric "
        "target block in both scripts/synthetic/generate_deepseek.py and "
        "scripts/synthetic/gemini_review.py. If the numbers above ever drift from this "
        "history again, re-run this script and update all four locations together -- "
        "nothing currently guards their agreement except tests/test_synthetic_distribution.py's "
        "hardcoded expectation of configs/synthetic_events.json's values, which only catches "
        "the config drifting from ITSELF, not from the panel."
    )


if __name__ == "__main__":
    main()
