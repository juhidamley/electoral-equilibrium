"""Measure the synthetic fine-tuning corpus's empirical bin-magnitude
distribution against the target in configs/synthetic_events.json, and verdict
whether it's close enough to trust (Phase 5, Step 5.4).

WHY THIS EXISTS: the generation and review prompts (scripts/synthetic/
generate_deepseek.py, scripts/synthetic/gemini_review.py) both *ask* the
teacher model to keep "strong"/"moderate" rare -- real per-bloc shifts
cluster small, per the VOTER panel (median 0.0029, the "slight" midpoint).
But "several blocs at strong is unrealistic" in gemini_review.py's review
prompt is pure LLM judgment; nothing measures the corpus that actually comes
out the other end. A corpus that quietly drifts toward more strong/moderate
labels than real data supports teaches the model a false base rate --
correct per-bin MAGNITUDES with a wrong per-bin FREQUENCY still produces an
overconfident model. That's the same overstatement failure mode the whole
Step 2.1 rescale fixed, just moved from "the numbers are wrong" to "the
numbers are right but occur too often."

TOLERANCE RATIONALE (read this before changing any threshold below):

1. Chi-square goodness-of-fit (scipy.stats.chisquare) against the target
   proportions is the standard test for "does this categorical count
   distribution match an expected proportions vector," and is reported for
   every check. But its raw p-value does NOT drive the verdict -- see #2.

2. THIS WAS THE FIRST DESIGN AND IT WAS WRONG; caught by this module's own
   required self-test (a "realistic" batch of 3000 bin-observations with
   every category within ~2pp of target -- neutral exact, slight +1.7pp,
   mild +2pp, moderate -3.3pp, strong -0.3pp) failed with p=8e-16. Chi-square
   power scales with n: at corpus-realistic sizes (a full generation run is
   ~64 seeds x ~20 expansion x 15 blocs, tens of thousands of observations),
   trivially small, practically-irrelevant deviations become "significant"
   by raw p-value alone. Gating FAIL on p<0.01 would make this check cry wolf
   on exactly the well-calibrated corpora it's supposed to wave through --
   which would just get it ignored, defeating the point. Fix: the omnibus
   check is gated on Cramer's V (effect size, size-corrected: V = sqrt(chi2 /
   (n * (k-1))), k=5 categories), not on p. V is ~scale-invariant -- the
   realistic fixture above scores V=0.08, a uniform-across-9-tokens skewed
   fixture scores V=0.90, a single-bloc-100%-strong fixture scores V=3.5.
   WARN at V>=0.15, FAIL at V>=0.30 (between Cohen's conventional "small" and
   "medium" effect-size bands) cleanly separates all three. p is still
   computed and reported for context, and is what determines chi_square_valid
   in #3 below, but cannot by itself escalate the verdict.

3. Chi-square (and therefore Cramer's V) is only valid when expected cell
   counts are large enough (Cochran's rule of thumb: >=80% of cells have
   expected count >=5, none below 1). Small batches -- e.g. auditing one
   shock's ~20 records against a 5-category target -- routinely violate
   this. When invalid, chi-square/V are skipped (reported as None) and the
   verdict falls back entirely to the per-bin bands below, which don't
   depend on sample-size assumptions.

4. Per-bin bands are asymmetric by bin, because the categories are not
   equally consequential. "moderate" and "strong" are the two bins this
   check actually exists to police: real data says they should be rare (5%
   and 2% of bins respectively), and it's specifically THESE bins
   overstating that quietly reproduces the original ±0.15-era overconfidence
   problem in miniature. Because their targets are themselves small,
   percentage-point deviation understates the problem (2% -> 6% is only 4
   points but is 3x the rate), so they're judged by RATIO to target: WARN at
   >=1.75x, FAIL at >=3x -- with an absolute floor (+4pp / +8pp) so a
   near-empty rare bin (target 2%, n small) doesn't trip the ratio check on
   a single stray record, and only OVERstatement triggers (a rare bin coming
   in under target is not the failure mode this exists to catch).
   "neutral"/"slight"/"mild" are large-percentage bins where a ratio is
   noisy and a percentage-point band is more legible: WARN at >=8pp absolute
   deviation, FAIL at >=15pp.

5. n=0 (nothing to measure) is neither PASS nor FAIL -- both would be
   misleading, since no evidence was observed either way. Reported as WARN
   with an explicit "nothing measured" reason instead of comparing an
   undefined empirical distribution against target.

6. The overall verdict is the worst of (FAIL > WARN > PASS) across the
   rare-bin ratio checks, the common-bin point checks, and the Cramer's-V
   omnibus check when valid. Every bin that triggered a check is named in
   `offending_bins`; the specific reason is in `reasons`. Nothing here is a
   hard gate -- see audit_corpus's docstring for why this ships as an audit,
   not a regenerate-on-fail loop.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scipy.stats import chisquare

from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    MAGNITUDE_CATEGORIES,
    bin_to_magnitude_category,
)

Verdict = Literal["PASS", "WARN", "FAIL"]

_VERDICT_RANK: dict[Verdict, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}

_RARE_BINS: tuple[str, ...] = ("moderate", "strong")
_COMMON_BINS: tuple[str, ...] = ("neutral", "slight", "mild")

_RARE_WARN_RATIO = 1.75
_RARE_FAIL_RATIO = 3.0
_RARE_WARN_FLOOR_PP = 4.0
_RARE_FAIL_FLOOR_PP = 8.0
_COMMON_WARN_PP = 8.0
_COMMON_FAIL_PP = 15.0
_CRAMERS_V_WARN = 0.15
_CRAMERS_V_FAIL = 0.30

_ALL_BLOCS: tuple[str, ...] = (
    tuple(CANONICAL_RACES) + tuple(CANONICAL_RELIGIONS) + tuple(CANONICAL_GENDERS)
)

DEFAULT_TARGET_CONFIG_PATH = "configs/synthetic_events.json"


def load_target(config_path: str | Path = DEFAULT_TARGET_CONFIG_PATH) -> dict[str, float]:
    """Read the canonical bin-distribution target from synthetic_events.json.

    Reads the structured `_meta.bin_distribution_target` object, not the
    prose in `_meta.magnitude_grounding` -- the prose says outright it's a
    hand-synced duplicate, and regex-parsing a second unstructured copy out
    of English text would just add a third place for these five numbers to
    drift apart. If the two ever disagree, the structured object wins; that's
    stated in its own `_note` field too.
    """
    data = json.loads(Path(config_path).read_text())
    target = data["_meta"]["bin_distribution_target"]
    target = {k: v for k, v in target.items() if not k.startswith("_")}
    missing = set(MAGNITUDE_CATEGORIES) - set(target)
    if missing:
        raise ValueError(f"bin_distribution_target missing categories: {missing}")
    total = sum(target.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"bin_distribution_target sums to {total}, not 1.0: {target}")
    return target


def extract_bloc_bins(record: dict[str, Any]) -> dict[str, str]:
    """Flatten one generated record's delta_bins_race/religion/gender into a
    single bloc_id -> bin_token dict (15 entries for a well-formed record).
    """
    out: dict[str, str] = {}
    out.update(record.get("delta_bins_race", {}) or {})
    out.update(record.get("delta_bins_religion", {}) or {})
    out.update(record.get("delta_bins_gender", {}) or {})
    return out


@dataclass(frozen=True)
class DistributionCheck:
    """One measured-vs-target comparison, at whatever granularity the caller
    asked for (the whole corpus, one bloc across all records, or one shock's
    records across all blocs).
    """

    label: str
    n: int
    observed_counts: dict[str, int]
    observed_pct: dict[str, float]  # 0-100
    target_pct: dict[str, float]  # 0-100, same keys
    deviation_pp: dict[str, float]  # observed_pct - target_pct
    chi_square: float | None
    chi_square_p: float | None
    cramers_v: float | None  # size-corrected effect size; see tolerance rationale #2
    chi_square_valid: bool
    verdict: Verdict
    offending_bins: tuple[str, ...]
    reasons: tuple[str, ...]


def _chi_square_check(
    counts: Counter[str], target_pct: dict[str, float], n: int
) -> tuple[float | None, float | None, float | None, bool]:
    if n == 0:
        return None, None, None, False
    observed = [counts.get(cat, 0) for cat in MAGNITUDE_CATEGORIES]
    expected = [target_pct[cat] * n for cat in MAGNITUDE_CATEGORIES]
    n_low = sum(1 for e in expected if e < 5)
    valid = n_low <= len(expected) * 0.2 and all(e >= 1 for e in expected)
    if not valid:
        return None, None, None, False
    result = chisquare(f_obs=observed, f_exp=expected)
    k = len(MAGNITUDE_CATEGORIES)
    cramers_v = math.sqrt(result.statistic / (n * (k - 1)))
    return float(result.statistic), float(result.pvalue), float(cramers_v), True


def _verdict(
    observed_pct: dict[str, float],
    target_pct: dict[str, float],
    cramers_v: float | None,
    chi_valid: bool,
) -> tuple[Verdict, tuple[str, ...], tuple[str, ...]]:
    severity: Verdict = "PASS"
    offending: list[str] = []
    reasons: list[str] = []

    def _escalate(new: Verdict) -> None:
        nonlocal severity
        if _VERDICT_RANK[new] > _VERDICT_RANK[severity]:
            severity = new

    for cat in _RARE_BINS:
        obs = observed_pct.get(cat, 0.0)
        tgt_pct = target_pct[cat] * 100
        dev = obs - tgt_pct
        ratio = obs / tgt_pct if tgt_pct > 0 else (math.inf if obs > 0 else 1.0)
        if ratio >= _RARE_FAIL_RATIO and dev >= _RARE_FAIL_FLOOR_PP:
            _escalate("FAIL")
            offending.append(cat)
            reasons.append(
                f"{cat}: {obs:.1f}% observed vs {tgt_pct:.1f}% target "
                f"({ratio:.1f}x) -- overstatement risk, the exact failure "
                f"mode this check exists to catch"
            )
        elif ratio >= _RARE_WARN_RATIO and dev >= _RARE_WARN_FLOOR_PP:
            _escalate("WARN")
            offending.append(cat)
            reasons.append(f"{cat}: {obs:.1f}% observed vs {tgt_pct:.1f}% target ({ratio:.1f}x)")

    for cat in _COMMON_BINS:
        obs = observed_pct.get(cat, 0.0)
        tgt_pct = target_pct[cat] * 100
        dev = abs(obs - tgt_pct)
        if dev >= _COMMON_FAIL_PP:
            _escalate("FAIL")
            offending.append(cat)
            reasons.append(f"{cat}: {obs:.1f}% observed vs {tgt_pct:.1f}% target ({dev:+.1f}pp)")
        elif dev >= _COMMON_WARN_PP:
            _escalate("WARN")
            offending.append(cat)
            reasons.append(f"{cat}: {obs:.1f}% observed vs {tgt_pct:.1f}% target ({dev:+.1f}pp)")

    if chi_valid:
        assert cramers_v is not None
        if cramers_v >= _CRAMERS_V_FAIL:
            _escalate("FAIL")
            reasons.append(
                f"Cramer's V={cramers_v:.3f} >= {_CRAMERS_V_FAIL} -- distribution differs from target (large effect)"
            )
        elif cramers_v >= _CRAMERS_V_WARN:
            _escalate("WARN")
            reasons.append(
                f"Cramer's V={cramers_v:.3f} >= {_CRAMERS_V_WARN} -- distribution differs from target (small-medium effect)"
            )
    else:
        reasons.append(
            "chi-square/Cramer's V not computed: expected cell counts too small for this n (see tolerance rationale #3)"
        )

    if severity == "PASS" and not reasons:
        reasons.append(
            "within tolerance on all bins and (where valid) the Cramer's V omnibus check"
        )

    return severity, tuple(offending), tuple(reasons)


def measure_distribution(
    bin_tokens: list[str], target: dict[str, float], label: str
) -> DistributionCheck:
    """Core measurement: given a flat list of 9-token bin labels (already
    extracted from whatever records/blocs they came from) and a target,
    return counts, percentages, deviations, and a PASS/WARN/FAIL verdict.
    """
    n = len(bin_tokens)
    target_pct_display = {cat: target[cat] * 100 for cat in MAGNITUDE_CATEGORIES}

    if n == 0:
        # Nothing observed -- not PASS (that would misleadingly imply
        # confidence), not FAIL (there's no actual disagreement, just no
        # data). See tolerance rationale #5.
        zero = {cat: 0.0 for cat in MAGNITUDE_CATEGORIES}
        return DistributionCheck(
            label=label,
            n=0,
            observed_counts={cat: 0 for cat in MAGNITUDE_CATEGORIES},
            observed_pct=zero,
            target_pct=target_pct_display,
            deviation_pp=dict(zero),
            chi_square=None,
            chi_square_p=None,
            cramers_v=None,
            chi_square_valid=False,
            verdict="WARN",
            offending_bins=(),
            reasons=(
                "n=0: nothing measured, cannot verdict against a target with no observations",
            ),
        )

    categories = [bin_to_magnitude_category(t) for t in bin_tokens]
    counts = Counter(categories)
    observed_pct = {cat: 100.0 * counts.get(cat, 0) / n for cat in MAGNITUDE_CATEGORIES}
    deviation_pp = {
        cat: observed_pct[cat] - target_pct_display[cat] for cat in MAGNITUDE_CATEGORIES
    }
    chi2, chi_p, cramers_v, chi_valid = _chi_square_check(counts, target, n)
    verdict, offending, reasons = _verdict(observed_pct, target, cramers_v, chi_valid)
    return DistributionCheck(
        label=label,
        n=n,
        observed_counts={cat: counts.get(cat, 0) for cat in MAGNITUDE_CATEGORIES},
        observed_pct=observed_pct,
        target_pct=target_pct_display,
        deviation_pp=deviation_pp,
        chi_square=chi2,
        chi_square_p=chi_p,
        cramers_v=cramers_v,
        chi_square_valid=chi_valid,
        verdict=verdict,
        offending_bins=offending,
        reasons=reasons,
    )


@dataclass(frozen=True)
class CorpusDistributionReport:
    """Full audit of a generated batch: overall, per-bloc, and per-shock.

    Per-bloc and per-shock are not optional extras -- an aggregate can look
    right while individual blocs or individual shocks are skewed (e.g. one
    seed archetype the teacher consistently over-dramatizes, diluted to
    invisibility in the corpus-wide aggregate by 63 well-calibrated seeds).
    `worst_verdict` is the single number to look at first; the breakdowns are
    for finding out *what* is off once it isn't PASS.
    """

    overall: DistributionCheck
    per_bloc: dict[str, DistributionCheck]
    per_shock: dict[str, DistributionCheck]
    worst_verdict: Verdict


def audit_corpus(
    records: list[dict[str, Any]], target: dict[str, float] | None = None
) -> CorpusDistributionReport:
    """Measure a generated corpus (list of records in the
    scripts/synthetic/generate_deepseek.py output schema) against `target`
    (defaults to configs/synthetic_events.json's configured target).
    """
    if target is None:
        target = load_target()

    overall_tokens: list[str] = []
    per_bloc_tokens: dict[str, list[str]] = defaultdict(list)
    per_shock_tokens: dict[str, list[str]] = defaultdict(list)

    for rec in records:
        shock_id = rec.get("shock_id", "<unknown_shock>")
        for bloc, token in extract_bloc_bins(rec).items():
            overall_tokens.append(token)
            per_bloc_tokens[bloc].append(token)
            per_shock_tokens[shock_id].append(token)

    overall = measure_distribution(overall_tokens, target, label="overall")
    per_bloc = {
        bloc: measure_distribution(per_bloc_tokens[bloc], target, label=bloc)
        for bloc in _ALL_BLOCS
        if bloc in per_bloc_tokens
    }
    per_shock = {
        shock_id: measure_distribution(tokens, target, label=shock_id)
        for shock_id, tokens in per_shock_tokens.items()
    }

    worst: Verdict = overall.verdict
    for check in list(per_bloc.values()) + list(per_shock.values()):
        if _VERDICT_RANK[check.verdict] > _VERDICT_RANK[worst]:
            worst = check.verdict

    return CorpusDistributionReport(
        overall=overall, per_bloc=per_bloc, per_shock=per_shock, worst_verdict=worst
    )


def format_report(report: CorpusDistributionReport) -> str:
    """Human-readable summary for the post-generation audit script (option
    (a): flag drift for a person to review, don't auto-reject -- see
    scripts/synthetic/audit_distribution.py).
    """
    lines: list[str] = []

    def _fmt_check(check: DistributionCheck) -> list[str]:
        out = [f"  n={check.n}  verdict={check.verdict}"]
        for cat in MAGNITUDE_CATEGORIES:
            out.append(
                f"    {cat:10s} observed={check.observed_pct[cat]:5.1f}%  "
                f"target={check.target_pct[cat]:5.1f}%  dev={check.deviation_pp[cat]:+5.1f}pp"
            )
        if check.chi_square_valid:
            out.append(
                f"    chi-square: stat={check.chi_square:.2f}  p={check.chi_square_p:.4f}  "
                f"Cramer's V={check.cramers_v:.3f} (verdict driver, not p -- see tolerance rationale #2)"
            )
        for reason in check.reasons:
            out.append(f"    - {reason}")
        return out

    lines.append(
        f"=== OVERALL: {report.overall.verdict} (worst across all breakdowns: {report.worst_verdict}) ==="
    )
    lines.extend(_fmt_check(report.overall))

    if report.per_bloc:
        lines.append("")
        lines.append("=== PER-BLOC ===")
        for bloc, check in sorted(report.per_bloc.items()):
            if check.verdict != "PASS":
                lines.append(f"-- {bloc} --")
                lines.extend(_fmt_check(check))

    if report.per_shock:
        lines.append("")
        lines.append("=== PER-SHOCK (flagged only) ===")
        flagged = {sid: c for sid, c in report.per_shock.items() if c.verdict != "PASS"}
        if not flagged:
            lines.append("  (none -- every shock's own bin distribution is within tolerance)")
        for shock_id, check in sorted(flagged.items()):
            lines.append(f"-- {shock_id} --")
            lines.extend(_fmt_check(check))

    return "\n".join(lines)
