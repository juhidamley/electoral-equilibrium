"""Type aliases and canonical constants for the Electoral Equilibrium pipeline.

═══════════════════════════════════════════════════════════════════════════════
THE DOMAIN VOCABULARY (this file defines the words the whole project speaks)
═══════════════════════════════════════════════════════════════════════════════
Before any code makes sense, you need the core nouns. This file is where they
are pinned down once so every other module agrees on spelling and meaning.

  • BLOC  — a group of voters that share a demographic trait, e.g. "latino" or
            "evangelical". A bloc is identified by a short lowercase string.

  • STRATUM (plural STRATA) — one *way* of slicing the whole electorate into
            blocs. We use THREE independent strata:
                1. Race/ethnicity  (5 blocs)
                2. Religion        (7 blocs)
                3. Gender          (3 blocs)
            "Independent" is key: each stratum on its own covers ~100% of voters.
            We do NOT cross them (there is no "latino × evangelical × woman"
            cell). That keeps the model tractable; the paper notes this is an
            approximation (the "ecological fallacy").

  • VOTE SHARE (μ, "mu") — the fraction of a bloc that votes for the party we're
            modeling, a number in [0, 1]. e.g. African-American Democratic vote
            share ≈ 0.90.

  • COALITION WEIGHT (w) — how much a campaign chooses to "invest" in / rely on
            each *race* bloc. These are the optimizer's decision variables; the
            5 race weights sum to 1.0. (Religion and gender weights are fixed
            from the voter panel, not optimized.)

  • SHOCK — a hypothetical political event ("an assassination attempt", "a
            recession") whose effect on each bloc's vote share we predict.

  • DELTA (Δ) — the predicted *change* in a bloc's vote share caused by a shock.
            The language model doesn't output a raw number; it outputs one of 9
            discrete DELTA BINS (see DELTA_BINS below), which we then convert to
            a number via its midpoint.

  • V_eq ("V equilibrium") — the win threshold: the effective vote share a
            coalition must reach to win. Roughly 0.52–0.53 for Democrats,
            0.49–0.51 for Republicans, derived from the voter panel per party.

  • LAYER WEIGHTS (λ₁, λ₂, λ₃, "lambda") — how much each of the three strata
            counts toward the single "effective loyalty" scalar. They sum to 1.0.

FORMATTING INVARIANTS enforced project-wide:
  • All demographic identifiers are lowercase snake_case strings.
  • All election cycles are int in YYYY format (2020, never "2020" or a date).
  • All share/weight values are floats in [0, 1].
"""

from __future__ import annotations

from enum import Enum
from typing import Final  # available since Python 3.8; no try/except needed

# TypeAlias was added to typing in Python 3.10.
# The fallback (TypeAlias = type) is a no-op at runtime: annotations are not
# evaluated at import time (from __future__ import annotations), so assigning
# type to TypeAlias has no practical effect other than silencing type checkers
# on 3.9. Remove the try/except once the project drops Python 3.9 support.
try:
    from typing import Literal, TypeAlias
except ImportError:
    from typing import Literal  # type: ignore[assignment]

    TypeAlias = type  # type: ignore[assignment,misc]


# ── Canonical bloc enumerations ───────────────────────────────────────────────
#
# str mixin: enum members ARE strings.
#   Race.AFRICAN_AMERICAN == "african_american"  → True
#   isinstance(Race.AFRICAN_AMERICAN, str)       → True
#   json.dumps(Race.AFRICAN_AMERICAN)            → '"african_american"'
#
# This means Enum members work transparently as JSON dict keys and as artifact
# dict keys — no .value calls needed downstream. Kernel code should prefer
# Race.AFRICAN_AMERICAN over the bare string literal; user-facing serialization
# (JSONL, Parquet) receives plain str automatically.
#
# Member order is authoritative and matches CANONICAL_* list order below.


class Race(str, Enum):
    """The five canonical race/ethnicity strata blocs."""

    AFRICAN_AMERICAN = "african_american"  # ~12% of electorate
    LATINO = "latino"  # ~11%
    ASIAN = "asian"  # ~5%
    WHITE = "white"  # ~62% (non-Hispanic White only)
    OTHER_RACE = "other_race"  # ~10%


class Religion(str, Enum):
    """The seven canonical religion/affiliation strata blocs."""

    EVANGELICAL = "evangelical"  # ~24%
    CATHOLIC = "catholic"  # ~21%
    PROTESTANT = "protestant"  # ~13% (mainline Protestant, non-Evangelical)
    SECULAR = "secular"  # ~26% (none/unaffiliated)
    JEWISH = "jewish"  # ~2%
    MUSLIM = "muslim"  # ~1%
    OTHER_REL = "other_rel"  # ~13%


class Gender(str, Enum):
    """The three canonical gender strata blocs."""

    WOMEN = "women"  # ~52%
    MEN = "men"  # ~47%
    OTHER_GENDER = "other_gender"  # ~1%


# ── Canonical identifier lists ────────────────────────────────────────────────
# Derived from the enums above: a single source of truth.
# Final prevents reassignment; the enum definition is the authoritative order.

CANONICAL_RACES: Final[list[str]] = [e.value for e in Race]
CANONICAL_RELIGIONS: Final[list[str]] = [e.value for e in Religion]
CANONICAL_GENDERS: Final[list[str]] = [e.value for e in Gender]


# ── Scalar type aliases ───────────────────────────────────────────────────────

Cycle: TypeAlias = int  # YYYY format (e.g. 2020); never datetime
BlocId: TypeAlias = str  # lowercase snake_case (e.g. "evangelical")
Weight: TypeAlias = float  # vote share or coalition weight in [0, 1]
ElasticityScore: TypeAlias = float  # RoBERTa sentiment elasticity in [-1, 1]
ShockDelta: TypeAlias = float  # LLM-estimated vote-share change per bloc
LoraRank: TypeAlias = int  # QLoRA rank parameter (e.g. 16 or 32)
Platform: TypeAlias = str  # social media platform identifier
LagDays: TypeAlias = int  # days between shock event and favorability poll
BlocWeight: TypeAlias = float  # inferred probability that an author belongs to a bloc
BlocWeights: TypeAlias = dict  # dict[BlocId, BlocWeight], values sum to 1.0

# Demographic ID aliases remain str for JSON-serializable artifact dict keys.
# In new kernel code prefer Race / Religion / Gender enum members for safety.
RaceId: TypeAlias = str
ReligionId: TypeAlias = str
GenderId: TypeAlias = str

Party: TypeAlias = Literal["democrat", "republican"]
# Democrat V_eq ~0.52-0.53; Republican V_eq ~0.49-0.51 (from voter panel,
# build_constraint_spec). Every component that reads vote share or sets a win
# condition must receive party as an explicit argument — never hardcode.


# ── Delta bin constants ───────────────────────────────────────────────────────
# WHY BINS INSTEAD OF RAW NUMBERS?
# The language model (Stage 1) predicts how a shock moves each bloc's vote share.
# Asking an LLM for a precise float like "-0.0734" is unreliable — LLMs are bad
# at fine numeric precision and may emit something unparseable. Instead we give
# it a small fixed menu of 9 labels ("strong_neg" ... "strong_pos") and force it
# to pick exactly one (this is "constrained decoding", done with the `outlines`
# library). Discrete, always-valid, and easy to reason about.
#
# Each label corresponds to a numeric *range*; downstream code turns the chosen
# label back into a single number using the range's midpoint (see BIN_MIDPOINTS
# and bin_to_delta below). The ranges are contiguous and non-overlapping, so the
# 9 bins tile the interval [-0.15, +0.15] with no gaps.
#
# Standardized on "slight" (not "weak") — see DECISIONS.md §4.

DELTA_BINS: Final[tuple[str, ...]] = (
    "strong_neg",  # numeric range [-0.15, -0.09)
    "mod_neg",  # [-0.09, -0.05)
    "mild_neg",  # [-0.05, -0.02)
    "slight_neg",  # [-0.02, -0.005)
    "neutral",  # [-0.005, +0.005]
    "slight_pos",  # (+0.005, +0.02]
    "mild_pos",  # (+0.02, +0.05]
    "mod_pos",  # (+0.05, +0.09]
    "strong_pos",  # (+0.09, +0.15]
)

# Midpoints used by bin_to_delta()
BIN_MIDPOINTS: Final[dict[str, float]] = {
    "strong_neg": -0.120,
    "mod_neg": -0.070,
    "mild_neg": -0.035,
    "slight_neg": -0.012,
    "neutral": 0.000,
    "slight_pos": +0.012,
    "mild_pos": +0.035,
    "mod_pos": +0.070,
    "strong_pos": +0.120,
}

# Layer weight keys (must sum to 1.0). These name the three λ values that blend
# the race / religion / gender strata into one "effective loyalty" number:
#   lambda_1 → race, lambda_2 → religion, lambda_3 → gender.
# Stored in configs/layer_weights.json, calibrated from historical elections.
LAYER_WEIGHT_KEYS: Final[tuple[str, ...]] = ("lambda_1", "lambda_2", "lambda_3")

# Valid source tags for ShockResponseData — records WHICH model produced a
# delta estimate, so we can compare approaches in the paper:
#   "llm_unified"         → the fine-tuned Mistral model using all signals
#   "roberta_news_only"   → the RoBERTa baseline fed only news text
#   "roberta_social_only" → the RoBERTa baseline fed only social-media text
VALID_SOURCES: Final[frozenset[str]] = frozenset(
    ["llm_unified", "roberta_news_only", "roberta_social_only"]
)

# Valid pipeline modes:
#   "historical" → replay known past shocks (for validation against real outcomes)
#   "continuous" → run live on new incoming events
VALID_PIPELINE_MODES: Final[frozenset[str]] = frozenset(["historical", "continuous"])


def bin_to_delta(token: str) -> float:
    """Convert a delta-bin label back into a single numeric vote-share change.

    The LLM gives us a label like "mod_neg"; the optimizer and Monte Carlo need
    an actual number to compute with. We use the midpoint of the label's range
    (e.g. "mod_neg" covers [-0.09, -0.05), so its representative value is -0.07).

    Args:
        token: One of the 9 strings in DELTA_BINS.

    Returns:
        The midpoint vote-share delta for that bin (a float in [-0.12, +0.12]).

    Raises:
        ValueError: if `token` is not a recognized bin — fail loudly rather than
        silently treating a typo as 0.0, which would hide a real bug.
    """
    if token not in BIN_MIDPOINTS:
        raise ValueError(f"Unknown delta bin token {token!r}. Must be one of {DELTA_BINS}")
    return BIN_MIDPOINTS[token]
