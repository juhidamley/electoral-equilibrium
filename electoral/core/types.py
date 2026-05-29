"""Type aliases and canonical constants for the Electoral Equilibrium pipeline.

All demographic identifiers are lowercase snake_case strings.
All election cycles are int in YYYY format.
All share/weight values are floats in [0, 1].
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
# 9-token discrete magnitude bins for LLM constrained decoding.
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

# Layer weight keys (must sum to 1.0)
LAYER_WEIGHT_KEYS: Final[tuple[str, ...]] = ("lambda_1", "lambda_2", "lambda_3")

# Valid source tags for ShockResponseData
VALID_SOURCES: Final[frozenset[str]] = frozenset(
    ["llm_unified", "roberta_news_only", "roberta_social_only"]
)

# Valid pipeline modes
VALID_PIPELINE_MODES: Final[frozenset[str]] = frozenset(["historical", "continuous"])


def bin_to_delta(token: str) -> float:
    """Map a delta bin token to its numeric midpoint."""
    if token not in BIN_MIDPOINTS:
        raise ValueError(f"Unknown delta bin token {token!r}. Must be one of {DELTA_BINS}")
    return BIN_MIDPOINTS[token]
