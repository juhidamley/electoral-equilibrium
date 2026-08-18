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
#
# RESCALED Step 2.1 (see DECISIONS.md "Step 2.1 — rescale delta-bin midpoints to
# measured reality"): the original [-0.15, +0.15] range was an ungrounded design
# assumption. Panel ground truth (data/ground_truth/panel_deltas.json) measures
# real per-bloc shifts with a median of ~0.0036 and a largest trustworthy cell of
# 0.0201; the old scale overstated real movement by roughly an order of magnitude.
# Every value below is the old value x0.25, mapping the old ±0.12 max midpoint onto
# a ±0.03 ceiling. The bins tile [-0.0375, +0.0375] with no gaps, and 0.0375 (not
# 0.03) is the clip ceiling used downstream — it is the strong bin's outer EDGE,
# not its midpoint.
#
# The neutral/slight boundary is ±0.00125, NOT a scaled midpoint. Anything that
# thresholds "is this effectively zero" must use that edge — see
# electoral/llm/eval.py::_sign().

DELTA_BINS: Final[tuple[str, ...]] = (
    "strong_neg",  # numeric range [-0.0375, -0.0225)
    "mod_neg",  # [-0.0225, -0.0125)
    "mild_neg",  # [-0.0125, -0.005)
    "slight_neg",  # [-0.005, -0.00125)
    "neutral",  # [-0.00125, +0.00125]
    "slight_pos",  # (+0.00125, +0.005]
    "mild_pos",  # (+0.005, +0.0125]
    "mod_pos",  # (+0.0125, +0.0225]
    "strong_pos",  # (+0.0225, +0.0375]
)

# Midpoints used by bin_to_delta(). Rescaled Step 2.1 — old value x0.25.
# THIS IS THE ONLY DEFINITION. Do not copy these numbers into another module;
# import them. tests/test_bin_midpoints_sync.py enforces that tree-wide.
BIN_MIDPOINTS: Final[dict[str, float]] = {
    "strong_neg": -0.0300,
    "mod_neg": -0.0175,
    "mild_neg": -0.00875,
    "slight_neg": -0.00300,
    "neutral": 0.000,
    "slight_pos": +0.00300,
    "mild_pos": +0.00875,
    "mod_pos": +0.0175,
    "strong_pos": +0.0300,
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
