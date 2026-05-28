"""Type aliases for the Electoral Equilibrium pipeline.

All demographic identifiers are lowercase snake_case strings.
All election cycles are int in YYYY format.
All share/weight values are floats in [0, 1].
"""
from __future__ import annotations

try:
    from typing import Literal, TypeAlias
except ImportError:
    from typing import Literal  # type: ignore[assignment]
    TypeAlias = type  # type: ignore[assignment,misc]

# ── Scalar aliases ──────────────────────────────────────────────────────────
Cycle: TypeAlias = int          # YYYY format (e.g. 2020); never datetime
BlocId: TypeAlias = str         # lowercase snake_case (e.g. "evangelical")
Weight: TypeAlias = float       # vote share or coalition weight in [0, 1]
ElasticityScore: TypeAlias = float   # RoBERTa sentiment elasticity in [-1, 1]
ShockDelta: TypeAlias = float   # LLM-estimated vote-share change per bloc
LoraRank: TypeAlias = int       # QLoRA rank parameter (e.g. 16 or 32)
Platform: TypeAlias = str       # social media platform identifier
LagDays: TypeAlias = int        # days between shock event and favorability poll
BlocWeight: TypeAlias = float   # inferred probability that an author belongs to a bloc
BlocWeights: TypeAlias = dict   # dict[BlocId, BlocWeight], values sum to 1.0

RaceId: TypeAlias = str         # one of the 5 canonical race identifiers
ReligionId: TypeAlias = str     # one of the 7 canonical religion identifiers
GenderId: TypeAlias = str       # one of the 3 canonical gender identifiers

Party: TypeAlias = Literal["democrat", "republican"]
# Democrat V_eq ~0.52-0.53; Republican V_eq ~0.49-0.51 (from voter panel,
# build_constraint_spec). Every component that reads vote share or sets a win
# condition must receive party as an explicit argument — never hardcode.

# ── Canonical identifiers ───────────────────────────────────────────────────
# Do not deviate from these — confirmed authoritative source for the pipeline.

CANONICAL_RACES: list[str] = [
    "african_american",  # ~12% of electorate
    "latino",            # ~11%
    "asian",             # ~5%
    "white",             # ~62% (non-Hispanic White only)
    "other_race",        # ~10%
]

CANONICAL_RELIGIONS: list[str] = [
    "evangelical",   # ~24%
    "catholic",      # ~21%
    "protestant",    # ~13% (mainline Protestant, non-Evangelical)
    "secular",       # ~26% (none/unaffiliated)
    "jewish",        # ~2%
    "muslim",        # ~1%
    "other_rel",     # ~13%
]

CANONICAL_GENDERS: list[str] = [
    "women",         # ~52%
    "men",           # ~47%
    "other_gender",  # ~1%
]

# 9-token discrete magnitude bins used for LLM constrained decoding.
# Standardized on "slight" (not "weak") — see DECISIONS.md §4.
DELTA_BINS: tuple[str, ...] = (
    "strong_neg",   # numeric range [-0.15, -0.09)
    "mod_neg",      # [-0.09, -0.05)
    "mild_neg",     # [-0.05, -0.02)
    "slight_neg",   # [-0.02, -0.005)
    "neutral",      # [-0.005, +0.005]
    "slight_pos",   # (+0.005, +0.02]
    "mild_pos",     # (+0.02, +0.05]
    "mod_pos",      # (+0.05, +0.09]
    "strong_pos",   # (+0.09, +0.15]
)

# Midpoints used by bin_to_delta()
BIN_MIDPOINTS: dict[str, float] = {
    "strong_neg": -0.120,
    "mod_neg":    -0.070,
    "mild_neg":   -0.035,
    "slight_neg": -0.012,
    "neutral":     0.000,
    "slight_pos": +0.012,
    "mild_pos":   +0.035,
    "mod_pos":    +0.070,
    "strong_pos": +0.120,
}

# Layer weight keys (must sum to 1.0)
LAYER_WEIGHT_KEYS: tuple[str, ...] = ("lambda_1", "lambda_2", "lambda_3")

# Valid source tags for ShockResponseData
VALID_SOURCES: frozenset[str] = frozenset(
    ["llm_unified", "roberta_news_only", "roberta_social_only"]
)

# Valid pipeline modes
VALID_PIPELINE_MODES: frozenset[str] = frozenset(["historical", "continuous"])


def bin_to_delta(token: str) -> float:
    """Map a delta bin token to its numeric midpoint."""
    if token not in BIN_MIDPOINTS:
        raise ValueError(
            f"Unknown delta bin token {token!r}. Must be one of {DELTA_BINS}"
        )
    return BIN_MIDPOINTS[token]