"""Panel cleaning — coerce, normalise, drop, sort, and dedup a raw survey panel."""

from __future__ import annotations

import logging
import re

import pandas as pd

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

log = logging.getLogger(__name__)

# Columns that must be non-null to keep a row.
_REQUIRED: tuple[str, ...] = ("cycle", "bloc")

# Ordered substitution rules applied to raw bloc strings.
_BLOC_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\s/\-\.]+"), "_"),  # whitespace / slashes / hyphens / dots → _
    (re.compile(r"[^a-z0-9_]"), ""),  # strip remaining non-word chars
    (re.compile(r"_+"), "_"),  # collapse runs of underscores
    (re.compile(r"^_+|_+$"), ""),  # strip leading/trailing underscores
]

# ── Bloc name normalisation ───────────────────────────────────────────────────

# The 15 canonical bloc IDs — derived from the authoritative enums in
# electoral.core.types so they can never diverge from Race/Religion/Gender.
CANONICAL_BLOCS: frozenset[str] = frozenset(
    {*CANONICAL_RACES, *CANONICAL_RELIGIONS, *CANONICAL_GENDERS}
)

# Maps snake_case-normalised labels → canonical bloc ID.
# Keys are produced by applying _BLOC_SUBS to the lowercased raw label.
# Ambiguous labels (e.g. bare "other") are intentionally absent — callers
# must use the stratum-qualified form ("other_race", "other_rel", "other_gender").
_BLOC_MAP: dict[str, str] = {
    # ── Race (DECISIONS.md §Demographic Architecture) ─────────────────────────
    # canonical
    "african_american": "african_american",
    "latino": "latino",
    "asian": "asian",
    "white": "white",
    "other_race": "other_race",
    # survey aliases
    "black": "african_american",
    "black_african_american": "african_american",
    "hispanic": "latino",
    "hispanic_latino": "latino",
    "latino_hispanic": "latino",
    "latina": "latino",
    "latinx": "latino",
    "chicano": "latino",
    "chicana": "latino",
    "asian_american": "asian",
    "non_hispanic_white": "white",
    "white_non_hispanic": "white",
    "other_races": "other_race",
    "all_others": "other_race",  # NEP exit-poll label
    "multiracial": "other_race",
    "two_or_more_races": "other_race",
    "indigenous": "other_race",
    "native_american": "other_race",
    # ── Religion (DECISIONS.md §Seven religion groups) ────────────────────────
    # canonical
    "evangelical": "evangelical",
    "catholic": "catholic",
    "protestant": "protestant",
    "secular": "secular",
    "jewish": "jewish",
    "muslim": "muslim",
    "other_rel": "other_rel",
    # evangelical aliases — "White" prefix is common in NEP religion rows;
    # these map to the religion-only stratum (not a race×religion cross-tab)
    "white_evangelical": "evangelical",
    "born_again": "evangelical",
    "born_again_christian": "evangelical",
    "evangelical_born_again": "evangelical",
    "white_evangelical_born_again": "evangelical",
    "evangelical_christian": "evangelical",
    "white_evangelical_christian": "evangelical",
    "protestant_evangelical": "evangelical",
    "white_evangelical_born_again_christian": "evangelical",
    "white_evangelical_or_born_again_christian": "evangelical",
    # catholic aliases
    "roman_catholic": "catholic",
    "catholics": "catholic",
    # protestant aliases
    "protestants": "protestant",
    "mainline_protestant": "protestant",
    "other_protestant": "protestant",
    "non_evangelical_protestant": "protestant",
    "protestants_non_evangelical": "protestant",  # NEP parenthetical label
    # secular aliases
    "none": "secular",
    "no_religion": "secular",
    "unaffiliated": "secular",
    "religiously_unaffiliated": "secular",
    "agnostic": "secular",
    "atheist": "secular",
    "no_religion_secular": "secular",
    "nones": "secular",
    # jewish aliases
    "jew": "jewish",
    "judaism": "jewish",
    # muslim aliases
    "islam": "muslim",
    "islamic": "muslim",
    # other_rel aliases
    "other_religion": "other_rel",
    "other_religions": "other_rel",
    "other_faith": "other_rel",
    "other_faiths": "other_rel",
    "other_dk": "other_rel",
    "something_else": "other_rel",
    # ── Gender (DECISIONS.md §Three-stratum architecture) ─────────────────────
    # canonical
    "women": "women",
    "men": "men",
    "other_gender": "other_gender",
    # aliases
    "woman": "women",
    "female": "women",
    "man": "men",
    "male": "men",
    "non_binary": "other_gender",
    "nonbinary": "other_gender",
    "transgender": "other_gender",
}


def _to_key(raw: str) -> str:
    """Apply the same snake_case normalisation as _normalize_bloc to a scalar."""
    s = raw.strip().lower()
    for pattern, repl in _BLOC_SUBS:
        s = pattern.sub(repl, s)
    return s


def normalize_bloc(raw: str) -> str:
    """Map a source-specific demographic label to its canonical bloc ID.

    Applies snake_case normalisation, then looks up the result against the
    project's bloc alias table (built from DECISIONS.md and survey lexicons).
    Canonical IDs (e.g. ``"evangelical"``) are accepted directly.

    Ambiguous bare labels such as ``"other"`` are not in the table — use the
    stratum-qualified form: ``"other_race"``, ``"other_rel"``, or
    ``"other_gender"``.

    Parameters
    ----------
    raw:
        Raw demographic label as it appears in the source (e.g. from a NEP
        exit-poll sub_category column).  Case and whitespace are ignored.

    Returns
    -------
    str
        One of the 15 canonical bloc IDs.

    Raises
    ------
    ValueError
        If *raw* does not resolve to any known alias or canonical ID.
    """
    if not isinstance(raw, str):
        raise ValueError(f"normalize_bloc: expected a non-null str, got {type(raw).__name__}")
    key = _to_key(raw)
    if key in CANONICAL_BLOCS:
        return key
    canonical = _BLOC_MAP.get(key)
    if canonical is None:
        raise ValueError(
            f"normalize_bloc: unrecognized bloc label {raw!r} "
            f"(normalized key: {key!r}). "
            f"Canonical blocs: {sorted(CANONICAL_BLOCS)}."
        )
    return canonical


def _normalize_bloc(s: pd.Series) -> pd.Series:
    """Lowercase and snake_case a string Series; preserve NA as pd.NA."""
    null_mask = s.isna()
    # fillna("") so str operations see an empty string for nulls, not "nan".
    out = s.fillna("").astype(str).str.lower().str.strip()
    for pattern, repl in _BLOC_SUBS:
        out = out.str.replace(pattern, repl, regex=True)
    out = out.astype("string")
    # Restore original nulls and values that normalised to empty string.
    # Use .eq("") rather than == "" to avoid Python-level comparison on StringArray.
    out[null_mask | out.eq("")] = pd.NA
    return out


def clean_raw_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to a raw survey panel DataFrame.

    Steps (in order):
    1. Coerce ``cycle`` to nullable Int64 (YYYY); non-numeric → NA.
    2. Normalise ``bloc`` to lowercase snake_case nullable string.
    3. Coerce ``vote_share`` and ``turnout`` to nullable Float64; non-numeric → NA.
    4. Drop rows where ``cycle`` or ``bloc`` is null; log count at INFO.
    5. Sort by (cycle, bloc); reset index.
    6. Raise ValueError on duplicate (cycle, bloc, source) tuples.
       Falls back to (cycle, bloc) when ``source`` column is absent.
       Note: ``pd.NA`` in ``source`` is treated as equal by pandas
       ``duplicated()`` — rows with missing source can be flagged as
       duplicates of each other even if they represent distinct provenance.

    Returns a new DataFrame; the input is not modified.
    """
    df = df.copy()

    # ── Step 1: cycle → Int64 ────────────────────────────────────────────────
    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce").astype("Int64")

    # ── Step 2: bloc → snake_case → canonical ID ─────────────────────────────
    if "bloc" in df.columns:
        df["bloc"] = _normalize_bloc(df["bloc"])
        df["bloc"] = (
            df["bloc"].map(lambda x: normalize_bloc(x) if pd.notna(x) else pd.NA).astype("string")
        )

    # ── Step 3: vote_share / turnout → Float64 ────────────────────────────────
    for col in ("vote_share", "turnout"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    # ── Step 4: drop rows missing cycle or bloc ───────────────────────────────
    required = [c for c in _REQUIRED if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.info(
            "clean_raw_panel: dropped %d row(s) with null %s",
            n_dropped,
            required,
        )

    # ── Step 5: sort by (cycle, bloc) ────────────────────────────────────────
    sort_keys = [c for c in ("cycle", "bloc") if c in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)

    # ── Step 6: raise on duplicate key tuples ────────────────────────────────
    dup_keys = [c for c in ("cycle", "bloc", "source") if c in df.columns]
    if len(dup_keys) >= 2:
        dupes = df.duplicated(subset=dup_keys, keep=False)
        if dupes.any():
            examples = df.loc[dupes, dup_keys].drop_duplicates().head(3).to_dict(orient="records")
            raise ValueError(
                f"clean_raw_panel: duplicate ({', '.join(dup_keys)}) tuples detected — "
                f"deduplicate before cleaning. Examples: {examples}"
            )

    return df
