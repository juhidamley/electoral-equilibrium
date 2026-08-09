"""lexicon_match: word-boundary-aware keyword matching for bio lexicons.

Phase 4 precision audit found 33,653 of 153,215 "usable" bio_bloc records
(21.96%) were assigned via an INTERNAL-SUBSTRING match, not a genuine
word-boundary match -- e.g. "son" matching inside "Ferguson"/"Mason"/
"Bolsonaro"/"reason", "desi" matching inside "Afrodesiac"/"design",
"trans" matching inside "transnational"/"transplant". Plain `kw in text`
substring checks cannot distinguish these from real matches.

This module is the SINGLE SOURCE OF TRUTH for that check, imported by both
scripts/sample_archives.py::keyword_bio_bloc() (the function that actually
produced the corpus's bio_bloc field) and
electoral/nlp/bio_classifier.py::_stage1_keyword() (feeds
build_bio_labels.py / SetFit training). These two independently
reimplemented near-identical keyword-scan logic once already (different
priority order, different threshold behavior) and drifted -- see Phase 4
Step 4.4 diagnosis. Do not let this specific piece drift too; if you need to
change matching behavior, change it here.

STRICT word-boundary matching alone is not enough: it also flags legitimate
inflections as false positives -- "episcopal" not matching inside
"episcopalian", "vatican" not matching inside "vaticana"/"vaticano" (Italian),
"pagan" not matching inside "paganism", "fundamentalist" not matching inside
"fundamentalists" -- all real, correct extensions of the same word. The
audit's own first pass over-flagged on exactly this (25.40% -> 21.96% after
correcting for it). So this checks strict boundaries FIRST, then falls back
to keyword+suffix boundaries for a small set of common inflections.

KNOWN RESIDUAL LIMITATION: the suffix list below intentionally includes
bare "a"/"o" to catch Spanish/Italian gender-adjective inflections
("latino"/"latina", "orgulloso"/"orgullosa"). This also legitimizes a few
unrelated homographs for OTHER keys -- e.g. "son" + "o" matches Italian
"sono" ("I am"), a verb form with no connection to the English word "son".
This is exactly why "son" was removed from gender_lexicon.json rather than
kept and boundary-fixed: word-boundary matching cannot rescue a key that
collides with unrelated words/verb-forms in other languages even at a clean
boundary. See gender_lexicon.json's note for the per-key precision findings
that drove each keep/remove decision.
"""

from __future__ import annotations

import re

# Suffixes tolerated after a keyword before requiring a hard boundary --
# plurals, adjectival/demonym forms, and Spanish/Italian gender inflections.
# Deliberately small and conservative: each entry here was added because a
# real corpus example needed it (see module docstring), not speculatively.
_COMMON_INFLECTION_SUFFIXES: tuple[str, ...] = (
    "s",
    "es",
    "ing",
    "ed",
    "ism",
    "ist",
    "ian",
    "ianism",
    "a",
    "o",
    "os",
    "as",
    "my",
)

_boundary_cache: dict[tuple[str, str], bool] = {}
_CACHE_MAX = 200_000


def has_word_boundary_match(keyword: str, text_lower: str) -> bool:
    """True if `keyword` occurs in `text_lower` at a word boundary (allowing
    common inflections), not only as a substring inside a longer word.

    `text_lower` must already be lowercased (callers already lowercase bio
    text once per classification; doing it again per-keyword here would be
    wasteful across a lexicon scan).

    Symbols with no alphanumeric characters (e.g. gender-sign emoji) have no
    substring-collision risk and always return True on a plain `in` hit --
    callers should still gate on `keyword in text_lower` first for those, as
    the existing keyword_bio_bloc()/_stage1_keyword() loops already do.
    """
    if not any(c.isalnum() for c in keyword):
        return True

    cache_key = (keyword, text_lower)
    cached = _boundary_cache.get(cache_key)
    if cached is not None:
        return cached

    esc = re.escape(keyword)
    if re.search(r"(?<![a-z0-9])" + esc + r"(?![a-z0-9])", text_lower):
        result = True
    else:
        result = False
        for suffix in _COMMON_INFLECTION_SUFFIXES:
            if re.search(r"(?<![a-z0-9])" + esc + re.escape(suffix) + r"(?![a-z0-9])", text_lower):
                result = True
                break
        if not result:
            # Trailing digits: hashtag/handle styling ("dominican4life").
            if re.search(r"(?<![a-z0-9])" + esc + r"\d+(?![a-z0-9])", text_lower):
                result = True

    if len(_boundary_cache) < _CACHE_MAX:
        _boundary_cache[cache_key] = result
    return result
