"""Regression guard against internal-substring lexicon false positives.

This bug class has appeared twice: the male/female collision caught during
Phase 4 Step 5.1's lexicon work (before it ever shipped), and the son/desi/
trans/bright/lds/pagan/black substring collisions found afterward by a
corpus-wide precision audit (21.96% of "usable" bio_bloc assignments were
internal-substring matches, not genuine ones -- see
electoral/nlp/lexicon_match.py's module docstring and configs/*_lexicon.json's
`note` fields for the full writeup). Two independent discoveries of the same
bug class is the point at which you stop relying on catching it by hand a
third time and add a guard instead.

This test does NOT try to prove a new keyword is safe in general (that's not
mechanically decidable -- "vatican" is fine, "son" isn't, and both look like
ordinary short words). It proves a keyword doesn't collide with the SPECIFIC
false positives already found in the real corpus. Extend FALSE_POSITIVE_TRAPS
whenever a corpus scan or hand-verification pass finds a new one -- that's
the intended maintenance path, not a one-time fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from electoral.nlp.lexicon_match import has_word_boundary_match  # noqa: E402
from scripts.sample_archives import keyword_bio_bloc, load_lexicons  # noqa: E402

# Each trap: (text, the substring collision it's known to trigger, source).
# These are REAL strings pulled from the corpus during the Phase 4 precision
# audit -- not synthetic examples. bio text containing any of these must
# classify as "unknown"; if a lexicon key change makes one of these match,
# that key has reintroduced a known false-positive collision.
FALSE_POSITIVE_TRAPS: list[tuple[str, str]] = [
    ("Turd Ferguson", "son (surname)"),
    ("Howdy, Parson! Welcome to Hell!", "son (Parson)"),
    ("#ForaBolsonaro", "son (Bolsonaro)"),
    ("Family Medicine Doctor. Interests: Population Health.", "son (Population)"),
    ("Ms.Afrodesiac", "desi (Afrodesiac)"),
    ("Devout critic and former architect. @parsonsdesign", "desi (parsonsdesign) + son (parsons)"),
    ("Transnational Feminism", "trans (Transnational)"),
    ("Utah transplant", "trans (transplant)"),
    ("A very happy European looking forward to a bright and wonderful future", "bright (plain adjective)"),
    ("Graphic Designer. I love Bright Colors", "bright (plain adjective) + desi (Designer)"),
    ("Once There Was A Snowman is propaganda", "pagan (propaganda)"),
    ("In the Rabbit Hole", "rabbi (Rabbit)"),
    ("Going down the rabbit hole was the best thing ever!", "rabbi (rabbit)"),
    ("BlackDickVerified", "black (compound username, no boundary)"),
    ("Verified Blackman", "black (compound surname/username)"),
    ("La única ideología chingona es en la que uno cree, todas las demás son pendejadas.", "son (Spanish verb 'son' = 'are')"),
    ("Sono nato e vivo a Roma", "son (Italian 'sono' = 'I am')"),
]

# Legitimate inflections that must NOT be broken by the boundary fix --
# these are correct matches the audit's first (over-strict) pass wrongly
# flagged as false positives before suffix-tolerance was added.
LEGITIMATE_INFLECTIONS: list[tuple[str, str]] = [
    ("Episcopalian (Anglican)", "protestant"),
    ("Trying to avoid fundamentalists.", "evangelical"),
    ("Neo-Pagan, New Age Spiritist", "other_rel"),
    ("Radio Vaticana Italia", "catholic"),
    ("person, web writer, dissident, son, brother, friend", "men"),
]


@pytest.fixture(scope="module")
def lexicons():
    return load_lexicons()


@pytest.mark.parametrize("text,collision_source", FALSE_POSITIVE_TRAPS)
def test_known_false_positive_traps_stay_unknown(lexicons, text, collision_source):
    result = keyword_bio_bloc(text, lexicons)
    assert result == "unknown", (
        f"{text!r} classified as {result!r} via a known false-positive collision "
        f"({collision_source}). A lexicon key was added or changed that reintroduces "
        f"this internal-substring match -- see electoral/nlp/lexicon_match.py and "
        f"configs/*_lexicon.json's `note` fields for the precision audit this guards."
    )


@pytest.mark.parametrize("text,expected_bloc", LEGITIMATE_INFLECTIONS)
def test_legitimate_inflections_still_match(lexicons, text, expected_bloc):
    result = keyword_bio_bloc(text, lexicons)
    assert result == expected_bloc, (
        f"{text!r} classified as {result!r}, expected {expected_bloc!r}. The "
        f"word-boundary fix's inflection-suffix tolerance "
        f"(electoral.nlp.lexicon_match._COMMON_INFLECTION_SUFFIXES) may have "
        f"regressed -- these are genuine matches, not false positives, and must "
        f"keep matching."
    )


def test_new_lexicon_keys_checked_against_trap_corpus(lexicons):
    """Sweep every current lexicon key against every trap string directly
    (bypassing keyword_bio_bloc's priority/best-weight logic) so a collision
    is caught even if some other stratum would have won first and masked it
    in the end-to-end test above.

    A raw substring hit (e.g. 'afro' inside 'afrodesiac') is EXPECTED and
    fine on its own -- that's exactly the case has_word_boundary_match()
    exists to reject. The actual failure condition is a keyword matching a
    trap string AND being accepted as a word-boundary match -- that means
    the boundary guard itself failed to catch a known collision.
    """
    trap_texts_lower = [t.lower() for t, _ in FALSE_POSITIVE_TRAPS]
    offenders = []
    for stratum in ("race", "religion", "gender"):
        for keyword in lexicons.get(stratum, {}):
            for trap_lower in trap_texts_lower:
                if keyword in trap_lower and has_word_boundary_match(keyword, trap_lower):
                    offenders.append((stratum, keyword, trap_lower))
    assert not offenders, (
        "Lexicon key(s) are wrongly ACCEPTED as word-boundary matches inside known "
        "false-positive trap strings:\n"
        + "\n".join(f"  [{s}] {k!r} wrongly accepted inside {t!r}" for s, k, t in offenders)
    )
