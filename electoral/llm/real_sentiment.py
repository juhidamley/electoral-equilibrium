"""Optional inference-time real-sentiment refinement (additive, OFF by default).

For events that have real archive coverage in
``data/finetune/shock_sentiment_aggregates.json``, this module looks up the
measured per-bloc social sentiment (RoBERTa scores) so the caller can populate
the model's ``## Social Bio-Weighted Scores`` prompt field before prediction.

HONEST FRAMING — this is a *soft inference-time correction, not a retrained
driver.* The v3 model was fine-tuned on 982/982 records with EMPTY sentiment
fields, so it treats injected values as a weak hint, not a strong lever.
Measured behavior on 4 covered events (v3, greedy): ~1/3 of blocs shift, and of
those ~85% move *toward* the real reaction, magnitude ~1 bin — never nonsensical.
So refinement nudges predictions toward measured reactions; it does not override
the model's learned priors (e.g. it keeps its "BLM mobilizes Democrats" belief).

Everything degrades gracefully: if the aggregates file is absent (e.g. not
mounted in a deployment image) or no event matches, the lookup returns ``None``
and the caller runs exactly the unrefined path.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Default locations, resolved against both CWD and the repo root so this works
# whether the server is launched from the repo root or elsewhere.
_AGG_CANDIDATES = [
    Path("data/finetune/shock_sentiment_aggregates.json"),
    Path(__file__).parent.parent.parent / "data" / "finetune" / "shock_sentiment_aggregates.json",
]
_REGISTRY_CANDIDATES = [
    Path("configs/shocks.json"),
    Path(__file__).parent.parent.parent / "configs" / "shocks.json",
]

# Minimum real posts per bloc to trust a bloc's aggregate (matches the archive's
# own coverage gate). Blocs below this are omitted from the injected scores.
DEFAULT_MIN_BLOC_N = 30
# Minimum keyword overlap to accept a text→shock match (high precision: only
# refine when we're confident the query really is the covered event).
_MIN_MATCH_OVERLAP = 2


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


@lru_cache(maxsize=1)
def _load_aggregates() -> dict[str, Any]:
    """Load the shock→per-bloc-sentiment archive; empty dict if unavailable."""
    path = _first_existing(_AGG_CANDIDATES)
    if path is None:
        log.info("real_sentiment: aggregates file not found — refinement unavailable")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("aggregates", {})
    except Exception:
        log.warning("real_sentiment: failed to parse %s", path, exc_info=True)
        return {}


@lru_cache(maxsize=1)
def _load_registry() -> list[dict[str, Any]]:
    """Load configs/shocks.json (id, keywords, description) for text matching."""
    path = _first_existing(_REGISTRY_CANDIDATES)
    if path is None:
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return entries if isinstance(entries, list) else []
    except Exception:
        log.warning("real_sentiment: failed to parse shocks registry", exc_info=True)
        return []


def _tokens(text: str) -> set[str]:
    """Lowercase word set, dropping punctuation and short stop-tokens."""
    return {w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(w) > 2}


def match_shock_id(event_text: str) -> Optional[str]:
    """Return the best-matching covered shock_id for free-text ``event_text``.

    Matches the query's keywords against each registry entry's keywords + id
    tokens, keeping only shocks that actually have sentiment coverage. Requires
    ``_MIN_MATCH_OVERLAP`` overlapping keywords so unrelated events don't get
    spuriously "refined". Returns None when no confident match exists.
    """
    agg = _load_aggregates()
    if not agg:
        return None
    ev = _tokens(event_text)
    if not ev:
        return None
    best_id: Optional[str] = None
    best_overlap = 0
    for entry in _load_registry():
        sid = entry.get("id")
        if sid not in agg:  # only shocks with real sentiment coverage
            continue
        kw: set[str] = {w for w in str(sid).split("_") if not w.isdigit()}
        for k in entry.get("keywords", []):
            kw |= _tokens(str(k))
        overlap = len(ev & kw)
        if overlap > best_overlap:
            best_overlap, best_id = overlap, sid
    return best_id if best_overlap >= _MIN_MATCH_OVERLAP else None


def get_social(shock_id: str) -> dict[str, Any]:
    """Return the raw per-bloc social dict ``{bloc: {roberta, sentiment, n}}`` for
    a shock_id (empty dict if the shock is unknown/uncovered). Used by the
    empirical-support layer to read both scores and post counts."""
    return _load_aggregates().get(shock_id, {}).get("social", {})


def lookup_real_social_sentiment(
    event_text: str,
    *,
    min_bloc_n: int = DEFAULT_MIN_BLOC_N,
) -> Optional[dict[str, Any]]:
    """Return real per-bloc social sentiment for a covered event, else None.

    Result shape (None when no confident match or no covered blocs)::

        {
          "shock_id": "dobbs_2022",
          "scores": {bloc: roberta_mean, ...},   # blocs with n >= min_bloc_n
          "n_blocs": <int>,
          "min_bloc_n": <int>,
          "source": "shock_sentiment_aggregates.json (roberta mean)",
        }

    ``scores`` is what the caller injects into the prompt's social field.
    """
    sid = match_shock_id(event_text)
    if sid is None:
        return None
    social = _load_aggregates().get(sid, {}).get("social", {})
    scores = {
        bloc: round(float(info["roberta"]), 4)
        for bloc, info in social.items()
        if isinstance(info, dict) and info.get("n", 0) >= min_bloc_n and "roberta" in info
    }
    if not scores:
        return None
    return {
        "shock_id": sid,
        "scores": scores,
        "n_blocs": len(scores),
        "min_bloc_n": min_bloc_n,
        "source": "shock_sentiment_aggregates.json (roberta mean)",
    }
