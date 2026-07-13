"""Empirical grounding of a prediction against real archived reactions.

Two integrated, HONESTLY-FRAMED behaviors, both purely at inference/annotation
time (no retraining, no model changes):

1. EMPIRICAL SUPPORT (every covered event) — annotate each bloc's predicted
   delta with the real scored social sentiment (RoBERTa) and a per-bloc
   ``agreement`` flag (aligned | diverged | no_data). This is an OUT-OF-SAMPLE
   COMPARISON, not an accuracy score: sentiment polarity and Democratic-loyalty
   delta need not share a sign, so ``diverged`` is informative, not a failure.

2. SENTIMENT-REFINED PREDICTION (aligned regime ONLY) — for events the
   validation split labels "sentiment-aligned", the caller may inject the real
   sentiment to produce a second, refined prediction (the validated soft
   correction: ~85% of changed blocs move toward reality, ~1 bin). This is
   STRICTLY GATED: valence-divergent / mobilizing events (BLM, etc.) are NEVER
   refined — injecting negative reaction sentiment there pushes predictions the
   wrong way. Divergent and uncovered events get the base prediction only.

Regime source: data/validation/oos_sentiment_alignment.json (the divergent set).
Sentiment source: data/finetune/shock_sentiment_aggregates.json (via real_sentiment).
Real sentiment is NEVER fabricated — uncovered events report agreement=no_data.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from electoral.core.types import (
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)
from electoral.llm.real_sentiment import (
    DEFAULT_MIN_BLOC_N,
    get_social,
    lookup_real_social_sentiment,
)

log = logging.getLogger(__name__)

_ALL_BLOCS: list[str] = (
    list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)
)

_OOS_CANDIDATES = [
    Path("data/validation/oos_sentiment_alignment.json"),
    Path(__file__).parent.parent.parent / "data" / "validation" / "oos_sentiment_alignment.json",
]

_EMP_NOTE = (
    "Empirical support = out-of-sample comparison of the base prediction's per-bloc "
    "direction against real archived social sentiment (RoBERTa mean). It is NOT an "
    "accuracy score: sentiment polarity and Democratic-loyalty delta need not share a "
    "sign, so 'diverged' can be correct model behavior (see regime)."
)
_REFINE_APPLIED_NOTE = (
    "Soft inference-time correction on a SENTIMENT-ALIGNED event: real social sentiment "
    "was injected into the prompt. Validated behavior: ~85% of changed blocs move toward "
    "the real reaction, ~1 bin magnitude. This is NOT a retrained driver — the model was "
    "fine-tuned on empty sentiment fields, so injection is a soft hint, not a strong lever."
)
_REFINE_AVAILABLE_NOTE = (
    "This aligned event supports the soft sentiment refinement (off by default). "
    "Re-request with refine_with_real_sentiment=true to also get the refined prediction."
)
_REFINE_SUPPRESSED_NOTE = (
    "Refinement is deliberately gated to the sentiment-aligned regime. This event is "
    "valence-divergent (a mobilizing event where negative reaction sentiment coincides "
    "with Democratic GAINS), so injecting sentiment would push predictions the wrong way. "
    "Empirical support is still shown; only the base prediction is returned."
)


@lru_cache(maxsize=1)
def _load_regime() -> Optional[frozenset[str]]:
    """Set of valence-divergent shock_ids, or ``None`` if the regime file is
    missing/unreadable.

    Returning None (not an empty set) is deliberate and FAIL-SAFE: an empty set
    would misclassify every covered event as "aligned" and could silently refine
    a divergent event. None instead makes classify_event no-op the whole feature
    (base prediction only), so a missing/corrupt regime file can never break base
    predictions OR enable unsafe refinement.
    """
    for p in _OOS_CANDIDATES:
        if p.is_file():
            try:
                oos = json.loads(p.read_text(encoding="utf-8"))
                return frozenset(oos.get("valence_divergence", {}).get("shocks", {}).keys())
            except Exception:
                log.warning(
                    "empirical_support: failed to parse %s — grounding disabled", p, exc_info=True
                )
                return None
    log.info("empirical_support: regime file not found — grounding disabled (base predictions only)")
    return None


def _sign(x: Optional[float], eps: float = 1e-9) -> int:
    if x is None:
        return 0
    return 1 if x > eps else (-1 if x < -eps else 0)


def classify_event(event_text: str, *, min_bloc_n: int = DEFAULT_MIN_BLOC_N) -> tuple[Optional[str], str]:
    """Return (shock_id | None, regime) where regime ∈ {aligned, divergent, uncovered}.

    ``uncovered`` = no confident archive match, or matched but no bloc has enough
    real posts. ``divergent`` = matched + covered + in the valence-divergent set.
    ``aligned`` = matched + covered + not divergent (safe to refine).
    """
    # Fail-safe: if the regime file is unavailable, no-op the whole feature (base
    # prediction only) rather than risk refining a divergent event.
    divergent = _load_regime()
    if divergent is None:
        return None, "uncovered"
    match = lookup_real_social_sentiment(event_text, min_bloc_n=min_bloc_n)
    if match is None:
        return None, "uncovered"
    sid = match["shock_id"]
    return sid, ("divergent" if sid in divergent else "aligned")


def _empirical_rows(shock_id: str, base_bins: dict[str, str], min_bloc_n: int) -> tuple[list[dict], int, int, int]:
    """Per-bloc empirical-support rows + (n_aligned, n_diverged, n_no_data)."""
    social = get_social(shock_id)
    rows: list[dict[str, Any]] = []
    n_aligned = n_diverged = n_no_data = 0
    for bloc in _ALL_BLOCS:
        pred_bin = base_bins.get(bloc)
        pred_delta = BIN_MIDPOINTS.get(pred_bin) if pred_bin else None
        info = social.get(bloc)
        if info and info.get("n", 0) >= min_bloc_n and "roberta" in info:
            real = round(float(info["roberta"]), 4)
            n_posts = int(info["n"])
            if _sign(pred_delta) == 0 or _sign(real) == 0:
                agreement = "no_data"
                n_no_data += 1
            elif _sign(pred_delta) == _sign(real):
                agreement = "aligned"
                n_aligned += 1
            else:
                agreement = "diverged"
                n_diverged += 1
        else:
            real = None
            n_posts = int(info["n"]) if info else 0
            agreement = "no_data"
            n_no_data += 1
        rows.append(
            {
                "bloc": bloc,
                "predicted_delta": pred_delta,
                "predicted_bin": pred_bin,
                "real_social_sentiment": real,
                "n_posts": n_posts,
                "agreement": agreement,
            }
        )
    return rows, n_aligned, n_diverged, n_no_data


def build_grounding(
    event_text: str,
    base_bins: dict[str, str],
    *,
    refined_bins: Optional[dict[str, str]] = None,
    min_bloc_n: int = DEFAULT_MIN_BLOC_N,
) -> dict[str, Any]:
    """Assemble the empirical-support + refinement grounding for a prediction.

    Parameters
    ----------
    event_text:
        Free-text event description (used to match archive coverage/regime).
    base_bins:
        FLAT {bloc: delta_bin} of the BASE (un-refined) prediction, all 15 blocs.
    refined_bins:
        FLAT {bloc: delta_bin} of the refined prediction, supplied ONLY when the
        caller actually ran refinement (aligned regime). None otherwise.

    Returns a dict: {shock_id, regime, empirical_support, refinement}. For
    uncovered events empirical_support is None and refinement is marked base-only.
    """
    shock_id, regime = classify_event(event_text, min_bloc_n=min_bloc_n)

    if regime == "uncovered":
        return {
            "shock_id": None,
            "regime": "uncovered",
            "empirical_support": None,
            "refinement": {
                "regime": "uncovered",
                "refinable": False,
                "applied": False,
                "reason": "No real-reaction archive coverage for this event.",
                "refined_prediction": None,
                "note": "Base prediction only; real sentiment is never fabricated (agreement=no_data).",
            },
        }

    rows, n_aligned, n_diverged, n_no_data = _empirical_rows(shock_id, base_bins, min_bloc_n)
    empirical_support = {
        "shock_id": shock_id,
        "regime": regime,
        "n_aligned": n_aligned,
        "n_diverged": n_diverged,
        "n_no_data": n_no_data,
        "blocs": rows,
        "note": _EMP_NOTE,
    }

    if regime == "aligned":
        if refined_bins is not None:
            refined_deltas = {b: BIN_MIDPOINTS.get(v, 0.0) for b, v in refined_bins.items()}
            refinement = {
                "regime": "aligned",
                "refinable": True,
                "applied": True,
                "reason": None,
                "refined_prediction": {"bins": refined_bins, "deltas": refined_deltas},
                "note": _REFINE_APPLIED_NOTE,
            }
        else:
            refinement = {
                "regime": "aligned",
                "refinable": True,
                "applied": False,
                "reason": _REFINE_AVAILABLE_NOTE,
                "refined_prediction": None,
                "note": "Soft inference-time correction, off by default.",
            }
    else:  # divergent — refinement forbidden
        refinement = {
            "regime": "divergent",
            "refinable": False,
            "applied": False,
            "reason": _REFINE_SUPPRESSED_NOTE,
            "refined_prediction": None,
            "note": "Empirical support shown; refinement gated OUT of the divergent regime.",
        }

    return {
        "shock_id": shock_id,
        "regime": regime,
        "empirical_support": empirical_support,
        "refinement": refinement,
    }
