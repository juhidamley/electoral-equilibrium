"""bio_classifier: 3-stage author demographic inference.

Inference hierarchy (first successful stage wins):
  1. Keyword lexicon  — scans bio text against race/religion/gender lexicons
  2. Language prior   — language_priors.json when post is non-English
  3. SetFit (Pi)      — POST bio to Pi server for embedding + bloc assignment
  4. null             — excluded from mean/covariance estimation

Platform-level proxies ("platform_proxy", "subreddit_proxy") are set upstream
by the collector and short-circuit classification here.

Per CLAUDE.md: posts with inference_method="language_prior" are EXCLUDED from
both mean μ and covariance Σ_Δ estimation — use as held-out validation only.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RACE_LEXICON_PATH = _REPO_ROOT / "configs" / "race_lexicon.json"
_RELIGION_LEXICON_PATH = _REPO_ROOT / "configs" / "religion_lexicon.json"
_GENDER_LEXICON_PATH = _REPO_ROOT / "configs" / "gender_lexicon.json"
_LANGUAGE_PRIORS_PATH = _REPO_ROOT / "configs" / "language_priors.json"

# Methods set by collectors upstream — classifier is a no-op for these
_UPSTREAM_METHODS: frozenset[str] = frozenset(["platform_proxy", "subreddit_proxy"])

# BCP-47 lang prefix → language_priors.json key
_LANG_TO_PRIOR: dict[str, str] = {
    "es": "spanish",
    "ko": "korean",
    "zh": "chinese",
    "ar": "arabic",
}

# Languages we treat as English (no language-prior fallback)
_ENGLISH_LANGS: frozenset[str] = frozenset(["en", "en-gb", "en-us", "en-au", "en-ca", ""])


def _normalize_weights(accumulated: dict[str, float]) -> dict[str, float]:
    """Normalize accumulated keyword weights so values sum to 1.0."""
    total = sum(accumulated.values())
    if total == 0.0:
        return {}
    return {k: v / total for k, v in accumulated.items() if v > 0}


@dataclasses.dataclass
class BioClassification:
    """Result of 3-stage demographic inference for a single post author.

    Empty weight dicts mean "no signal" for that stratum. Posts with
    inference_method=None have no signal in any stratum and must be
    excluded from estimation entirely.

    CLAUDE.md rule: inference_method == "language_prior" → exclude from
    mean μ and Σ_Δ; use as held-out validation only.
    """

    inference_method: str | None
    race_weights: dict[str, float]
    religion_weights: dict[str, float]
    gender_weights: dict[str, float]

    def has_signal(self) -> bool:
        """True if any stratum has non-empty weights."""
        return bool(self.race_weights or self.religion_weights or self.gender_weights)

    def is_estimable(self) -> bool:
        """True if this post should contribute to μ/Σ_Δ estimation."""
        return self.has_signal() and self.inference_method != "language_prior"


class BioClassifier:
    """3-stage demographic inference from author bio strings.

    Thread-safe for concurrent reads. Pi server calls use a configurable
    timeout so a slow or unreachable Pi never stalls the pipeline.
    """

    def __init__(
        self,
        race_lexicon: dict[str, dict[str, float]],
        religion_lexicon: dict[str, dict[str, float]],
        gender_lexicon: dict[str, dict[str, float]],
        language_priors: dict[str, Any],
        pi_server_url: str | None = None,
        pi_timeout_secs: float = 2.0,
    ) -> None:
        # Store lexicons with lowercase keys for case-insensitive matching
        self._race_lex: dict[str, dict[str, float]] = {
            k.lower(): v for k, v in race_lexicon.items()
        }
        self._rel_lex: dict[str, dict[str, float]] = {
            k.lower(): v for k, v in religion_lexicon.items()
        }
        self._gen_lex: dict[str, dict[str, float]] = {
            k.lower(): v for k, v in gender_lexicon.items()
        }
        self._lang_priors: dict[str, Any] = language_priors.get("priors", {})
        self._pi_url = pi_server_url.rstrip("/") if pi_server_url else None
        self._pi_timeout = pi_timeout_secs
        # Circuit breaker: set False after first Pi failure to skip subsequent calls.
        self._pi_available: bool = self._pi_url is not None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        pi_server_url: str | None = None,
    ) -> "BioClassifier":
        """Load lexicons from standard config paths and optionally base.json."""
        race_lex = json.loads(_RACE_LEXICON_PATH.read_text(encoding="utf-8"))["keywords"]
        rel_lex = json.loads(_RELIGION_LEXICON_PATH.read_text(encoding="utf-8"))["keywords"]
        gen_lex = json.loads(_GENDER_LEXICON_PATH.read_text(encoding="utf-8"))["keywords"]
        lang_priors = json.loads(_LANGUAGE_PRIORS_PATH.read_text(encoding="utf-8"))

        if pi_server_url is None:
            # Resolution order: PI_BIO_SERVER_URL > PI_TAILSCALE_IP > base.json
            pi_server_url = os.environ.get("PI_BIO_SERVER_URL") or (
                f"http://{os.environ['PI_TAILSCALE_IP']}:9000"
                if os.environ.get("PI_TAILSCALE_IP")
                else None
            )
        if pi_server_url is None and config_path is not None:
            try:
                cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
                pi_server_url = cfg.get("pi_bio_server")
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read pi_bio_server from %s: %s", config_path, exc)

        return cls(
            race_lexicon=race_lex,
            religion_lexicon=rel_lex,
            gender_lexicon=gen_lex,
            language_priors=lang_priors,
            pi_server_url=pi_server_url,
        )

    def classify(
        self,
        bio: str | None,
        lang: str = "",
        inference_method: str | None = None,
    ) -> BioClassification:
        """Run the 3-stage inference pipeline on a single author bio.

        Args:
            bio: Author bio/description text (may be empty or None).
            lang: BCP-47 language code of the POST (not the bio).
            inference_method: If already set upstream ("platform_proxy",
                              "subreddit_proxy"), returned unchanged.

        Returns:
            BioClassification with inference_method and stratum weight dicts.
        """
        if inference_method in _UPSTREAM_METHODS:
            return BioClassification(
                inference_method=inference_method,
                race_weights={},
                religion_weights={},
                gender_weights={},
            )

        bio = (bio or "").strip()

        # Stage 1: keyword lexicon
        result = self._stage1_keyword(bio)
        if result is not None:
            return result

        # Stage 2: language prior (non-English posts only)
        lang_prefix = (lang[:2] if lang else "").lower()
        prior_key = _LANG_TO_PRIOR.get(lang_prefix)
        if prior_key and lang.lower() not in _ENGLISH_LANGS:
            result = self._stage2_language_prior(prior_key)
            if result is not None:
                return result

        # Stage 3: SetFit via Pi (English posts with non-empty bio)
        if bio and self._pi_available:
            result = self._stage3_setfit(bio)
            if result is not None:
                return result

        return BioClassification(
            inference_method=None,
            race_weights={},
            religion_weights={},
            gender_weights={},
        )

    def _stage1_keyword(self, bio: str) -> BioClassification | None:
        """Scan bio against all three lexicons. Returns None if no keywords match."""
        if not bio:
            return None

        bio_lower = bio.lower()
        race_acc: dict[str, float] = {}
        rel_acc: dict[str, float] = {}
        gen_acc: dict[str, float] = {}

        for kw, weights in self._race_lex.items():
            if kw in bio_lower:
                for bloc, w in weights.items():
                    race_acc[bloc] = race_acc.get(bloc, 0.0) + w

        for kw, weights in self._rel_lex.items():
            if kw in bio_lower:
                for bloc, w in weights.items():
                    rel_acc[bloc] = rel_acc.get(bloc, 0.0) + w

        for kw, weights in self._gen_lex.items():
            if kw in bio_lower:
                for bloc, w in weights.items():
                    gen_acc[bloc] = gen_acc.get(bloc, 0.0) + w

        if not race_acc and not rel_acc and not gen_acc:
            return None

        return BioClassification(
            inference_method="keyword_bio",
            race_weights=_normalize_weights(race_acc),
            religion_weights=_normalize_weights(rel_acc),
            gender_weights=_normalize_weights(gen_acc),
        )

    def _stage2_language_prior(self, lang_key: str) -> BioClassification | None:
        """Apply language-based demographic prior from language_priors.json."""
        prior = self._lang_priors.get(lang_key)
        if prior is None:
            return None
        race_weights = {k: float(v) for k, v in prior.get("race", {}).items()}
        religion_weights = {k: float(v) for k, v in prior.get("religion", {}).items()}
        gender_weights = {k: float(v) for k, v in prior.get("gender", {}).items()}
        if not race_weights and not religion_weights and not gender_weights:
            return None
        return BioClassification(
            inference_method="language_prior",
            race_weights=race_weights,
            religion_weights=religion_weights,
            gender_weights=gender_weights,
        )

    def _stage3_setfit(self, bio: str) -> BioClassification | None:
        """POST bio to Pi server; returns None on timeout or if bloc is null.

        Pi server currently returns {"bloc": null, "embedding": [...]} until
        SetFit centroids are trained (Week 4 Day 3). When bloc is non-null,
        expect format "stratum:bloc_id" e.g. "religion:evangelical".
        """
        if not self._pi_url:
            return None

        url = f"{self._pi_url}/classify"
        payload = json.dumps({"bio": bio}).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._pi_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("Pi bio server unavailable (%s); disabling SetFit for this run.", exc)
            self._pi_available = False
            return None

        bloc = data.get("bloc")
        if bloc is None:
            return None  # SetFit not yet trained — normal until Week 4 Day 3

        bloc_str = str(bloc)
        if ":" not in bloc_str:
            logger.warning("Pi server bloc format unexpected (no colon): %r", bloc)
            return None

        stratum, bloc_id = bloc_str.split(":", 1)
        race_w: dict[str, float] = {}
        rel_w: dict[str, float] = {}
        gen_w: dict[str, float] = {}

        if stratum == "race" and bloc_id in CANONICAL_RACES:
            race_w = {bloc_id: 1.0}
        elif stratum == "religion" and bloc_id in CANONICAL_RELIGIONS:
            rel_w = {bloc_id: 1.0}
        elif stratum == "gender" and bloc_id in CANONICAL_GENDERS:
            gen_w = {bloc_id: 1.0}
        else:
            logger.warning(
                "Pi server returned unknown bloc %r (stratum=%r, id=%r); skipping.",
                bloc,
                stratum,
                bloc_id,
            )
            return None

        return BioClassification(
            inference_method="setfit_bio",
            race_weights=race_w,
            religion_weights=rel_w,
            gender_weights=gen_w,
        )

    def classify_batch(
        self,
        posts: list[dict],
    ) -> list[BioClassification]:
        """Classify a list of canonical post dicts in one pass.

        Reads author_description, lang, and inference_method from each post.
        """
        results = []
        for post in posts:
            results.append(
                self.classify(
                    bio=post.get("author_description"),
                    lang=post.get("lang", ""),
                    inference_method=post.get("inference_method"),
                )
            )
        return results
