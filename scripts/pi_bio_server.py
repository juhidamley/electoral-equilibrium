"""FastAPI bio classifier server — runs on Raspberry Pi 5 (+ optional Hailo-8L NPU).

Endpoints
---------
POST /classify          Single bio classification.
POST /classify_batch    Batch classification; accepts up to 50 items per call.
GET  /health            Liveness probe; reports inference backend in use.

Three-stage inference pipeline (per stratum, per request)
----------------------------------------------------------
Stage 1 — SentenceTransformer embedding + exemplar cosine similarity.
    CPU path:  sentence-transformers/all-MiniLM-L6-v2 on Pi CPU.
    NPU path:  HEF-compiled model on Hailo-8L (sub-millisecond per bio).
    If max per-stratum confidence >= SETFIT_THRESHOLD → use; inference_method="setfit".

Stage 2 — Keyword lexicon fallback.
    Scans lowercased bio text against configs/{race,religion,gender}_lexicon.json.
    Blends weights when multiple keywords match (mean of matched entries).
    If any keyword matched → use; inference_method="keyword_lexicon".

Stage 3 — Language prior fallback.
    Maps BCP-47 lang code to bloc priors from configs/language_priors.json.
    Always produces a result; inference_method="language_prior".
    IMPORTANT: language_prior results are excluded from mean/covariance estimation
    downstream. They are held-out validation only. See CLAUDE.md §Language fallback.

Usage
-----
    # On Pi — install dependencies first:
    #   pip install fastapi uvicorn sentence-transformers numpy
    python scripts/pi_bio_server.py --config configs/base.json --port 9000

    # With NPU enabled (pi_npu_enabled must be true in config):
    python scripts/pi_bio_server.py --config configs/base.json --port 9000 --npu

    # CPU-only smoke test (runs on any machine):
    python scripts/pi_bio_server.py --config configs/base.json --port 9000 --cpu-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
log = logging.getLogger("pi_bio_server")

# ── Constants ─────────────────────────────────────────────────────────────────

SETFIT_THRESHOLD = 0.60         # minimum per-stratum max-confidence to accept Stage 1
BATCH_SIZE_LIMIT = 50           # enforced by /classify_batch
MODEL_NAME = "all-MiniLM-L6-v2"
HEF_DEFAULT_PATH = Path(__file__).parent.parent / "adapters" / "all-minilm-l6-v2.hef"

# Canonical canonical stratum fallback distributions (uniform) used when
# all three stages produce no signal (e.g. empty bio, unknown language).
_CANONICAL_RACES = ["african_american", "latino", "asian", "white", "other_race"]
_CANONICAL_RELIGIONS = [
    "evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel",
]
_CANONICAL_GENDERS = ["women", "men", "other_gender"]

# Representative exemplar sentences for cosine-similarity classification.
# Ordered to match the canonical bloc lists above.
_RACE_EXEMPLARS = [
    "Black African American community activist civil rights",
    "Latino Hispanic Latinx Chicano immigrant family",
    "Asian Chinese Korean Japanese Filipino Vietnamese American",
    "White American rural suburban Christian conservative",
    "Multiracial biracial mixed Indigenous Native American other",
]
_RELIGION_EXEMPLARS = [
    "evangelical born again Christian conservative biblical faith prayer",
    "Catholic mass Vatican rosary Notre Dame parish priest",
    "Methodist Presbyterian Lutheran Episcopal mainline Protestant church",
    "atheist agnostic secular humanist no religion science",
    "Jewish synagogue Torah Shabbat rabbi kosher Israel",
    "Muslim Islam mosque Quran prayer Ramadan halal",
    "Buddhist Hindu Sikh Unitarian spiritual new age other faith",
]
_GENDER_EXEMPLARS = [
    "she her woman mother feminist women's rights",
    "he him man father husband men's",
    "they them nonbinary transgender enby queer gender nonconforming",
]


# ── Config loading ────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_lexicons(repo_root: Path) -> tuple[dict, dict, dict]:
    """Load the three keyword lexicons from configs/. Keys are already lowercase."""
    race = _load_json(repo_root / "configs" / "race_lexicon.json")["keywords"]
    religion = _load_json(repo_root / "configs" / "religion_lexicon.json")["keywords"]
    gender = _load_json(repo_root / "configs" / "gender_lexicon.json")["keywords"]
    return race, religion, gender


def _load_language_priors(repo_root: Path) -> dict[str, dict]:
    """Return {lang_prefix: {stratum: {bloc: weight}}} from language_priors.json."""
    raw = _load_json(repo_root / "configs" / "language_priors.json")
    return raw.get("priors", {})


# ── Inference backend — CPU path ──────────────────────────────────────────────

class _CPUBackend:
    """SentenceTransformer on Pi CPU (or any machine for testing)."""

    def __init__(self) -> None:
        log.info("Loading SentenceTransformer(%s) on CPU...", MODEL_NAME)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        log.info("CPU backend ready.")

        # Pre-compute exemplar embeddings once at startup.
        self._race_emb = self._embed(_RACE_EXEMPLARS)
        self._religion_emb = self._embed(_RELIGION_EXEMPLARS)
        self._gender_emb = self._embed(_GENDER_EXEMPLARS)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (n, dim)."""
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def embed_bio(self, bio: str) -> np.ndarray:
        """Embed a single bio string; return L2-normalised row vector (dim,)."""
        vec = self.model.encode([bio], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].astype(np.float32)

    @property
    def exemplar_embeddings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._race_emb, self._religion_emb, self._gender_emb

    @property
    def backend_name(self) -> str:
        return "cpu"


# ── Inference backend — NPU path ──────────────────────────────────────────────

class _NPUBackend:
    """Hailo-8L NPU backend via Hailo SDK.

    The HEF file is compiled from all-MiniLM-L6-v2 ONNX by scripts/compile_hailo.py.
    The NPU handles embedding inference only; cosine similarity runs on CPU.
    """

    def __init__(self, hef_path: Path) -> None:
        log.info("Initialising Hailo NPU backend — HEF: %s", hef_path)
        if not hef_path.exists():
            raise FileNotFoundError(
                f"HEF model not found at {hef_path}. "
                "Run scripts/compile_hailo.py first to compile the model for the NPU."
            )

        # Hailo SDK import is deferred so the module can be imported on non-Pi
        # machines for testing with pi_npu_enabled=false without import errors.
        try:
            from hailo_platform import (  # type: ignore[import]
                HEF,
                VDevice,
                HailoStreamInterface,
                InferVStreams,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform is not installed. "
                "Install the Hailo SDK on the Pi before enabling NPU mode."
            ) from exc

        self._hef_path = hef_path

        # Load the HEF and configure the VDevice (Hailo runtime).
        target = VDevice()
        hef = HEF(str(hef_path))
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = target.configure(hef, configure_params)
        self._network_group = network_groups[0]

        input_vstreams_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        self._infer_pipeline = InferVStreams(
            self._network_group, input_vstreams_params, output_vstreams_params
        )

        # Infer input/output names from the HEF metadata.
        self._input_name = hef.get_input_vstream_infos()[0].name
        self._output_name = hef.get_output_vstream_infos()[0].name

        log.info("NPU backend ready — input: %s, output: %s",
                 self._input_name, self._output_name)

        # Pre-compute exemplar embeddings via NPU.
        self._race_emb = self._embed_batch(_RACE_EXEMPLARS)
        self._religion_emb = self._embed_batch(_RELIGION_EXEMPLARS)
        self._gender_emb = self._embed_batch(_GENDER_EXEMPLARS)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Tokenise and embed texts through the NPU; return L2-normalised (n, dim)."""
        # Tokenisation still runs on CPU (lightweight); only the embedding forward
        # pass is offloaded to the Hailo-8L.
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        tokenizer_model = SentenceTransformer(MODEL_NAME, device="cpu")
        features = tokenizer_model.tokenize(texts)
        input_ids = features["input_ids"].numpy().astype(np.float32)

        with self._network_group.activate():
            input_data = {self._input_name: input_ids}
            output_data = self._infer_pipeline.infer(input_data)

        vecs = output_data[self._output_name]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
        return (vecs / norms).astype(np.float32)

    def embed_bio(self, bio: str) -> np.ndarray:
        """Embed one bio string via NPU; return L2-normalised (dim,)."""
        return self._embed_batch([bio])[0]

    @property
    def exemplar_embeddings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._race_emb, self._religion_emb, self._gender_emb

    @property
    def backend_name(self) -> str:
        return "npu"


# ── Classifier ────────────────────────────────────────────────────────────────

def _cosine_softmax(bio_vec: np.ndarray, exemplar_matrix: np.ndarray) -> np.ndarray:
    """Return softmax of cosine similarities between bio_vec and each exemplar row."""
    sims = exemplar_matrix @ bio_vec          # (n,) — bio_vec and rows already L2-normed
    sims = np.clip(sims, -1.0, 1.0)
    # Temperature-scaled softmax (τ=0.1) sharpens the distribution.
    exp_sims = np.exp(sims / 0.1)
    return exp_sims / exp_sims.sum()


def _apply_lexicon(bio_lower: str, lexicon: dict[str, dict[str, float]],
                   canonical: list[str]) -> tuple[dict[str, float], bool]:
    """Scan bio_lower for lexicon keywords; return blended weights and hit flag."""
    matched: list[dict[str, float]] = []
    # Longest-keyword-first scan reduces false substring matches.
    for kw in sorted(lexicon, key=len, reverse=True):
        if kw in bio_lower:
            matched.append(lexicon[kw])

    if not matched:
        return {b: 1.0 / len(canonical) for b in canonical}, False

    # Blend: average over matched keyword weight dicts.
    blended: dict[str, float] = {b: 0.0 for b in canonical}
    for m in matched:
        for bloc, w in m.items():
            if bloc in blended:
                blended[bloc] += w
    total = sum(blended.values()) or 1.0
    return {b: blended[b] / total for b in canonical}, True


def _apply_language_prior(
    lang: str,
    priors: dict[str, dict],
    canonical_race: list[str],
    canonical_religion: list[str],
    canonical_gender: list[str],
) -> dict[str, dict[str, float]]:
    """Return per-stratum uniform-prior dicts, with race overridden if lang matches."""
    uniform_race = {b: 1.0 / len(canonical_race) for b in canonical_race}
    uniform_rel = {b: 1.0 / len(canonical_religion) for b in canonical_religion}
    uniform_gen = {b: 1.0 / len(canonical_gender) for b in canonical_gender}

    lang_prefix = lang.lower().split("-")[0] if lang else ""
    if lang_prefix in priors:
        race_prior = priors[lang_prefix].get("race", {})
        # Fill missing canonical blocs with zero; re-normalise.
        race_weights = {b: float(race_prior.get(b, 0.0)) for b in canonical_race}
        total = sum(race_weights.values()) or 1.0
        uniform_race = {b: race_weights[b] / total for b in canonical_race}

    return {"race": uniform_race, "religion": uniform_rel, "gender": uniform_gen}


class BioClassifier:
    """Three-stage bio classifier backed by either CPU or NPU inference."""

    def __init__(
        self,
        backend: _CPUBackend | _NPUBackend,
        race_lexicon: dict,
        religion_lexicon: dict,
        gender_lexicon: dict,
        language_priors: dict,
        canonical_races: list[str],
        canonical_religions: list[str],
        canonical_genders: list[str],
    ) -> None:
        self._backend = backend
        self._race_lex = race_lexicon
        self._rel_lex = religion_lexicon
        self._gen_lex = gender_lexicon
        self._lang_priors = language_priors
        self._races = canonical_races
        self._religions = canonical_religions
        self._genders = canonical_genders
        self._executor = ThreadPoolExecutor(max_workers=2)

    def classify_one(self, bio: str, lang: str) -> dict[str, Any]:
        """Run three-stage inference on a single bio string.

        Returns a dict with keys:
            race, religion, gender  — {bloc_id: weight} dicts (each sums to 1.0)
            inference_method        — "setfit" | "keyword_lexicon" | "language_prior"
        """
        bio_lower = bio.lower().strip() if bio else ""

        # ── Stage 1: SentenceTransformer / NPU ───────────────────────────────
        if bio_lower:
            bio_vec = self._backend.embed_bio(bio_lower)
            race_emb, rel_emb, gen_emb = self._backend.exemplar_embeddings

            race_probs = _cosine_softmax(bio_vec, race_emb)
            rel_probs = _cosine_softmax(bio_vec, rel_emb)
            gen_probs = _cosine_softmax(bio_vec, gen_emb)

            # Accept Stage 1 only if ALL three strata exceed the confidence threshold.
            if (race_probs.max() >= SETFIT_THRESHOLD
                    and rel_probs.max() >= SETFIT_THRESHOLD
                    and gen_probs.max() >= SETFIT_THRESHOLD):
                return {
                    "race": dict(zip(self._races, race_probs.tolist())),
                    "religion": dict(zip(self._religions, rel_probs.tolist())),
                    "gender": dict(zip(self._genders, gen_probs.tolist())),
                    "inference_method": "setfit",
                }

        # ── Stage 2: keyword lexicon ──────────────────────────────────────────
        if bio_lower:
            race_w, race_hit = _apply_lexicon(bio_lower, self._race_lex, self._races)
            rel_w, rel_hit = _apply_lexicon(bio_lower, self._rel_lex, self._religions)
            gen_w, gen_hit = _apply_lexicon(bio_lower, self._gen_lex, self._genders)

            if race_hit or rel_hit or gen_hit:
                return {
                    "race": race_w,
                    "religion": rel_w,
                    "gender": gen_w,
                    "inference_method": "keyword_lexicon",
                }

        # ── Stage 3: language prior ───────────────────────────────────────────
        strata = _apply_language_prior(
            lang, self._lang_priors, self._races, self._religions, self._genders
        )
        return {**strata, "inference_method": "language_prior"}

    def classify_batch(
        self, items: list[dict[str, str]]
    ) -> dict[str, dict[str, Any]]:
        """Classify a list of {did, bio, lang} dicts; return {did: result}."""
        results: dict[str, dict[str, Any]] = {}
        for item in items:
            did = item.get("did") or item.get("id") or str(id(item))
            bio = item.get("bio") or item.get("author_description") or ""
            lang = item.get("lang") or ""
            results[did] = self.classify_one(bio, lang)
        return results


# ── FastAPI application ───────────────────────────────────────────────────────

def _build_app(classifier: BioClassifier, backend_name: str):
    """Build and return the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore[import]
        from pydantic import BaseModel              # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "fastapi and pydantic are not installed. "
            "Run: pip install fastapi uvicorn pydantic"
        ) from exc

    app = FastAPI(
        title="Electoral Equilibrium — Bio Classifier",
        description="Three-stage demographic inference from social media bios.",
        version="1.0.0",
    )

    class ClassifyRequest(BaseModel):
        bio: str = ""
        lang: str = ""

    class BatchItem(BaseModel):
        did: str
        bio: str = ""
        lang: str = ""

    class BatchRequest(BaseModel):
        items: list[BatchItem]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "inference_backend": backend_name,
            "setfit_threshold": SETFIT_THRESHOLD,
            "batch_size_limit": BATCH_SIZE_LIMIT,
        }

    @app.post("/classify")
    def classify(req: ClassifyRequest):
        try:
            return classifier.classify_one(req.bio, req.lang)
        except Exception as exc:
            log.exception("classify error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/classify_batch")
    def classify_batch(req: BatchRequest):
        if len(req.items) > BATCH_SIZE_LIMIT:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(req.items)} exceeds limit {BATCH_SIZE_LIMIT}. "
                       f"Split into smaller batches.",
            )
        try:
            raw = [item.model_dump() for item in req.items]
            return classifier.classify_batch(raw)
        except Exception as exc:
            log.exception("classify_batch error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Electoral Equilibrium bio classifier server (Pi + optional Hailo NPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/base.json",
        help="Path to configs/base.json (or domain equivalent).",
    )
    parser.add_argument(
        "--port", type=int, default=9000,
        help="TCP port to bind. Must match pi_bio_server in configs/base.json.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind host. Use 0.0.0.0 on Pi; 127.0.0.1 for local smoke tests.",
    )
    parser.add_argument(
        "--hef",
        default=str(HEF_DEFAULT_PATH),
        help="Path to compiled HEF model (NPU mode only).",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU backend regardless of pi_npu_enabled in config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).parent.parent.resolve()

    # ── Load base config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = _load_json(config_path)
    pi_npu_enabled: bool = config.get("pi_npu_enabled", False)

    log.info("pi_npu_enabled=%s  cpu_only_flag=%s", pi_npu_enabled, args.cpu_only)

    # ── Initialise inference backend — branching is explicit and logged ────────
    if args.cpu_only or not pi_npu_enabled:
        # CPU-only path: SentenceTransformer on Pi CPU.
        # This path is always safe to use regardless of hardware.
        log.info("Selecting CPU backend (pi_npu_enabled=%s, --cpu-only=%s)",
                 pi_npu_enabled, args.cpu_only)
        backend: _CPUBackend | _NPUBackend = _CPUBackend()
    else:
        # NPU path: Hailo SDK + HEF model.
        # Only reached when pi_npu_enabled=true in configs/base.json AND
        # --cpu-only is not passed. The Hailo SDK is NEVER imported in the CPU path.
        log.info("Selecting NPU backend — HEF: %s", args.hef)
        backend = _NPUBackend(hef_path=Path(args.hef))

    # ── Load lexicons and language priors ─────────────────────────────────────
    race_lex, rel_lex, gen_lex = _load_lexicons(repo_root)
    lang_priors = _load_language_priors(repo_root)
    log.info(
        "Lexicons loaded — race: %d keywords, religion: %d, gender: %d",
        len(race_lex), len(rel_lex), len(gen_lex),
    )

    # ── Load canonical bloc lists from config ─────────────────────────────────
    canonical_races = config.get("races", _CANONICAL_RACES)
    canonical_religions = config.get("religions", _CANONICAL_RELIGIONS)
    canonical_genders = config.get("genders", _CANONICAL_GENDERS)

    # ── Build classifier and server ───────────────────────────────────────────
    classifier = BioClassifier(
        backend=backend,
        race_lexicon=race_lex,
        religion_lexicon=rel_lex,
        gender_lexicon=gen_lex,
        language_priors=lang_priors,
        canonical_races=canonical_races,
        canonical_religions=canonical_religions,
        canonical_genders=canonical_genders,
    )

    app = _build_app(classifier, backend_name=backend.backend_name)

    log.info("Starting server on %s:%d (backend=%s)", args.host, args.port, backend.backend_name)

    try:
        import uvicorn  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is not installed. Run: pip install uvicorn"
        ) from exc

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
