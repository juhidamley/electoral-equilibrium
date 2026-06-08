"""Bio classifier server — Raspberry Pi 5.

Endpoints
---------
GET  /health     Liveness probe.
                 Returns {"status": "ok", "mode": "cpu"|"npu", "model": str}
POST /classify   Classify an author bio.
                 Returns {"bloc": "stratum:bloc_id"|null, "embedding": list[float]}

Startup behaviour (reads configs/base.json["pi_npu_enabled"]):
  false → load all-MiniLM-L6-v2 via SentenceTransformer on CPU.
           Hailo SDK is NOT imported. Passing data to the Hailo interface
           when NPU is disabled causes an immediate crash — this path avoids
           it entirely.
  true  → load the embedding backbone from the compiled HEF via the Hailo SDK.
           The SetFit classification head always runs on CPU.

Trained models must exist at:
  models/setfit_race/
  models/setfit_religion/
  models/setfit_gender/

Run scripts/train_setfit.py to produce them.
The server fails at startup if models are missing.

Usage
-----
    # CPU (default; works on any machine without Hailo SDK):
    python scripts/pi_bio_server.py --cpu-only

    # Reads pi_npu_enabled from configs/base.json:
    python scripts/pi_bio_server.py

    # Health check:
    curl http://localhost:9000/health
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
log = logging.getLogger("pi_bio_server")

REPO_ROOT   = Path(__file__).resolve().parents[1]
MODELS_DIR  = REPO_ROOT / "models"
MODEL_NAME  = "all-MiniLM-L6-v2"
HEF_PATH    = MODELS_DIR / "setfit_embedding.hef"

# Server returns null if top-class probability is below this.
_MIN_CONF = 0.50

# Stratum → (model_dir, SetFit label field, server label prefix)
_STRATA = {
    "race":     ("setfit_race",     "race"),
    "religion": ("setfit_religion", "religion"),
    "gender":   ("setfit_gender",   "gender"),
}

# Gender internal labels ("F"/"M") → canonical bloc IDs
_GENDER_TO_BLOC = {"F": "women", "M": "men"}


# ── NPU backend ───────────────────────────────────────────────────────────────


class _HailoRunner:
    """Wraps the Hailo VDevice + InferVStreams for embedding inference.

    Only instantiated when pi_npu_enabled=true. This class imports
    hailo_platform inside __init__ so it is never imported on CPU-only paths
    — importing hailo_platform when the NPU hardware is absent causes a crash.

    The HEF is compiled from all-MiniLM-L6-v2 ONNX via Hailo Dataflow Compiler
    (run compile_hailo.py separately). Input: mean-pooled token embeddings after
    tokenisation (CPU). Output: 384-dim L2-normalised embedding vector.
    """

    def __init__(self, hef_path: Path) -> None:
        try:
            from hailo_platform import (  # type: ignore[import]
                HEF,
                VDevice,
                HailoStreamInterface,
                InferVStreams,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform is not installed. "
                "Install the Hailo Python SDK matching your HailoRT version."
            ) from exc

        if not hef_path.exists():
            raise FileNotFoundError(
                f"HEF not found: {hef_path}\n"
                "Compile with:  python scripts/compile_hailo.py"
            )

        log.info("Loading Hailo HEF from %s ...", hef_path)
        self._hef = HEF(str(hef_path))
        self._target = VDevice()
        self._configured = self._target.configure(self._hef)
        self._ng = self._configured[0]
        self._ng_params = self._ng.create_params()

        # Resolve input/output names from the compiled graph
        input_vstreams_params  = InputVStreamParams.make(self._ng)
        output_vstreams_params = OutputVStreamParams.make(self._ng)
        self._InferVStreams       = InferVStreams
        self._input_vstreams_p  = input_vstreams_params
        self._output_vstreams_p = output_vstreams_params

        # Determine the single input and output layer names
        input_names  = list(input_vstreams_params.keys())
        output_names = list(output_vstreams_params.keys())
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Expected HEF with 1 input and 1 output; got "
                f"inputs={input_names}, outputs={output_names}"
            )
        self._input_name  = input_names[0]
        self._output_name = output_names[0]
        log.info(
            "Hailo ready — input=%s output=%s", self._input_name, self._output_name
        )

    def encode(self, tokenizer: Any, text: str) -> np.ndarray:
        """Tokenize on CPU, embed on NPU. Returns (384,) float32 numpy array."""
        tokens = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        input_data = {
            self._input_name: tokens["input_ids"].astype(np.float32)
        }
        with self._InferVStreams(
            self._ng,
            self._input_vstreams_p,
            self._output_vstreams_p,
        ) as infer_pipeline:
            infer_results = infer_pipeline.infer(input_data)
        embedding = infer_results[self._output_name][0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_label_config(model_dir: Path) -> dict[int, str]:
    """Read label_config.json written by train_setfit.py → {int_id: label_str}."""
    cfg_path = model_dir / "label_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"label_config.json missing in {model_dir}. "
            "Re-run train_setfit.py to regenerate."
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in cfg["id2label"].items()}


def _load_cpu_state(models_dir: Path) -> dict:
    """Load SentenceTransformer backbone + three SetFit models for CPU mode.

    Returns a state dict with:
        backbone   : SentenceTransformer (unused in CPU mode — full models encode)
        models     : {stratum: {"sf_model": SetFitModel, "id2label": dict}}
        mode       : "cpu"
    """
    try:
        from setfit import SetFitModel  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "setfit is not installed. Run:  pip install 'setfit>=1.0'"
        ) from exc

    state: dict[str, Any] = {"mode": "cpu", "models": {}}

    for stratum, (dir_name, _) in _STRATA.items():
        model_dir = models_dir / dir_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"SetFit model not found: {model_dir}\n"
                "Run:  python scripts/train_setfit.py"
            )
        log.info("Loading CPU model for stratum '%s' from %s ...", stratum, model_dir)
        sf_model = SetFitModel.from_pretrained(str(model_dir))
        id2label = _load_label_config(model_dir)
        state["models"][stratum] = {"sf_model": sf_model, "id2label": id2label}
        log.info("  Stratum '%s' loaded (%d classes).", stratum, len(id2label))

    return state


def _load_npu_state(models_dir: Path) -> dict:
    """Load HEF backbone (NPU) + sklearn heads from SetFit models (CPU).

    Returns a state dict with:
        hailo_runner : _HailoRunner
        tokenizer    : HuggingFace tokenizer for all-MiniLM-L6-v2
        models       : {stratum: {"head": sklearn_estimator, "id2label": dict}}
        mode         : "npu"
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import]
        from setfit import SetFitModel          # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "transformers and setfit must be installed for NPU mode."
        ) from exc

    hailo_runner = _HailoRunner(HEF_PATH)

    log.info("Loading tokenizer %s for NPU mode ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    state: dict[str, Any] = {
        "mode": "npu",
        "hailo_runner": hailo_runner,
        "tokenizer": tokenizer,
        "models": {},
    }

    for stratum, (dir_name, _) in _STRATA.items():
        model_dir = models_dir / dir_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"SetFit model not found: {model_dir}\n"
                "Run:  python scripts/train_setfit.py"
            )
        log.info("Loading NPU head for stratum '%s' from %s ...", stratum, model_dir)
        sf_model = SetFitModel.from_pretrained(str(model_dir))
        head = sf_model.model_head
        id2label = _load_label_config(model_dir)
        state["models"][stratum] = {"head": head, "id2label": id2label}
        # Release the backbone immediately — NPU provides embeddings.
        del sf_model
        log.info("  Stratum '%s' head loaded (%d classes).", stratum, len(id2label))

    return state


# ── Inference ─────────────────────────────────────────────────────────────────


def _classify_cpu(bio: str, state: dict) -> tuple[str | None, list[float]]:
    """CPU-mode classification. Returns (bloc_str | None, embedding_list)."""
    models = state["models"]

    # Encode once using the race model's backbone (same for all strata)
    sf_race = models["race"]["sf_model"]
    embedding: np.ndarray = sf_race.model_body.encode(
        [bio], convert_to_numpy=True, normalize_embeddings=True,
    )[0]

    best_bloc: str | None = None
    best_conf: float = _MIN_CONF  # only update if confidence exceeds threshold

    for stratum, (_, prefix) in _STRATA.items():
        sf_model = models[stratum]["sf_model"]
        id2label = models[stratum]["id2label"]

        # predict_proba via the full model (handles internal normalization)
        probs = sf_model.predict_proba([bio])
        if isinstance(probs, np.ndarray):
            if probs.ndim == 2:
                probs_1d = probs[0]
            else:
                probs_1d = probs
        else:
            try:
                probs_1d = np.array(probs).flatten()
            except Exception:
                continue

        pred_idx = int(np.argmax(probs_1d))
        conf = float(probs_1d[pred_idx])

        if conf > best_conf and pred_idx in id2label:
            raw_label = id2label[pred_idx]
            # Gender uses internal "F"/"M" labels — map to canonical bloc IDs
            if stratum == "gender":
                canonical = _GENDER_TO_BLOC.get(raw_label)
                if canonical is None:
                    continue
            else:
                canonical = raw_label
            best_bloc = f"{prefix}:{canonical}"
            best_conf = conf

    return best_bloc, embedding.tolist()


def _classify_npu(bio: str, state: dict) -> tuple[str | None, list[float]]:
    """NPU-mode classification. Returns (bloc_str | None, embedding_list)."""
    hailo_runner = state["hailo_runner"]
    tokenizer    = state["tokenizer"]
    models       = state["models"]

    embedding = hailo_runner.encode(tokenizer, bio)

    best_bloc: str | None = None
    best_conf: float = _MIN_CONF

    for stratum, (_, prefix) in _STRATA.items():
        head     = models[stratum]["head"]
        id2label = models[stratum]["id2label"]

        try:
            probs_2d = head.predict_proba(embedding.reshape(1, -1))
            probs_1d = probs_2d[0]
        except Exception as exc:
            log.debug("Head prediction failed for stratum '%s': %s", stratum, exc)
            continue

        pred_idx = int(np.argmax(probs_1d))
        conf = float(probs_1d[pred_idx])

        if conf > best_conf and pred_idx in id2label:
            raw_label = id2label[pred_idx]
            if stratum == "gender":
                canonical = _GENDER_TO_BLOC.get(raw_label)
                if canonical is None:
                    continue
            else:
                canonical = raw_label
            best_bloc = f"{prefix}:{canonical}"
            best_conf = conf

    return best_bloc, embedding.tolist()


# ── FastAPI app ───────────────────────────────────────────────────────────────


def _build_app(state: dict):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "fastapi and pydantic are not installed.\n"
            "Run:  pip install fastapi uvicorn pydantic"
        ) from exc

    mode: str = state["mode"]
    classify_fn = _classify_npu if mode == "npu" else _classify_cpu

    app = FastAPI(
        title="Electoral Equilibrium — Bio Classifier",
        version="2.0.0",
    )

    class ClassifyRequest(BaseModel):
        bio: str = ""

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode":   mode,
            "model":  MODEL_NAME,
        }

    @app.post("/classify")
    def classify(req: ClassifyRequest):
        bio = req.bio.strip()
        if not bio:
            return {"bloc": None, "embedding": []}
        try:
            bloc, embedding = classify_fn(bio, state)
        except Exception as exc:
            log.exception("Classify error: %s", exc)
            return {"bloc": None, "embedding": []}
        return {"bloc": bloc, "embedding": embedding}

    return app


# ── Config helpers ────────────────────────────────────────────────────────────


def _load_config(path: Path) -> dict:
    if not path.exists():
        log.warning("Config not found at %s — using empty config.", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _set_npu_disabled(config_path: Path) -> None:
    config = _load_config(config_path)
    config["pi_npu_enabled"] = False
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    log.info("Set pi_npu_enabled=false in %s.", config_path)


# ── Entry point ───────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bio classifier server for the Raspberry Pi 5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/base.json",
        help="Path to configs/base.json.",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU mode and write pi_npu_enabled=false to config.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if args.cpu_only:
        _set_npu_disabled(config_path)
        npu_enabled = False
        log.info("--cpu-only: CPU mode selected.")
    else:
        cfg = _load_config(config_path)
        npu_enabled = bool(cfg.get("pi_npu_enabled", False))

    log.info("Mode: %s", "npu" if npu_enabled else "cpu")

    if npu_enabled:
        # IMPORTANT: Hailo SDK must only be touched here.
        # Any attempt to call Hailo APIs when NPU is disabled crashes the process.
        log.info("Loading NPU backend (Hailo HEF + sklearn heads) ...")
        state = _load_npu_state(MODELS_DIR)
    else:
        log.info("Loading CPU backend (SentenceTransformer + SetFit models) ...")
        state = _load_cpu_state(MODELS_DIR)

    app = _build_app(state)

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is not installed. Run:  pip install uvicorn") from exc

    log.info(
        "Bio classifier server ready. mode=%s  host=%s  port=%d",
        state["mode"], args.host, args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
