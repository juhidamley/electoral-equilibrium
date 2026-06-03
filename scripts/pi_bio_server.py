"""Bio classifier server — Raspberry Pi 5 (CPU mode, SetFit stub).

Endpoints
---------
GET  /health     Liveness probe. Returns mode, model name, and status.
POST /classify   Embed a bio string. Returns the embedding vector; bloc is null
                 until SetFit is trained in Week 4 Day 3.

Usage
-----
    # CPU-only (works on any machine, no Hailo SDK needed):
    python scripts/pi_bio_server.py --cpu-only

    # Default — reads pi_npu_enabled from configs/base.json:
    python scripts/pi_bio_server.py --config configs/base.json

    # Health check:
    curl http://localhost:9000/health
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
log = logging.getLogger("pi_bio_server")

MODEL_NAME = "all-MiniLM-L6-v2"


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed.\n"
            "Run: pip install sentence-transformers"
        ) from exc

    log.info("Loading %s on CPU (float32)...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    # Ensure float32 inference — sentence-transformers defaults to float32 on CPU,
    # but be explicit for clarity.
    model = model.float()
    log.info("Model ready.")
    return model


# ── Config helpers ────────────────────────────────────────────────────────────


def _load_config(path: Path) -> dict:
    if not path.exists():
        log.warning("Config not found at %s — using empty config.", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _set_npu_disabled(config_path: Path) -> None:
    """Write pi_npu_enabled=false to config on disk."""
    config = _load_config(config_path)
    config["pi_npu_enabled"] = False
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    log.info("Set pi_npu_enabled=false in %s.", config_path)


# ── FastAPI app ───────────────────────────────────────────────────────────────


def _build_app(model):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "fastapi and pydantic are not installed.\n"
            "Run: pip install fastapi uvicorn pydantic"
        ) from exc

    app = FastAPI(
        title="Electoral Equilibrium — Bio Classifier",
        version="1.0.0",
    )

    class ClassifyRequest(BaseModel):
        bio: str = ""

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": "cpu",
            "inference_backend": "cpu",
            "model": MODEL_NAME,
        }

    @app.post("/classify")
    def classify(req: ClassifyRequest):
        bio = req.bio.strip()
        if bio:
            vec = model.encode([bio], convert_to_numpy=True, normalize_embeddings=True)
            embedding = vec[0].tolist()
        else:
            embedding = []
        # bloc classification is null until SetFit is trained (Week 4 Day 3)
        return {"bloc": None, "embedding": embedding}

    return app


# ── Entry point ───────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bio classifier server for the Pi (CPU mode, SetFit stub).",
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
        help="Force CPU backend and write pi_npu_enabled=false to config.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).parent.parent.resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    if args.cpu_only:
        _set_npu_disabled(config_path)
        log.info("--cpu-only: CPU backend selected.")
    else:
        config = _load_config(config_path)
        if config.get("pi_npu_enabled", False):
            log.warning(
                "pi_npu_enabled=true in config but NPU support is not implemented in "
                "this stub. Falling back to CPU. Run compile_hailo.py first, then "
                "upgrade this server for NPU inference."
            )

    model = _load_model()
    app = _build_app(model)

    log.info("Starting server on %s:%d", args.host, args.port)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is not installed. Run: pip install uvicorn") from exc

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
