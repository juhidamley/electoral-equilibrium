"""deploy/modal_app.py — Modal deployment of the Electoral Equilibrium FastAPI backend.

=== ADAPTER SOURCE ===
The r=16 QLoRA adapter (adapter_model.safetensors, ~52 MB) is stored in the Modal
Volume "electoral-adapters".  Populate it ONCE from this machine before deploying:

    modal volume create electoral-adapters
    cd /path/to/electoral-equilibrium            # repo root
    modal volume put electoral-adapters models/mistral-r16/ /mistral-r16/

The base model (mistralai/Mistral-7B-v0.3, ~14 GB) is downloaded from HuggingFace
during image build and baked into the cached image layer.  This is a one-time ~30 min
build; subsequent deploys are instant.

=== SECRETS ===
Create one Modal secret before deploying:

    modal secret create electoral-secrets \
        ALLOWED_ORIGIN="https://<your-vercel-subdomain>.vercel.app" \
    # optional:
        HF_TOKEN="hf_..."

ALLOWED_ORIGIN controls CORS — the exact Vercel URL, not a wildcard.
HF_TOKEN is only needed if the HuggingFace model requires authentication
(mistralai/Mistral-7B-v0.3 is public, so usually not required).

=== DEPLOYMENT ===
    modal deploy deploy/modal_app.py

Copy the printed web endpoint URL from the Modal dashboard and set:
    Vercel project → Settings → Environment Variables:
        NEXT_PUBLIC_API_URL = https://<workspace>--electoral-equilibrium-serve.modal.run

=== VERCEL SETUP ===
1. Push this repo to GitHub.
2. In Vercel: New Project → Import repo → set Root Directory = "webapp".
3. Framework preset: Next.js (auto-detected).
4. Add env var: NEXT_PUBLIC_API_URL = <Modal endpoint URL from above>.
5. Deploy.  Verify /estimate/stream in both "democrat" and "republican" modes.

=== COST ===
GPU: A10G (24 GB VRAM) — Mistral 7B in float16 uses ~14 GB, leaving headroom.
Modal bills only for active GPU time; scales to zero when idle.
Cold start (model load + adapter merge): ~30–60 s.
Warm request (SSE stream): ~5–15 s end-to-end.

=== PINNED DEPENDENCY RATIONALE ===
outlines==0.0.37   — HPC-tested version; newer outlines changed the Transformers API
numpy<2.0          — cvxpy 1.3.x uses np.bool8 (removed in numpy 2.0)
scipy<1.14         — API break in sparse matrix handling used by cvxpy
cvxpy==1.3.4       — DQCP interface changed in 1.4.x; not yet validated
transformers==4.46.3 — tested with this version on Laguna (MAE = 0.0362)
"""

from __future__ import annotations

from typing import Any

import modal

# ── Pinned dependency set ──────────────────────────────────────────────────────
# Matches the USC Laguna L40S training environment.
# Do NOT bump without re-running full eval — outlines/transformers/numpy interact
# tightly with the constrained-decoding pipeline.

_PACKAGES = [
    # PyTorch ecosystem
    "torch==2.4.0",
    "transformers==4.46.3",
    "peft==0.13.2",
    "accelerate==0.33.0",
    "bitsandbytes==0.43.3",
    "safetensors==0.4.5",
    "sentencepiece==0.2.0",  # Mistral tokenizer is sentencepiece-based
    "protobuf==4.25.3",  # sentencepiece tokenizer conversion needs it
    # Constrained decoding — PINNED to HPC-tested version
    "outlines==0.0.37",
    # Numerical — PINNED below breaking API changes
    "numpy<2.0",
    "scipy<1.14",
    # Optimizer — PINNED to 1.3.x (1.4 changed DQCP interface, not yet validated)
    "cvxpy==1.3.4",
    # API stack
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.6",
    "pydantic==2.10.3",
    "python-multipart==0.0.9",
    "duckdb==0.10.3",
    # Data
    "pandas==2.2.2",
    "pyarrow==17.0.0",
    # HuggingFace Hub (model download during image build)
    "huggingface-hub==0.24.7",
]

_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
_MODEL_CACHE = "/root/model-cache"

# ── Image ──────────────────────────────────────────────────────────────────────


def _download_base_model() -> None:
    """Download base model weights into the image layer (runs once at build time)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=_BASE_MODEL,
        local_dir=_MODEL_CACHE,
        ignore_patterns=[
            "*.msgpack",
            "*.h5",
            "flax_model*",
            "tf_model*",
            "rust_model*",
        ],
    )
    print(f"[modal build] Base model cached at {_MODEL_CACHE}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_PACKAGES)
    .env(
        {
            # Point transformers + HF hub at the baked-in cache.
            "TRANSFORMERS_CACHE": _MODEL_CACHE,
            "HF_HOME": _MODEL_CACHE,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .run_function(
        _download_base_model,
        # Mistral-7B-v0.3 is public — no HF token needed for download.
    )
    # Local-dir adds must be the final image layers in Modal 1.x (runtime mount).
    .add_local_dir("electoral", remote_path="/root/electoral",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("artifacts", remote_path="/root/artifacts",
                   ignore=["**/*.parquet"])
)

# ── Modal app ──────────────────────────────────────────────────────────────────

modal_app = modal.App("electoral-equilibrium", image=image)
app = modal_app  # alias so `modal deploy` auto-discovers

# ── Volumes ────────────────────────────────────────────────────────────────────
# adapter_vol — adapter weights; populated once with `modal volume put` (see header).
adapter_vol = modal.Volume.from_name("electoral-adapters", create_if_missing=True)

# audit_vol — persists the DuckDB audit log across container restarts.
# NOTE: single-writer DuckDB constraint is enforced by allow_concurrent_inputs=1.
# If concurrency is ever raised, switch to a PostgreSQL-backed audit store.
audit_vol = modal.Volume.from_name("electoral-audit", create_if_missing=True)

# ── Local-code mounts ──────────────────────────────────────────────────────────
# The electoral/ package and runtime configs are mounted from the local checkout.
# This means `modal deploy` bundles the current HEAD of these directories —
# re-deploy after any local change to pick them up.


# ── Container paths ────────────────────────────────────────────────────────────
# These are the paths inside the running Modal container.
_ADAPTER_PATH = "/adapters/mistral-r16-v2"
_AUDIT_DB_PATH = "/audit/audit.duckdb"


# ── FastAPI ASGI endpoint ──────────────────────────────────────────────────────


@modal_app.function(
    # A10G: 24 GB VRAM — Mistral 7B float16 uses ~14 GB; leaves ~10 GB for KV cache.
    # Alternatives: gpu=modal.gpu.L4() (similar price), gpu=modal.gpu.A100() (overkill).
    gpu="A10G",
    memory=20480,  # 20 GB system RAM — base model + overhead
    cpu=2.0,
    timeout=900,  # generous for cold start (~60 s model load + adapter merge)
    volumes={
        "/adapters": adapter_vol,
        "/audit": audit_vol,
    },
    secrets=[
        modal.Secret.from_name("electoral-secrets"),
    ],
    # Single-writer constraint for both DuckDB and the CVXPY ProcessPoolExecutor.
    # Raise only after switching audit store to a concurrent-safe backend.
    max_containers=1,
    scaledown_window=600,  # keep warm 5 min — saves cold-start latency
)
@modal.asgi_app()
def serve() -> Any:
    """Return the Electoral Equilibrium FastAPI app for Modal ASGI serving.

    Called once per container startup (not per request).  Sets up the environment
    before importing shock_endpoint so module-level env reads (ADAPTER_PATH,
    BASE_MODEL) resolve correctly.
    """
    import multiprocessing as mp
    import os
    import sys

    # Set start method BEFORE ProcessPoolExecutor is created in _lifespan.
    # "spawn" is required because the main process has an active CUDA context
    # (LLM loaded on GPU); forking after CUDA init is unsupported by PyTorch.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set in a previous invocation on this container

    # Put the mounted source on sys.path so `import electoral` resolves
    # to /root/electoral (the mounted local package, not site-packages).
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    # Inject paths that shock_endpoint reads at MODULE IMPORT TIME.
    # Must be set before `from electoral.api.shock_endpoint import app`.
    os.environ["ADAPTER_PATH"] = _ADAPTER_PATH
    os.environ["BASE_MODEL"] = _BASE_MODEL

    # Build the CORS allowed-origins list.
    # ALLOWED_ORIGIN comes from the "electoral-secrets" Modal secret.
    vercel_origin = os.environ.get("ALLOWED_ORIGIN", "").strip()
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3002",
    ]
    if vercel_origin:
        allowed_origins.append(vercel_origin)
        # Also allow preview deployments (branch URLs) from the same Vercel project.
        # Vercel preview URLs follow: https://<project>-<hash>-<team>.vercel.app
        # If the ALLOWED_ORIGIN is the production URL, add the *.vercel.app pattern
        # for previews by adding it explicitly below:
        # allowed_origins.append("https://*.vercel.app")  # uncomment for previews

    from electoral.api.shock_endpoint import app as fastapi_app
    from fastapi.middleware.cors import CORSMiddleware

    # Replace the dev-only CORS middleware added in shock_endpoint.py.
    # user_middleware is populated by add_middleware(); middleware_stack is built
    # lazily on first request — modifying user_middleware here is safe.
    fastapi_app.user_middleware = [
        m for m in fastapi_app.user_middleware if m.cls is not CORSMiddleware
    ]
    # Reset cached stack so it's rebuilt from the updated user_middleware.
    fastapi_app.middleware_stack = None

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        allow_credentials=False,
    )

    return fastapi_app
