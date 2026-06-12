"""FastAPI service: ShockEstimator inference endpoint.

Routes:
  POST /estimate  — run ShockEstimator.estimate() and return ShockResponseData
  GET  /health    — liveness check with device info
  GET  /blocs     — canonical bloc names for all three strata

ShockEstimator is loaded once at startup via the FastAPI lifespan context and
stored in app.state.estimator. The same ShockResponseSchema Pydantic model used
for constrained decoding is also declared as the /estimate response_model,
guaranteeing the two schemas are always identical.
"""

from __future__ import annotations

import logging
import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from electoral.artifacts import ShockResponseData, ShockResponseSchema
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.llm.inference import ShockEstimator

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_ADAPTER_PATH: str = os.environ.get("ADAPTER_PATH", "models/mistral-r16")
_BASE_MODEL: str = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.3")

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.estimator = ShockEstimator(
        adapter_path=_ADAPTER_PATH,
        base_model=_BASE_MODEL,
    )
    yield
    # Nothing to tear down — model is held in process memory.


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Electoral Equilibrium — Shock Estimator API",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response models ─────────────────────────────────────────────────


class EstimateRequest(BaseModel):
    event: dict[str, Any]
    intensity: float = 1.0


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/estimate", response_model=ShockResponseSchema)
def estimate(request: EstimateRequest) -> Any:
    """Run ShockEstimator.estimate() and return constrained delta bins.

    The response_model is ShockResponseSchema — the same Pydantic model used
    by outlines for constrained decoding — so the API contract and the
    generation constraint are always identical.
    """
    event_text = (
        request.event.get("description", "")
        if isinstance(request.event, dict)
        else str(request.event)
    )
    if not event_text or len(event_text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Event description too short")
    if not (0.1 <= request.intensity <= 3.0):
        raise HTTPException(status_code=422, detail="Intensity out of range")

    estimator: ShockEstimator = app.state.estimator
    try:
        result: ShockResponseData = estimator.estimate(
            request.event, intensity=request.intensity
        )
    except Exception:
        log.exception("Unhandled error in /estimate")
        raise HTTPException(status_code=500, detail="Internal inference error")

    # Convert frozen dataclass → nested dict matching ShockResponseSchema shape.
    return {
        "delta_bins_race": result.delta_bins_race,
        "delta_bins_religion": result.delta_bins_religion,
        "delta_bins_gender": result.delta_bins_gender,
        "delta_eff": result.delta_eff,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — returns model path and active device."""
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "model": _ADAPTER_PATH, "device": device}


@app.get("/blocs")
def blocs() -> dict[str, list[str]]:
    """Return canonical bloc names for all three demographic strata."""
    return {
        "race": list(CANONICAL_RACES),
        "religion": list(CANONICAL_RELIGIONS),
        "gender": list(CANONICAL_GENDERS),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
