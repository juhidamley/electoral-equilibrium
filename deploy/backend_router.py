"""deploy/backend_router.py — Route inference traffic between Modal and HPC vLLM.

STATUS: DEACTIVATED — scaffolded for future use, NOT wired into any active
deployment path.  Activate when the HPC vLLM backend (hpc_vllm.sh) lands.
See: docs/devplan.pdf §7 "HPC vLLM backend"

WHY DEACTIVATED:
    The current production backend is Modal (deploy/modal_app.py).  The router
    becomes relevant once the HPC vLLM path is validated and the CMC Hopper
    allocation is confirmed.

INTENDED BEHAVIOUR (when active):
    INFERENCE_BACKEND=modal      → forward to MODAL_URL
    INFERENCE_BACKEND=hpc_vllm  → forward to VLLM_URL

    The router is a thin HTTP proxy — it does NOT run inference itself.
    It translates Electoral API calls (/estimate/stream, /estimate, etc.)
    to the correct downstream format and preserves SSE streaming.

ACTIVATION CHECKLIST:
    1. hpc_vllm.sh is running and the vLLM endpoint is reachable.
    2. Set env vars:
         INFERENCE_BACKEND=hpc_vllm
         VLLM_URL=http://<hopper-tailscale-ip>:8080
         MODAL_URL=https://<workspace>--electoral-equilibrium-serve.modal.run
    3. Uncomment the code below and delete this NotImplementedError.
    4. Update shock_endpoint.py CORS to allow the router's origin.
    5. Point NEXT_PUBLIC_API_URL at the router, not directly at Modal.

DEPLOYMENT (when active):
    uvicorn deploy.backend_router:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

raise NotImplementedError(
    "backend_router is DEACTIVATED — not yet wired into any deployment path. "
    "See deploy/hpc_vllm.sh and docs/devplan.pdf §7 for the activation plan."
)

# ── DEACTIVATED SCAFFOLD ── Uncomment when HPC vLLM task lands ───────────────
#
# import os
# from typing import AsyncGenerator
#
# import httpx
# from fastapi import FastAPI, Request, Response
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
#
# _BACKEND = os.environ.get("INFERENCE_BACKEND", "modal")
# _MODAL_URL = os.environ.get("MODAL_URL", "").rstrip("/")
# _VLLM_URL = os.environ.get("VLLM_URL", "").rstrip("/")
#
#
# def _target_base() -> str:
#     """Return the base URL of the active backend."""
#     if _BACKEND == "hpc_vllm":
#         if not _VLLM_URL:
#             raise EnvironmentError("VLLM_URL must be set when INFERENCE_BACKEND=hpc_vllm")
#         return _VLLM_URL
#     if not _MODAL_URL:
#         raise EnvironmentError("MODAL_URL must be set when INFERENCE_BACKEND=modal")
#     return _MODAL_URL
#
#
# app = FastAPI(
#     title="Electoral Equilibrium — Backend Router",
#     description=(
#         f"Active backend: {_BACKEND!r}. "
#         "Set INFERENCE_BACKEND=modal|hpc_vllm to switch."
#     ),
# )
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[os.environ.get("ALLOWED_ORIGIN", "*")],
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["Content-Type", "Authorization"],
#     allow_credentials=False,
# )
#
#
# @app.get("/health")
# async def health() -> dict[str, str]:
#     return {"status": "ok", "backend": _BACKEND, "target": _target_base()}
#
#
# @app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
# async def proxy(request: Request, path: str) -> Response:
#     """Transparent HTTP proxy — preserves SSE streaming for /estimate/stream."""
#     target = f"{_target_base()}/{path}"
#
#     # SSE streaming: pipe bytes as they arrive rather than buffering.
#     if request.headers.get("accept") == "text/event-stream" or path == "estimate/stream":
#         async def _stream() -> AsyncGenerator[bytes, None]:
#             async with httpx.AsyncClient(timeout=None) as client:
#                 async with client.stream(
#                     method=request.method,
#                     url=target,
#                     headers={
#                         k: v for k, v in request.headers.items()
#                         if k.lower() not in ("host", "content-length")
#                     },
#                     content=await request.body(),
#                     params=dict(request.query_params),
#                 ) as resp:
#                     async for chunk in resp.aiter_bytes():
#                         yield chunk
#
#         return StreamingResponse(
#             _stream(),
#             media_type="text/event-stream",
#             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
#         )
#
#     # Standard request: buffer response and forward.
#     async with httpx.AsyncClient(timeout=120.0) as client:
#         resp = await client.request(
#             method=request.method,
#             url=target,
#             headers={
#                 k: v for k, v in request.headers.items()
#                 if k.lower() not in ("host", "content-length")
#             },
#             content=await request.body(),
#             params=dict(request.query_params),
#         )
#     return Response(
#         content=resp.content,
#         status_code=resp.status_code,
#         headers={k: v for k, v in resp.headers.items() if k.lower() != "transfer-encoding"},
#         media_type=resp.headers.get("content-type"),
#     )
