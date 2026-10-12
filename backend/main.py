"""AMAN Sentinel Phase 4 API -- a FastAPI wrapper around the single Phase
1-3 integration point, `phase3.aman_pipeline.verify_content()`.

This file does not modify, retrain, or re-implement any classifier,
novelty-detector, router, retrieval, or LLM-verification logic -- it only
imports and calls `verify_content()` (via `backend/pipeline_wrapper.py`)
and shapes its real return value to the Phase 4 API contract.

Run from the repo root (so `phase3`/`backend` import as top-level
packages):

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Then see interactive docs at http://localhost:8000/docs

Timeout policy
--------------
`verify_content()` is synchronous (not async) and its latency varies hugely
by path:
  - ~30-90s, but ONLY on the very first call ever made -- this service
    avoids paying that cost per-request by warming the pipeline up once at
    startup (see `lifespan` below), before any request is accepted.
  - ~1-2s on the trusted (non-escalated, VERIFIED_AUTOMATICALLY) path.
  - up to roughly 2 minutes on the escalation path (evidence retrieval +
    a live Ollama LLM call for VERIFIED_WITH_EVIDENCE / UNCERTAIN).

Because it's a blocking call, each request runs it in a worker thread
(`run_in_executor`) and the whole thing is bounded by
`VERIFY_TIMEOUT_SECONDS = 180.0` via `asyncio.wait_for`. 180s was chosen as
roughly 1.5x the documented worst case (~120s) -- generous enough to
absorb a slow/cold Ollama response without being unbounded, while still
guaranteeing every request eventually gets an HTTP response (504) instead
of hanging forever if the LLM provider wedges or a network call stalls.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]  # Code files/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline_wrapper import (  # noqa: E402
    is_pipeline_loaded,
    run_verification,
    warm_up_pipeline,
)
from backend.schemas import (  # noqa: E402
    ErrorResponse,
    HealthResponse,
    VerifyRequest,
    VerifyResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aman_backend")

# See module docstring "Timeout policy" above for the reasoning.
VERIFY_TIMEOUT_SECONDS = 180.0

# A small fixed-size pool: verify_content()'s trusted path touches shared
# module-level singleton models, so unbounded per-request threads buy
# nothing but memory. max_workers=4 lets a handful of requests queue
# concurrently without the service falling over under light load.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aman-verify")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the SVM/BiLSTM/MARBERT-probe/novelty-detector singletons ONCE,
    before the app starts accepting requests, so the one-time 30-90s model
    load cost is never charged to a caller."""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(_executor, warm_up_pipeline)
    except Exception:
        # Don't crash the app if warm-up fails -- /health will correctly
        # report pipeline_loaded=False, and /api/verify will surface the
        # real error (lazily retried on first use) instead of the service
        # never starting at all.
        logger.exception("Pipeline warm-up failed at startup; /health will report pipeline_loaded=False.")
    yield
    # _executor is intentionally NOT shut down here: it is a module-level
    # singleton meant to live for the process's lifetime. (Tests create
    # multiple TestClient lifespan cycles against this same app/executor;
    # shutting it down on the first one would break every subsequent
    # request.) Its worker threads exit naturally on process shutdown.


app = FastAPI(
    title="AMAN Sentinel Verification API",
    description=(
        "Phase 4 API wrapper around the Phase 1-3 AMAN misinformation "
        "verification pipeline (SVM/BiLSTM/MARBERT ensemble + confidence "
        "router + novelty detector + evidence-retrieval/LLM escalation)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Turns Pydantic's validation errors into the same clean
    {error, detail} shape as every other error response, instead of
    FastAPI's default verbose error-list body."""
    messages = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
    )
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": messages})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Reports whether the one-time startup load has finished. Never
    re-runs that load itself."""
    return HealthResponse(status="ok", pipeline_loaded=is_pipeline_loaded())


@app.post(
    "/api/verify",
    response_model=VerifyResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid input (empty, whitespace-only, or too long)."},
        503: {"model": ErrorResponse, "description": "The verification pipeline raised an error."},
        504: {"model": ErrorResponse, "description": "Verification did not complete within the timeout."},
    },
)
async def verify(payload: VerifyRequest) -> VerifyResponse:
    """Runs the full AMAN verification pipeline on `payload.text`
    (optionally with `payload.evidence_text`), bounded by
    VERIFY_TIMEOUT_SECONDS. See module docstring for the timeout
    rationale."""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, run_verification, payload.text, payload.evidence_text),
            timeout=VERIFY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("verify_content() exceeded the %.0fs timeout.", VERIFY_TIMEOUT_SECONDS)
        return JSONResponse(
            status_code=504,
            content={
                "error": "verification_timeout",
                "detail": (
                    f"Verification did not complete within {VERIFY_TIMEOUT_SECONDS:.0f} seconds. "
                    "This usually means the escalation path (evidence retrieval + LLM call) is "
                    "unusually slow or the configured LLM provider is unreachable."
                ),
            },
        )
    except Exception:  # noqa: BLE001 - deliberately broad: never leak a raw traceback to the client
        logger.exception("verify_content() raised an unhandled exception.")
        return JSONResponse(
            status_code=503,
            content={
                "error": "verification_failed",
                "detail": (
                    "The verification pipeline encountered an internal error "
                    "(model load failure, LLM provider unreachable, or missing "
                    "retrieval index). See server logs for details."
                ),
            },
        )
