"""Wraps `phase3.aman_pipeline.verify_content()` for the FastAPI service:

  - a one-time startup warm-load of the lazy SVM/BiLSTM/MARBERT/novelty
    singletons (so no request pays that 30-90s cost; see `warm_up_pipeline`)
  - translation from verify_content()'s raw dict into the Phase 4
    `VerifyResponse` schema (see `backend/schemas.py`)
  - the sync-to-async boundary: verify_content() is a blocking, synchronous
    call and must run off the asyncio event loop (done by the caller in
    `backend/main.py` via a threadpool executor + `asyncio.wait_for`).

Nothing under phase1/2/3 is modified here -- this module only imports and
calls `phase3.aman_pipeline.verify_content()` (and its two private lazy
loaders, used only to force the warm-up timing, never to change behavior).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Code files/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas import (  # noqa: E402
    EvidenceItem,
    ModelVote,
    NoveltyInfo,
    VerifyMetadata,
    VerifyResponse,
)

logger = logging.getLogger("aman_backend.pipeline")

_pipeline_loaded = False


def is_pipeline_loaded() -> bool:
    """Used by GET /health. Reports the warm-up flag only -- never
    re-triggers loading."""
    return _pipeline_loaded


def warm_up_pipeline() -> None:
    """Force the lazy ensemble (SVM/BiLSTM/MARBERT-probe) and novelty/router
    threshold singletons in `phase3.aman_pipeline` to load NOW, at app
    startup, instead of on the first request. This is the one-time 30-90s
    cost documented in the Phase 4 brief.

    Deliberately does NOT warm the evidence-retrieval index or an LLM
    provider: `phase3.aman_pipeline.verify_content()` imports and loads
    those lazily itself, only on the escalation path, by explicit design
    (see that module's docstring: "the trusted path above never pays for
    loading the retrieval index or an LLM provider unless escalation
    actually happens"). Warming them unconditionally here would contradict
    that design and pay a cost most requests never need.
    """
    global _pipeline_loaded
    import phase3.aman_pipeline as pipeline

    logger.info("Warming up AMAN pipeline (SVM/BiLSTM/MARBERT-probe/novelty detector)...")
    start = time.time()
    pipeline._get_ensemble()
    pipeline._get_threshold()
    _pipeline_loaded = True
    logger.info("Pipeline warm-up complete in %.1fs", time.time() - start)


def run_verification(text: str, evidence_text: str) -> VerifyResponse:
    """Synchronous call into verify_content(). Must be invoked off the
    asyncio event loop -- backend/main.py runs this in a threadpool with
    an asyncio.wait_for timeout wrapped around it."""
    import phase3.aman_pipeline as pipeline

    start = time.time()
    raw = pipeline.verify_content(text, evidence_text)
    elapsed = time.time() - start
    return _translate(raw, elapsed)


def _translate(raw: dict, elapsed_seconds: float) -> VerifyResponse:
    """Maps verify_content()'s real return shape onto the Phase 4 API
    contract: nests novelty_score/novelty_category under `novelty`, derives
    `model_agreement` from `_router_detail.all_agree`, and flattens the
    router detail + a timing figure into `metadata`."""
    router = raw.get("_router_detail", {}) or {}
    votes = router.get("votes", []) or []
    model_votes = [
        ModelVote(model=v["model"], prediction=v["prediction"], confidence=v["confidence"])
        for v in votes
    ]
    return VerifyResponse(
        status=raw["status"],
        classification=raw.get("classification"),
        confidence=raw["confidence"],
        novelty=NoveltyInfo(level=raw["novelty_category"], score=raw["novelty_score"]),
        model_agreement="ALL_AGREE" if router.get("all_agree") else "PARTIAL_AGREEMENT",
        deep_verification_used=raw["deep_verification_used"],
        evidence_verdict=raw.get("evidence_verdict"),
        explanation=raw.get("explanation", ""),
        evidence=[EvidenceItem(**e) for e in raw.get("evidence", [])],
        processing_stages=raw.get("processing_stages", []),
        metadata=VerifyMetadata(
            trust_score=router.get("trust_score", 0.0),
            router_threshold=router.get("threshold", 0.0),
            model_votes=model_votes,
            processing_time_seconds=round(elapsed_seconds, 4),
        ),
    )
