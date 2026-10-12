"""Pydantic v2 request/response schemas for the AMAN Sentinel Phase 4 API.

Every field here maps directly to something `phase3.aman_pipeline.verify_content()`
actually returns (confirmed by reading that module, ~145 lines, before writing this
file) — nothing is invented. See `backend/pipeline_wrapper.py` for the
raw-dict -> schema translation.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Input is capped to keep the pipeline (classifiers + retrieval + LLM prompt
# construction) operating on bounded-size text; 5000 chars is generous for a
# single news claim/post while still rejecting pasted-in whole articles that
# would blow out LLM prompt size on the escalation path.
MAX_TEXT_LENGTH = 5000


class VerifyRequest(BaseModel):
    """POST /api/verify request body."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="The claim text to verify (Arabic or English).",
    )
    evidence_text: str = Field(
        default="",
        max_length=MAX_TEXT_LENGTH,
        description="Optional caller-supplied evidence/context text. Defaults to empty.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v

    @field_validator("evidence_text")
    @classmethod
    def evidence_text_normalize(cls, v: str) -> str:
        # evidence_text is optional and legitimately empty — only guard
        # against it being nothing but whitespace padding.
        return v.strip() if v and not v.strip() else v


class EvidenceItem(BaseModel):
    """One retrieved evidence passage, as returned inside
    `verify_content()["evidence"]`."""

    id: str
    text: str
    source: str
    retrieval_method: str
    retrieval_score: float


class NoveltyInfo(BaseModel):
    level: Literal["FAMILIAR", "MODERATELY NOVEL", "HIGHLY NOVEL"]
    score: float


class ModelVote(BaseModel):
    """One ensemble member's vote, derived from
    `verify_content()["_router_detail"]["votes"]`."""

    model: str
    prediction: str
    confidence: float


class VerifyMetadata(BaseModel):
    trust_score: float
    router_threshold: float
    model_votes: list[ModelVote]
    processing_time_seconds: float


class VerifyResponse(BaseModel):
    """POST /api/verify response body — adapts
    `phase3.aman_pipeline.verify_content()`'s real return shape to the
    Phase 4 API contract (nesting novelty/router-detail, renaming
    all_agree -> model_agreement)."""

    status: Literal["VERIFIED_AUTOMATICALLY", "VERIFIED_WITH_EVIDENCE", "UNCERTAIN"]
    classification: Optional[Literal["Real", "Fake"]] = None
    confidence: float
    novelty: NoveltyInfo
    model_agreement: Literal["ALL_AGREE", "PARTIAL_AGREEMENT"]
    deep_verification_used: bool
    evidence_verdict: Optional[Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]] = None
    explanation: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    processing_stages: list[str] = Field(default_factory=list)
    metadata: VerifyMetadata


class HealthResponse(BaseModel):
    status: Literal["ok"]
    pipeline_loaded: bool


class ErrorResponse(BaseModel):
    """Uniform error body for 422/503/500/504 — never a raw traceback."""

    error: str
    detail: str
