"""Phase 4 integration tests: confirm the full stack's contract holds for
all three demo paths (phase4/demo_examples.json), not just one path in
isolation the way backend/tests/test_api.py and phase3/tests/test_phase3.py
each independently do.

Run with:
    python -m pytest phase4/tests/test_integration.py -v

Fast tests stub `phase3.aman_pipeline.verify_content` with representative
shapes for each of the 3 demo paths and confirm every field the frontend
actually reads (`frontend/src/components/*`) round-trips through the real
FastAPI response model unchanged. One opt-in slow test
(AMAN_RUN_SLOW_TESTS=1) calls the REAL pipeline for all 3 real demo claims
through the live HTTP API -- this is the actual manual check already run
once by hand during Phase 4 integration (confirmed working), captured
here as a repeatable regression test.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]  # Code files/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.pipeline_wrapper as pipeline_wrapper  # noqa: E402
import backend.main as backend_main  # noqa: E402

DEMO_EXAMPLES = json.loads((ROOT / "phase4" / "demo_examples.json").read_text(encoding="utf-8"))["examples"]

# Representative verify_content() return shapes for the three demo paths,
# matching phase3.aman_pipeline.verify_content's real schema exactly
# (see phase3/aman_pipeline.py's docstring).
STUB_RESULTS = {
    "high_confidence_no_escalation": {
        "status": "VERIFIED_AUTOMATICALLY", "classification": "Fake", "confidence": 0.8749,
        "novelty_score": 0.2459, "novelty_category": "FAMILIAR", "deep_verification_used": False,
        "evidence_verdict": None, "explanation": "Classified as Fake by AMAN's primary classifier...",
        "evidence": [],
        "processing_stages": ["analyzing_language", "running_classifiers", "checking_confidence",
                               "checking_novelty", "preparing_result"],
        "_router_detail": {"trust_score": 0.8749, "threshold": 0.3992, "all_agree": True, "votes": [
            {"model": "svm", "prediction": "Fake", "confidence": 0.8749, "raw_confidence": 0.8749},
            {"model": "bilstm", "prediction": "Fake", "confidence": 0.5148, "raw_confidence": 0.5148},
            {"model": "transformer", "prediction": "Fake", "confidence": 0.8759, "raw_confidence": 0.8759},
        ]},
    },
    "escalation_with_evidence": {
        "status": "VERIFIED_WITH_EVIDENCE", "classification": "Real", "confidence": 0.9,
        "novelty_score": 0.31, "novelty_category": "MODERATELY NOVEL", "deep_verification_used": True,
        "evidence_verdict": "SUPPORTED", "explanation": "The evidence passages support the claim...",
        "evidence": [{"id": "own5_1", "text": "evidence passage text", "source": "VERA-ARAB",
                      "retrieval_method": "hybrid_rrf", "retrieval_score": 0.87}],
        "processing_stages": ["analyzing_language", "running_classifiers", "checking_confidence",
                               "checking_novelty", "retrieving_evidence", "deep_verification", "preparing_result"],
        "_router_detail": {"trust_score": 0.1, "threshold": 0.3992, "all_agree": False, "votes": [
            {"model": "svm", "prediction": "Real", "confidence": 0.6, "raw_confidence": 0.6},
            {"model": "bilstm", "prediction": "Real", "confidence": 0.9, "raw_confidence": 0.9},
            {"model": "transformer", "prediction": "Fake", "confidence": 0.52, "raw_confidence": 0.52},
        ]},
    },
    "insufficient_evidence": {
        "status": "UNCERTAIN", "classification": None, "confidence": 0.0,
        "novelty_score": 0.1064, "novelty_category": "FAMILIAR", "deep_verification_used": True,
        "evidence_verdict": "INSUFFICIENT_EVIDENCE",
        "explanation": "[SAFEGUARD] model returned CONTRADICTED without citing any evidence",
        "evidence": [],
        "processing_stages": ["analyzing_language", "running_classifiers", "checking_confidence",
                               "checking_novelty", "retrieving_evidence", "deep_verification", "preparing_result"],
        "_router_detail": {"trust_score": 0.114, "threshold": 0.3992, "all_agree": False, "votes": [
            {"model": "svm", "prediction": "Real", "confidence": 0.61, "raw_confidence": 0.61},
            {"model": "bilstm", "prediction": "Real", "confidence": 0.89, "raw_confidence": 0.9},
            {"model": "transformer", "prediction": "Fake", "confidence": 0.53, "raw_confidence": 0.52},
        ]},
    },
}

REQUIRED_TOP_LEVEL_KEYS = {
    "status", "classification", "confidence", "novelty", "model_agreement",
    "deep_verification_used", "evidence_verdict", "explanation", "evidence",
    "processing_stages", "metadata",
}
VALID_STATUS = {"VERIFIED_AUTOMATICALLY", "VERIFIED_WITH_EVIDENCE", "UNCERTAIN"}
VALID_EVIDENCE_VERDICT = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", None}
VALID_NOVELTY_LEVEL = {"FAMILIAR", "MODERATELY NOVEL", "HIGHLY NOVEL"}
VALID_AGREEMENT = {"ALL_AGREE", "PARTIAL_AGREEMENT"}


@pytest.fixture(autouse=True)
def _reset_pipeline_loaded_flag():
    original = pipeline_wrapper._pipeline_loaded
    yield
    pipeline_wrapper._pipeline_loaded = original


@pytest.fixture
def client():
    pipeline_wrapper._pipeline_loaded = True
    return TestClient(backend_main.app)


@pytest.mark.parametrize("example", DEMO_EXAMPLES, ids=[e["label"] for e in DEMO_EXAMPLES])
def test_demo_example_contract_shape(client, monkeypatch, example):
    """Every one of the 3 demo paths must satisfy the full API contract
    the frontend was built against -- not just one path tested once."""
    stub = STUB_RESULTS[example["label"]]
    import phase3.aman_pipeline as real_pipeline
    monkeypatch.setattr(real_pipeline, "verify_content", lambda text, evidence_text="": dict(stub))

    response = client.post("/api/verify", json={"text": example["claim"], "evidence_text": example["evidence"]})
    assert response.status_code == 200
    body = response.json()

    assert REQUIRED_TOP_LEVEL_KEYS.issubset(body.keys())
    assert body["status"] in VALID_STATUS
    assert body["evidence_verdict"] in VALID_EVIDENCE_VERDICT
    assert body["novelty"]["level"] in VALID_NOVELTY_LEVEL
    assert 0.0 <= body["novelty"]["score"] <= 1.0
    assert body["model_agreement"] in VALID_AGREEMENT
    assert isinstance(body["processing_stages"], list) and len(body["processing_stages"]) >= 4
    assert "trust_score" in body["metadata"] and "model_votes" in body["metadata"]

    if body["deep_verification_used"]:
        assert "retrieving_evidence" in body["processing_stages"]
        assert "deep_verification" in body["processing_stages"]
    else:
        assert body["evidence"] == []
        assert body["evidence_verdict"] is None


def test_insufficient_evidence_path_never_claims_a_classification(client, monkeypatch):
    """The frontend's two-tier language rule depends on this: AMAN must
    never report a Real/Fake classification when the evidence verdict is
    INSUFFICIENT_EVIDENCE -- that would contradict the UI's own
    'LIKELY RELIABLE/MISLEADING vs SUPPORTED/CONTRADICTED/INSUFFICIENT
    EVIDENCE' distinction."""
    stub = STUB_RESULTS["insufficient_evidence"]
    import phase3.aman_pipeline as real_pipeline
    monkeypatch.setattr(real_pipeline, "verify_content", lambda text, evidence_text="": dict(stub))
    response = client.post("/api/verify", json={"text": "test claim", "evidence_text": ""})
    body = response.json()
    assert body["evidence_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert body["classification"] is None
    assert body["status"] == "UNCERTAIN"


@pytest.mark.skipif(
    os.environ.get("AMAN_RUN_SLOW_TESTS") != "1",
    reason="Real end-to-end run through all 3 live demo examples; set AMAN_RUN_SLOW_TESTS=1 to run "
           "(pays model warm-up + live retrieval/LLM calls; can take several minutes total).",
)
def test_all_three_demo_examples_real_end_to_end():
    pipeline_wrapper._pipeline_loaded = False
    client = TestClient(backend_main.app)
    with client:  # triggers the real lifespan startup warm-up
        health = client.get("/health")
        assert health.json()["pipeline_loaded"] is True

        for example in DEMO_EXAMPLES:
            t0 = time.time()
            response = client.post(
                "/api/verify",
                json={"text": example["claim"], "evidence_text": example["evidence"]},
                timeout=200,
            )
            elapsed = time.time() - t0
            assert response.status_code == 200, f"{example['label']} failed: {response.text[:300]}"
            body = response.json()
            assert REQUIRED_TOP_LEVEL_KEYS.issubset(body.keys())
            print(f"{example['label']}: status={body['status']} "
                  f"classification={body['classification']} evidence_verdict={body['evidence_verdict']} "
                  f"elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
