"""Tests for the Phase 4 FastAPI service (backend/main.py). Run with:

    python -m pytest backend/tests/test_api.py -v

from the repo root (so the `backend`/`phase3` packages import correctly).

Most tests stub out `phase3.aman_pipeline.verify_content()` directly,
mirroring the stubbing pattern already established in
`phase3/tests/test_phase3.py`'s `EndToEndTests` (swap the function, call
through, restore it) -- so this suite is fast, deterministic, and never
touches the real SVM/BiLSTM/MARBERT models, retrieval index, or an LLM
provider. One opt-in test calls the real pipeline end-to-end through the
live HTTP API for a genuine integration check (slow: pays the one-time
model-load cost), gated behind AMAN_RUN_SLOW_TESTS=1 so normal runs stay
fast.
"""
from __future__ import annotations

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


@pytest.fixture(autouse=True)
def _reset_pipeline_loaded_flag():
    """Every test controls the warm-up flag explicitly via the `client`
    fixture below; restore whatever it was before so tests don't leak
    state into each other."""
    original = pipeline_wrapper._pipeline_loaded
    yield
    pipeline_wrapper._pipeline_loaded = original


@pytest.fixture
def client(monkeypatch):
    """A TestClient whose startup warm-up is stubbed to a fast no-op, so
    running these tests never pays the real 30-90s SVM/BiLSTM/MARBERT load
    cost. (The real warm_up_pipeline is exercised separately by the opt-in
    slow test at the bottom of this file.)"""

    def _fake_warmup() -> None:
        pipeline_wrapper._pipeline_loaded = True

    monkeypatch.setattr(pipeline_wrapper, "warm_up_pipeline", _fake_warmup)
    monkeypatch.setattr(backend_main, "warm_up_pipeline", _fake_warmup)
    with TestClient(backend_main.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_pipeline_loaded_after_startup(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "pipeline_loaded": True}


# ---------------------------------------------------------------------------
# Trusted (VERIFIED_AUTOMATICALLY) path -- stubbed
# ---------------------------------------------------------------------------


def test_verify_trusted_path_maps_contract_fields(client, monkeypatch):
    def _fake_verify_content(claim_text, evidence_text=""):
        return {
            "status": "VERIFIED_AUTOMATICALLY",
            "classification": "Real",
            "confidence": 0.95,
            "novelty_score": 0.1,
            "novelty_category": "FAMILIAR",
            "deep_verification_used": False,
            "evidence_verdict": None,
            "explanation": "stubbed trusted-path result",
            "evidence": [],
            "processing_stages": [
                "analyzing_language", "running_classifiers",
                "checking_confidence", "checking_novelty", "preparing_result",
            ],
            "_router_detail": {
                "trust_score": 0.95, "threshold": 0.4, "all_agree": True,
                "votes": [{"model": "svm", "prediction": "Real", "confidence": 0.95, "raw_confidence": 0.95}],
            },
        }

    import phase3.aman_pipeline as real_pipeline
    monkeypatch.setattr(real_pipeline, "verify_content", _fake_verify_content)

    resp = client.post("/api/verify", json={"text": "مثال خبر", "evidence_text": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "VERIFIED_AUTOMATICALLY"
    assert body["classification"] == "Real"
    assert body["deep_verification_used"] is False
    assert body["evidence_verdict"] is None
    assert body["model_agreement"] == "ALL_AGREE"
    assert body["novelty"] == {"level": "FAMILIAR", "score": 0.1}
    assert body["evidence"] == []
    assert body["metadata"]["trust_score"] == 0.95
    assert body["metadata"]["router_threshold"] == 0.4
    assert body["metadata"]["model_votes"] == [{"model": "svm", "prediction": "Real", "confidence": 0.95}]
    assert body["metadata"]["processing_time_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# Escalated (VERIFIED_WITH_EVIDENCE) path -- stubbed
# ---------------------------------------------------------------------------


def test_verify_escalated_path_maps_contract_fields(client, monkeypatch):
    def _fake_verify_content(claim_text, evidence_text=""):
        return {
            "status": "VERIFIED_WITH_EVIDENCE",
            "classification": "Fake",
            "confidence": 0.8,
            "novelty_score": 0.7,
            "novelty_category": "HIGHLY NOVEL",
            "deep_verification_used": True,
            "evidence_verdict": "CONTRADICTED",
            "explanation": "stubbed escalation result",
            "evidence": [
                {"id": "e1", "text": "نص دليل", "source": "test",
                 "retrieval_method": "hybrid_rrf", "retrieval_score": 0.9},
            ],
            "processing_stages": [
                "analyzing_language", "running_classifiers", "checking_confidence",
                "checking_novelty", "retrieving_evidence", "deep_verification", "preparing_result",
            ],
            "_router_detail": {
                "trust_score": 0.2, "threshold": 0.4, "all_agree": False,
                "votes": [
                    {"model": "svm", "prediction": "Fake", "confidence": 0.6, "raw_confidence": 0.6},
                    {"model": "bilstm", "prediction": "Real", "confidence": 0.55, "raw_confidence": 0.55},
                ],
            },
        }

    import phase3.aman_pipeline as real_pipeline
    monkeypatch.setattr(real_pipeline, "verify_content", _fake_verify_content)

    resp = client.post("/api/verify", json={"text": "خبر مثير للجدل"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "VERIFIED_WITH_EVIDENCE"
    assert body["deep_verification_used"] is True
    assert body["evidence_verdict"] == "CONTRADICTED"
    assert body["model_agreement"] == "PARTIAL_AGREEMENT"
    assert len(body["evidence"]) == 1
    assert body["evidence"][0]["id"] == "e1"
    assert len(body["metadata"]["model_votes"]) == 2


# ---------------------------------------------------------------------------
# Input validation -> 422
# ---------------------------------------------------------------------------


def test_verify_rejects_empty_text(client):
    resp = client.post("/api/verify", json={"text": ""})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_input"


def test_verify_rejects_whitespace_only_text(client):
    resp = client.post("/api/verify", json={"text": "   \n\t  "})
    assert resp.status_code == 422


def test_verify_rejects_overlong_text(client):
    resp = client.post("/api/verify", json={"text": "a" * 5001})
    assert resp.status_code == 422


def test_verify_rejects_missing_text_field(client):
    resp = client.post("/api/verify", json={})
    assert resp.status_code == 422


def test_verify_accepts_max_length_text(client, monkeypatch):
    """5000 chars exactly is the documented limit and must be accepted."""
    def _fake_verify_content(claim_text, evidence_text=""):
        return {
            "status": "UNCERTAIN", "classification": None, "confidence": 0.5,
            "novelty_score": 0.5, "novelty_category": "MODERATELY NOVEL",
            "deep_verification_used": False, "evidence_verdict": None,
            "explanation": "ok", "evidence": [], "processing_stages": [],
            "_router_detail": {"trust_score": 0.5, "threshold": 0.4, "all_agree": True, "votes": []},
        }

    import phase3.aman_pipeline as real_pipeline
    monkeypatch.setattr(real_pipeline, "verify_content", _fake_verify_content)

    resp = client.post("/api/verify", json={"text": "a" * 5000})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error handling -> clean 503, never a raw traceback
# ---------------------------------------------------------------------------


def test_verify_provider_failure_returns_clean_503(client, monkeypatch):
    """Simulates verify_content() raising (model load failure, LLM
    provider unreachable, missing retrieval index, etc.) -- must come back
    as a clean JSON 503, never a raw Python traceback."""

    def _raising_verify_content(claim_text, evidence_text=""):
        raise RuntimeError("simulated: LLM provider unreachable at http://localhost:11434")

    import phase3.aman_pipeline as real_pipeline
    monkeypatch.setattr(real_pipeline, "verify_content", _raising_verify_content)

    resp = client.post("/api/verify", json={"text": "خبر"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "verification_failed"
    assert "Traceback" not in body["detail"]
    assert "RuntimeError" not in body["detail"]
    assert "11434" not in body["detail"]  # no internal detail leaked


# ---------------------------------------------------------------------------
# Timeout -> clean 504
# ---------------------------------------------------------------------------


def test_verify_timeout_returns_504(client, monkeypatch):
    """Simulates the escalation path exceeding the configured timeout."""

    def _slow_verify_content(claim_text, evidence_text=""):
        time.sleep(0.5)
        return {}

    import phase3.aman_pipeline as real_pipeline
    monkeypatch.setattr(real_pipeline, "verify_content", _slow_verify_content)
    monkeypatch.setattr(backend_main, "VERIFY_TIMEOUT_SECONDS", 0.05)

    resp = client.post("/api/verify", json={"text": "خبر"})
    assert resp.status_code == 504
    assert resp.json()["error"] == "verification_timeout"


# ---------------------------------------------------------------------------
# LLM_PROVIDER env-var dispatch (phase3/llm_provider.py additions)
# ---------------------------------------------------------------------------


class TestLLMProviderEnvSwitch:
    def test_default_env_falls_back_to_mock_when_ollama_unreachable(self, monkeypatch):
        from phase3 import llm_provider
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setattr(llm_provider, "is_ollama_reachable", lambda *a, **k: False)
        provider = llm_provider.get_default_provider()
        assert isinstance(provider, llm_provider.MockProvider)

    def test_default_env_uses_ollama_when_reachable(self, monkeypatch):
        from phase3 import llm_provider
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setattr(llm_provider, "is_ollama_reachable", lambda *a, **k: True)
        provider = llm_provider.get_default_provider()
        assert isinstance(provider, llm_provider.OllamaProvider)

    def test_explicit_mock_env(self, monkeypatch):
        from phase3 import llm_provider
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        provider = llm_provider.get_default_provider()
        assert isinstance(provider, llm_provider.MockProvider)

    def test_vllm_env_constructs_vllm_provider(self, monkeypatch):
        from phase3 import llm_provider
        monkeypatch.setenv("LLM_PROVIDER", "vllm")
        monkeypatch.setenv("VLLM_ENDPOINT", "http://example-rocm-host:8000")
        provider = llm_provider.get_default_provider()
        assert isinstance(provider, llm_provider.VLLMProvider)
        assert provider.endpoint == "http://example-rocm-host:8000"

    def test_vllm_provider_without_endpoint_fails_honestly_not_silently(self, monkeypatch):
        """No AMD/vLLM endpoint exists in this sandbox -- the provider must
        raise rather than pretend to produce a real model response."""
        from phase3 import llm_provider
        provider = llm_provider.VLLMProvider(endpoint="")
        with pytest.raises(RuntimeError):
            provider.generate("test prompt")


# ---------------------------------------------------------------------------
# Opt-in: one real, slow, end-to-end call through the live pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AMAN_RUN_SLOW_TESTS") != "1",
    reason=(
        "Pays the real one-time ~30-90s SVM/BiLSTM/MARBERT model-load cost "
        "plus real inference; set AMAN_RUN_SLOW_TESTS=1 to run it."
    ),
)
def test_real_pipeline_end_to_end_through_live_api():
    """No stubbing: exercises the actual phase3.aman_pipeline.verify_content()
    through the real HTTP app, including the real startup warm-up."""
    with TestClient(backend_main.app) as c:
        health_resp = c.get("/health")
        assert health_resp.json()["pipeline_loaded"] is True

        resp = c.post("/api/verify", json={"text": "اختبار حقيقي لخط الأنابيب الكامل"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in {"VERIFIED_AUTOMATICALLY", "VERIFIED_WITH_EVIDENCE", "UNCERTAIN"}
        assert isinstance(body["processing_stages"], list) and len(body["processing_stages"]) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
