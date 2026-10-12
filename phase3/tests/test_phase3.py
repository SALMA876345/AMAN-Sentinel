"""Phase 3 automated checks. Run with:
    python -m unittest phase3.tests.test_phase3 -v

Uses MockProvider (deterministic, no network/model dependency) for all
LLM-schema/safeguard tests, so this suite runs fast and reproducibly
regardless of whether Ollama is reachable. Retrieval tests use the
already-built, cached index (phase3/data/retrieval_index/) so they don't
rebuild it. Each test is explicitly skipped (not silently passed) if its
required artifact is missing.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # Code files/
RETRIEVAL_INDEX = ROOT / "phase3" / "data" / "retrieval_index"
ROUTER_RESULTS = ROOT / "phase2" / "reports" / "router_results.json"


def _retrieval_index_present() -> bool:
    return (RETRIEVAL_INDEX / "corpus.csv").exists()


class RetrievalTests(unittest.TestCase):
    @unittest.skipUnless(_retrieval_index_present(), "run phase3/evaluate_retrieval.py (or retrieve_evidence once) first")
    def test_bm25_retrieval_returns_evidence_pack(self):
        from phase3.retrieval import retrieve_evidence
        pack = retrieve_evidence("الحكومة تعلن عن قرار جديد", top_k=3, method="bm25")
        self._assert_valid_evidence_pack(pack, expected_method="bm25")

    @unittest.skipUnless(_retrieval_index_present(), "run phase3/evaluate_retrieval.py (or retrieve_evidence once) first")
    def test_dense_retrieval_returns_evidence_pack(self):
        from phase3.retrieval import retrieve_evidence
        pack = retrieve_evidence("الحكومة تعلن عن قرار جديد", top_k=3, method="dense")
        self._assert_valid_evidence_pack(pack, expected_method="dense")

    @unittest.skipUnless(_retrieval_index_present(), "run phase3/evaluate_retrieval.py (or retrieve_evidence once) first")
    def test_hybrid_rrf_retrieval_returns_evidence_pack(self):
        from phase3.retrieval import retrieve_evidence
        pack = retrieve_evidence("الحكومة تعلن عن قرار جديد", top_k=3, method="hybrid_rrf")
        self._assert_valid_evidence_pack(pack, expected_method="hybrid_rrf")

    @unittest.skipUnless(_retrieval_index_present(), "run phase3/evaluate_retrieval.py (or retrieve_evidence once) first")
    def test_evidence_pack_schema(self):
        from phase3.retrieval import retrieve_evidence
        pack = retrieve_evidence("خبر تجريبي للاختبار", top_k=5, method="hybrid_rrf")
        self.assertIn("claim", pack)
        self.assertIn("retrieved_evidence", pack)
        self.assertLessEqual(len(pack["retrieved_evidence"]), 5)
        for item in pack["retrieved_evidence"]:
            for key in ("id", "text", "source", "retrieval_method", "retrieval_score"):
                self.assertIn(key, item)
            self.assertIsInstance(item["retrieval_score"], float)

    def _assert_valid_evidence_pack(self, pack: dict, expected_method: str) -> None:
        self.assertIn("retrieved_evidence", pack)
        for item in pack["retrieved_evidence"]:
            self.assertEqual(item["retrieval_method"], expected_method)
            self.assertIsInstance(item["id"], str)


class LLMSchemaTests(unittest.TestCase):
    def test_llm_output_schema_valid(self):
        from phase3.llm_provider import MockProvider
        from phase3.verification_llm import verify_claim_against_evidence

        evidence_pack = {
            "claim": "اختبار الخبر",
            "retrieved_evidence": [
                {"id": "e1", "text": "نص دليل يدعم اختبار الخبر بوضوح", "source": "test",
                 "retrieval_method": "bm25", "retrieval_score": 1.0},
            ],
        }
        provider = MockProvider()
        provider.set_context("اختبار الخبر", ["e1"])
        result = verify_claim_against_evidence("اختبار الخبر", evidence_pack, provider)

        for key in ("verdict", "confidence", "explanation", "evidence_ids", "claim_language"):
            self.assertIn(key, result)
        self.assertIn(result["verdict"], {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"})
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn(result["claim_language"], {"ar", "en", "mixed"})

    def test_invalid_citation_rejected(self):
        """A provider that cites an id NOT in the evidence pack must be
        caught and downgraded to INSUFFICIENT_EVIDENCE, never passed
        through."""
        from phase3.llm_provider import LLMProvider
        from phase3.verification_llm import verify_claim_against_evidence

        class _HallucinatingProvider(LLMProvider):
            def generate(self, prompt: str) -> str:
                return json.dumps({
                    "verdict": "SUPPORTED", "confidence": 0.9,
                    "explanation": "cites a fabricated id",
                    "evidence_ids": ["this_id_does_not_exist_in_the_pack"],
                    "claim_language": "ar",
                })

        evidence_pack = {
            "claim": "خبر",
            "retrieved_evidence": [
                {"id": "real_id_1", "text": "نص دليل حقيقي", "source": "test",
                 "retrieval_method": "bm25", "retrieval_score": 1.0},
            ],
        }
        result = verify_claim_against_evidence("خبر", evidence_pack, _HallucinatingProvider())
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("this_id_does_not_exist_in_the_pack", result["evidence_ids"])

    def test_insufficient_evidence_when_no_citation(self):
        """A SUPPORTED/CONTRADICTED verdict with an empty evidence_ids list
        must be force-downgraded -- a verdict without a citation is not
        acceptable per the brief's hallucination safeguards."""
        from phase3.llm_provider import LLMProvider
        from phase3.verification_llm import verify_claim_against_evidence

        class _UncitedProvider(LLMProvider):
            def generate(self, prompt: str) -> str:
                return json.dumps({
                    "verdict": "CONTRADICTED", "confidence": 0.8,
                    "explanation": "no citation provided",
                    "evidence_ids": [], "claim_language": "ar",
                })

        evidence_pack = {
            "claim": "خبر", "retrieved_evidence": [
                {"id": "e1", "text": "نص", "source": "test", "retrieval_method": "bm25", "retrieval_score": 1.0},
            ],
        }
        result = verify_claim_against_evidence("خبر", evidence_pack, _UncitedProvider())
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_arabic_claim_language_detection(self):
        from phase3.verification_llm import detect_language
        self.assertEqual(detect_language("هذا خبر تجريبي باللغة العربية فقط"), "ar")

    def test_english_claim_language_detection(self):
        from phase3.verification_llm import detect_language
        self.assertEqual(detect_language("this is a test claim written entirely in english"), "en")

    def test_mixed_language_claim_detection(self):
        from phase3.verification_llm import detect_language
        result = detect_language("breaking news: حريق كبير in Dubai today")
        self.assertEqual(result, "mixed")

    def test_malformed_json_falls_back_to_insufficient_evidence(self):
        from phase3.llm_provider import LLMProvider
        from phase3.verification_llm import verify_claim_against_evidence

        class _BrokenProvider(LLMProvider):
            def generate(self, prompt: str) -> str:
                return "this is not json at all, no braces here"

        evidence_pack = {"claim": "خبر", "retrieved_evidence": []}
        result = verify_claim_against_evidence("خبر", evidence_pack, _BrokenProvider())
        self.assertEqual(result["verdict"], "INSUFFICIENT_EVIDENCE")


class Phase2RoutingTests(unittest.TestCase):
    @unittest.skipUnless(ROUTER_RESULTS.exists(), "Phase 2 router_results.json not found")
    def test_router_threshold_loadable(self):
        from phase3.router_decision import load_default_threshold
        threshold = load_default_threshold()
        self.assertIsInstance(threshold, float)

    def test_high_trust_score_routes_to_verified(self):
        from phase3.router_decision import route, VERIFIED_PREDICTION
        self.assertEqual(route(0.99, threshold=0.4), VERIFIED_PREDICTION)

    def test_low_trust_score_routes_to_escalate(self):
        from phase3.router_decision import route, ESCALATE_FOR_DEEP_VERIFICATION
        self.assertEqual(route(0.01, threshold=0.4), ESCALATE_FOR_DEEP_VERIFICATION)


class EndToEndTests(unittest.TestCase):
    """Exercises the full phase3.aman_pipeline.verify_content() contract
    using MockProvider substituted for the real LLM, so this test is fast
    and has no network dependency. The real-provider path is exercised
    separately in phase3/evaluate_deep_verification.py (25 real held-out
    examples; see phase3/reports/deep_verification_eval_summary.json)."""

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "live_inference.py").exists(),
        "phase3/live_inference.py not found",
    )
    def test_verify_content_output_schema_trusted_path(self):
        """A claim the ensemble agrees on with high confidence should take
        the trusted path and never touch retrieval/LLM at all."""
        import phase3.aman_pipeline as pipeline

        class _StubEnsemble:
            def score(self, claim_text, evidence_text=""):
                return {
                    "votes": [{"model": "svm", "prediction": "Real", "confidence": 0.99, "raw_confidence": 0.99}],
                    "all_agree": True, "svm_prediction": "Real", "svm_confidence": 0.99,
                    "novelty_score": 0.01, "novelty_category": "FAMILIAR", "trust_score": 0.99,
                }

        original_get_ensemble, original_get_threshold = pipeline._get_ensemble, pipeline._get_threshold
        pipeline._get_ensemble = lambda: _StubEnsemble()
        pipeline._get_threshold = lambda: 0.4
        try:
            result = pipeline.verify_content("مثال خبر", "مثال دليل")
        finally:
            pipeline._get_ensemble, pipeline._get_threshold = original_get_ensemble, original_get_threshold

        self.assertEqual(result["status"], "VERIFIED_AUTOMATICALLY")
        self.assertFalse(result["deep_verification_used"])
        self.assertEqual(result["evidence"], [])
        for key in ("status", "classification", "confidence", "novelty_score", "novelty_category",
                    "deep_verification_used", "evidence_verdict", "explanation", "evidence"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
