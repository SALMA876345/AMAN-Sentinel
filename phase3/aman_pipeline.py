"""Phase 3 integration point: `verify_content(text, evidence_text="")`.

This is the single Python entry point Phase 4 (API/frontend) will call.
It wires together everything built across all three phases:

  Phase 1: SVM (production classifier, macro F1 0.8640)
  Phase 2: BiLSTM + MARBERT-probe (disagreement signals only, not
           delivered as verdicts), novelty detector, router trust_score
  Phase 3: evidence retrieval (BM25/dense/hybrid) + evidence-grounded LLM
           verification, invoked ONLY when the router escalates

Routing is never bypassed: every input goes through
phase3.live_inference.LiveModelEnsemble + phase3.router_decision.route
first. The LLM is the escalation layer, not the primary classifier, per
AMAN_PHASE2_INTELLIGENCE_REPORT.md's explicit recommendation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

from phase3.live_inference import LiveModelEnsemble  # noqa: E402
from phase3.router_decision import (  # noqa: E402
    ESCALATE_FOR_DEEP_VERIFICATION,
    VERIFIED_PREDICTION,
    load_default_threshold,
    route,
)

_ensemble: Optional[LiveModelEnsemble] = None
_threshold: Optional[float] = None


def _get_ensemble() -> LiveModelEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = LiveModelEnsemble()
    return _ensemble


def _get_threshold() -> float:
    global _threshold
    if _threshold is None:
        _threshold = load_default_threshold()
    return _threshold


def verify_content(claim_text: str, evidence_text: str = "") -> dict:
    """Returns the final AMAN result schema:

    {
      "status": "VERIFIED_AUTOMATICALLY" | "VERIFIED_WITH_EVIDENCE" | "UNCERTAIN",
      "classification": "Real" | "Fake" | null,
      "confidence": float,
      "novelty_score": float,
      "novelty_category": str,
      "deep_verification_used": bool,
      "evidence_verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE" | null,
      "explanation": str,
      "evidence": [ ... ]  # Evidence Pack entries actually cited, empty if not escalated
    }
    """
    stages = ["analyzing_language", "running_classifiers", "checking_confidence", "checking_novelty"]
    scoring = _get_ensemble().score(claim_text, evidence_text)
    threshold = _get_threshold()
    decision = route(scoring["trust_score"], threshold)

    if decision == VERIFIED_PREDICTION:
        stages.append("preparing_result")
        return {
            "status": "VERIFIED_AUTOMATICALLY",
            "classification": scoring["svm_prediction"],
            "confidence": round(scoring["svm_confidence"], 4),
            "novelty_score": round(scoring["novelty_score"], 4),
            "novelty_category": scoring["novelty_category"],
            "deep_verification_used": False,
            "evidence_verdict": None,
            "explanation": (
                f"Classified as {scoring['svm_prediction']} by AMAN's primary classifier "
                f"with sufficient confidence and model agreement; no deep verification needed."
            ),
            "evidence": [],
            "processing_stages": stages,
            "_router_detail": {
                "trust_score": round(scoring["trust_score"], 4),
                "threshold": round(threshold, 4),
                "all_agree": scoring["all_agree"],
                "votes": scoring["votes"],
            },
        }

    # ESCALATE_FOR_DEEP_VERIFICATION: evidence retrieval + LLM reasoning.
    # Imports are local so the trusted path above never pays for loading
    # the retrieval index or an LLM provider unless escalation actually
    # happens.
    from phase3.retrieval import retrieve_evidence
    from phase3.verification_llm import verify_claim_against_evidence
    from phase3.llm_provider import get_default_provider

    stages.append("retrieving_evidence")
    # hybrid_rrf won on Recall@5/@10/MRR in phase3/reports/retrieval_evaluation.json
    evidence_pack = retrieve_evidence(claim_text, top_k=5, method="hybrid_rrf")
    stages.append("deep_verification")
    provider = get_default_provider()
    llm_result = verify_claim_against_evidence(claim_text, evidence_pack, provider)
    stages.append("preparing_result")

    evidence_verdict = llm_result["verdict"]
    if evidence_verdict == "SUPPORTED":
        status, classification = "VERIFIED_WITH_EVIDENCE", "Real"
    elif evidence_verdict == "CONTRADICTED":
        status, classification = "VERIFIED_WITH_EVIDENCE", "Fake"
    else:
        status, classification = "UNCERTAIN", None

    cited_ids = set(llm_result.get("evidence_ids", []))
    cited_evidence = [e for e in evidence_pack["retrieved_evidence"] if e["id"] in cited_ids]

    return {
        "status": status,
        "classification": classification,
        "confidence": round(float(llm_result.get("confidence", 0.0)), 4),
        "novelty_score": round(scoring["novelty_score"], 4),
        "novelty_category": scoring["novelty_category"],
        "deep_verification_used": True,
        "evidence_verdict": evidence_verdict,
        "explanation": llm_result.get("explanation", ""),
        "evidence": cited_evidence,
        "processing_stages": stages,
        "_router_detail": {
            "trust_score": round(scoring["trust_score"], 4),
            "threshold": round(threshold, 4),
            "all_agree": scoring["all_agree"],
            "votes": scoring["votes"],
        },
    }


if __name__ == "__main__":
    import json
    result = verify_content(
        "وصول ميسي والمنتخب الأرجنتيني إلى قطر",
        "صحيفة الخليج الخليج الرياضي",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
