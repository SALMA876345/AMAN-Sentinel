"""Phase 3 Step 11: evaluate the deep-verification pipeline on a bounded,
fixed-seed sample of real ESCALATED rows from the TEST split.

Reuses Phase 2's already-computed routing decisions
(phase2/reports/router_decisions_test.csv) to find which TEST rows
actually escalate -- no need to re-run the SVM/BiLSTM/MARBERT ensemble
live for this; that path is already verified correct in
phase3/live_inference.py and tested separately. This script only
exercises the NEW Phase 3 components: retrieval + LLM verification.

These are genuinely held-out examples: the verification prompt itself was
designed and iterated against VALIDATION-split examples only (see
phase3/reports/llm_smoketest_results.json), never against TEST. This
script is the one and only pass over TEST for Phase 3.

Metrics reported (Section 11 of the brief):
  - retrieval success rate (>=1 evidence item returned)
  - structured-output validity rate (valid JSON matching the schema)
  - evidence citation validity rate (every cited id exists in the pack)
  - unsupported-citation rate (fraction of citations that referenced a
    nonexistent id, before the safeguard corrected it)
  - insufficient-evidence rate
  - end-to-end success rate (pipeline completed without an exception)
  - verdict correctness, restricted to rows where the LLM committed to
    SUPPORTED/CONTRADICTED (compared against the row's real true_label;
    INSUFFICIENT_EVIDENCE rows are not scored for correctness -- they are
    an abstention, not a wrong answer)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

from phase3.retrieval import retrieve_evidence  # noqa: E402
from phase3.verification_llm import verify_claim_against_evidence  # noqa: E402
from phase3.llm_provider import get_default_provider  # noqa: E402

CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
ROUTER_DECISIONS = ROOT / "phase2" / "reports" / "router_decisions_test.csv"
OUT_DIR = ROOT / "phase3" / "reports"
SAMPLE_SIZE = 25
SEED = 42

VERDICT_TO_LABEL = {"SUPPORTED": "Real", "CONTRADICTED": "Fake"}


def main() -> None:
    decisions = pd.read_csv(ROUTER_DECISIONS, low_memory=False)
    escalated = decisions[decisions["route_decision"] == "ESCALATE_FOR_DEEP_VERIFICATION"]
    print(f"Test rows that escalate: {len(escalated):,} / {len(decisions):,} "
          f"({len(escalated) / len(decisions) * 100:.1f}%)")

    rng = np.random.RandomState(SEED)
    sample_ids = rng.choice(escalated["sample_id"].to_numpy(), size=min(SAMPLE_SIZE, len(escalated)), replace=False)

    test = pd.read_csv(CLEAN_DIR / "source_binary_test.csv", low_memory=False).fillna("").set_index("sample_id")

    provider = get_default_provider()
    print(f"Using provider: {type(provider).__name__}")

    rows = []
    t_start = time.time()
    for i, sid in enumerate(sample_ids, start=1):
        row = test.loc[sid]
        claim_text, true_label = row["claim_text"], row["task_label"]
        t0 = time.time()
        record = {"sample_id": sid, "true_label": true_label, "claim_preview": claim_text[:60]}
        try:
            evidence_pack = retrieve_evidence(claim_text, top_k=5, method="hybrid_rrf")
            n_evidence = len(evidence_pack["retrieved_evidence"])
            record["retrieval_success"] = n_evidence > 0
            record["n_evidence_retrieved"] = n_evidence

            result = verify_claim_against_evidence(claim_text, evidence_pack, provider)
            record["end_to_end_success"] = True
            record["structured_output_valid"] = isinstance(result, dict) and "verdict" in result
            record["verdict"] = result.get("verdict")
            record["confidence"] = result.get("confidence")
            record["evidence_ids_cited"] = result.get("evidence_ids", [])
            valid_ids = {e["id"] for e in evidence_pack["retrieved_evidence"]}
            cited = set(record["evidence_ids_cited"])
            record["citation_validity"] = cited.issubset(valid_ids)
            record["safeguard_triggered"] = "SAFEGUARD" in str(result.get("explanation", ""))
            record["elapsed_seconds"] = round(time.time() - t0, 1)

            if record["verdict"] in VERDICT_TO_LABEL:
                predicted_label = VERDICT_TO_LABEL[record["verdict"]]
                record["verdict_correct"] = predicted_label == true_label
            else:
                record["verdict_correct"] = None  # not scored for INSUFFICIENT_EVIDENCE
        except Exception as exc:  # noqa: BLE001
            record["end_to_end_success"] = False
            record["error"] = str(exc)[:200]
        rows.append(record)
        print(f"  [{i}/{len(sample_ids)}] {sid}  verdict={record.get('verdict')}  "
              f"elapsed={record.get('elapsed_seconds', '?')}s")

    results = pd.DataFrame(rows)
    total_elapsed = time.time() - t_start

    n = len(results)
    completed = results[results["end_to_end_success"] == True]  # noqa: E712
    summary = {
        "sample_size": n,
        "escalated_fraction_of_test": float(len(escalated) / len(decisions)),
        "provider_used": type(provider).__name__,
        "end_to_end_success_rate": float((results["end_to_end_success"] == True).mean()),  # noqa: E712
        "retrieval_success_rate": float(completed["retrieval_success"].mean()) if len(completed) else None,
        "structured_output_validity_rate": float(completed["structured_output_valid"].mean()) if len(completed) else None,
        "citation_validity_rate": float(completed["citation_validity"].mean()) if len(completed) else None,
        "safeguard_triggered_rate": float(completed["safeguard_triggered"].mean()) if len(completed) else None,
        "insufficient_evidence_rate": float((completed["verdict"] == "INSUFFICIENT_EVIDENCE").mean()) if len(completed) else None,
        "verdict_distribution": completed["verdict"].value_counts().to_dict() if len(completed) else {},
        "verdict_accuracy_when_committed": (
            float(completed["verdict_correct"].dropna().mean())
            if len(completed) and completed["verdict_correct"].notna().any() else None
        ),
        "n_committed_verdicts_scored": int(completed["verdict_correct"].notna().sum()) if len(completed) else 0,
        "total_wall_clock_seconds": round(total_elapsed, 1),
        "sample_seed": SEED,
        "note": (
            "Sample drawn from TEST-split rows the Phase 2 router already escalated "
            "(phase2/reports/router_decisions_test.csv), fixed seed, size capped at "
            f"{SAMPLE_SIZE} for hackathon time budget. verdict_accuracy_when_committed "
            "excludes INSUFFICIENT_EVIDENCE rows (abstention, not a wrong answer) by design."
        ),
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "deep_verification_eval_examples.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "deep_verification_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {OUT_DIR / 'deep_verification_eval_examples.csv'}")
    print(f"Saved: {OUT_DIR / 'deep_verification_eval_summary.json'}")


if __name__ == "__main__":
    main()
