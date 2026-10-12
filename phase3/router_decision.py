"""Phase 3: live routing decision, reusing Phase 2's established
trust_score threshold exactly (fit on VALIDATION only in Phase 2; not
refit here -- a live request must use a threshold already chosen without
ever seeing this request).

Default operating point: 90% target coverage, matching
phase2/router.py's own stated default. Threshold value read directly
from phase2/reports/router_results.json (Phase 2's output), not
recomputed, so a live decision cannot drift from the audited Phase 2
number.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Code files/
ROUTER_RESULTS = ROOT / "phase2" / "reports" / "router_results.json"

VERIFIED_PREDICTION = "VERIFIED_PREDICTION"
ESCALATE_FOR_DEEP_VERIFICATION = "ESCALATE_FOR_DEEP_VERIFICATION"


def load_default_threshold() -> float:
    results = json.loads(ROUTER_RESULTS.read_text(encoding="utf-8"))
    return float(results["coverage_sweep"]["target_90pct"]["trust_score_threshold"])


def route(trust_score: float, threshold: float | None = None) -> str:
    if threshold is None:
        threshold = load_default_threshold()
    return VERIFIED_PREDICTION if trust_score >= threshold else ESCALATE_FOR_DEEP_VERIFICATION


if __name__ == "__main__":
    threshold = load_default_threshold()
    print(f"Default (90% target coverage) trust_score threshold: {threshold:.4f}")
    for example_score in [0.9, 0.5, 0.3, 0.0, -0.2]:
        print(f"  trust_score={example_score:+.2f} -> {route(example_score, threshold)}")
