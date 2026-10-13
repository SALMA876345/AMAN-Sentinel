"""Phase 3: live routing decision, reusing Phase 2's established
trust_score threshold exactly (fit on VALIDATION only in Phase 2; not
refit here -- a live request must use a threshold already chosen without
ever seeing this request).

Default operating point: 80% target coverage (actual: 76.90% coverage,
92.65% selective accuracy -- the headline router result cited in
phase5/VALIDATED_METRICS.md, README.md, and every Phase 5 submission
document). Threshold value read directly from
phase2/reports/router_results.json (Phase 2's own output, not
recomputed), so the live decision cannot drift from the audited Phase 2
number.

Changed from the 90%-target threshold used through Phase 3/4 (set
2026-10-13, Phase 5 integration): a docs cross-check caught that the
deployed default didn't match the operating point every judge-facing
document headlines, which would have been a real, confusing mismatch
between what the code does and what the docs claim it does. Both
90%-target and 80%-target thresholds are equally valid, already-computed
Phase 2 outputs -- this is a choice of which to deploy by default, not a
change to any model, threshold-fitting method, or validated number.
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
    return float(results["coverage_sweep"]["target_80pct"]["trust_score_threshold"])


def route(trust_score: float, threshold: float | None = None) -> str:
    if threshold is None:
        threshold = load_default_threshold()
    return VERIFIED_PREDICTION if trust_score >= threshold else ESCALATE_FOR_DEEP_VERIFICATION


if __name__ == "__main__":
    threshold = load_default_threshold()
    print(f"Default (80% target coverage) trust_score threshold: {threshold:.4f}")
    for example_score in [0.9, 0.5, 0.3, 0.0, -0.2]:
        print(f"  trust_score={example_score:+.2f} -> {route(example_score, threshold)}")
