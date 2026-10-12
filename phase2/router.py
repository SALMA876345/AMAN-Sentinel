"""Phase 2 Step 8: AMAN decision router (experimental).

Produces VERIFIED_PREDICTION or ESCALATE_FOR_DEEP_VERIFICATION for every
row, ranked by a single composite TRUST SCORE built from:
  - calibrated confidence of the strongest available individual model
    (the dominant term -- the original FAAV project's own audit found
    hand-built multi-signal gates lose to plain softmax/calibrated
    confidence in 22/24 comparisons, so confidence is not diluted by
    averaging it with other signals; agreement and novelty instead
    PENALIZE the score, demoting risky rows without hard-excluding them)
  - a fixed penalty when models disagree (disagreement makes a row
    strictly less trustworthy than any confidence value alone implies --
    see phase2/reports/disagreement_results.json: 2-of-3 agreement rows
    are only 64.9% accurate vs 92.6% when all 3 agree)
  - a graduated penalty for novelty category (HIGHLY NOVEL penalized more
    than MODERATELY NOVEL, FAMILIAR unpenalized)

Earlier version of this script used agreement/novelty as HARD pass/fail
filters before ranking by confidence. That is a methodological bug, not a
design choice: the hard filters alone reject ~36% of rows unconditionally,
making 70/80/90% target coverage mathematically unreachable regardless of
confidence threshold (measured actual coverage was 40-48% against those
targets). The composite-score version below ranks ALL rows on one
continuous scale, so any coverage target from 0-100% is reachable by
moving the threshold -- this is the standard selective-prediction
construction (and matches the shape of the original FAAV project's own
risk-coverage curves in improvements/selective_evaluation.py).

Threshold selection discipline: the composite-score threshold needed to
hit a target coverage level (100/90/80/70%) is the (1 - target_coverage)
quantile of the VALIDATION split's composite-score distribution -- no
validation OR test LABELS are used to pick it. The threshold is applied
once to the TEST split to report selective accuracy and risk.

Requires (reads only, writes nothing elsewhere):
  phase2/reports/{svm,bilstm,transformer}_predictions_{validation,test}[_calibrated].csv
  artifacts/novelty/{validation,test}_scores.csv
Gracefully degrades to whichever models are actually present.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Code files/
REPORTS_DIR = ROOT / "phase2" / "reports"
NOVELTY_DIR = ROOT / "artifacts" / "novelty"
COVERAGE_TARGETS = [1.00, 0.90, 0.80, 0.70]

MODEL_FILES = {
    "svm": {
        "validation": (REPORTS_DIR / "svm_predictions_validation_calibrated.csv", REPORTS_DIR / "svm_predictions_validation.csv"),
        "test": (REPORTS_DIR / "svm_predictions_test_calibrated.csv", REPORTS_DIR / "svm_predictions_test.csv"),
    },
    "bilstm": {
        "validation": (REPORTS_DIR / "bilstm_predictions_validation_calibrated.csv", REPORTS_DIR / "bilstm_predictions_validation.csv"),
        "test": (REPORTS_DIR / "bilstm_predictions_test_calibrated.csv", REPORTS_DIR / "bilstm_predictions_test.csv"),
    },
    "transformer": {
        "validation": (REPORTS_DIR / "transformer_predictions_validation_calibrated.csv", REPORTS_DIR / "transformer_predictions_validation.csv"),
        "test": (REPORTS_DIR / "transformer_predictions_test_calibrated.csv", REPORTS_DIR / "transformer_predictions_test.csv"),
    },
}


def load_model(name: str, split: str) -> pd.DataFrame | None:
    calibrated_path, raw_path = MODEL_FILES[name][split]
    path = calibrated_path if calibrated_path.exists() else raw_path
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False).set_index("sample_id")
    if "calibrated_confidence" not in df.columns:
        df["calibrated_confidence"] = df["confidence"]
    return df


def build_frame(split: str, model_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    loaded = {n: load_model(n, split) for n in model_names}
    available = [n for n, df in loaded.items() if df is not None]
    common_idx = loaded[available[0]].index
    for n in available[1:]:
        common_idx = common_idx.intersection(loaded[n].index)

    novelty_path = NOVELTY_DIR / f"{split}_scores.csv"
    novelty = pd.read_csv(novelty_path, low_memory=False).set_index("sample_id") if novelty_path.exists() else None
    if novelty is not None:
        common_idx = common_idx.intersection(novelty.index)

    frame = pd.DataFrame(index=common_idx)
    frame["true_label"] = loaded[available[0]].loc[common_idx, "true_label"]
    for n in available:
        frame[f"{n}_pred"] = loaded[n].loc[common_idx, "prediction"]
        frame[f"{n}_conf"] = loaded[n].loc[common_idx, "calibrated_confidence"]

    pred_cols = [f"{n}_pred" for n in available]
    frame["all_agree"] = frame[pred_cols].nunique(axis=1) == 1

    # Primary model = highest mean calibrated confidence among available models
    # on this split (a simple, defensible, data-driven choice rather than a
    # hardcoded preference).
    best_name = max(available, key=lambda n: frame[f"{n}_conf"].mean())
    frame["primary_model"] = best_name
    frame["primary_pred"] = frame[f"{best_name}_pred"]
    frame["primary_conf"] = frame[f"{best_name}_conf"]

    if novelty is not None:
        frame["novelty_score"] = novelty.loc[common_idx, "novelty_score"]
        frame["novelty_category"] = novelty.loc[common_idx, "novelty_category"]
    else:
        frame["novelty_score"] = np.nan
        frame["novelty_category"] = "FAMILIAR"  # no penalty if unavailable, rather than silently excluding everything

    novelty_penalty = frame["novelty_category"].map(
        {"FAMILIAR": 0.0, "MODERATELY NOVEL": 0.10, "HIGHLY NOVEL": 0.30}
    ).fillna(0.0)
    disagreement_penalty = np.where(frame["all_agree"], 0.0, 0.50)
    frame["trust_score"] = frame["primary_conf"] - disagreement_penalty - novelty_penalty

    return frame, available


def selective_metrics(frame: pd.DataFrame, accepted: np.ndarray) -> dict:
    n_total = len(frame)
    n_accepted = int(accepted.sum())
    coverage = n_accepted / n_total if n_total else 0.0
    if n_accepted == 0:
        return {"coverage": 0.0, "selective_accuracy": float("nan"), "selective_risk": float("nan"), "n_accepted": 0}
    correct = (frame.loc[accepted, "primary_pred"] == frame.loc[accepted, "true_label"]).mean()
    return {
        "coverage": float(coverage), "selective_accuracy": float(correct),
        "selective_risk": float(1 - correct), "n_accepted": n_accepted,
    }


def main() -> None:
    model_names = ["svm", "bilstm", "transformer"]
    val_frame, val_available = build_frame("validation", model_names)
    test_frame, test_available = build_frame("test", model_names)
    print(f"Router using models: {val_available} (validation), {test_available} (test)")
    print(f"Validation rows where all models agree: {val_frame['all_agree'].mean()*100:.1f}%  "
          f"(not a hard filter -- a penalty term in the composite trust_score)")

    results = {}
    thresholds = {}
    for target_coverage in COVERAGE_TARGETS:
        if target_coverage >= 1.0:
            threshold = -np.inf
        else:
            threshold = float(np.quantile(val_frame["trust_score"], 1 - target_coverage))
        thresholds[target_coverage] = threshold

        test_accepted = test_frame["trust_score"].to_numpy() >= threshold
        metrics = selective_metrics(test_frame, test_accepted)
        results[f"target_{int(target_coverage * 100)}pct"] = {
            "target_coverage": target_coverage, "trust_score_threshold": threshold, **metrics,
        }
        print(f"  target_coverage={target_coverage:.0%}  threshold={threshold:.4f}  "
              f"actual_coverage={metrics['coverage']:.4f}  selective_accuracy={metrics['selective_accuracy']:.4f}  "
              f"selective_risk={metrics['selective_risk']:.4f}  n_accepted={metrics['n_accepted']}")

    # Full per-row routing decision at the 90% target (a reasonable default
    # operating point), persisted for inspection / the test suite.
    default_threshold = thresholds[0.90]
    test_frame["route_decision"] = np.where(
        test_frame["trust_score"] >= default_threshold,
        "VERIFIED_PREDICTION", "ESCALATE_FOR_DEEP_VERIFICATION",
    )
    out_cols = ["true_label", "primary_model", "primary_pred", "primary_conf",
                "all_agree", "novelty_category", "trust_score", "route_decision"]
    test_frame[out_cols].to_csv(REPORTS_DIR / "router_decisions_test.csv", encoding="utf-8-sig")

    out = {
        "models_used": test_available, "primary_model_selection": "highest mean calibrated confidence per split",
        "composite_score": "trust_score = primary_calibrated_confidence - disagreement_penalty(0.50) - novelty_penalty(0.10/0.30)",
        "threshold_selection": "quantile of VALIDATION trust_score distribution over ALL rows; no labels used",
        "default_operating_point": "90% target coverage",
        "coverage_sweep": results,
    }
    out_path = REPORTS_DIR / "router_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {REPORTS_DIR / 'router_decisions_test.csv'}")


if __name__ == "__main__":
    main()
