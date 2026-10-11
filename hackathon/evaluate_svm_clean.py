"""Phase 1 Step 7: evaluate the persisted clean SVM on the clean test split
ONLY, after model selection is already complete (train_svm_clean.py never
touches test). Reports Accuracy, Precision, Recall, F1, Macro F1, confusion
matrix, and per-class metrics, then prints the historical-vs-clean
comparison required by the brief — with an explicit caveat that the
historical number may be inflated by the evidence leakage documented in
hackathon/reports/leakage_check_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
ARTIFACT_DIR = ROOT / "artifacts" / "svm"
HISTORICAL_METRICS_FILE = (
    ROOT / "results" / "final_studies"
    / "Study 2 - Source-domain_binary_linear_svm_metrics.json"
)
TASK = "binary"
LABELS = ["Real", "Fake"]


def main() -> None:
    vectorizer_path = ARTIFACT_DIR / "vectorizer.joblib"
    model_path = ARTIFACT_DIR / "model.joblib"
    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError("Run hackathon/train_svm_clean.py first — artifacts not found.")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    test_path = CLEAN_DIR / f"source_{TASK}_test.csv"
    test = pd.read_csv(test_path, low_memory=False).fillna("")
    test["model_text"] = test["claim_norm"].astype(str) + " [SEP] " + test["evidence_norm"].astype(str)

    matrix = vectorizer.transform(test["model_text"])
    predictions = model.predict(matrix)
    scores = model.decision_function(matrix)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    from scipy.special import softmax
    confidence = softmax(scores, axis=1).max(axis=1)

    y_true, y_pred = test["task_label"].values, predictions
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("=" * 70)
    print(f"CLEAN SVM — test set (hackathon/data/clean_split/source_{TASK}_test.csv)")
    print("=" * 70)
    print(f"Test rows        : {len(test):,}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Macro Precision  : {macro_p:.4f}")
    print(f"Macro Recall     : {macro_r:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")
    print(f"MCC              : {mcc:.4f}")
    print(f"\nPer-class:")
    for label in LABELS:
        r = report[label]
        print(f"  {label:<6} precision={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1-score']:.4f}  support={int(r['support'])}")
    print(f"\nConfusion matrix (rows=true, cols=pred, order={LABELS}):")
    print(cm)

    clean_metrics = {
        "task": TASK, "labels": LABELS, "test_rows": int(len(test)),
        "accuracy": float(acc), "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1), "mcc": float(mcc),
        "classification_report": report, "confusion_matrix": cm.tolist(),
        "confidence_derivation_note": (
            "confidence = softmax(decision_function).max(axis=1): a monotonic "
            "transform of the signed margin, NOT a calibrated probability. "
            "LinearSVC has no predict_proba."
        ),
        "data_source": "hackathon/data/clean_split/ — leakage-checked clean split, NOT the historical split",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "clean_evaluation_metrics.json").write_text(
        json.dumps(clean_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # per-row predictions, for the sanity-check script and for audit
    output = test[["sample_id", "source", "claim_text", "evidence_text", "task_label"]].copy()
    output["prediction"] = predictions
    output["confidence"] = confidence
    output["is_correct"] = output["task_label"] == output["prediction"]
    output.to_csv(ARTIFACT_DIR / "clean_test_predictions.csv", index=False, encoding="utf-8-sig")

    # Historical comparison
    print("\n" + "=" * 70)
    print("HISTORICAL vs CLEAN — Linear SVM, Study-2-equivalent (source-domain, binary)")
    print("=" * 70)
    if HISTORICAL_METRICS_FILE.exists():
        hist = json.loads(HISTORICAL_METRICS_FILE.read_text(encoding="utf-8"))
        hm = hist["test_metrics"]
        print(f"{'Metric':<18}{'Historical':>14}{'Clean':>14}{'Delta':>14}")
        for key, hist_key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("MCC", "mcc"), ("Weighted F1", "weighted_f1")]:
            hv = hm[hist_key]
            cv = {"accuracy": acc, "macro_f1": macro_f1, "mcc": mcc, "weighted_f1": weighted_f1}[hist_key]
            print(f"{key:<18}{hv:>14.4f}{cv:>14.4f}{cv - hv:>+14.4f}")
        print(f"\nHistorical test rows: {hist['test_rows']:,}   Clean test rows: {len(test):,}")
        comparison = {
            "historical": {k: hm[v] for k, v in [("accuracy", "accuracy"), ("macro_f1", "macro_f1"), ("mcc", "mcc"), ("weighted_f1", "weighted_f1")]},
            "historical_test_rows": hist["test_rows"],
            "clean": {"accuracy": acc, "macro_f1": macro_f1, "mcc": mcc, "weighted_f1": weighted_f1},
            "clean_test_rows": int(len(test)),
            "caveat": (
                "The historical number comes from data/final_studies/, a split grouped "
                "by claim only. The Phase 1 audit measured 57.2% of its source_binary_test "
                "rows reusing evidence text seen verbatim in training (84.7% train-majority-"
                "label agreement on that shared evidence), so the historical figure may be "
                "inflated by an evidence-side shortcut rather than reflecting claim "
                "understanding. The clean figure is measured on a split with zero claim, "
                "evidence, and claim-evidence-pair overlap (see leakage_check_results.json). "
                "Any drop from historical to clean is consistent with — not proof of — the "
                "leakage hypothesis; other differences (different exact row set, different "
                "component sizes after grouping) also contribute."
            ),
        }
        (ARTIFACT_DIR / "historical_vs_clean_comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print(f"Historical metrics file not found: {HISTORICAL_METRICS_FILE}")

    print(f"\nSaved: {ARTIFACT_DIR / 'clean_evaluation_metrics.json'}")
    print(f"Saved: {ARTIFACT_DIR / 'clean_test_predictions.csv'}")


if __name__ == "__main__":
    main()
