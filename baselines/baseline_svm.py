"""
SVM baseline — claim-level fake-news classification.

Pure text classifier: TF-IDF over the claim text -> LinearSVC.
No evidence, no heuristics — this is the claim-only baseline that the
evidence-based RAG / FAAV models are compared against.

Runs in 2-class (Real/Fake) or 3-class (Real/Fake/Unverified) mode:
    CLASS_MODE=2 python baselines/baseline_svm.py
    CLASS_MODE=3 python baselines/baseline_svm.py   (default)
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET,
    CLASS_MODE, CLASS_LABELS, result_tagged,
)

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, cohen_kappa_score,
)

OUTPUT_FILE  = result_tagged("svm_claim_results.csv")
METRICS_FILE = result_tagged("svm_metrics.txt")
N_CAP        = 15_000   # per-class cap


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"=== SVM baseline — {CLASS_MODE}-class {CLASS_LABELS} ===")

    # ── Train ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(TRAIN_FILE, usecols=["text", "label"], low_memory=False)
    df["content"] = df["text"].fillna("").apply(normalize_arabic)
    df = df[(df["content"].str.strip() != "") & df["label"].isin(CLASS_LABELS)]
    df = pd.concat([
        g.sample(min(len(g), N_CAP), random_state=42)
        for _, g in df.groupby("label")
    ]).reset_index(drop=True)
    print(f"Training rows: {len(df):,}  |  {df['label'].value_counts().to_dict()}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   LinearSVC(max_iter=3000, C=1.0, class_weight="balanced")),
    ])
    print("Training SVM ...")
    model.fit(df["content"], df["label"])

    # ── Test ───────────────────────────────────────────────────────────────────
    dt = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
    dt = dt[dt["gold_label"].isin(CLASS_LABELS)].reset_index(drop=True)
    dt["claim_norm"] = dt["claim_ar"].fillna("").apply(normalize_arabic)
    dt["svm_prediction"] = model.predict(dt["claim_norm"])
    dt["is_correct"] = dt["gold_label"] == dt["svm_prediction"]

    y_true, y_pred = dt["gold_label"], dt["svm_prediction"]
    acc   = accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wf1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    print(f"\nAccuracy={acc:.4f}  MacroF1={mf1:.4f}  MCC={mcc:.4f}")
    print(report)

    dt.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(f"SVM Baseline Results ({CLASS_MODE}-class)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Classes  : {CLASS_LABELS}\n")
        f.write(f"Test rows: {len(dt)}\n\n")
        f.write(f"Accuracy        : {acc:.4f}\n")
        f.write(f"Macro F1        : {mf1:.4f}\n")
        f.write(f"Weighted F1     : {wf1:.4f}\n")
        f.write(f"MCC             : {mcc:.4f}\n")
        f.write(f"Cohen Kappa     : {kappa:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

    print(f"\nResults -> {OUTPUT_FILE}")
    print(f"Metrics -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
