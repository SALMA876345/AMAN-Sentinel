"""Phase 1 Step 6: train the clean SVM baseline on the CLEAN HACKATHON SPLIT.

Reproduces the existing SVM methodology from
experiments/final_studies_classical.py as closely as possible:

    claim_norm + " [SEP] " + evidence_norm
        -> FeatureUnion( word TF-IDF (1,2)-grams, char_wb TF-IDF (3,5)-grams )
        -> LinearSVC

C is tuned over {0.1, 0.25, 0.5, 1.0, 2.0, 4.0} x class_weight in
{None, "balanced"}, selected by (macro_f1, mcc, accuracy) lexicographic
order on the VALIDATION split only — the test split is never touched
during model selection. The final model is refit on train+validation with
the selected hyperparameters, matching the historical script's
train_final_model() step.

This trains on hackathon/data/clean_split/source_binary_*.csv, NOT on
data/final_studies/source_binary_*.csv (the historical, leak-affected
split). See hackathon/leakage_checks.py for the measured difference.

Persists, under artifacts/svm/ (a NEW directory — no historical result
file is overwritten):
    vectorizer.joblib        fitted FeatureUnion
    model.joblib              fitted LinearSVC
    config.json               hyperparameters + feature config
    label_mapping.json        class label <-> index
    split_metadata.json       row counts, fingerprints of the exact
                              train/validation/test files used
    validation_sweep.json     full hyperparameter sweep (auditable)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics, set_seed  # noqa: E402

CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
ARTIFACT_DIR = ROOT / "artifacts" / "svm"
SEED = 42
TASK = "binary"
LABELS = ["Real", "Fake"]
C_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
CLASS_WEIGHTS = [None, "balanced"]


def fingerprint(df: pd.DataFrame) -> str:
    payload = "\n".join(sorted(df["sample_id"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_split(name: str) -> pd.DataFrame:
    path = CLEAN_DIR / f"source_{TASK}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run hackathon/build_clean_split.py first."
        )
    df = pd.read_csv(path, low_memory=False).fillna("")
    df["model_text"] = df["claim_norm"].astype(str) + " [SEP] " + df["evidence_norm"].astype(str)
    return df


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
            max_features=70_000, sublinear_tf=True, strip_accents=None,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=90_000, sublinear_tf=True,
        )),
    ])


def predict_with_confidence(model: LinearSVC, matrix) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(matrix)
    scores = model.decision_function(matrix)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    from scipy.special import softmax
    confidence = softmax(scores, axis=1).max(axis=1)
    return predictions, confidence


def main() -> None:
    set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    print(f"Clean split rows -> train={len(train):,}  validation={len(validation):,}  test={len(test):,}")

    vectorizer = build_vectorizer()
    train_matrix = vectorizer.fit_transform(train["model_text"])
    validation_matrix = vectorizer.transform(validation["model_text"])

    sweep = []
    best = None
    for class_weight in CLASS_WEIGHTS:
        for c_value in C_VALUES:
            model = LinearSVC(C=c_value, class_weight=class_weight, max_iter=10_000, random_state=SEED)
            model.fit(train_matrix, train["task_label"])
            predictions = model.predict(validation_matrix)
            metrics = compute_classification_metrics(validation["task_label"], predictions, LABELS)
            row = {
                "c": c_value, "class_weight": class_weight or "none",
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "mcc": metrics["mcc"],
            }
            sweep.append(row)
            score = (metrics["macro_f1"], metrics["mcc"], metrics["accuracy"])
            if best is None or score > best["score"]:
                best = {"score": score, "c": c_value, "class_weight": class_weight, "validation_metrics": metrics}
            print(f"  C={c_value:<5} class_weight={class_weight or 'none':<9} "
                  f"val_acc={metrics['accuracy']:.4f} val_macroF1={metrics['macro_f1']:.4f} val_mcc={metrics['mcc']:.4f}")

    print(f"\nSelected (validation-only): C={best['c']}  class_weight={best['class_weight'] or 'none'}  "
          f"val_macro_f1={best['validation_metrics']['macro_f1']:.4f}")

    # Refit on train + validation with the selected hyperparameters
    final_train = pd.concat([train, validation], ignore_index=True)
    final_vectorizer = build_vectorizer()
    final_matrix = final_vectorizer.fit_transform(final_train["model_text"])
    final_model = LinearSVC(C=best["c"], class_weight=best["class_weight"], max_iter=10_000, random_state=SEED)
    final_model.fit(final_matrix, final_train["task_label"])

    # Persist
    joblib.dump(final_vectorizer, ARTIFACT_DIR / "vectorizer.joblib")
    joblib.dump(final_model, ARTIFACT_DIR / "model.joblib")

    config = {
        "model": "LinearSVC",
        "selected_C": best["c"],
        "selected_class_weight": best["class_weight"] or "none",
        "max_iter": 10_000,
        "random_state": SEED,
        "task": TASK,
        "labels": LABELS,
        "input_format": "claim_norm + ' [SEP] ' + evidence_norm",
        "feature_union": [
            {"name": "word", "analyzer": "word", "ngram_range": [1, 2], "min_df": 2, "max_df": 0.995, "max_features": 70000, "sublinear_tf": True},
            {"name": "char", "analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2, "max_features": 90000, "sublinear_tf": True},
        ],
        "selection_criterion": "lexicographic (macro_f1, mcc, accuracy) on validation only",
        "refit_policy": "train + validation with selected hyperparameters (test never used for selection)",
        "confidence_derivation": (
            "softmax(decision_function) — NOT a calibrated probability. "
            "LinearSVC has no predict_proba; this is a monotonic transform "
            "of the signed distance to the separating hyperplane, scaled to "
            "sum to 1 across classes for reporting convenience only."
        ),
        "data_source": "hackathon/data/clean_split/ (leakage-checked), NOT data/final_studies/ (historical, leak-affected)",
    }
    (ARTIFACT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    label_mapping = {label: idx for idx, label in enumerate(sorted(LABELS))}
    (ARTIFACT_DIR / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")

    split_metadata = {
        "task": TASK,
        "source_files": {
            "train": str(CLEAN_DIR / f"source_{TASK}_train.csv"),
            "validation": str(CLEAN_DIR / f"source_{TASK}_validation.csv"),
            "test": str(CLEAN_DIR / f"source_{TASK}_test.csv"),
        },
        "row_counts": {"train": len(train), "validation": len(validation), "test": len(test),
                       "final_train_(train+validation)": len(final_train)},
        "fingerprints_sha256": {
            "train": fingerprint(train), "validation": fingerprint(validation), "test": fingerprint(test),
        },
    }
    (ARTIFACT_DIR / "split_metadata.json").write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "validation_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    print(f"\nArtifacts saved under {ARTIFACT_DIR}")
    for f in sorted(ARTIFACT_DIR.glob("*")):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
