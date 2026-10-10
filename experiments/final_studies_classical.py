"""Run LR and Linear SVM for the final three-study evaluation.

Study 1: source-only training -> complete UAE task dataset (zero-shot).
Study 2: source-only training -> frozen source test (source-domain).
Study 3: source + UAE training -> same frozen UAE test (adaptation).

Each study is evaluated as binary claim veracity and as a three-class
faithfulness-aware final decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics, set_seed


DATA_DIR = ROOT / "data" / "final_studies"
RESULTS_DIR = ROOT / "results" / "final_studies"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SEED = 42
TASKS = {
    "binary": {
        "label_col": "binary_label",
        "labels": ["Real", "Fake"],
    },
    "three_class": {
        "label_col": "three_class_label",
        "labels": ["Real", "Fake", "Unverified"],
    },
}
C_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
CLASS_WEIGHTS = [None, "balanced"]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", low_memory=False).fillna("")


def filter_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    label_col = TASKS[task]["label_col"]
    labels = TASKS[task]["labels"]
    out = df[df[label_col].isin(labels)].copy()
    out["label"] = out[label_col]
    out["model_text"] = (
        out["claim_norm"].astype(str)
        + " [SEP] "
        + out["evidence_norm"].astype(str)
    )
    return out.reset_index(drop=True)


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=70_000,
                    sublinear_tf=True,
                    strip_accents=None,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=90_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_model(model_name: str, c_value: float, class_weight: str | None):
    if model_name == "Logistic Regression":
        base = LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight=class_weight,
            max_iter=2_000,
            random_state=SEED,
        )
        return OneVsRestClassifier(base)
    if model_name == "Linear SVM":
        return LinearSVC(
            C=c_value,
            class_weight=class_weight,
            max_iter=10_000,
            random_state=SEED,
        )
    raise ValueError(f"Unknown model: {model_name}")


def predict_with_confidence(model, matrix) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(matrix)
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(matrix).max(axis=1)
    else:
        scores = model.decision_function(matrix)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        confidence = softmax(scores, axis=1).max(axis=1)
    return predictions, confidence


def tune_model(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    vectorizer: FeatureUnion,
) -> tuple[dict, list[dict]]:
    train_matrix = vectorizer.fit_transform(train["model_text"])
    validation_matrix = vectorizer.transform(validation["model_text"])
    sweep = []
    best = None
    for class_weight in CLASS_WEIGHTS:
        for c_value in C_VALUES:
            model = build_model(model_name, c_value, class_weight)
            model.fit(train_matrix, train["label"])
            predictions = model.predict(validation_matrix)
            metrics = compute_classification_metrics(
                validation["label"],
                predictions,
                labels=sorted(train["label"].unique().tolist()),
            )
            row = {
                "c": c_value,
                "class_weight": class_weight or "none",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mcc": metrics["mcc"],
            }
            sweep.append(row)
            score = (metrics["macro_f1"], metrics["mcc"], metrics["accuracy"])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "c": c_value,
                    "class_weight": class_weight,
                    "validation_metrics": metrics,
                }
    if best is None:
        raise RuntimeError(f"No hyperparameter candidate completed for {model_name}")
    return best, sweep


def fingerprint(df: pd.DataFrame) -> str:
    payload = "\n".join(sorted(df["sample_id"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_and_save(
    *,
    model_name: str,
    study: str,
    task: str,
    model,
    vectorizer: FeatureUnion,
    test: pd.DataFrame,
    tuning: dict,
    sweep: list[dict],
    train_rows: int,
    validation_rows: int,
) -> dict:
    labels = TASKS[task]["labels"]
    matrix = vectorizer.transform(test["model_text"])
    predictions, confidence = predict_with_confidence(model, matrix)
    metrics = compute_classification_metrics(test["label"], predictions, labels)

    output = test[
        [
            "sample_id",
            "dataset_family",
            "source",
            "claim_text",
            "evidence_text",
            "label",
        ]
    ].copy()
    output["prediction"] = predictions
    output["confidence"] = confidence
    output["is_correct"] = output["label"] == output["prediction"]
    slug = (
        f"{study}_{task}_{model_name.lower().replace(' ', '_')}"
        .replace("+", "plus")
    )
    output.to_csv(
        PREDICTIONS_DIR / f"{slug}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "study": study,
        "task": task,
        "model": model_name,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": int(len(test)),
        "test_fingerprint_sha256": fingerprint(test),
        "selected_hyperparameters": {
            "c": tuning["c"],
            "class_weight": tuning["class_weight"] or "none",
        },
        "validation_metrics": tuning["validation_metrics"],
        "test_metrics": metrics,
        "validation_sweep": sweep,
    }
    (RESULTS_DIR / f"{slug}_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = metrics["classification_report"]
    return {
        "Study": study,
        "Task": task,
        "Model": model_name,
        "Train N": train_rows,
        "Validation N": validation_rows,
        "Test N": int(len(test)),
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"],
        "MCC": metrics["mcc"],
        "Real F1": report.get("Real", {}).get("f1-score", np.nan),
        "Fake F1": report.get("Fake", {}).get("f1-score", np.nan),
        "Unverified F1": report.get("Unverified", {}).get(
            "f1-score", np.nan
        ),
        "Validation Macro F1": tuning["validation_metrics"]["macro_f1"],
        "Best C": tuning["c"],
        "Class Weight": tuning["class_weight"] or "none",
        "Test Fingerprint": fingerprint(test),
    }


def majority_result(
    *,
    study: str,
    task: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    labels = TASKS[task]["labels"]
    majority = train["label"].value_counts().idxmax()
    predictions = np.repeat(majority, len(test))
    metrics = compute_classification_metrics(test["label"], predictions, labels)
    report = metrics["classification_report"]
    return {
        "Study": study,
        "Task": task,
        "Model": f"Majority ({majority})",
        "Train N": int(len(train)),
        "Validation N": 0,
        "Test N": int(len(test)),
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"],
        "MCC": metrics["mcc"],
        "Real F1": report.get("Real", {}).get("f1-score", np.nan),
        "Fake F1": report.get("Fake", {}).get("f1-score", np.nan),
        "Unverified F1": report.get("Unverified", {}).get(
            "f1-score", np.nan
        ),
        "Validation Macro F1": np.nan,
        "Best C": np.nan,
        "Class Weight": "none",
        "Test Fingerprint": fingerprint(test),
    }


def result_from_metrics_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["test_metrics"]
    report = metrics["classification_report"]
    selected = payload["selected_hyperparameters"]
    return {
        "Study": payload["study"],
        "Task": payload["task"],
        "Model": payload["model"],
        "Train N": payload["train_rows"],
        "Validation N": payload["validation_rows"],
        "Test N": payload["test_rows"],
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"],
        "MCC": metrics["mcc"],
        "Real F1": report.get("Real", {}).get("f1-score", np.nan),
        "Fake F1": report.get("Fake", {}).get("f1-score", np.nan),
        "Unverified F1": report.get("Unverified", {}).get(
            "f1-score", np.nan
        ),
        "Validation Macro F1": payload["validation_metrics"]["macro_f1"],
        "Best C": selected["c"],
        "Class Weight": selected["class_weight"],
        "Test Fingerprint": payload["test_fingerprint_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=sorted(TASKS),
        help="Tasks to train in this invocation. Existing metric files are retained.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["source", "combined"],
        default=["source", "combined"],
        help="source creates Studies 1/2; combined creates Study 3.",
    )
    return parser.parse_args()


def train_final_model(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tuning: dict,
) -> tuple[FeatureUnion, object, pd.DataFrame]:
    final_train = pd.concat([train, validation], ignore_index=True)
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(final_train["model_text"])
    model = build_model(
        model_name,
        tuning["c"],
        tuning["class_weight"],
    )
    model.fit(matrix, final_train["label"])
    return vectorizer, model, final_train


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    raw = {
        f"{domain}_{task}_{part}": load_split(
            f"{domain}_{task}_{part}"
        )
        for task in TASKS
        for domain in ["source", "uae", "combined"]
        for part in ["train", "validation", "test"]
    }
    raw.update(
        {
            f"uae_{task}_full": load_split(f"uae_{task}_full")
            for task in TASKS
        }
    )
    majority_rows = []

    for task in args.tasks:
        source_train = filter_task(raw[f"source_{task}_train"], task)
        source_validation = filter_task(
            raw[f"source_{task}_validation"], task
        )
        source_test = filter_task(raw[f"source_{task}_test"], task)
        uae_test = filter_task(raw[f"uae_{task}_full"], task)

        if "source" in args.families:
            majority_rows.append(
                majority_result(
                    study="Study 1 - Zero-shot UAE",
                    task=task,
                    train=source_train,
                    test=uae_test,
                )
            )
            majority_rows.append(
                majority_result(
                    study="Study 2 - Source-domain",
                    task=task,
                    train=source_train,
                    test=source_test,
                )
            )

            for model_name in ["Logistic Regression", "Linear SVM"]:
                print(f"\nTuning {model_name}: source-only / {task}", flush=True)
                tuning, sweep = tune_model(
                    model_name,
                    source_train,
                    source_validation,
                    build_vectorizer(),
                )
                print(
                    f"  selected C={tuning['c']} "
                    f"class_weight={tuning['class_weight'] or 'none'} "
                    f"val_macro_f1={tuning['validation_metrics']['macro_f1']:.4f}",
                    flush=True,
                )
                vectorizer, model, final_train = train_final_model(
                    model_name,
                    source_train,
                    source_validation,
                    tuning,
                )
                evaluate_and_save(
                    model_name=model_name,
                    study="Study 1 - Zero-shot UAE",
                    task=task,
                    model=model,
                    vectorizer=vectorizer,
                    test=uae_test,
                    tuning=tuning,
                    sweep=sweep,
                    train_rows=len(final_train),
                    validation_rows=len(source_validation),
                )
                evaluate_and_save(
                    model_name=model_name,
                    study="Study 2 - Source-domain",
                    task=task,
                    model=model,
                    vectorizer=vectorizer,
                    test=source_test,
                    tuning=tuning,
                    sweep=sweep,
                    train_rows=len(final_train),
                    validation_rows=len(source_validation),
                )

        if "combined" in args.families:
            combined_train = filter_task(
                raw[f"combined_{task}_train"], task
            )
            combined_validation = filter_task(
                raw[f"combined_{task}_validation"], task
            )
            combined_test = filter_task(raw[f"combined_{task}_test"], task)
            majority_rows.append(
                majority_result(
                    study="Study 3 - Combined adaptation",
                    task=task,
                    train=combined_train,
                    test=combined_test,
                )
            )

            for model_name in ["Logistic Regression", "Linear SVM"]:
                print(f"\nTuning {model_name}: combined / {task}", flush=True)
                tuning, sweep = tune_model(
                    model_name,
                    combined_train,
                    combined_validation,
                    build_vectorizer(),
                )
                print(
                    f"  selected C={tuning['c']} "
                    f"class_weight={tuning['class_weight'] or 'none'} "
                    f"val_macro_f1={tuning['validation_metrics']['macro_f1']:.4f}",
                    flush=True,
                )
                vectorizer, model, final_train = train_final_model(
                    model_name,
                    combined_train,
                    combined_validation,
                    tuning,
                )
                evaluate_and_save(
                    model_name=model_name,
                    study="Study 3 - Combined adaptation",
                    task=task,
                    model=model,
                    vectorizer=vectorizer,
                    test=combined_test,
                    tuning=tuning,
                    sweep=sweep,
                    train_rows=len(final_train),
                    validation_rows=len(combined_validation),
                )

    all_majority_rows = []
    for task in TASKS:
        source_train = filter_task(raw[f"source_{task}_train"], task)
        source_test = filter_task(raw[f"source_{task}_test"], task)
        uae_test = filter_task(raw[f"uae_{task}_full"], task)
        combined_train = filter_task(
            raw[f"combined_{task}_train"], task
        )
        combined_test = filter_task(raw[f"combined_{task}_test"], task)
        all_majority_rows.extend(
            [
                majority_result(
                    study="Study 1 - Zero-shot UAE",
                    task=task,
                    train=source_train,
                    test=uae_test,
                ),
                majority_result(
                    study="Study 2 - Source-domain",
                    task=task,
                    train=source_train,
                    test=source_test,
                ),
                majority_result(
                    study="Study 3 - Combined adaptation",
                    task=task,
                    train=combined_train,
                    test=combined_test,
                ),
            ]
        )
    metric_rows = [
        result_from_metrics_file(path)
        for path in RESULTS_DIR.glob("*_metrics.json")
        if path.name != "manifest_audit.json"
        and json.loads(path.read_text(encoding="utf-8")).get("model")
        in {"Logistic Regression", "Linear SVM"}
    ]
    comparison = pd.DataFrame(all_majority_rows + metric_rows)
    comparison = comparison.sort_values(
        ["Task", "Study", "Macro F1"],
        ascending=[True, True, False],
    )
    output_path = RESULTS_DIR / "classical_model_comparison.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\nFinal classical comparison:")
    print(
        comparison[
            ["Study", "Task", "Model", "Test N", "Accuracy", "Macro F1", "MCC"]
        ].to_string(index=False)
    )
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
