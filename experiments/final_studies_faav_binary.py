"""Evaluate binary FAAV-Light and FAAV-Full on the frozen final studies.

Binary FAAV always returns the AraBERT Real/Fake verdict and adds a separate
Verified/Unverified faithfulness status. Overall classification metrics
therefore remain directly comparable with the binary verifier, while coverage
and selective metrics quantify performance on verdicts accepted by the gate.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final_studies_faav import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    build_gate_vectorizer,
    fingerprint,
    verifier_predictions,
)
from utils.faav_common import (
    compute_classification_metrics,
    lexical_overlap,
    set_seed,
)


DATA_DIR = ROOT / "data" / "final_studies"
LABELS = ["Real", "Fake"]
SEED = 42
MIN_VALIDATION_COVERAGE = 0.80


def load_split(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv", low_memory=False).fillna("")
    frame = frame[frame["binary_label"].isin(LABELS)].copy()
    frame["label"] = frame["binary_label"]
    frame["model_text"] = (
        frame["claim_norm"].astype(str)
        + " [SEP] "
        + frame["evidence_norm"].astype(str)
    )
    frame["evidence_chars"] = frame["evidence_norm"].astype(str).str.len()
    frame["lexical_overlap"] = [
        lexical_overlap(claim, evidence)
        for claim, evidence in zip(
            frame["claim_norm"],
            frame["evidence_norm"],
        )
    ]
    return frame.reset_index(drop=True)


def selective_metrics(
    frame: pd.DataFrame,
    accepted: np.ndarray,
) -> dict:
    accepted = np.asarray(accepted, dtype=bool)
    coverage = float(accepted.mean())
    if not accepted.any():
        return {
            "coverage": coverage,
            "selective_accuracy": 0.0,
            "selective_macro_f1": 0.0,
            "selective_mcc": 0.0,
            "accepted_rows": 0,
        }
    metrics = compute_classification_metrics(
        frame.loc[accepted, "label"],
        frame.loc[accepted, "verifier_prediction"],
        LABELS,
    )
    return {
        "coverage": coverage,
        "selective_accuracy": metrics["accuracy"],
        "selective_macro_f1": metrics["macro_f1"],
        "selective_mcc": metrics["mcc"],
        "accepted_rows": int(accepted.sum()),
    }


def light_acceptance(
    frame: pd.DataFrame,
    *,
    confidence_threshold: float,
    overlap_threshold: float,
    evidence_char_threshold: int,
) -> np.ndarray:
    return (
        (frame["verifier_confidence"].values >= confidence_threshold)
        & (frame["lexical_overlap"].values >= overlap_threshold)
        & (frame["evidence_chars"].values >= evidence_char_threshold)
    )


def candidate_score(metrics: dict) -> tuple[float, float, float]:
    if metrics["coverage"] + 1e-12 < MIN_VALIDATION_COVERAGE:
        return (-1.0, -1.0, -1.0)
    return (
        metrics["selective_macro_f1"],
        metrics["selective_mcc"],
        metrics["coverage"],
    )


def tune_light(validation: pd.DataFrame) -> tuple[dict, list[dict]]:
    sweep = []
    best = None
    for confidence, overlap, evidence_chars in product(
        [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        [0, 20, 40, 80, 120],
    ):
        accepted = light_acceptance(
            validation,
            confidence_threshold=float(confidence),
            overlap_threshold=float(overlap),
            evidence_char_threshold=int(evidence_chars),
        )
        metrics = selective_metrics(validation, accepted)
        row = {
            "confidence_threshold": float(confidence),
            "overlap_threshold": float(overlap),
            "evidence_char_threshold": int(evidence_chars),
            **metrics,
        }
        sweep.append(row)
        score = candidate_score(metrics)
        if best is None or score > best["score"]:
            best = {**row, "score": score}
    if best is None or best["score"][0] < 0:
        raise RuntimeError("No FAAV-Light candidate met the coverage floor.")
    return best, sweep


def gate_text(frame: pd.DataFrame) -> pd.Series:
    return (
        "verdict_"
        + frame["verifier_prediction"].astype(str)
        + " [SEP] "
        + frame["model_text"].astype(str)
    )


def tune_full(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[object, LogisticRegression, dict, list[dict]]:
    train = train.copy()
    train["trustworthy"] = (
        train["verifier_prediction"].values == train["label"].values
    ).astype(int)
    if train["trustworthy"].nunique() < 2:
        raise RuntimeError(
            "FAAV-Full requires both correct and incorrect training verdicts."
        )

    vectorizer = build_gate_vectorizer()
    train_matrix = vectorizer.fit_transform(gate_text(train))
    validation_matrix = vectorizer.transform(gate_text(validation))

    sweep = []
    best = None
    for c_value in [0.1, 0.5, 1.0, 2.0, 4.0]:
        gate = LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
        )
        gate.fit(train_matrix, train["trustworthy"])
        trust_probability = gate.predict_proba(validation_matrix)[:, 1]
        for gate_threshold, confidence_threshold in product(
            [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
            [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        ):
            accepted = (
                (trust_probability >= gate_threshold)
                & (
                    validation["verifier_confidence"].values
                    >= confidence_threshold
                )
            )
            metrics = selective_metrics(validation, accepted)
            row = {
                "c": float(c_value),
                "gate_threshold": float(gate_threshold),
                "confidence_threshold": float(confidence_threshold),
                **metrics,
            }
            sweep.append(row)
            score = candidate_score(metrics)
            if best is None or score > best["score"]:
                best = {**row, "score": score}
    if best is None or best["score"][0] < 0:
        raise RuntimeError("No FAAV-Full candidate met the coverage floor.")

    selected_gate = LogisticRegression(
        C=best["c"],
        solver="liblinear",
        class_weight="balanced",
        max_iter=2_000,
        random_state=SEED,
    )
    selected_gate.fit(train_matrix, train["trustworthy"])
    return vectorizer, selected_gate, best, sweep


def full_acceptance(
    frame: pd.DataFrame,
    vectorizer,
    gate: LogisticRegression,
    settings: dict,
) -> tuple[np.ndarray, np.ndarray]:
    trust_probability = gate.predict_proba(
        vectorizer.transform(gate_text(frame))
    )[:, 1]
    accepted = (
        (trust_probability >= settings["gate_threshold"])
        & (
            frame["verifier_confidence"].values
            >= settings["confidence_threshold"]
        )
    )
    return accepted, trust_probability


def save_result(
    *,
    study: str,
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    accepted: np.ndarray,
    settings: dict,
    sweep: list[dict],
    gate_probability: np.ndarray | None = None,
) -> dict:
    predictions = test["verifier_prediction"].to_numpy()
    metrics = compute_classification_metrics(test["label"], predictions, LABELS)
    selective = selective_metrics(test, accepted)
    output = test[
        [
            "sample_id",
            "dataset_family",
            "source",
            "claim_text",
            "evidence_text",
            "label",
            "verifier_prediction",
            "verifier_confidence",
            "lexical_overlap",
            "evidence_chars",
        ]
    ].copy()
    if gate_probability is not None:
        output["gate_trust_probability"] = gate_probability
    output["prediction"] = predictions
    output["faithfulness_status"] = np.where(
        accepted,
        "Verified",
        "Unverified",
    )
    output["accepted"] = accepted
    output["is_correct"] = output["label"] == output["prediction"]

    file_slug = (
        f"{study}_binary_{model_name.lower().replace(' ', '_')}"
        .replace("+", "plus")
    )
    output.to_csv(
        PREDICTIONS_DIR / f"{file_slug}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "study": study,
        "task": "binary",
        "model": model_name,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_fingerprint_sha256": fingerprint(test),
        "settings": {
            **settings,
            "minimum_validation_coverage": MIN_VALIDATION_COVERAGE,
            "binary_output_protocol": (
                "Real/Fake verdict plus separate Verified/Unverified status"
            ),
        },
        "validation_sweep": sweep,
        "test_metrics": metrics,
        "abstention_rate": 0.0,
        "faithfulness_status_unverified_rate": 1.0 - selective["coverage"],
        **selective,
        "feature_leakage_statement": (
            "test labels and gold retrieval-quality/faithfulness fields were "
            "not used by the gate"
        ),
    }
    (RESULTS_DIR / f"{file_slug}_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = metrics["classification_report"]
    return {
        "Study": study,
        "Task": "binary",
        "Model": model_name,
        "Train N": len(train),
        "Validation N": len(validation),
        "Test N": len(test),
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["macro_f1"],
        "MCC": metrics["mcc"],
        "Coverage": selective["coverage"],
        "Selective Accuracy": selective["selective_accuracy"],
        "Selective Macro F1": selective["selective_macro_f1"],
        "Selective MCC": selective["selective_mcc"],
        "Real F1": report["Real"]["f1-score"],
        "Fake F1": report["Fake"]["f1-score"],
        "Test Fingerprint": fingerprint(test),
    }


def run_family(
    family: str,
    raw: dict[str, pd.DataFrame],
) -> list[dict]:
    if family == "source":
        train = raw["source_binary_train"]
        validation = raw["source_binary_validation"]
        tests = {
            "Study 1 - Zero-shot UAE": raw["uae_binary_full"],
            "Study 2 - Source-domain": raw["source_binary_test"],
        }
        checkpoint = MODELS_DIR / "arabert_source_binary"
    else:
        train = raw["combined_binary_train"]
        validation = raw["combined_binary_validation"]
        tests = {
            "Study 3 - Combined adaptation": raw["combined_binary_test"],
        }
        checkpoint = MODELS_DIR / "arabert_combined_binary"

    inference_frames = {
        "train": train,
        "validation": validation,
        **tests,
    }
    inferred = verifier_predictions(checkpoint, inference_frames)
    train_inferred = inferred.pop("train")
    validation_inferred = inferred.pop("validation")

    light_settings, light_sweep = tune_light(validation_inferred)
    vectorizer, gate, full_settings, full_sweep = tune_full(
        train_inferred,
        validation_inferred,
    )
    print(
        f"{family}: Light validation coverage="
        f"{light_settings['coverage']:.4f}, selective macro F1="
        f"{light_settings['selective_macro_f1']:.4f}; Full coverage="
        f"{full_settings['coverage']:.4f}, selective macro F1="
        f"{full_settings['selective_macro_f1']:.4f}",
        flush=True,
    )

    rows = []
    for study, test in inferred.items():
        light_accepted = light_acceptance(
            test,
            confidence_threshold=light_settings["confidence_threshold"],
            overlap_threshold=light_settings["overlap_threshold"],
            evidence_char_threshold=light_settings["evidence_char_threshold"],
        )
        light_row = save_result(
            study=study,
            model_name="FAAV-Light",
            train=train,
            validation=validation,
            test=test,
            accepted=light_accepted,
            settings={
                key: value
                for key, value in light_settings.items()
                if key != "score"
            },
            sweep=light_sweep,
        )
        rows.append(light_row)

        full_accepted, trust_probability = full_acceptance(
            test,
            vectorizer,
            gate,
            full_settings,
        )
        full_row = save_result(
            study=study,
            model_name="FAAV-Full",
            train=train,
            validation=validation,
            test=test,
            accepted=full_accepted,
            settings={
                key: value
                for key, value in full_settings.items()
                if key != "score"
            },
            sweep=full_sweep,
            gate_probability=trust_probability,
        )
        rows.append(full_row)
        print(
            f"  {study}: Light coverage={light_row['Coverage']:.4f}, "
            f"selective macro F1={light_row['Selective Macro F1']:.4f}; "
            f"Full coverage={full_row['Coverage']:.4f}, "
            f"selective macro F1={full_row['Selective Macro F1']:.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        f"{domain}_binary_{part}": load_split(f"{domain}_binary_{part}")
        for domain in ["source", "uae", "combined"]
        for part in ["train", "validation", "test"]
    }
    raw["uae_binary_full"] = load_split("uae_binary_full")
    rows = run_family("source", raw) + run_family("combined", raw)
    comparison = pd.DataFrame(rows).sort_values(
        ["Study", "Model"],
        ascending=[True, True],
    )
    output = RESULTS_DIR / "faav_binary_model_comparison.csv"
    comparison.to_csv(output, index=False, encoding="utf-8-sig")
    print("\nBinary FAAV comparison:")
    print(
        comparison[
            [
                "Study",
                "Model",
                "Test N",
                "Accuracy",
                "Macro F1",
                "MCC",
                "Coverage",
                "Selective Accuracy",
                "Selective Macro F1",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
