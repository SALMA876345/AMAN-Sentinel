"""Evaluate FAAV-Light and FAAV-Full on the frozen three-class studies.

FAAV-Light applies a validation-tuned deterministic gate using evidence length,
lexical overlap, and binary-verifier confidence.

FAAV-Full trains a TF-IDF evidence-usability gate and combines it with the
binary AraBERT verifier. Gold retrieval-quality and faithfulness fields are
never used as model inputs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import (
    compute_classification_metrics,
    lexical_overlap,
    set_seed,
)


DATA_DIR = ROOT / "data" / "final_studies"
RESULTS_DIR = ROOT / "results" / "final_studies"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["Real", "Fake", "Unverified"]
SEED = 42


def load_split(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv", low_memory=False).fillna("")
    frame = frame[frame["three_class_label"].isin(LABELS)].copy()
    frame["label"] = frame["three_class_label"]
    frame["model_text"] = (
        frame["claim_norm"].astype(str)
        + " [SEP] "
        + frame["evidence_norm"].astype(str)
    )
    frame["evidence_chars"] = frame["evidence_norm"].astype(str).str.len()
    frame["lexical_overlap"] = [
        lexical_overlap(claim, evidence)
        for claim, evidence in zip(
            frame["claim_norm"], frame["evidence_norm"]
        )
    ]
    return frame.reset_index(drop=True)


def fingerprint(frame: pd.DataFrame) -> str:
    payload = "\n".join(sorted(frame["sample_id"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def transformer_loader(tokenizer, frame: pd.DataFrame, batch_size: int = 64):
    encoded = tokenizer(
        frame["claim_norm"].astype(str).str[:400].tolist(),
        frame["evidence_norm"].astype(str).str[:1200].tolist(),
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if "token_type_ids" in encoded:
        tensors.append(encoded["token_type_ids"])
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size)


def verifier_predictions(
    checkpoint: Path,
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(DEVICE)
    id_to_label = {
        int(index): label for index, label in model.config.id2label.items()
    }
    model.eval()
    outputs = {}
    for name, frame in frames.items():
        probabilities = []
        with torch.inference_mode():
            for batch in transformer_loader(tokenizer, frame):
                inputs = {
                    "input_ids": batch[0].to(DEVICE),
                    "attention_mask": batch[1].to(DEVICE),
                }
                if len(batch) == 3:
                    inputs["token_type_ids"] = batch[2].to(DEVICE)
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=DEVICE.type == "cuda",
                ):
                    logits = model(**inputs).logits
                probabilities.append(
                    torch.softmax(logits.float(), dim=1).cpu().numpy()
                )
        probability = np.vstack(probabilities)
        predicted_indices = probability.argmax(axis=1)
        augmented = frame.copy()
        for index, label in id_to_label.items():
            augmented[f"verifier_{label.lower()}_probability"] = probability[
                :, index
            ]
        augmented["verifier_confidence"] = probability.max(axis=1)
        augmented["verifier_prediction"] = [
            id_to_label[int(index)] for index in predicted_indices
        ]
        outputs[name] = augmented
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def light_predictions(
    frame: pd.DataFrame,
    *,
    confidence_threshold: float,
    overlap_threshold: float,
    evidence_char_threshold: int,
) -> np.ndarray:
    accept = (
        (frame["verifier_confidence"] >= confidence_threshold)
        & (frame["lexical_overlap"] >= overlap_threshold)
        & (frame["evidence_chars"] >= evidence_char_threshold)
        & (frame["verifier_prediction"] != "Unverified")
    )
    return np.where(
        frame["verifier_prediction"] == "Unverified",
        "Unverified",
        np.where(accept, frame["verifier_prediction"], "Unverified"),
    )


def tune_light(validation: pd.DataFrame) -> tuple[dict, list[dict]]:
    sweep = []
    best = None
    for confidence, overlap, evidence_chars in product(
        [0.0, 0.50, 0.60, 0.70, 0.80, 0.90],
        [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
        [0, 20, 40, 80, 120],
    ):
        predictions = light_predictions(
            validation,
            confidence_threshold=float(confidence),
            overlap_threshold=overlap,
            evidence_char_threshold=evidence_chars,
        )
        metrics = compute_classification_metrics(
            validation["label"], predictions, LABELS
        )
        row = {
            "confidence_threshold": float(confidence),
            "overlap_threshold": overlap,
            "evidence_char_threshold": evidence_chars,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
        }
        sweep.append(row)
        score = (metrics["macro_f1"], metrics["mcc"], metrics["accuracy"])
        if best is None or score > best["score"]:
            best = {**row, "score": score}
    if best is None:
        raise RuntimeError("No FAAV-Light threshold candidate completed.")
    return best, sweep


def build_gate_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def tune_full(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[FeatureUnion, LogisticRegression, dict, list[dict]]:
    train = train.copy()
    validation = validation.copy()
    train["usable"] = (train["label"] != "Unverified").astype(int)
    validation["usable"] = (validation["label"] != "Unverified").astype(int)
    vectorizer = build_gate_vectorizer()
    train_matrix = vectorizer.fit_transform(train["model_text"])
    validation_matrix = vectorizer.transform(validation["model_text"])

    sweep = []
    best = None
    best_model = None
    for c_value in [0.1, 0.5, 1.0, 2.0, 4.0]:
        gate = LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
        )
        gate.fit(train_matrix, train["usable"])
        usable_probability = gate.predict_proba(validation_matrix)[:, 1]
        for gate_threshold, confidence_threshold in product(
            [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            [0.0, 0.50, 0.60, 0.70, 0.80, 0.90],
        ):
            accept = (
                (usable_probability >= gate_threshold)
                & (
                    validation["verifier_confidence"].values
                    >= confidence_threshold
                )
                & (
                    validation["verifier_prediction"].values
                    != "Unverified"
                )
            )
            predictions = np.where(
                validation["verifier_prediction"].values == "Unverified",
                "Unverified",
                np.where(
                    accept,
                    validation["verifier_prediction"],
                    "Unverified",
                ),
            )
            metrics = compute_classification_metrics(
                validation["label"], predictions, LABELS
            )
            row = {
                "c": c_value,
                "gate_threshold": float(gate_threshold),
                "confidence_threshold": float(confidence_threshold),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mcc": metrics["mcc"],
            }
            sweep.append(row)
            score = (metrics["macro_f1"], metrics["mcc"], metrics["accuracy"])
            if best is None or score > best["score"]:
                best = {**row, "score": score}
                best_model = gate
    if best is None or best_model is None:
        raise RuntimeError("No FAAV-Full candidate completed.")

    # Refit only the selected gate configuration; validation remains reserved
    # for threshold selection and is not added to gate training.
    selected_gate = LogisticRegression(
        C=best["c"],
        solver="liblinear",
        class_weight="balanced",
        max_iter=2_000,
        random_state=SEED,
    )
    selected_gate.fit(train_matrix, train["usable"])
    return vectorizer, selected_gate, best, sweep


def full_predictions(
    frame: pd.DataFrame,
    vectorizer: FeatureUnion,
    gate: LogisticRegression,
    settings: dict,
) -> tuple[np.ndarray, np.ndarray]:
    gate_probability = gate.predict_proba(
        vectorizer.transform(frame["model_text"])
    )[:, 1]
    accept = (
        gate_probability >= settings["gate_threshold"]
    ) & (
        frame["verifier_confidence"].values
        >= settings["confidence_threshold"]
    ) & (
        frame["verifier_prediction"].values != "Unverified"
    )
    predictions = np.where(
        frame["verifier_prediction"].values == "Unverified",
        "Unverified",
        np.where(
            accept,
            frame["verifier_prediction"],
            "Unverified",
        ),
    )
    return predictions, gate_probability


def save_faav_result(
    *,
    study: str,
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
    settings: dict,
    sweep: list[dict],
    gate_probability: np.ndarray | None = None,
) -> dict:
    metrics = compute_classification_metrics(test["label"], predictions, LABELS)
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
        output["gate_usable_probability"] = gate_probability
    output["prediction"] = predictions
    output["is_correct"] = output["label"] == output["prediction"]
    output["abstained"] = output["prediction"] == "Unverified"
    file_slug = (
        f"{study}_three_class_{model_name.lower().replace(' ', '_')}"
        .replace("+", "plus")
    )
    output.to_csv(
        PREDICTIONS_DIR / f"{file_slug}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "study": study,
        "task": "three_class",
        "model": model_name,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_fingerprint_sha256": fingerprint(test),
        "settings": settings,
        "validation_sweep": sweep,
        "test_metrics": metrics,
        "abstention_rate": float(np.mean(predictions == "Unverified")),
        "feature_leakage_statement": (
            "retrieval_quality and faithfulness_label were not used as inputs"
        ),
    }
    (RESULTS_DIR / f"{file_slug}_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = metrics["classification_report"]
    return {
        "Study": study,
        "Task": "three_class",
        "Model": model_name,
        "Train N": len(train),
        "Validation N": len(validation),
        "Test N": len(test),
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"],
        "MCC": metrics["mcc"],
        "Real F1": report["Real"]["f1-score"],
        "Fake F1": report["Fake"]["f1-score"],
        "Unverified F1": report["Unverified"]["f1-score"],
        "Abstention Rate": float(np.mean(predictions == "Unverified")),
        "Test Fingerprint": fingerprint(test),
    }


def run_family(
    family: str,
    raw: dict[str, pd.DataFrame],
) -> list[dict]:
    if family == "source":
        train = raw["source_three_class_train"]
        validation = raw["source_three_class_validation"]
        tests = {
            "Study 1 - Zero-shot UAE": raw["uae_three_class_full"],
            "Study 2 - Source-domain": raw["source_three_class_test"],
        }
        checkpoint = MODELS_DIR / "arabert_source_three_class"
    else:
        train = raw["combined_three_class_train"]
        validation = raw["combined_three_class_validation"]
        tests = {
            "Study 3 - Combined adaptation": raw[
                "combined_three_class_test"
            ]
        }
        checkpoint = MODELS_DIR / "arabert_combined_three_class"

    inference_frames = {"validation": validation, **tests}
    inferred = verifier_predictions(checkpoint, inference_frames)
    validation_inferred = inferred.pop("validation")

    light_settings, light_sweep = tune_light(validation_inferred)
    gate_vectorizer, gate, full_settings, full_sweep = tune_full(
        train,
        validation_inferred,
    )
    print(
        f"{family}: FAAV-Light val macro F1={light_settings['macro_f1']:.4f}; "
        f"FAAV-Full val macro F1={full_settings['macro_f1']:.4f}",
        flush=True,
    )

    rows = []
    for study, test in inferred.items():
        light_pred = light_predictions(
            test,
            confidence_threshold=light_settings["confidence_threshold"],
            overlap_threshold=light_settings["overlap_threshold"],
            evidence_char_threshold=light_settings["evidence_char_threshold"],
        )
        light_metrics = save_faav_result(
            study=study,
            model_name="FAAV-Light",
            train=train,
            validation=validation,
            test=test,
            predictions=light_pred,
            settings={
                key: value
                for key, value in light_settings.items()
                if key != "score"
            },
            sweep=light_sweep,
        )
        rows.append(light_metrics)

        full_pred, gate_probability = full_predictions(
            test,
            gate_vectorizer,
            gate,
            full_settings,
        )
        full_metrics = save_faav_result(
            study=study,
            model_name="FAAV-Full",
            train=train,
            validation=validation,
            test=test,
            predictions=full_pred,
            settings={
                key: value
                for key, value in full_settings.items()
                if key != "score"
            },
            sweep=full_sweep,
            gate_probability=gate_probability,
        )
        rows.append(full_metrics)
        print(
            f"  {study}: Light macro_f1={light_metrics['Macro F1']:.4f}; "
            f"Full macro_f1={full_metrics['Macro F1']:.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        f"{domain}_three_class_{part}": load_split(
            f"{domain}_three_class_{part}"
        )
        for domain in ["source", "uae", "combined"]
        for part in ["train", "validation", "test"]
    }
    raw["uae_three_class_full"] = load_split("uae_three_class_full")
    rows = run_family("source", raw) + run_family("combined", raw)
    comparison = pd.DataFrame(rows).sort_values(
        ["Study", "Macro F1"],
        ascending=[True, False],
    )
    output = RESULTS_DIR / "faav_model_comparison.csv"
    comparison.to_csv(output, index=False, encoding="utf-8-sig")
    print("\nFAAV comparison:")
    print(
        comparison[
            [
                "Study",
                "Model",
                "Test N",
                "Accuracy",
                "Macro F1",
                "MCC",
                "Abstention Rate",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
