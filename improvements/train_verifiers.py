"""Retrain the AraBERT verifiers and persist row-level test probabilities.

The submitted package ships metric JSONs but not the checkpoints or the
row-level predictions, so selective-prediction analysis, significance testing,
and gate ablations cannot be reproduced from it. This script rebuilds only what
those analyses need, using the same protocol as
``experiments/final_studies_neural.py`` (AraBERT, max_len 128, batch 32,
3 epochs, AdamW lr 2e-5, weight decay 0.01, 10% linear warmup, sqrt-balanced
class weights, fp16, best epoch by validation macro F1, seed 42).

Everything is written under ``results/improvements`` so the submitted
``results/final_studies`` artifacts are never modified.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from experiments.final_studies_neural import (  # noqa: E402
    ARABERT_NAME,
    DEVICE,
    SEED,
    TASKS,
    class_weights,
    encode_transformer_frame,
    filter_task,
    fingerprint,
    load_split,
    transformer_loader,
    unpack_transformer_batch,
)
from utils.faav_common import compute_classification_metrics, set_seed  # noqa: E402

OUT_DIR = ROOT / "results" / "improvements"
PREDICTIONS_DIR = OUT_DIR / "predictions"

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3

# Which test frames each trained family is evaluated on, using the study names
# from the submitted report (zero-shot UAE is retained here as a diagnostic).
FAMILY_TESTS = {
    "source": {
        "Study 1 - Zero-shot UAE": "uae_{task}_full",
        "Study 2 - Source-domain": "source_{task}_test",
    },
    "combined": {
        "Study 3 - Combined adaptation": "combined_{task}_test",
    },
}

# Columns carried into the prediction dumps so downstream analysis can build
# gate features and provenance controls without re-reading the big CSVs.
CARRY = [
    "sample_id",
    "dataset_family",
    "source",
    "claim_norm",
    "evidence_norm",
    "claim_group",
    "evidence_group",
    "evidence_url",
    "retrieval_quality",
    "faithfulness_label",
    "label",
]


def slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


def predict_probabilities(model, tokenizer, frame: pd.DataFrame) -> np.ndarray:
    """Return the full softmax matrix for ``frame`` (rows stay in order)."""
    encoded = encode_transformer_frame(tokenizer, frame, MAX_LEN)
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if "token_type_ids" in encoded:
        tensors.append(encoded["token_type_ids"])
    dataset = torch.utils.data.TensorDataset(*tensors)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE * 2, shuffle=False
    )
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in loader:
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
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(chunks)


def dump_predictions(
    frame: pd.DataFrame,
    probability: np.ndarray,
    labels: list[str],
    path: Path,
) -> dict:
    out = frame[[c for c in CARRY if c in frame.columns]].copy()
    for index, label in enumerate(labels):
        out[f"probability_{label}"] = probability[:, index]
    predicted = [labels[i] for i in probability.argmax(axis=1)]
    out["prediction"] = predicted
    out["confidence"] = probability.max(axis=1)
    ordered = np.sort(probability, axis=1)
    # Margin between the top two classes: the standard selective-prediction
    # confidence signal, and one the plain max-probability score discards.
    out["margin"] = ordered[:, -1] - ordered[:, -2]
    out["entropy"] = -(probability * np.log(probability + 1e-12)).sum(axis=1)
    out["correct"] = (out["prediction"] == frame["label"].values).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return compute_classification_metrics(frame["label"], predicted, labels)


def run_family(family: str, task: str) -> dict:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(SEED)
    labels = TASKS[task]["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    train = filter_task(load_split(f"{family}_{task}_train"), task)
    validation = filter_task(load_split(f"{family}_{task}_validation"), task)

    tokenizer = AutoTokenizer.from_pretrained(ARABERT_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ARABERT_NAME,
        local_files_only=True,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    train_loader = transformer_loader(
        tokenizer, train, label_to_id, MAX_LEN, BATCH_SIZE, True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * 0.10), 1),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(class_weights(train["label"], labels))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_state = None
    best_epoch = 0
    best_score = -1.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inputs, targets = unpack_transformer_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(**inputs).logits
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.item()))

        probability = predict_probabilities(model, tokenizer, validation)
        predicted = [labels[i] for i in probability.argmax(axis=1)]
        metrics = compute_classification_metrics(
            validation["label"], predicted, labels
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": metrics["macro_f1"],
            }
        )
        print(
            f"  [{family}/{task}] epoch {epoch}"
            f" loss={np.mean(losses):.4f}"
            f" val_macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "family": family,
        "task": task,
        "base_model": ARABERT_NAME,
        "seed": SEED,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "training_history": history,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "tests": {},
    }

    # Validation probabilities are needed to tune every gate on validation only.
    validation_probability = predict_probabilities(model, tokenizer, validation)
    validation_metrics = dump_predictions(
        validation,
        validation_probability,
        labels,
        PREDICTIONS_DIR / f"validation_{family}_{task}_arabert.csv",
    )

    # Train-split probabilities are dumped purely as a diagnostic. The shipped
    # FAAV-Full gate fits its "is the verdict trustworthy?" target on exactly
    # these rows, which the verifier has already fitted, so the target is
    # near-constant and the gate degenerates to always-accept. Recording train
    # accuracy alongside validation accuracy quantifies that gap.
    train_probability = predict_probabilities(model, tokenizer, train)
    train_metrics = dump_predictions(
        train,
        train_probability,
        labels,
        PREDICTIONS_DIR / f"train_{family}_{task}_arabert.csv",
    )
    summary["train_accuracy_in_sample"] = train_metrics["accuracy"]
    summary["validation_accuracy_held_out"] = validation_metrics["accuracy"]
    summary["gate_target_positive_rate_train"] = train_metrics["accuracy"]
    summary["gate_target_positive_rate_validation"] = validation_metrics["accuracy"]
    print(
        f"  [{family}/{task}] verifier accuracy:"
        f" train(in-sample)={train_metrics['accuracy']:.4f}"
        f" validation(held-out)={validation_metrics['accuracy']:.4f}",
        flush=True,
    )

    for study, template in FAMILY_TESTS[family].items():
        frame = filter_task(load_split(template.format(task=task)), task)
        probability = predict_probabilities(model, tokenizer, frame)
        metrics = dump_predictions(
            frame,
            probability,
            labels,
            PREDICTIONS_DIR / f"{slug(study)}_{task}_arabert.csv",
        )
        summary["tests"][study] = {
            "rows": int(len(frame)),
            "test_fingerprint_sha256": fingerprint(frame),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mcc": metrics["mcc"],
        }
        print(
            f"  [{family}/{task}] {study}:"
            f" acc={metrics['accuracy']:.4f}"
            f" macro_f1={metrics['macro_f1']:.4f}"
            f" mcc={metrics['mcc']:.4f}",
            flush=True,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}", flush=True)
    summaries = []
    for family in ["source", "combined"]:
        for task in ["binary", "three_class"]:
            print(f"\n=== training AraBERT [{family} / {task}] ===", flush=True)
            summaries.append(run_family(family, task))
    path = OUT_DIR / "verifier_replication_summary.json"
    path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
