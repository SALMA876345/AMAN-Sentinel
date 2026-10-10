"""Run BiLSTM and AraBERT on the frozen final-study manifests."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics, set_seed


DATA_DIR = ROOT / "data" / "final_studies"
RESULTS_DIR = ROOT / "results" / "final_studies"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"
ARABERT_NAME = "aubmindlab/bert-base-arabertv02"
MARBERT_NAME = "UBC-NLP/MARBERTv2"
TRANSFORMERS = {
    "arabert": {
        "display_name": "AraBERT",
        "base_model": ARABERT_NAME,
        "batch_size": 32,
    },
    "marbert": {
        "display_name": "MARBERT",
        "base_model": MARBERT_NAME,
        "batch_size": 24,
    },
}
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bilstm", "arabert", "marbert"],
        default=["bilstm", "arabert", "marbert"],
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=sorted(TASKS),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["source", "combined"],
        default=["source", "combined"],
        help="source creates Studies 1/2; combined creates Study 3.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", low_memory=False).fillna("")


def filter_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    config = TASKS[task]
    out = df[df[config["label_col"]].isin(config["labels"])].copy()
    out["label"] = out[config["label_col"]]
    out["model_text"] = (
        out["claim_norm"].astype(str)
        + " [SEP] "
        + out["evidence_norm"].astype(str)
    )
    return out.reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    payload = "\n".join(sorted(df["sample_id"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def class_weights(labels: pd.Series, label_order: list[str]) -> torch.Tensor:
    counts = labels.value_counts().to_dict()
    total = len(labels)
    weights = [
        math.sqrt(total / (len(label_order) * max(counts.get(label, 1), 1)))
        for label in label_order
    ]
    weights = np.asarray(weights, dtype=np.float32)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def slug(study: str, task: str, model_name: str) -> str:
    return (
        f"{study}_{task}_{model_name.lower().replace(' ', '_')}"
        .replace("+", "plus")
    )


def metrics_path(study: str, task: str, model_name: str) -> Path:
    return RESULTS_DIR / f"{slug(study, task, model_name)}_metrics.json"


def family_complete(
    family: str,
    task: str,
    model_name: str,
) -> bool:
    if family == "source":
        studies = ["Study 1 - Zero-shot UAE", "Study 2 - Source-domain"]
    else:
        studies = ["Study 3 - Combined adaptation"]
    return all(metrics_path(study, task, model_name).exists() for study in studies)


def save_result(
    *,
    study: str,
    task: str,
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    predictions: list[str],
    confidence: np.ndarray,
    training_history: list[dict],
    best_epoch: int,
    settings: dict,
) -> dict:
    labels = TASKS[task]["labels"]
    metrics = compute_classification_metrics(test["label"], predictions, labels)
    validation_best = max(
        (row["validation_macro_f1"] for row in training_history),
        default=float("nan"),
    )
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
    file_slug = slug(study, task, model_name)
    output.to_csv(
        PREDICTIONS_DIR / f"{file_slug}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "study": study,
        "task": task,
        "model": model_name,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_fingerprint_sha256": fingerprint(test),
        "best_epoch": best_epoch,
        "validation_macro_f1": validation_best,
        "training_history": training_history,
        "settings": settings,
        "test_metrics": metrics,
    }
    metrics_path(study, task, model_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


class SequenceDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_len: int,
    ) -> None:
        self.rows = []
        for text, label in zip(texts, labels):
            ids = [vocab.get(token, 1) for token in text.split()[:max_len]]
            length = max(len(ids), 1)
            ids = ids or [1]
            ids += [0] * (max_len - len(ids))
            self.rows.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(length, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.dropout = nn.Dropout(0.30)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(representation))


def build_vocab(texts: list[str], max_size: int = 50_000) -> dict[str, int]:
    counts = Counter(token for text in texts for token in text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counts.most_common(max_size - 2):
        vocab[token] = len(vocab)
    return vocab


def bilstm_predict(
    model: BiLSTMClassifier,
    loader: DataLoader,
    labels: list[str],
) -> tuple[list[str], np.ndarray]:
    model.eval()
    predictions = []
    confidence = []
    with torch.inference_mode():
        for ids, lengths, _ in loader:
            logits = model(ids.to(DEVICE), lengths.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)
            values, indices = probabilities.max(dim=1)
            predictions.extend(labels[i] for i in indices.cpu().tolist())
            confidence.extend(values.cpu().tolist())
    return predictions, np.asarray(confidence)


def run_bilstm_family(
    family: str,
    task: str,
    raw: dict[str, pd.DataFrame],
) -> None:
    labels = TASKS[task]["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    if family == "source":
        train = filter_task(raw[f"source_{task}_train"], task)
        validation = filter_task(
            raw[f"source_{task}_validation"], task
        )
        tests = {
            "Study 1 - Zero-shot UAE": filter_task(
                raw[f"uae_{task}_full"], task
            ),
            "Study 2 - Source-domain": filter_task(
                raw[f"source_{task}_test"], task
            ),
        }
    else:
        train = filter_task(
            raw[f"combined_{task}_train"], task
        )
        validation = filter_task(
            raw[f"combined_{task}_validation"], task
        )
        tests = {
            "Study 3 - Combined adaptation": filter_task(
                raw[f"combined_{task}_test"], task
            )
        }

    max_len = 180
    batch_size = 128
    vocab = build_vocab(train["model_text"].tolist())
    train_dataset = SequenceDataset(
        train["model_text"].tolist(),
        [label_to_id[x] for x in train["label"]],
        vocab,
        max_len,
    )
    val_dataset = SequenceDataset(
        validation["model_text"].tolist(),
        [label_to_id[x] for x in validation["label"]],
        vocab,
        max_len,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    model = BiLSTMClassifier(len(vocab), len(labels)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(class_weights(train["label"], labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    history = []
    best_state = None
    best_epoch = 0
    best_score = -1.0
    stale = 0
    for epoch in range(1, 9):
        model.train()
        losses = []
        for ids, lengths, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids.to(DEVICE), lengths.to(DEVICE))
            loss = criterion(logits, targets.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_predictions, _ = bilstm_predict(model, val_loader, labels)
        val_macro_f1 = f1_score(
            validation["label"],
            val_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_macro_f1": float(val_macro_f1),
        }
        history.append(row)
        print(
            f"  BiLSTM {family}/{task} epoch={epoch} "
            f"loss={row['train_loss']:.4f} val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )
        if val_macro_f1 > best_score + 1e-5:
            best_score = val_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is None:
        raise RuntimeError("BiLSTM did not produce a checkpoint.")
    model.load_state_dict(best_state)

    settings = {
        "device": str(DEVICE),
        "vocab_size": len(vocab),
        "max_len": max_len,
        "batch_size": batch_size,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "class_weight": "sqrt_inverse_frequency",
        "early_stopping_patience": 2,
    }
    for study, test in tests.items():
        test_dataset = SequenceDataset(
            test["model_text"].tolist(),
            [label_to_id[x] for x in test["label"]],
            vocab,
            max_len,
        )
        predictions, confidence = bilstm_predict(
            model,
            DataLoader(test_dataset, batch_size=batch_size),
            labels,
        )
        metrics = save_result(
            study=study,
            task=task,
            model_name="BiLSTM",
            train=train,
            validation=validation,
            test=test,
            predictions=predictions,
            confidence=confidence,
            training_history=history,
            best_epoch=best_epoch,
            settings=settings,
        )
        print(
            f"  {study}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )


def encode_transformer_frame(tokenizer, frame: pd.DataFrame, max_len: int):
    # Bound raw strings before WordPiece tokenization. Several source rows hold
    # full articles; tokenizing the unused tail is extremely expensive even
    # though the transformer input is truncated to max_len afterward.
    claims = frame["claim_norm"].astype(str).str[:400].tolist()
    evidence = frame["evidence_norm"].astype(str).str[:1200].tolist()
    encoded = tokenizer(
        claims,
        evidence,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded


def transformer_loader(
    tokenizer,
    frame: pd.DataFrame,
    label_to_id: dict[str, int],
    max_len: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    encoded = encode_transformer_frame(tokenizer, frame, max_len)
    label_ids = torch.tensor(
        [label_to_id[label] for label in frame["label"]],
        dtype=torch.long,
    )
    tensors = [
        encoded["input_ids"],
        encoded["attention_mask"],
    ]
    if "token_type_ids" in encoded:
        tensors.append(encoded["token_type_ids"])
    tensors.append(label_ids)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def unpack_transformer_batch(batch):
    if len(batch) == 4:
        input_ids, attention_mask, token_type_ids, targets = batch
    else:
        input_ids, attention_mask, targets = batch
        token_type_ids = None
    inputs = {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
    }
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids.to(DEVICE)
    return inputs, targets.to(DEVICE)


def transformer_predict(model, loader, labels: list[str]):
    model.eval()
    predictions = []
    confidence = []
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = unpack_transformer_batch(batch)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits.float(), dim=1)
            values, indices = probabilities.max(dim=1)
            predictions.extend(labels[i] for i in indices.cpu().tolist())
            confidence.extend(values.cpu().tolist())
    return predictions, np.asarray(confidence)


def run_transformer_family(
    model_key: str,
    family: str,
    task: str,
    raw: dict[str, pd.DataFrame],
) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    transformer_config = TRANSFORMERS[model_key]
    display_name = transformer_config["display_name"]
    base_model = transformer_config["base_model"]
    labels = TASKS[task]["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    if family == "source":
        train = filter_task(raw[f"source_{task}_train"], task)
        validation = filter_task(
            raw[f"source_{task}_validation"], task
        )
        tests = {
            "Study 1 - Zero-shot UAE": filter_task(
                raw[f"uae_{task}_full"], task
            ),
            "Study 2 - Source-domain": filter_task(
                raw[f"source_{task}_test"], task
            ),
        }
    else:
        train = filter_task(
            raw[f"combined_{task}_train"], task
        )
        validation = filter_task(
            raw[f"combined_{task}_validation"], task
        )
        tests = {
            "Study 3 - Combined adaptation": filter_task(
                raw[f"combined_{task}_test"], task
            )
        }

    max_len = 128
    batch_size = transformer_config["batch_size"]
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        local_files_only=True,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    train_loader = transformer_loader(
        tokenizer, train, label_to_id, max_len, batch_size, True
    )
    val_loader = transformer_loader(
        tokenizer, validation, label_to_id, max_len, batch_size * 2, False
    )
    epochs = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * 0.10), 1),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(class_weights(train["label"], labels))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    history = []
    best_state = None
    best_epoch = 0
    best_score = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
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
        val_predictions, _ = transformer_predict(model, val_loader, labels)
        val_macro_f1 = f1_score(
            validation["label"],
            val_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_macro_f1": float(val_macro_f1),
        }
        history.append(row)
        print(
            f"  {display_name} {family}/{task} epoch={epoch} "
            f"loss={row['train_loss']:.4f} val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )
        if val_macro_f1 > best_score + 1e-5:
            best_score = val_macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 1:
                break
    if best_state is None:
        raise RuntimeError(f"{display_name} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    settings = {
        "base_model": base_model,
        "device": str(DEVICE),
        "max_len": max_len,
        "batch_size": batch_size,
        "claim_character_cap": 400,
        "evidence_character_cap": 1200,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "class_weight": "sqrt_inverse_frequency",
        "early_stopping_patience": 1,
    }
    for study, test in tests.items():
        test_loader = transformer_loader(
            tokenizer,
            test,
            label_to_id,
            max_len,
            batch_size * 2,
            False,
        )
        predictions, confidence = transformer_predict(model, test_loader, labels)
        metrics = save_result(
            study=study,
            task=task,
            model_name=display_name,
            train=train,
            validation=validation,
            test=test,
            predictions=predictions,
            confidence=confidence,
            training_history=history,
            best_epoch=best_epoch,
            settings=settings,
        )
        print(
            f"  {study}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    final_dir = MODELS_DIR / f"{model_key}_{family}_{task}"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  Saved final {display_name} checkpoint to {final_dir}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
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
    print(f"Device: {DEVICE}", flush=True)
    for model_name in args.models:
        display_name = (
            "BiLSTM"
            if model_name == "bilstm"
            else TRANSFORMERS[model_name]["display_name"]
        )
        for task in args.tasks:
            for family in args.families:
                if not args.force and family_complete(family, task, display_name):
                    print(
                        f"Skipping completed {display_name} {family}/{task}",
                        flush=True,
                    )
                    continue
                print(
                    f"\nRunning {display_name} {family}/{task}",
                    flush=True,
                )
                if model_name == "bilstm":
                    run_bilstm_family(family, task, raw)
                else:
                    run_transformer_family(model_name, family, task, raw)


if __name__ == "__main__":
    main()
