"""Phase 1 Step 3 (priority #2): attempt the BiLSTM baseline on the CLEAN
split, since its required data is now available (unlike the transformer
checkpoints, BiLSTM needs no external pretrained weights).

Architecture matches experiments/final_studies_neural.py's
BiLSTMClassifier unchanged: Embedding(128) -> Dropout(0.30) ->
pack_padded_sequence -> BiLSTM(128 hidden, 1 layer, bidirectional) ->
concat(h[-2], h[-1]) -> Dropout -> Linear.

Unlike the historical script, this ALWAYS persists the trained weights
(artifacts/bilstm/model.pt + vocab.json + config.json), since the audit
found the historical run never saved a BiLSTM checkpoint at all.

This runs on CPU (torch 2.12.0+cpu confirmed in the Phase 1 audit) with a
capped epoch budget and early stopping. If it does not complete in a
practical time, that is reported honestly in
AMAN_PHASE1_RESTORATION_REPORT.md rather than silently abandoned or faked.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics, set_seed  # noqa: E402

CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
ARTIFACT_DIR = ROOT / "artifacts" / "bilstm"
SEED = 42
TASK = "binary"
LABELS = ["Real", "Fake"]
MAX_LEN = 128
VOCAB_SIZE = 30_000
BATCH_SIZE = 128
MAX_EPOCHS = 4
EARLY_STOP_PATIENCE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(name: str) -> pd.DataFrame:
    path = CLEAN_DIR / f"source_{TASK}_{name}.csv"
    df = pd.read_csv(path, low_memory=False).fillna("")
    df["model_text"] = df["claim_norm"].astype(str) + " [SEP] " + df["evidence_norm"].astype(str)
    return df


def build_vocab(texts: list[str], max_size: int) -> dict[str, int]:
    counts = Counter(token for text in texts for token in text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counts.most_common(max_size - 2):
        vocab[token] = len(vocab)
    return vocab


class SequenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int) -> None:
        self.rows = []
        for text, label in zip(texts, labels):
            ids = [vocab.get(tok, 1) for tok in text.split()[:max_len]]
            length = max(len(ids), 1)
            ids = ids or [1]
            ids += [0] * (max_len - len(ids))
            self.rows.append((torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long), torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.dropout = nn.Dropout(0.30)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, ids, lengths):
        embedded = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(representation))


def predict(model, loader, labels):
    model.eval()
    preds = []
    with torch.inference_mode():
        for ids, lengths, _ in loader:
            logits = model(ids.to(DEVICE), lengths.to(DEVICE))
            idx = torch.softmax(logits, dim=1).argmax(dim=1)
            preds.extend(labels[i] for i in idx.cpu().tolist())
    return preds


def main() -> None:
    t0 = time.time()
    set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    print(f"Clean split rows -> train={len(train):,} validation={len(validation):,} test={len(test):,}")

    vocab = build_vocab(train["model_text"].tolist(), VOCAB_SIZE)
    train_ds = SequenceDataset(train["model_text"].tolist(), [label_to_id[x] for x in train["task_label"]], vocab, MAX_LEN)
    val_ds = SequenceDataset(validation["model_text"].tolist(), [label_to_id[x] for x in validation["task_label"]], vocab, MAX_LEN)
    test_ds = SequenceDataset(test["model_text"].tolist(), [label_to_id[x] for x in test["task_label"]], vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = BiLSTMClassifier(len(vocab), len(LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
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
        val_preds = predict(model, val_loader, LABELS)
        val_macro_f1 = f1_score(validation["task_label"], val_preds, labels=LABELS, average="macro", zero_division=0)
        elapsed = time.time() - t0
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": float(val_macro_f1), "elapsed_seconds": round(elapsed, 1)}
        history.append(row)
        print(f"  epoch={epoch} loss={row['train_loss']:.4f} val_macro_f1={val_macro_f1:.4f} elapsed={elapsed:.0f}s", flush=True)
        if val_macro_f1 > best_score + 1e-5:
            best_score, best_epoch, stale = val_macro_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    test_preds = predict(model, test_loader, LABELS)
    metrics = compute_classification_metrics(test["task_label"], test_preds, LABELS)
    print(f"\nTest (clean split): accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} mcc={metrics['mcc']:.4f}")

    torch.save(best_state, ARTIFACT_DIR / "model.pt")
    (ARTIFACT_DIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    config = {
        "model": "BiLSTMClassifier", "vocab_size": len(vocab), "max_len": MAX_LEN,
        "embedding_dim": 128, "hidden_dim": 128, "num_layers": 1, "bidirectional": True,
        "labels": LABELS, "best_epoch": best_epoch, "device_trained_on": str(DEVICE),
        "training_history": history, "total_wall_clock_seconds": round(time.time() - t0, 1),
        "data_source": "hackathon/data/clean_split/",
    }
    (ARTIFACT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_out = {"test_metrics": metrics, "test_rows": int(len(test))}
    (ARTIFACT_DIR / "clean_evaluation_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    print(f"\nArtifacts saved under {ARTIFACT_DIR}")
    print(f"Total wall-clock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
