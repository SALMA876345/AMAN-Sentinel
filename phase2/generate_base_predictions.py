"""Phase 2: regenerate SVM and BiLSTM per-row predictions on BOTH the
validation and test clean splits, using the Phase 1 persisted artifacts
unchanged (no retraining, no artifact overwritten).

Phase 1 only persisted SVM predictions for test
(artifacts/svm/clean_test_predictions.csv) and never persisted per-row
BiLSTM predictions at all (only aggregate metrics). Phase 2's calibration,
disagreement, and ensemble steps all need validation-split predictions
(to fit calibration / thresholds) and a consistent BiLSTM prediction file
for both splits — this script produces exactly that, reusing the exact
same loading/inference code paths Phase 1 used.

Output (new, does not touch any Phase 1 file):
    phase2/reports/svm_predictions_validation.csv
    phase2/reports/svm_predictions_test.csv
    phase2/reports/bilstm_predictions_validation.csv
    phase2/reports/bilstm_predictions_test.csv

Columns: sample_id, true_label, prediction, confidence, raw_score
    (raw_score = SVM decision_function value; BiLSTM max softmax prob,
    duplicated into confidence for schema symmetry)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
SVM_DIR = ROOT / "artifacts" / "svm"
BILSTM_DIR = ROOT / "artifacts" / "bilstm"
OUT_DIR = ROOT / "phase2" / "reports"
TASK = "binary"
LABELS = ["Real", "Fake"]
MAX_LEN = 128  # must match hackathon/train_bilstm_clean.py


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DIR / f"source_{TASK}_{name}.csv", low_memory=False).fillna("")
    df["model_text"] = df["claim_norm"].astype(str) + " [SEP] " + df["evidence_norm"].astype(str)
    return df


def svm_predict(split_name: str, df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = joblib.load(SVM_DIR / "vectorizer.joblib")
    model = joblib.load(SVM_DIR / "model.joblib")
    matrix = vectorizer.transform(df["model_text"])
    predictions = model.predict(matrix)
    scores = model.decision_function(matrix)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    from scipy.special import softmax
    confidence = softmax(scores, axis=1).max(axis=1)
    raw_score = scores[:, 1] if scores.shape[1] == 2 else scores.max(axis=1)
    return pd.DataFrame({
        "sample_id": df["sample_id"],
        "true_label": df["task_label"],
        "prediction": predictions,
        "confidence": confidence,
        "raw_score": raw_score,
    })


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


class SequenceDataset(Dataset):
    def __init__(self, texts, vocab, max_len):
        self.rows = []
        for text in texts:
            ids = [vocab.get(tok, 1) for tok in text.split()[:max_len]]
            length = max(len(ids), 1)
            ids = ids or [1]
            ids += [0] * (max_len - len(ids))
            self.rows.append((torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def bilstm_predict(split_name: str, df: pd.DataFrame) -> pd.DataFrame:
    vocab = json.loads((BILSTM_DIR / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads((BILSTM_DIR / "config.json").read_text(encoding="utf-8"))
    model = BiLSTMClassifier(config["vocab_size"], len(LABELS))
    state = torch.load(BILSTM_DIR / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    ds = SequenceDataset(df["model_text"].tolist(), vocab, MAX_LEN)
    loader = DataLoader(ds, batch_size=128)
    preds, confs, logits_real, logits_fake = [], [], [], []
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    with torch.inference_mode():
        for ids, lengths in loader:
            logits = model(ids, lengths)
            probs = torch.softmax(logits, dim=1)
            values, indices = probs.max(dim=1)
            preds.extend(LABELS[i] for i in indices.tolist())
            confs.extend(values.tolist())
            logits_real.extend(logits[:, label_to_id["Real"]].tolist())
            logits_fake.extend(logits[:, label_to_id["Fake"]].tolist())
    return pd.DataFrame({
        "sample_id": df["sample_id"],
        "true_label": df["task_label"],
        "prediction": preds,
        "confidence": confs,
        "raw_score": confs,  # BiLSTM has no separate decision-margin score; softmax confidence doubles here
        "logit_real": logits_real,
        "logit_fake": logits_fake,
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ["validation", "test"]:
        df = load_split(split_name)
        svm_out = svm_predict(split_name, df)
        svm_out.to_csv(OUT_DIR / f"svm_predictions_{split_name}.csv", index=False, encoding="utf-8-sig")
        print(f"SVM {split_name}: {len(svm_out):,} rows -> {OUT_DIR / f'svm_predictions_{split_name}.csv'}")

        bilstm_out = bilstm_predict(split_name, df)
        bilstm_out.to_csv(OUT_DIR / f"bilstm_predictions_{split_name}.csv", index=False, encoding="utf-8-sig")
        print(f"BiLSTM {split_name}: {len(bilstm_out):,} rows -> {OUT_DIR / f'bilstm_predictions_{split_name}.csv'}")

        # Quick sanity check against Phase 1's own test metrics for SVM
        if split_name == "test":
            acc = (svm_out["prediction"] == svm_out["true_label"]).mean()
            print(f"  [sanity] SVM test accuracy recomputed here: {acc:.4f} (Phase 1 reported 0.8645)")
            bacc = (bilstm_out["prediction"] == bilstm_out["true_label"]).mean()
            print(f"  [sanity] BiLSTM test accuracy recomputed here: {bacc:.4f} (Phase 1 reported ~0.8003)")


if __name__ == "__main__":
    main()
