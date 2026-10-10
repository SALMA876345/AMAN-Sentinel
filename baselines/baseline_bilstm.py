"""
BiLSTM baseline — trains on AFND, evaluates on manual claim test set.

Architecture:
  Embedding (128-d)  ->  BiLSTM (128 units × 2 layers)
  ->  concat(forward_hn, backward_hn)  ->  Dropout  ->  Linear  ->  Softmax

The model learns reliable / unreliable from AFND article text, then
transfers to Supported / Refuted / Not Enough Evidence via confidence
thresholding (same logic as SVM and LR baselines).
"""
import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (
    DATA_DIR, RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET,
    CLASS_MODE, CLASS_LABELS, result_tagged,
)
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


OUTPUT_FILE  = result_tagged("bilstm_claim_results.csv")
METRICS_FILE = result_tagged("bilstm_metrics.txt")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
VOCAB_SIZE    = 20_000
MAX_LEN       = 200
EMBED_DIM     = 128
HIDDEN_DIM    = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
BATCH_SIZE    = 32
EPOCHS        = 5
LR            = 1e-3
THRESHOLD     = 0.60          # confidence needed to commit to Supported/Refuted
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Utilities ─────────────────────────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts, max_size):
    counter = Counter(tok for text in texts for tok in text.split())
    vocab = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common(max_size - 2)]
    return {w: i for i, w in enumerate(vocab)}


def encode(text, vocab, max_len):
    ids = [vocab.get(t, 1) for t in text.split()[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


# ── Dataset class ─────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.X = [torch.tensor(encode(t, vocab, max_len), dtype=torch.long)
                  for t in texts]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.drop(self.embedding(x))
        _, (hn, _) = self.lstm(emb)
        # hn shape: (num_layers*2, batch, hidden) — take last layer
        hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(self.drop(hidden))


# ── Load training data ({CLASS_MODE}-class per config) ────────────────────────
print(f"=== BiLSTM baseline — {CLASS_MODE}-class {CLASS_LABELS} ===")
df_train = pd.read_csv(TRAIN_FILE, usecols=["text", "label"], low_memory=False)
df_train["content"] = df_train["text"].fillna("").apply(normalize_arabic)
df_train = df_train[df_train["content"].str.strip() != ""].dropna(subset=["label"])
df_train = df_train[df_train["label"].isin(CLASS_LABELS)].reset_index(drop=True)

N_CAP = 15_000
df_train = pd.concat([
    g.sample(min(len(g), N_CAP), random_state=42)
    for _, g in df_train.groupby("label")
]).reset_index(drop=True)

le = LabelEncoder()
df_train["label_id"] = le.fit_transform(df_train["label"])
num_classes = len(le.classes_)

X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = train_test_split(
    df_train["content"].tolist(),
    df_train["label_id"].tolist(),
    test_size=0.2, random_state=42,
    stratify=df_train["label_id"].tolist(),
)

vocab = build_vocab(X_tr_raw, VOCAB_SIZE)

train_ds  = TextDataset(X_tr_raw,  y_tr_raw,  vocab, MAX_LEN)
val_ds    = TextDataset(X_val_raw, y_val_raw, vocab, MAX_LEN)
train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ── Train ─────────────────────────────────────────────────────────────────────
bilstm_model = BiLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM,
                      NUM_LAYERS, num_classes, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"Training BiLSTM on {DEVICE} ...")
for epoch in range(EPOCHS):
    bilstm_model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_dl:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(bilstm_model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bilstm_model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_dl)
    print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}")

# ── AFND validation split evaluation ─────────────────────────────────────────
bilstm_model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in val_dl:
        preds = bilstm_model(X_batch.to(DEVICE)).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_true.extend(y_batch.tolist())

y_pred_names = le.inverse_transform(all_preds)
y_true_names = le.inverse_transform(all_true)

afnd_acc    = accuracy_score(y_true_names, y_pred_names)
afnd_f1     = f1_score(y_true_names, y_pred_names, average="weighted", zero_division=0)
afnd_report = classification_report(y_true_names, y_pred_names, labels=CLASS_LABELS, zero_division=0)
afnd_matrix = confusion_matrix(y_true_names, y_pred_names, labels=CLASS_LABELS)

print(f"\nNLI Validation  Accuracy={afnd_acc:.4f}  F1={afnd_f1:.4f}")
print(afnd_report)

# ── Manual claim test-set evaluation ─────────────────────────────────────────
df_test = pd.read_excel(TEST_FILE, sheet_name="Tabelle1")
df_test = df_test[df_test["gold_label"].isin(CLASS_LABELS)].reset_index(drop=True)
df_test["claim_norm"] = df_test["claim_ar"].fillna("").apply(normalize_arabic)

all_claim_preds = []
bilstm_model.eval()
with torch.no_grad():
    for text in df_test["claim_norm"].tolist():
        ids    = torch.tensor([encode(text, vocab, MAX_LEN)], dtype=torch.long).to(DEVICE)
        logits = bilstm_model(ids)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # Return the class with highest probability (Real / Fake / Unverified)
        all_claim_preds.append(le.classes_[int(np.argmax(probs))])

df_test["bilstm_prediction"] = all_claim_preds
df_test["is_correct"]        = df_test["gold_label"] == df_test["bilstm_prediction"]

claim_acc    = accuracy_score(df_test["gold_label"], df_test["bilstm_prediction"])
claim_f1     = f1_score(df_test["gold_label"], df_test["bilstm_prediction"],
                        average="weighted", zero_division=0)
claim_report = classification_report(df_test["gold_label"], df_test["bilstm_prediction"],
                                     labels=CLASS_LABELS, zero_division=0)
claim_matrix = confusion_matrix(df_test["gold_label"], df_test["bilstm_prediction"], labels=CLASS_LABELS)

print(f"\nClaim Verification  Accuracy={claim_acc:.4f}  F1={claim_f1:.4f}")
print(claim_report)

# ── Persist ───────────────────────────────────────────────────────────────────
df_test.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("BiLSTM Baseline Results\n")
    f.write("=======================\n\n")
    f.write("=== Enriched NLI Validation (claim truthfulness) ===\n")
    f.write(f"Accuracy : {afnd_acc:.4f}\n")
    f.write(f"F1-score : {afnd_f1:.4f}\n\n")
    f.write(afnd_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(afnd_matrix))
    f.write("\n\n=== Claim Verification — manual test dataset ===\n")
    f.write(f"Accuracy : {claim_acc:.4f}\n")
    f.write(f"F1-score : {claim_f1:.4f}\n\n")
    f.write(claim_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(claim_matrix))

print(f"\nResults saved -> {OUTPUT_FILE}")
print(f"Metrics saved -> {METRICS_FILE}")
