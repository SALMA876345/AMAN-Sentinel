"""
MARBERT fine-tuned baseline — 3-class (Real / Fake / Unverified)
Model : UBC-NLP/MARBERTv2  (pretrained on 1B Arabic tweets)
Shows : before fine-tuning (random head) vs after fine-tuning results
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_FILE, TEST_FILE, TEST_SHEET, RESULTS_DIR,
    MARBERT_CSV, MARBERT_METRICS, MARBERT_SAVE,
    MARBERT_BEFORE_CSV, MARBERT_BEFORE_METRICS,
    CLASS_MODE, CLASS_LABELS, MODEL_INPUT_MODE, SEED,
)
from utils.faav_common import normalize_arabic, set_seed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)

MODEL_NAME = "UBC-NLP/MARBERTv2"
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS     = CLASS_LABELS


class ArabicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, evidences=None):
        self.texts     = texts
        self.evidences = evidences
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        ev  = self.evidences[idx] if self.evidences is not None else None
        enc = self.tokenizer(
            self.texts[idx], ev,
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_all_metrics(y_true, y_pred, y_proba, label_names):
    acc    = accuracy_score(y_true, y_pred)
    mf1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    wf1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)
    kappa  = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=label_names, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=label_names)
    try:
        auc = roc_auc_score(
            pd.get_dummies(y_true).values,
            y_proba, multi_class="ovr", average="macro",
        )
    except Exception:
        auc = float("nan")
    return {"acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "mcc": mcc, "kappa": kappa, "auc": auc,
            "report": report, "matrix": matrix}


def save_metrics(path, title, m, model_name, n_train, phase):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n{'='*len(title)}\n\n")
        f.write(f"Model     : {model_name}\n")
        f.write(f"Device    : {DEVICE}\n")
        f.write(f"Phase     : {phase}\n")
        f.write(f"Input mode: {MODEL_INPUT_MODE}\n")
        f.write(f"Train rows: {n_train}\n\n")
        f.write(f"Accuracy        : {m['acc']:.4f}\n")
        f.write(f"Macro F1        : {m['macro_f1']:.4f}\n")
        f.write(f"Weighted F1     : {m['weighted_f1']:.4f}\n")
        f.write(f"MCC             : {m['mcc']:.4f}\n")
        f.write(f"Cohen Kappa     : {m['kappa']:.4f}\n")
        f.write(f"AUC-ROC (macro) : {m['auc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(m["report"])
        f.write("\nConfusion Matrix:\n")
        f.write(str(m["matrix"]))


def run_inference(model, tokenizer, claims, evidences, le):
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for i in range(0, len(claims), BATCH_SIZE):
            bc = claims[i:i+BATCH_SIZE]
            be = evidences[i:i+BATCH_SIZE] if evidences is not None else None
            enc = tokenizer(
                bc, be, max_length=MAX_LEN, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            ).logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(proba, axis=1)
            all_proba.extend(proba.tolist())
            all_preds.extend(le.inverse_transform(preds))
    return all_preds, np.array(all_proba)


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"=== MARBERT baseline — {CLASS_MODE}-class {LABELS} | input={MODEL_INPUT_MODE} ===")

    # ── Load training data ────────────────────────────────────────────────────
    print(f"Loading training data from {TRAIN_FILE} ...")
    df_train = pd.read_csv(TRAIN_FILE, usecols=["text", "label", "evidence_text"], low_memory=False)
    df_train = df_train[df_train["label"].isin(LABELS)].dropna(subset=["text", "label"])
    df_train["text"] = df_train["text"].apply(normalize_arabic)
    df_train["evidence_text"] = df_train["evidence_text"].fillna("").apply(normalize_arabic)
    print(f"Training rows: {len(df_train)}  |  {df_train['label'].value_counts().to_dict()}")

    le = LabelEncoder().fit(LABELS)
    df_train["label_id"] = le.transform(df_train["label"])
    num_classes = len(le.classes_)

    # ── Load test data ────────────────────────────────────────────────────────
    print("Loading test data ...")
    df_test = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
    df_test = df_test[df_test["gold_label"].isin(LABELS)].reset_index(drop=True)
    test_claims = [normalize_arabic(str(c)) for c in df_test["claim_ar"]]
    test_evs    = [normalize_arabic(str(e)) for e in df_test["evidence_text"].fillna("")]
    if MODEL_INPUT_MODE == "claim_only":
        train_evs = None
        inference_evs = None
    else:
        train_evs = df_train["evidence_text"].tolist()
        inference_evs = test_evs

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1 — BEFORE FINE-TUNING
    # ════════════════════════════════════════════════════════════════════════
    print("\n--- PHASE 1: Before fine-tuning ---")
    model_before = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(DEVICE)

    preds_before, proba_before = run_inference(model_before, tokenizer, test_claims, inference_evs, le)
    m_before = compute_all_metrics(
        df_test["gold_label"].tolist(), preds_before, proba_before, LABELS
    )
    print(f"Before fine-tuning — Accuracy: {m_before['acc']:.4f}  Macro F1: {m_before['macro_f1']:.4f}")

    df_before = df_test.copy()
    df_before["marbert_prediction"] = preds_before
    df_before["is_correct"] = df_before["gold_label"] == df_before["marbert_prediction"]
    df_before.to_csv(MARBERT_BEFORE_CSV, index=False, encoding="utf-8-sig")
    save_metrics(MARBERT_BEFORE_METRICS, "MARBERT Before Fine-tuning",
                 m_before, MODEL_NAME, len(df_train), phase="Before fine-tuning (random head)")
    del model_before
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2 — FINE-TUNING
    # ════════════════════════════════════════════════════════════════════════
    print("\n--- PHASE 2: Fine-tuning ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(DEVICE)

    train_dl = DataLoader(
        ArabicDataset(
            df_train["text"].tolist(), df_train["label_id"].tolist(),
            tokenizer, MAX_LEN, evidences=train_evs
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )

    counts   = Counter(df_train["label_id"].tolist())
    total    = sum(counts.values())
    class_w  = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float, device=DEVICE,
    )
    loss_fn   = nn.CrossEntropyLoss(weight=class_w)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    epoch_losses = []
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in train_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labs = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        avg_loss = running / len(train_dl)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}")

    pd.DataFrame({"epoch": range(1, EPOCHS+1), "loss": epoch_losses}).to_csv(
        os.path.join(RESULTS_DIR, "marbert_training_loss.csv"), index=False
    )

    os.makedirs(MARBERT_SAVE, exist_ok=True)
    model.save_pretrained(MARBERT_SAVE)
    tokenizer.save_pretrained(MARBERT_SAVE)
    print(f"Model saved -> {MARBERT_SAVE}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3 — AFTER FINE-TUNING EVALUATION
    # ════════════════════════════════════════════════════════════════════════
    print("\n--- PHASE 3: After fine-tuning evaluation ---")
    preds_after, proba_after = run_inference(model, tokenizer, test_claims, inference_evs, le)
    m_after = compute_all_metrics(
        df_test["gold_label"].tolist(), preds_after, proba_after, LABELS
    )
    print(f"After fine-tuning — Accuracy: {m_after['acc']:.4f}  Macro F1: {m_after['macro_f1']:.4f}")
    print(m_after["report"])

    df_after = df_test.copy()
    df_after["marbert_prediction"]        = preds_after
    for label in LABELS:
        df_after[f"marbert_proba_{label.lower()}"] = proba_after[:, le.transform([label])[0]]
    df_after["is_correct"] = df_after["gold_label"] == df_after["marbert_prediction"]
    df_after.to_csv(MARBERT_CSV, index=False, encoding="utf-8-sig")
    save_metrics(MARBERT_METRICS, "MARBERT After Fine-tuning",
                 m_after, MODEL_NAME, len(df_train), phase="After fine-tuning")

    print("\n=== MARBERT SUMMARY ===")
    print(f"{'Metric':<20} {'Before':>10} {'After':>10}")
    print("-" * 42)
    for k, label in [("acc","Accuracy"), ("macro_f1","Macro F1"), ("mcc","MCC"), ("kappa","Kappa")]:
        print(f"{label:<20} {m_before[k]:>10.4f} {m_after[k]:>10.4f}")

    print(f"\nResults saved -> {MARBERT_CSV}")
    print(f"Metrics saved -> {MARBERT_METRICS}")


if __name__ == "__main__":
    main()
