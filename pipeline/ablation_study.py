"""
Ablation study for MARBERT — removes one component at a time.
Runs 7 configurations and saves a comparison table.

Configurations:
  1. Full MARBERT          — all components (baseline best)
  2. No class weights      — equal loss weight for all classes
  3. No Unverified train   — drop Unverified rows from training
  4. Claim only            — no evidence_text passed to model
  5. Binary only           — Real vs Fake only (no Unverified)
  6. Half training data    — 50% of training rows
  7. No Arabic normalize   — raw text without normalization
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_FILE, TEST_FILE, TEST_SHEET, RESULTS_DIR, ABLATION_CSV, ABLATION_METRICS

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

MODEL_NAME = "UBC-NLP/MARBERTv2"
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS     = ["Real", "Fake", "Unverified"]


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ArabicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, evidences=None):
        self.texts     = texts
        self.evidences = evidences
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        ev  = self.evidences[idx] if self.evidences else None
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


def train_and_eval(config_name, df_train, df_test, tokenizer,
                   use_class_weights=True, use_evidence=True, label_list=None):
    if label_list is None:
        label_list = LABELS

    le = LabelEncoder().fit(label_list)
    df_train = df_train[df_train["label"].isin(label_list)].copy()
    df_train["label_id"] = le.transform(df_train["label"])
    num_classes = len(le.classes_)

    df_test_f = df_test[df_test["gold_label"].isin(label_list)].copy()
    test_claims = df_test_f["claim_norm"].tolist()
    test_evs    = df_test_f["ev_norm"].tolist() if use_evidence else [""] * len(test_claims)

    train_dl = DataLoader(
        ArabicDataset(
            df_train["text"].tolist(), df_train["label_id"].tolist(),
            tokenizer, MAX_LEN
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(DEVICE)

    if use_class_weights:
        counts  = Counter(df_train["label_id"].tolist())
        total   = sum(counts.values())
        class_w = torch.tensor(
            [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
            dtype=torch.float, device=DEVICE,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"\n  Training: {config_name} | {len(df_train)} rows | classes={label_list}")
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
        print(f"    Epoch {epoch+1}/{EPOCHS}  loss={running/len(train_dl):.4f}")

    # Inference
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_claims), BATCH_SIZE):
            bc = test_claims[i:i+BATCH_SIZE]
            be = test_evs[i:i+BATCH_SIZE]
            enc = tokenizer(
                bc, be, max_length=MAX_LEN, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            ).logits
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(le.inverse_transform(preds))

    y_true = df_test_f["gold_label"].tolist()
    acc  = accuracy_score(y_true, all_preds)
    mf1  = f1_score(y_true, all_preds, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(y_true, all_preds)

    # Per-class F1
    per_class = f1_score(y_true, all_preds, average=None,
                         labels=label_list, zero_division=0)
    per_class_dict = dict(zip(label_list, per_class))

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {"config": config_name, "acc": acc, "macro_f1": mf1, "mcc": mcc,
            **{f"f1_{k.lower()}": v for k, v in per_class_dict.items()}}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading training data ...")
    df_train_full = pd.read_csv(TRAIN_FILE, usecols=["text", "label"], low_memory=False)
    df_train_full = df_train_full[df_train_full["label"].isin(LABELS)].dropna()
    df_train_full["text"] = df_train_full["text"].apply(normalize_arabic)

    print("Loading test data ...")
    df_test = pd.read_excel(TEST_FILE, sheet_name="Tabelle1")
    df_test["claim_norm"] = df_test["claim_ar"].apply(normalize_arabic)
    df_test["ev_norm"]    = df_test["evidence_text"].fillna("").apply(normalize_arabic)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = []

    # 1. Full MARBERT
    r = train_and_eval("Full MARBERT", df_train_full.copy(), df_test, tokenizer)
    results.append(r)

    # 2. No class weights
    r = train_and_eval("No Class Weights", df_train_full.copy(), df_test, tokenizer,
                       use_class_weights=False)
    results.append(r)

    # 3. No Unverified in training
    df_no_unv = df_train_full[df_train_full["label"] != "Unverified"].copy()
    r = train_and_eval("No Unverified Train", df_no_unv, df_test, tokenizer,
                       label_list=["Real", "Fake", "Unverified"])
    results.append(r)

    # 4. Claim only (no evidence)
    r = train_and_eval("Claim Only", df_train_full.copy(), df_test, tokenizer,
                       use_evidence=False)
    results.append(r)

    # 5. Binary only (Real vs Fake)
    df_binary = df_train_full[df_train_full["label"].isin(["Real", "Fake"])].copy()
    df_test_bin = df_test[df_test["gold_label"].isin(["Real", "Fake"])].copy()
    r = train_and_eval("Binary Only (Real/Fake)", df_binary, df_test_bin, tokenizer,
                       label_list=["Real", "Fake"])
    results.append(r)

    # 6. Half training data
    df_half = df_train_full.sample(frac=0.5, random_state=42).copy()
    r = train_and_eval("Half Training Data", df_half, df_test, tokenizer)
    results.append(r)

    # 7. No Arabic normalization
    df_raw = pd.read_csv(TRAIN_FILE, usecols=["text", "label"], low_memory=False)
    df_raw = df_raw[df_raw["label"].isin(LABELS)].dropna()
    r = train_and_eval("No Arabic Normalization", df_raw, df_test, tokenizer)
    results.append(r)

    df_results = pd.DataFrame(results)
    df_results.to_csv(ABLATION_CSV, index=False)

    with open(ABLATION_METRICS, "w", encoding="utf-8") as f:
        f.write("Ablation Study Results (MARBERT)\n")
        f.write("=================================\n\n")
        for r in results:
            f.write(f"Config: {r['config']}\n")
            f.write(f"  Accuracy : {r['acc']:.4f}\n")
            f.write(f"  Macro F1 : {r['macro_f1']:.4f}\n")
            f.write(f"  MCC      : {r['mcc']:.4f}\n")
            for k, v in r.items():
                if k.startswith("f1_"):
                    f.write(f"  {k:<20}: {v:.4f}\n")
            f.write("\n")

    print("\n=== ABLATION SUMMARY ===")
    print(f"{'Config':<30} {'Acc':>8} {'MacroF1':>10} {'MCC':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:<30} {r['acc']:>8.4f} {r['macro_f1']:>10.4f} {r['mcc']:>8.4f}")
    print(f"\nSaved -> {ABLATION_CSV}")


if __name__ == "__main__":
    main()
