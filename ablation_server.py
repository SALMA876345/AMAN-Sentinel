#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation study for the MARBERT verifier — SELF-CONTAINED, Linux server ready.

No imports from config.py or other project files. Auto-detects data paths.
Trains 7 MARBERT configurations and saves a comparison table.

Configurations
  1. Full MARBERT          — claim + evidence, class weights, 3-class (reference)
  2. No class weights      — equal loss weight for all classes
  3. No Unverified train   — Unverified dropped from TRAINING (still tested 3-class)
  4. Claim only            — NO evidence (claim text alone) in train AND test
  5. Binary only           — Real vs Fake only (2-class)
  6. Half training data    — 50 % of training rows
  7. No Arabic normalize   — raw text, normalization disabled

USAGE (Linux)
  # place data files under ./data/  (relative to this script), then:
  python3 ablation_server.py

  # or point to a custom data directory / files via env vars:
  ABLATION_DATA_DIR=/path/to/data python3 ablation_server.py
  TRAIN_FILE=/abs/claim_training.csv TEST_FILE=/abs/uae_test_dataset_final.xlsx python3 ablation_server.py

REQUIREMENTS
  pip install torch transformers scikit-learn pandas numpy openpyxl

OUTPUTS (written to ./results/)
  ablation_results.csv
  ablation_metrics.txt
  ablation_run_log.txt
"""
import os, re, sys, time, json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score

# ── Paths (robust, no config.py) ──────────────────────────────────────────────
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()

DATA_DIR    = os.environ.get("ABLATION_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_DIR = os.environ.get("ABLATION_RESULTS_DIR", os.path.join(ROOT, "results"))
TRAIN_FILE  = os.environ.get("TRAIN_FILE", os.path.join(DATA_DIR, "claim_training.csv"))
TEST_FILE   = os.environ.get("TEST_FILE",  os.path.join(DATA_DIR, "uae_test_dataset_final.xlsx"))
TEST_SHEET  = os.environ.get("TEST_SHEET", "Tabelle1")

os.makedirs(RESULTS_DIR, exist_ok=True)
ABLATION_CSV     = os.path.join(RESULTS_DIR, "ablation_results.csv")
ABLATION_METRICS = os.path.join(RESULTS_DIR, "ablation_metrics.txt")
LOG_FILE         = os.path.join(RESULTS_DIR, "ablation_run_log.txt")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "UBC-NLP/MARBERTv2"
MAX_LEN    = int(os.environ.get("MAX_LEN", 128))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))   # GX10: try 128 or 256
EPOCHS     = int(os.environ.get("EPOCHS", 3))
LR         = float(os.environ.get("LR", 2e-5))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS     = ["Real", "Fake", "Unverified"]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ArabicDataset(Dataset):
    def __init__(self, texts, evidences, labels, tokenizer, max_len):
        self.texts = texts; self.evidences = evidences
        self.labels = labels; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ev = self.evidences[idx] if self.evidences is not None else None
        enc = self.tokenizer(self.texts[idx], ev, max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def train_and_eval(name, df_train, df_test, tokenizer,
                   use_class_weights=True, use_evidence=True, label_list=None):
    label_list = label_list or LABELS
    le = LabelEncoder().fit(label_list)
    dtr = df_train[df_train["label"].isin(label_list)].copy()
    dtr["label_id"] = le.transform(dtr["label"])
    nc = len(le.classes_)

    dte = df_test[df_test["gold_label"].isin(label_list)].copy()
    test_claims = dte["claim_proc"].tolist()
    test_evs    = dte["ev_proc"].tolist() if use_evidence else [""] * len(dte)

    train_ev = dtr["ev_proc"].tolist() if use_evidence else None
    train_dl = DataLoader(
        ArabicDataset(dtr["text_proc"].tolist(), train_ev, dtr["label_id"].tolist(),
                      tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=nc, ignore_mismatched_sizes=True).to(DEVICE)

    if use_class_weights:
        counts = Counter(dtr["label_id"].tolist()); tot = sum(counts.values())
        cw = torch.tensor([tot / (nc * counts.get(i, 1)) for i in range(nc)],
                          dtype=torch.float, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(train_dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, max(1, steps // 10), steps)

    log(f"  train [{name}] rows={len(dtr)} classes={label_list} evidence={use_evidence} cw={use_class_weights}")
    for ep in range(EPOCHS):
        model.train(); run = 0.0
        for b in train_dl:
            ids = b["input_ids"].to(DEVICE); mask = b["attention_mask"].to(DEVICE)
            labs = b["label"].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); run += loss.item()
        log(f"    epoch {ep+1}/{EPOCHS} loss={run/len(train_dl):.4f}")

    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(test_claims), BATCH_SIZE):
            enc = tokenizer(test_claims[i:i+BATCH_SIZE], test_evs[i:i+BATCH_SIZE],
                            max_length=MAX_LEN, padding="max_length",
                            truncation=True, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(DEVICE),
                           attention_mask=enc["attention_mask"].to(DEVICE)).logits
            preds.extend(le.inverse_transform(logits.argmax(1).cpu().numpy()))

    yt = dte["gold_label"].tolist()
    acc = accuracy_score(yt, preds)
    mf1 = f1_score(yt, preds, average="macro", labels=label_list, zero_division=0)
    mcc = matthews_corrcoef(yt, preds)
    kappa = cohen_kappa_score(yt, preds, labels=label_list)
    per = dict(zip(label_list, f1_score(yt, preds, average=None, labels=label_list, zero_division=0)))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"config": name, "acc": acc, "macro_f1": mf1, "mcc": mcc, "kappa": kappa,
            **{f"f1_{k.lower()}": float(v) for k, v in per.items()}}


def main():
    log("=" * 60)
    log(f"ABLATION STUDY — device={DEVICE}")
    log(f"TRAIN={TRAIN_FILE}")
    log(f"TEST ={TEST_FILE}")
    log("=" * 60)
    for f in (TRAIN_FILE, TEST_FILE):
        if not os.path.exists(f):
            log(f"ABORT — missing file: {f}"); sys.exit(1)

    cols = pd.read_csv(TRAIN_FILE, nrows=1).columns
    use = [c for c in ["text", "label", "evidence_text"] if c in cols]
    df = pd.read_csv(TRAIN_FILE, usecols=use, low_memory=False)
    df = df[df["label"].isin(LABELS)].dropna(subset=["text", "label"])
    if "evidence_text" not in df.columns:
        df["evidence_text"] = ""
    df["evidence_text"] = df["evidence_text"].fillna("")
    # normalized columns + raw columns (config 7 uses raw)
    df["text_proc"] = df["text"].apply(normalize_arabic)
    df["ev_proc"]   = df["evidence_text"].apply(normalize_arabic)
    df["text_raw"]  = df["text"].astype(str)
    df["ev_raw"]    = df["evidence_text"].astype(str)
    log(f"train rows: {len(df):,}  {df['label'].value_counts().to_dict()}")

    dt = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
    dt["claim_proc"] = dt["claim_ar"].apply(normalize_arabic)
    dt["ev_proc"]    = dt["evidence_text"].fillna("").apply(normalize_arabic)
    log(f"test rows : {len(dt):,}  {dt['gold_label'].value_counts().to_dict()}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = []
    t_all = time.time()

    def run(cfg, **kw):
        t0 = time.time()
        r = train_and_eval(cfg, **kw)
        log(f"  done [{cfg}] acc={r['acc']:.4f} mf1={r['macro_f1']:.4f} mcc={r['mcc']:.4f} ({(time.time()-t0)/60:.1f} min)")
        results.append(r)
        # checkpoint after every config
        pd.DataFrame(results).to_csv(ABLATION_CSV, index=False)

    # 1. Full (claim + evidence)
    run("Full MARBERT", df_train=df, df_test=dt, tokenizer=tok)
    # 2. No class weights
    run("No Class Weights", df_train=df, df_test=dt, tokenizer=tok, use_class_weights=False)
    # 3. No Unverified in training (still evaluated 3-class)
    run("No Unverified Train", df_train=df[df["label"] != "Unverified"], df_test=dt, tokenizer=tok)
    # 4. Claim only (no evidence in train or test)
    run("Claim Only", df_train=df, df_test=dt, tokenizer=tok, use_evidence=False)
    # 5. Binary only
    run("Binary Only (Real/Fake)", df_train=df, df_test=dt, tokenizer=tok,
        label_list=["Real", "Fake"])
    # 6. Half data
    run("Half Training Data", df_train=df.sample(frac=0.5, random_state=42), df_test=dt, tokenizer=tok)
    # 7. No Arabic normalization (swap proc cols for raw)
    df_raw = df.copy(); df_raw["text_proc"] = df_raw["text_raw"]; df_raw["ev_proc"] = df_raw["ev_raw"]
    dt_raw = dt.copy(); dt_raw["claim_proc"] = dt_raw["claim_ar"].astype(str)
    dt_raw["ev_proc"] = dt_raw["evidence_text"].fillna("").astype(str)
    run("No Arabic Normalization", df_train=df_raw, df_test=dt_raw, tokenizer=tok)

    # ── Save summary ──────────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(ABLATION_CSV, index=False)
    with open(ABLATION_METRICS, "w", encoding="utf-8") as f:
        f.write("Ablation Study Results (MARBERT verifier)\n")
        f.write("=" * 42 + "\n\n")
        f.write(f"Base model : {MODEL_NAME}\nDevice     : {DEVICE}\n")
        f.write(f"Train file : {os.path.basename(TRAIN_FILE)}\n")
        f.write(f"Test file  : {os.path.basename(TEST_FILE)}\n\n")
        for r in results:
            f.write(f"Config: {r['config']}\n")
            f.write(f"  Accuracy : {r['acc']:.4f}\n  Macro F1 : {r['macro_f1']:.4f}\n")
            f.write(f"  MCC      : {r['mcc']:.4f}\n  Kappa    : {r['kappa']:.4f}\n")
            for k, v in r.items():
                if k.startswith("f1_"):
                    f.write(f"  {k:<16}: {v:.4f}\n")
            f.write("\n")

    log("\n=== ABLATION SUMMARY ===")
    log(f"{'Config':<28} {'Acc':>8} {'MacroF1':>9} {'MCC':>8}")
    for r in results:
        log(f"{r['config']:<28} {r['acc']:>8.4f} {r['macro_f1']:>9.4f} {r['mcc']:>8.4f}")
    log(f"\nTotal time: {(time.time()-t_all)/60:.1f} min")
    log(f"Saved -> {ABLATION_CSV}")
    log(f"Saved -> {ABLATION_METRICS}")


if __name__ == "__main__":
    main()
