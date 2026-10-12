"""Phase 2: fine-tune (or linear-probe) a strong Arabic transformer on the
clean, leakage-free SOURCE-DOMAIN-ONLY binary split that Phase 1 produced.

Follows the same overall shape as hackathon/train_bilstm_clean.py: load the
three clean CSVs, build model_text from claim_norm/evidence_norm, train with
validation-macro-F1 early stopping (patience 1), persist a config.json with
full training history + device + wall-clock, and a clean_evaluation_metrics
.json computed with utils.faav_common.compute_classification_metrics so
numbers are directly comparable to the SVM/BiLSTM baselines.

Model choice
------------
Prefer UBC-NLP/MARBERTv2 (MARBERT). Fall back to aubmindlab/bert-base-arabertv02
(AraBERT) only if MARBERT cannot be obtained. Both are already supported by
this repo's historical pipeline (experiments/final_studies_neural.py), so
staying within that pair keeps Phase 2 within the project's existing scope.

Hardware-aware strategy selection
----------------------------------
This machine is torch 2.12.0+cpu (no CUDA). Full BERT-base fine-tuning over
~14k rows can be very slow on CPU. Before committing to a full run we time a
handful of real forward+backward+optimizer-step batches on the actual model
and extrapolate a full-epoch wall clock. If 3 epochs of full fine-tuning
(plus data loading/tokenization/eval) looks tractable inside a ~45-60 minute
total budget, we do full fine-tuning. Otherwise we fall back to a frozen
backbone linear probe: freeze all transformer parameters, forward each split
through the frozen encoder ONCE to cache mean-pooled sentence embeddings,
then train only a small linear classification head on those cached
features (cheap: no backprop through the transformer, so many epochs are
affordable). Whichever path is taken is recorded honestly in config.json
and in the console output -- never silently relabelled.

Tokenization convention (matches experiments/final_studies_neural.py lines
~428-442): claim_norm capped to 400 chars, evidence_norm capped to 1200
chars before tokenization, sentence-pair encoding (claim, evidence) with
max_length=128, truncation=True, padding="max_length".
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    torch.set_num_threads(max(os.cpu_count() or 1, 1))
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]  # "Code files/"
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics, set_seed  # noqa: E402

CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
ARTIFACT_DIR = ROOT / "artifacts" / "transformers"
SEED = 42
TASK = "binary"
LABELS = ["Real", "Fake"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

CLAIM_CHAR_CAP = 400
EVIDENCE_CHAR_CAP = 1200
MAX_LEN = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EMBED_BATCH_SIZE = 64  # forward-only (no backward) caching pass for the linear probe
MAX_EPOCHS = 3
EARLY_STOP_PATIENCE = 1
LINEAR_PROBE_MAX_EPOCHS = 60
LINEAR_PROBE_PATIENCE = 5
LEARNING_RATE_FULL = 2e-5
LEARNING_RATE_PROBE = 1e-3
WEIGHT_DECAY = 0.01

# Total wall-clock budget for the decision between full fine-tuning and the
# frozen linear-probe fallback (data loading + tokenization + N epochs + eval).
TIME_BUDGET_SECONDS = 50 * 60

CANDIDATE_MODELS = [
    ("marbert", "UBC-NLP/MARBERTv2"),
    ("arabert", "aubmindlab/bert-base-arabertv02"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_split(name: str) -> pd.DataFrame:
    path = CLEAN_DIR / f"source_{TASK}_{name}.csv"
    df = pd.read_csv(path, low_memory=False).fillna("")
    df["claim_capped"] = df["claim_norm"].astype(str).str.slice(0, CLAIM_CHAR_CAP)
    df["evidence_capped"] = df["evidence_norm"].astype(str).str.slice(0, EVIDENCE_CHAR_CAP)
    return df


def try_load_base_model():
    """Try MARBERT first, fall back to AraBERT. Returns (model_key, base_model_name,
    tokenizer, reason) or raises if neither is obtainable."""
    last_error = None
    for model_key, base_model in CANDIDATE_MODELS:
        try:
            from transformers import AutoTokenizer

            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            log(f"Loaded tokenizer for {base_model} in {time.time() - t0:.1f}s")
            return model_key, base_model, tokenizer, None
        except Exception as exc:  # noqa: BLE001
            last_error = f"{base_model}: {type(exc).__name__}: {exc}"
            log(f"Could not obtain {base_model}: {last_error}")
    raise RuntimeError(f"Neither MARBERT nor AraBERT could be obtained. Last error: {last_error}")


def encode_frame(tokenizer, frame: pd.DataFrame):
    encoded = tokenizer(
        frame["claim_capped"].tolist(),
        frame["evidence_capped"].tolist(),
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded


def make_loader(encoded, labels_ids, batch_size, shuffle):
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if "token_type_ids" in encoded:
        tensors.append(encoded["token_type_ids"])
    tensors.append(torch.tensor(labels_ids, dtype=torch.long))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def unpack_batch(batch):
    if len(batch) == 4:
        input_ids, attention_mask, token_type_ids, targets = batch
        inputs = {
            "input_ids": input_ids.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE),
            "token_type_ids": token_type_ids.to(DEVICE),
        }
    else:
        input_ids, attention_mask, targets = batch
        inputs = {
            "input_ids": input_ids.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE),
        }
    return inputs, targets.to(DEVICE)


def probe_full_finetune_epoch_seconds(model, train_loader, n_probe_batches=3) -> float:
    """Time a handful of real forward+backward+optimizer-step batches on the
    actual model/data to extrapolate a full-epoch wall clock for full
    fine-tuning. Returns estimated seconds for one full epoch over the whole
    train_loader."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_FULL, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    times = []
    it = iter(train_loader)
    for i in range(n_probe_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        inputs, targets = unpack_batch(batch)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).logits
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        times.append(time.time() - t0)
    per_batch = float(np.mean(times)) if times else float("inf")
    num_batches = len(train_loader)
    est_epoch_seconds = per_batch * num_batches
    log(
        f"Probe: {len(times)} batches, per-batch={per_batch:.2f}s, "
        f"num_batches/epoch={num_batches}, estimated full-epoch={est_epoch_seconds:.0f}s"
    )
    return est_epoch_seconds, per_batch


@torch.inference_mode()
def predict_full_model(model, loader, labels):
    model.eval()
    preds, confs, probs = [], [], []
    for batch in loader:
        inputs, _ = unpack_batch(batch)
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits.float(), dim=1)
        values, indices = probabilities.max(dim=1)
        preds.extend(labels[i] for i in indices.cpu().tolist())
        confs.extend(values.cpu().tolist())
        probs.extend(probabilities.cpu().tolist())
    return preds, confs, probs


def run_full_finetune(tokenizer, base_model, train, validation, test):
    from transformers import AutoModelForSequenceClassification

    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(DEVICE)

    train_enc = encode_frame(tokenizer, train)
    val_enc = encode_frame(tokenizer, validation)
    test_enc = encode_frame(tokenizer, test)
    log(f"Tokenization done in {time.time() - t0:.1f}s")

    train_ids = [LABEL_TO_ID[x] for x in train["task_label"]]
    val_ids = [LABEL_TO_ID[x] for x in validation["task_label"]]
    test_ids = [LABEL_TO_ID[x] for x in test["task_label"]]

    train_loader = make_loader(train_enc, train_ids, BATCH_SIZE, True)
    val_loader = make_loader(val_enc, val_ids, EVAL_BATCH_SIZE, False)
    test_loader = make_loader(test_enc, test_ids, EVAL_BATCH_SIZE, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_FULL, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inputs, targets = unpack_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_preds, _, _ = predict_full_model(model, val_loader, LABELS)
        from sklearn.metrics import f1_score

        val_macro_f1 = f1_score(validation["task_label"], val_preds, labels=LABELS, average="macro", zero_division=0)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_macro_f1": float(val_macro_f1),
            "elapsed_seconds": round(elapsed, 1),
        }
        history.append(row)
        log(f"  epoch={epoch} loss={row['train_loss']:.4f} val_macro_f1={val_macro_f1:.4f} elapsed={elapsed:.0f}s")
        if val_macro_f1 > best_score + 1e-5:
            best_score, best_epoch, stale = val_macro_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.to(DEVICE)

    val_preds, val_confs, val_probs = predict_full_model(model, val_loader, LABELS)
    test_preds, test_confs, test_probs = predict_full_model(model, test_loader, LABELS)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "best_epoch": best_epoch,
        "val_preds": val_preds,
        "val_confs": val_confs,
        "val_probs": val_probs,
        "test_preds": test_preds,
        "test_confs": test_confs,
        "test_probs": test_probs,
    }


class LinearProbeHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.classifier(x)


@torch.inference_mode()
def mean_pool_embeddings(backbone, loader, split_name: str = "") -> torch.Tensor:
    backbone.eval()
    chunks = []
    n_batches = len(loader)
    t0 = time.time()
    for i, batch in enumerate(loader, start=1):
        inputs, _ = unpack_batch(batch)
        attention_mask = inputs["attention_mask"]
        outputs = backbone(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts
        chunks.append(pooled.cpu())
        if i == 1 or i % 20 == 0 or i == n_batches:
            log(f"    [{split_name}] embedding batch {i}/{n_batches} elapsed={time.time() - t0:.0f}s")
    return torch.cat(chunks, dim=0)


def run_linear_probe(tokenizer, base_model, train, validation, test):
    from transformers import AutoModel

    t0 = time.time()
    backbone = AutoModel.from_pretrained(base_model).to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad = False
    hidden_size = backbone.config.hidden_size

    train_enc = encode_frame(tokenizer, train)
    val_enc = encode_frame(tokenizer, validation)
    test_enc = encode_frame(tokenizer, test)
    log(f"Tokenization done in {time.time() - t0:.1f}s")

    train_ids = [LABEL_TO_ID[x] for x in train["task_label"]]
    val_ids = [LABEL_TO_ID[x] for x in validation["task_label"]]
    test_ids = [LABEL_TO_ID[x] for x in test["task_label"]]

    train_loader = make_loader(train_enc, train_ids, EMBED_BATCH_SIZE, False)
    val_loader = make_loader(val_enc, val_ids, EMBED_BATCH_SIZE, False)
    test_loader = make_loader(test_enc, test_ids, EMBED_BATCH_SIZE, False)

    t_embed = time.time()
    train_feats = mean_pool_embeddings(backbone, train_loader, "train")
    val_feats = mean_pool_embeddings(backbone, val_loader, "validation")
    test_feats = mean_pool_embeddings(backbone, test_loader, "test")
    log(f"Cached frozen embeddings for train/val/test in {time.time() - t_embed:.1f}s")

    train_targets = torch.tensor(train_ids, dtype=torch.long)
    val_targets = torch.tensor(val_ids, dtype=torch.long)
    test_targets = torch.tensor(test_ids, dtype=torch.long)

    head = LinearProbeHead(hidden_size, len(LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE_PROBE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    feat_loader = DataLoader(
        TensorDataset(train_feats, train_targets), batch_size=128, shuffle=True
    )

    from sklearn.metrics import f1_score

    history = []
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    for epoch in range(1, LINEAR_PROBE_MAX_EPOCHS + 1):
        head.train()
        losses = []
        for feats, targets in feat_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = head(feats)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        head.eval()
        with torch.inference_mode():
            val_logits = head(val_feats.to(DEVICE))
            val_pred_ids = val_logits.argmax(dim=1).cpu().tolist()
        val_preds = [LABELS[i] for i in val_pred_ids]
        val_macro_f1 = f1_score(validation["task_label"], val_preds, labels=LABELS, average="macro", zero_division=0)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_macro_f1": float(val_macro_f1),
            "elapsed_seconds": round(elapsed, 1),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            log(f"  epoch={epoch} loss={row['train_loss']:.4f} val_macro_f1={val_macro_f1:.4f} elapsed={elapsed:.0f}s")
        if val_macro_f1 > best_score + 1e-5:
            best_score, best_epoch, stale = val_macro_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= LINEAR_PROBE_PATIENCE:
                break
    log(f"  linear probe best_epoch={best_epoch} best_val_macro_f1={best_score:.4f}")

    head.load_state_dict(best_state)
    head.to(DEVICE)

    def predict_from_feats(feats):
        head.eval()
        with torch.inference_mode():
            logits = head(feats.to(DEVICE))
            probs = torch.softmax(logits.float(), dim=1)
            values, indices = probs.max(dim=1)
        preds = [LABELS[i] for i in indices.cpu().tolist()]
        return preds, values.cpu().tolist(), probs.cpu().tolist()

    val_preds, val_confs, val_probs = predict_from_feats(val_feats)
    test_preds, test_confs, test_probs = predict_from_feats(test_feats)

    return {
        "backbone": backbone,
        "head": head,
        "tokenizer": tokenizer,
        "history": history,
        "best_epoch": best_epoch,
        "val_preds": val_preds,
        "val_confs": val_confs,
        "val_probs": val_probs,
        "test_preds": test_preds,
        "test_confs": test_confs,
        "test_probs": test_probs,
    }


def build_predictions_df(frame, preds, confs, probs):
    prob_real = [p[LABEL_TO_ID["Real"]] for p in probs]
    prob_fake = [p[LABEL_TO_ID["Fake"]] for p in probs]
    return pd.DataFrame(
        {
            "sample_id": frame["sample_id"].tolist(),
            "true_label": frame["task_label"].tolist(),
            "prediction": preds,
            "confidence": confs,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
        }
    )


def main() -> None:
    t_start = time.time()
    set_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Device: {DEVICE}")

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    log(f"Clean split rows -> train={len(train):,} validation={len(validation):,} test={len(test):,}")

    model_key, base_model, tokenizer, _ = try_load_base_model()
    log(f"Selected base model: {model_key} ({base_model})")

    # --- Probe: time a handful of real full-fine-tune batches to decide strategy ---
    from transformers import AutoModelForSequenceClassification

    probe_sample = train.sample(n=min(64, len(train)), random_state=SEED)
    probe_enc = encode_frame(tokenizer, probe_sample)
    probe_ids = [LABEL_TO_ID[x] for x in probe_sample["task_label"]]
    probe_loader = make_loader(probe_enc, probe_ids, BATCH_SIZE, True)

    probe_model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    ).to(DEVICE)
    full_train_loader_len = int(np.ceil(len(train) / BATCH_SIZE))
    est_epoch_seconds, per_batch_seconds = probe_full_finetune_epoch_seconds(
        probe_model, probe_loader, n_probe_batches=3
    )
    # extrapolate using the real train-set batch count (probe_loader uses a tiny subset)
    est_epoch_seconds_full = per_batch_seconds * full_train_loader_len
    est_total_full_finetune = est_epoch_seconds_full * MAX_EPOCHS
    del probe_model
    log(
        f"Probe-based estimate: per_batch={per_batch_seconds:.2f}s, "
        f"train_batches/epoch={full_train_loader_len}, "
        f"est_epoch={est_epoch_seconds_full:.0f}s, "
        f"est_total_for_{MAX_EPOCHS}_epochs={est_total_full_finetune:.0f}s "
        f"(budget={TIME_BUDGET_SECONDS}s)"
    )

    elapsed_so_far = time.time() - t_start
    remaining_budget = TIME_BUDGET_SECONDS - elapsed_so_far
    use_full_finetune = est_total_full_finetune < remaining_budget * 0.85

    if use_full_finetune:
        strategy = "full_fine_tune"
        strategy_reason = (
            f"Probe measured {per_batch_seconds:.2f}s/batch on the real model; "
            f"extrapolated full-epoch time {est_epoch_seconds_full:.0f}s and "
            f"{MAX_EPOCHS}-epoch total {est_total_full_finetune:.0f}s fit inside the "
            f"remaining ~{remaining_budget:.0f}s budget, so full fine-tuning "
            f"(all parameters trainable) was judged tractable."
        )
        log(f"STRATEGY: {strategy} -- {strategy_reason}")
        result = run_full_finetune(tokenizer, base_model, train, validation, test)
    else:
        strategy = "frozen_linear_probe"
        strategy_reason = (
            f"Probe measured {per_batch_seconds:.2f}s/batch on the real model; "
            f"extrapolated full-epoch time {est_epoch_seconds_full:.0f}s and "
            f"{MAX_EPOCHS}-epoch total {est_total_full_finetune:.0f}s EXCEEDED the "
            f"remaining ~{remaining_budget:.0f}s budget on this CPU-only machine, so "
            f"full fine-tuning was judged intractable. Falling back to a frozen-backbone "
            f"linear probe: all transformer parameters frozen, mean-pooled last-hidden-state "
            f"embeddings cached once per split, and only a linear classification head "
            f"trained on those cached features."
        )
        log(f"STRATEGY: {strategy} -- {strategy_reason}")
        result = run_linear_probe(tokenizer, base_model, train, validation, test)

    val_metrics = compute_classification_metrics(validation["task_label"], result["val_preds"], LABELS)
    test_metrics = compute_classification_metrics(test["task_label"], result["test_preds"], LABELS)
    log(
        f"\nValidation (clean split): accuracy={val_metrics['accuracy']:.4f} "
        f"macro_f1={val_metrics['macro_f1']:.4f} mcc={val_metrics['mcc']:.4f}"
    )
    log(
        f"Test (clean split): accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} mcc={test_metrics['mcc']:.4f}"
    )

    # --- Persist model + tokenizer ---
    model_dir = ARTIFACT_DIR / f"{model_key}_clean_binary"
    model_dir.mkdir(parents=True, exist_ok=True)
    if strategy == "full_fine_tune":
        result["model"].save_pretrained(model_dir)
        result["tokenizer"].save_pretrained(model_dir)
    else:
        # Save the frozen backbone (unmodified weights) + tokenizer so the
        # encoder can be reloaded with AutoModel, plus the trained linear head.
        result["backbone"].save_pretrained(model_dir)
        result["tokenizer"].save_pretrained(model_dir)
        torch.save(result["head"].state_dict(), model_dir / "linear_probe_head.pt")
        (model_dir / "linear_probe_head_config.json").write_text(
            json.dumps(
                {
                    "hidden_size": result["backbone"].config.hidden_size,
                    "num_labels": len(LABELS),
                    "labels": LABELS,
                    "pooling": "mean_pool_last_hidden_state_masked",
                    "note": "Backbone parameters are frozen/unmodified from the base "
                    "checkpoint; only linear_probe_head.pt was trained. Reload backbone "
                    "with AutoModel.from_pretrained(this_dir), then apply mean-pooling "
                    "over the masked last_hidden_state, then this Linear head.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    log(f"Saved model/tokenizer to {model_dir}")

    # --- Predictions CSVs ---
    val_df = build_predictions_df(validation, result["val_preds"], result["val_confs"], result["val_probs"])
    test_df = build_predictions_df(test, result["test_preds"], result["test_confs"], result["test_probs"])
    val_df.to_csv(ARTIFACT_DIR / "clean_validation_predictions.csv", index=False, encoding="utf-8")
    test_df.to_csv(ARTIFACT_DIR / "clean_test_predictions.csv", index=False, encoding="utf-8")

    # --- config.json ---
    total_wall_clock = time.time() - t_start
    config = {
        "base_model_key": model_key,
        "base_model_name": base_model,
        "model_choice_reason": (
            "MARBERT (UBC-NLP/MARBERTv2) was preferred per the Phase 2 brief and was "
            "successfully obtained (tokenizer/weights downloaded or found in local HF "
            "cache)."
            if model_key == "marbert"
            else "UBC-NLP/MARBERTv2 could not be obtained within budget; fell back to "
            "aubmindlab/bert-base-arabertv02, the other transformer already supported "
            "by this repo's historical pipeline (experiments/final_studies_neural.py)."
        ),
        "strategy": strategy,
        "strategy_reason": strategy_reason,
        "probe": {
            "probe_batches_timed": 3,
            "probe_batch_size": BATCH_SIZE,
            "measured_seconds_per_batch": per_batch_seconds,
            "train_batches_per_epoch": full_train_loader_len,
            "estimated_full_finetune_epoch_seconds": est_epoch_seconds_full,
            "estimated_full_finetune_total_seconds_for_3_epochs": est_total_full_finetune,
            "time_budget_seconds": TIME_BUDGET_SECONDS,
        },
        "hyperparameters": {
            "max_len": MAX_LEN,
            "claim_character_cap": CLAIM_CHAR_CAP,
            "evidence_character_cap": EVIDENCE_CHAR_CAP,
            "batch_size": BATCH_SIZE if strategy == "full_fine_tune" else EMBED_BATCH_SIZE,
            "learning_rate": LEARNING_RATE_FULL if strategy == "full_fine_tune" else LEARNING_RATE_PROBE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS if strategy == "full_fine_tune" else LINEAR_PROBE_MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOP_PATIENCE if strategy == "full_fine_tune" else LINEAR_PROBE_PATIENCE,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "pooling_if_linear_probe": "mean_pool_last_hidden_state_masked",
        },
        "labels": LABELS,
        "best_epoch": result["best_epoch"],
        "training_history": result["history"],
        "device_trained_on": str(DEVICE),
        "total_wall_clock_seconds": round(total_wall_clock, 1),
        "data_source": "hackathon/data/clean_split/",
        "validation_metrics_at_best_epoch": {
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "mcc": val_metrics["mcc"],
        },
    }
    (ARTIFACT_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- clean_evaluation_metrics.json ---
    metrics_out = {
        "test_metrics": test_metrics,
        "test_rows": int(len(test)),
        "calibration_note": (
            "These confidence/probability values come directly from the trained "
            "classifier head's softmax output (raw softmax over 2 classes). They are "
            "NOT temperature-scaled or otherwise post-hoc calibrated -- that calibration "
            "step, if any, happens later in Phase 2 and is out of scope for this script. "
            "Treat 'confidence'/'prob_real'/'prob_fake' in the predictions CSVs as "
            "pre-calibration raw softmax outputs."
        ),
    }
    (ARTIFACT_DIR / "clean_evaluation_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log(f"\nArtifacts saved under {ARTIFACT_DIR}")
    log(f"Total wall-clock: {total_wall_clock:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
