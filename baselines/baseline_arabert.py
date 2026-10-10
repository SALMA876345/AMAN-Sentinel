"""
AraBERT — before fine-tuning evaluation only (3-class: Real / Fake / Unverified)
Model : aubmindlab/bert-base-arabertv02

Runs Phase 1 only: loads the pre-trained model with a RANDOM classification
head (no training on our data) and evaluates on the test set.
This gives the zero-shot Arabic NLP baseline, used as a comparison point
to show how much fine-tuning improves MARBERT.

Fine-tuning AraBERT is skipped to save time — MARBERT covers the
fine-tuned Arabic transformer comparison.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TEST_FILE, TEST_SHEET, RESULTS_DIR,
    CLASS_MODE, CLASS_LABELS, MODEL_INPUT_MODE, SEED, result_tagged,
)
from utils.faav_common import normalize_arabic, set_seed

ARABERT_BEFORE_CSV     = result_tagged("arabert_before_finetune_results.csv")
ARABERT_BEFORE_METRICS = result_tagged("arabert_before_finetune_metrics.txt")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS     = CLASS_LABELS



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
            y_proba,
            multi_class="ovr", average="macro",
        )
    except Exception:
        auc = float("nan")
    return {"acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "mcc": mcc, "kappa": kappa, "auc": auc,
            "report": report, "matrix": matrix}


def save_metrics(path, title, m, model_name, n_train, phase="After fine-tuning"):
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

    le          = LabelEncoder().fit(LABELS)
    num_classes = len(le.classes_)

    # ── Load test data ────────────────────────────────────────────────────────
    print(f"=== AraBERT before fine-tuning — {CLASS_MODE}-class {LABELS} | input={MODEL_INPUT_MODE} ===")
    df_test     = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
    df_test      = df_test[df_test["gold_label"].isin(LABELS)].reset_index(drop=True)
    test_claims = [normalize_arabic(str(c)) for c in df_test["claim_ar"]]
    test_evs    = [normalize_arabic(str(e)) for e in df_test["evidence_text"].fillna("")]
    inference_evs = None if MODEL_INPUT_MODE == "claim_only" else test_evs
    print(f"Test rows: {len(df_test)}  |  {df_test['gold_label'].value_counts().to_dict()}")

    # ── Tokenizer + model (pre-trained only, random head) ─────────────────────
    print(f"\nLoading pre-trained model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(DEVICE)
    print(f"Device: {DEVICE}  |  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1 ONLY — Before fine-tuning (random classification head)
    # Gives zero-shot baseline: how well does pre-trained Arabic NLP do
    # on fake news detection without any task-specific training?
    # ════════════════════════════════════════════════════════════════════════
    print("\n--- Evaluating (before fine-tuning / no training) ---")
    preds, proba = run_inference(model, tokenizer, test_claims, inference_evs, le)
    m = compute_all_metrics(df_test["gold_label"].tolist(), preds, proba, LABELS)

    print(f"Accuracy : {m['acc']:.4f}")
    print(f"Macro F1 : {m['macro_f1']:.4f}")
    print(f"MCC      : {m['mcc']:.4f}")
    print(f"Kappa    : {m['kappa']:.4f}")
    print(f"AUC-ROC  : {m['auc']:.4f}")
    print("\nClassification Report:")
    print(m["report"])

    df_out = df_test.copy()
    df_out["arabert_prediction"] = preds
    df_out["is_correct"]         = df_out["gold_label"] == df_out["arabert_prediction"]
    df_out.to_csv(ARABERT_BEFORE_CSV, index=False, encoding="utf-8-sig")
    save_metrics(ARABERT_BEFORE_METRICS, "AraBERT — Before Fine-tuning (Zero-shot)",
                 m, MODEL_NAME, n_train=0, phase="Before fine-tuning (random head)")

    print(f"\nResults saved -> {ARABERT_BEFORE_CSV}")
    print(f"Metrics saved -> {ARABERT_BEFORE_METRICS}")
    print("\nNote: AraBERT fine-tuning skipped — MARBERT covers the fine-tuned transformer comparison.")


if __name__ == "__main__":
    main()
