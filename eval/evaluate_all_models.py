"""
Unified evaluation & comparison across all models.

Run AFTER all model scripts have completed.  Loads every model's CSV,
computes a unified metric set on the same manual_claim_test_dataset.xlsx,
and prints / saves the comparison table requested in the assignment:

    Model               | Accuracy | Precision | Recall | F1 | Faithfulness | Halluc. | Grounding
    SVM                 |          |           |        |    |              |         |
    Logistic Regression |          |           |        |    |              |         |
    BiLSTM              |          |           |        |    |              |         |
    AraBERT             |          |           |        |    |              |         |
    Standard RAG        |          |           |        |    |              |         |
    FAAV-Light          |          |           |        |    |              |         |
    FAAV-Full (Proposed)|          |           |        |    |              |         |
"""
import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    matthews_corrcoef,
)

REPORT_TXT   = os.path.join(RESULTS_DIR, "all_models_comparison.txt")
REPORT_CSV   = os.path.join(RESULTS_DIR, "all_models_comparison.csv")

# ── Gold labels (full test set — used for models that evaluated all rows) ─────
df_gold            = pd.read_excel(TEST_FILE, sheet_name="Tabelle1")
gold_labels        = df_gold["gold_label"].tolist()
gold_faithfulness  = df_gold["faithfulness_label"].tolist() if "faithfulness_label" in df_gold.columns else []
N                  = len(gold_labels)

# ── Registry: model_name -> (filename, pred_col, faith_col) ───────────────────
# faith_col is the column that holds the predicted faithfulness label (or None)
MODEL_REGISTRY = [
    ("SVM",                  "svm_claim_results.csv",               "svm_prediction",        None),
    ("Logistic Regression",  "lr_claim_results.csv",                "lr_prediction",         None),
    ("BiLSTM",               "bilstm_claim_results.csv",            "bilstm_prediction",     None),
    ("AraBERT",              "arabert_claim_results.csv",           "arabert_prediction",    None),
    ("Standard RAG",         "standard_rag_results.csv",            "standard_rag_prediction", None),
    ("FAAV-Light (rule)",    "faithfulness_aware_rag_results.csv",  "faav_light_prediction", "predicted_faithfulness"),
    ("FAAV-Full (proposed)", "faav_full_results.csv",               "faav_prediction",       "predicted_faithfulness"),
    ("MARBERT",             "marbert_claim_results.csv",           "marbert_prediction",    None),
    ("SBERT-RAG",           "sbert_rag_results.csv",               "sbert_rag_prediction",  None),
]

rows = []

for model_name, filename, pred_col, faith_col in MODEL_REGISTRY:
    fpath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(fpath):
        print(f"[SKIP] {model_name}: {filename} not found — run the model script first.")
        continue

    df = pd.read_csv(fpath)

    if pred_col not in df.columns:
        print(f"[SKIP] {model_name}: column '{pred_col}' not found in {filename}.")
        continue

    preds = df[pred_col].tolist()

    # Use the gold labels from the model's own CSV if available (handles sampled subsets)
    if "gold_label" in df.columns:
        gold = df["gold_label"].tolist()
    else:
        n    = min(len(preds), N)
        gold = gold_labels[:n]
        preds = preds[:n]

    acc                         = accuracy_score(gold, preds)
    prec, rec, f1, _            = precision_recall_fscore_support(
        gold, preds, average="weighted", zero_division=0
    )
    macro_f1                    = f1_score(gold, preds, average="macro", zero_division=0)
    mcc                         = matthews_corrcoef(gold, preds)
    n = len(preds)

    # ── Faithfulness accuracy: filter out "Unknown" gold labels ──────────────
    faith_str = "N/A"
    if gold_faithfulness and faith_col and faith_col in df.columns:
        faith_preds_all = df[faith_col].tolist()[:n]
        gf_all          = gold_faithfulness[:n]
        # Only evaluate on rows where gold label is High or Low (not Unknown)
        pairs = [(g, p) for g, p in zip(gf_all, faith_preds_all) if g in ("High", "Low")]
        if pairs:
            gf_filt  = [p[0] for p in pairs]
            fp_filt  = [p[1] for p in pairs]
            faith_str = f"{accuracy_score(gf_filt, fp_filt):.4f} (n={len(pairs)})"

    # ── Hallucination rate ─────────────────────────────────────────────────
    halluc_str = "N/A"
    if "hallucination" in df.columns:
        halluc_str = f"{df['hallucination'].mean():.4f}"

    # ── Evidence grounding score ───────────────────────────────────────────
    ground_str = "N/A"
    if "grounding_score" in df.columns:
        ground_str = f"{df['grounding_score'].mean():.4f}"

    # ── Evidence relevance (retrieval quality) ─────────────────────────────
    ev_rel_str = "N/A"
    if "evidence_similarity" in df.columns:
        ev_rel_str = f"{df['evidence_similarity'].mean():.4f}"

    rows.append({
        "Model":              model_name,
        "N":                  n,
        "Accuracy":           round(acc,  4),
        "Precision":          round(prec, 4),
        "Recall":             round(rec,  4),
        "F1-Score":           round(f1,   4),
        "Macro F1":           round(macro_f1, 4),
        "MCC":                round(mcc, 4),
        "Faithfulness Acc":   faith_str,
        "Hallucination Rate": halluc_str,
        "Grounding Score":    ground_str,
        "Evidence Relevance": ev_rel_str,
    })
    print(f"[OK] {model_name:30s}  N={n:5d}  Acc={acc:.4f}  MacroF1={macro_f1:.4f}  MCC={mcc:.4f}  Faith={faith_str}")

if not rows:
    print("\nNo results found. Run all model scripts first.")
else:
    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    # ── Pretty-print table ─────────────────────────────────────────────────────
    W = 121
    sep = "=" * W

    header = (
        f"{'Model':<28} {'N':>5} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'MacroF1':>8} {'MCC':>8} {'Faithfulness':>14} {'Halluc.':>9} {'Grounding':>10} {'Ev.Rel.':>9}"
    )

    print(f"\n{sep}")
    print("FULL MODEL COMPARISON — configured test dataset")
    print(sep)
    print(header)
    print("-" * W)
    for _, r in df_cmp.iterrows():
        print(
            f"{r['Model']:<28} {str(r['N']):>5} {str(r['Accuracy']):>9} {str(r['Precision']):>10} "
            f"{str(r['Recall']):>8} {str(r['F1-Score']):>8} "
            f"{str(r['Macro F1']):>8} {str(r['MCC']):>8} "
            f"{str(r['Faithfulness Acc']):>14} {str(r['Hallucination Rate']):>9} "
            f"{str(r['Grounding Score']):>10} {str(r['Evidence Relevance']):>9}"
        )
    print(sep)
    print(f"Note: rows are read from {TEST_FILE}; models with their own gold_label column use that CSV's row count.")

    # ── Analysis narrative ─────────────────────────────────────────────────────
    def best(col):
        numeric = df_cmp[df_cmp[col].apply(lambda v: isinstance(v, (int, float)))]
        if numeric.empty:
            return "N/A"
        return numeric.loc[numeric[col].idxmax(), "Model"]

    ranked = df_cmp.sort_values(["Macro F1", "MCC"], ascending=False)
    best_model = ranked.iloc[0]["Model"]
    best_macro_f1 = ranked.iloc[0]["Macro F1"]
    best_mcc = ranked.iloc[0]["MCC"]
    narrative = f"""
ANALYSIS
========

This summary is generated from the CSVs loaded in this run; it does not reuse
fixed scores from earlier experiments.

Best model by Macro F1: {best_model} (Macro F1={best_macro_f1}, MCC={best_mcc})

Interpretation notes:
1. Treat Macro F1 and MCC as primary metrics because the Real/Fake/Unverified
   distribution can be imbalanced.
2. If evidence-aware models underperform claim-only models, inspect retrieval
   quality before changing the classifier head.
3. Use the per-domain outputs from experiments/faav_cross_domain.py for
   source/domain-specific conclusions; this aggregate table can hide transfer
   failures.
"""

    print(narrative)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("FULL MODEL COMPARISON REPORT\n")
        f.write("============================\n\n")
        f.write(header + "\n")
        f.write("-" * W + "\n")
        for _, r in df_cmp.iterrows():
            f.write(
                f"{r['Model']:<28} {str(r['N']):>5} {str(r['Accuracy']):>9} {str(r['Precision']):>10} "
                f"{str(r['Recall']):>8} {str(r['F1-Score']):>8} "
                f"{str(r['Macro F1']):>8} {str(r['MCC']):>8} "
                f"{str(r['Faithfulness Acc']):>14} {str(r['Hallucination Rate']):>9} "
                f"{str(r['Grounding Score']):>10} {str(r['Evidence Relevance']):>9}\n"
            )
        f.write("=" * W + "\n")
        f.write(f"Note: rows are read from {TEST_FILE}; models with their own gold_label column use that CSV's row count.\n")
        f.write(narrative)

    print(f"\nComparison table saved ->\n  {REPORT_TXT}\n  {REPORT_CSV}")
