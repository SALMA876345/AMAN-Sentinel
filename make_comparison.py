"""
Build side-by-side comparison tables (2-class and 3-class) by computing all
metrics DIRECTLY from each model's prediction CSV — uniform and reliable,
instead of parsing differently-formatted metrics text files.

Outputs:
    results/comparison_2class.csv / .txt
    results/comparison_3class.csv / .txt
    results/comparison_all.txt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, cohen_kappa_score)

ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

# In-distribution run reads/writes the *_indist variants and ignores the
# cross-domain Kaggle OVERRIDES.
INDIST = os.environ.get("INDIST", "0").strip() == "1"
SUF    = "_indist" if INDIST else ""

LABELS3 = ["Real", "Fake", "Unverified"]
LABELS2 = ["Real", "Fake"]

# Results obtained on Kaggle and reported manually before the CSV is downloaded.
# These take precedence over any local CSV until the official files are dropped in.
# Keyed by (mode, model display name).
OVERRIDES = {
    # --- Kaggle 2-class results on the new balanced test (5,053), claim+evidence ---
    ("2", "BiLSTM"): {"Accuracy": 0.4989, "Precision": 0.51, "Recall": 0.51,
                      "Macro F1": 0.4854, "Weighted F1": 0.48, "MCC": 0.0146, "Kappa": None},
    ("2", "MARBERT (before)"): {"Accuracy": 0.4736, "Precision": 0.44, "Recall": 0.49,
                      "Macro F1": 0.3874, "Weighted F1": 0.41, "MCC": -0.0309, "Kappa": -0.0182},
    ("2", "MARBERT (after)"): {"Accuracy": 0.5652, "Precision": 0.56, "Recall": 0.56,
                      "Macro F1": 0.5536, "Weighted F1": 0.56, "MCC": 0.1234, "Kappa": 0.1199},
    ("3", "BiLSTM"): {"Accuracy": 0.2805, "Precision": 0.32, "Recall": 0.31,
                      "Macro F1": 0.2666, "Weighted F1": 0.31, "MCC": -0.0210, "Kappa": None},
    ("3", "MARBERT (after)"): {"Accuracy": 0.4570, "Precision": 0.37, "Recall": 0.35,
                      "Macro F1": 0.3481, "Weighted F1": 0.4743, "MCC": 0.0366, "Kappa": 0.0357},
}

# (display name, results-csv stem, prediction column)
MODELS = [
    ("SVM",                  "svm_claim_results",                 "svm_prediction"),
    ("Logistic Regression",  "lr_claim_results",                  "lr_prediction"),
    ("BiLSTM",               "bilstm_claim_results",              "bilstm_prediction"),
    ("AraBERT (before)",     "arabert_before_finetune_results",   "arabert_prediction"),
    ("AraBERT (after FT)",   "arabert_claim_results",             "arabert_prediction"),
    ("MARBERT (before)",     "marbert_before_finetune_results",   "marbert_prediction"),
    ("MARBERT (after)",      "marbert_claim_results",             "marbert_prediction"),
    ("Standard RAG",         "standard_rag_results",              "standard_rag_prediction"),
    ("SBERT-RAG",            "sbert_rag_results",                 "sbert_rag_prediction"),
    ("FAAV-Light",           "faithfulness_aware_rag_results",    "faav_light_prediction"),
    ("FAAV-Full (proposed)", "faav_full_results",                 "faav_prediction"),
]


def metrics_for(stem, pred_col, mode, labels):
    path = os.path.join(RESULTS_DIR, f"{stem}_{mode}class{SUF}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    if "gold_label" not in df.columns or pred_col not in df.columns:
        return None
    df = df[df["gold_label"].isin(labels) & df[pred_col].isin(labels)]
    if df.empty:
        return None
    yt, yp = df["gold_label"], df[pred_col]
    return {
        "Accuracy":    accuracy_score(yt, yp),
        "Precision":   precision_score(yt, yp, average="macro", labels=labels, zero_division=0),
        "Recall":      recall_score(yt, yp, average="macro", labels=labels, zero_division=0),
        "Macro F1":    f1_score(yt, yp, average="macro", labels=labels, zero_division=0),
        "Weighted F1": f1_score(yt, yp, average="weighted", labels=labels, zero_division=0),
        "MCC":         matthews_corrcoef(yt, yp),
        "Kappa":       cohen_kappa_score(yt, yp, labels=labels),
        "N":           len(df),
    }


def build(mode):
    labels = LABELS2 if mode == "2" else LABELS3
    rows = []
    for name, stem, col in MODELS:
        ov = None if INDIST else OVERRIDES.get((mode, name))
        m = ov if ov is not None else metrics_for(stem, col, mode, labels)
        if m is None:
            rows.append({"Model": name, "Accuracy": None, "Macro F1": None,
                         "Weighted F1": None, "MCC": None, "Kappa": None})
        else:
            rows.append({"Model": name, **{k: m[k] for k in
                        ["Accuracy", "Macro F1", "Weighted F1", "MCC", "Kappa"]}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"comparison_{mode}class{SUF}.csv"), index=False)
    return df


def main():
    blocks = []
    for mode in ["2", "3"]:
        df = build(mode)
        block = f"\n=== {mode}-CLASS COMPARISON (test on UAE-MVD 20% split) ===\n"
        block += df.to_string(index=False, na_rep="pending",
                              float_format=lambda v: f"{v:.4f}")
        print(block)
        blocks.append(block)
        with open(os.path.join(RESULTS_DIR, f"comparison_{mode}class{SUF}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(block)
    with open(os.path.join(RESULTS_DIR, "comparison_all.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print("\nSaved -> results/comparison_{2,3}class.* , comparison_all.txt")


if __name__ == "__main__":
    main()
