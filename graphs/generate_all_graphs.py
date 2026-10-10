"""
Generates all 9 research-quality graphs from saved results files.
Run AFTER all models have been executed.

Output: graphs/*.png  (300 DPI, publication ready)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, GRAPHS_DIR

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc

os.makedirs(GRAPHS_DIR, exist_ok=True)

LABELS   = ["Real", "Fake", "Unverified"]
PALETTE  = {"Real": "#2ecc71", "Fake": "#e74c3c", "Unverified": "#f39c12"}
BG_COLOR = "#f8f9fa"
DPI      = 300


def savefig(name):
    path = os.path.join(GRAPHS_DIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


def load_metrics_value(filepath, key):
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if key in line:
                    return float(line.split(":")[-1].strip())
    except Exception:
        pass
    return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 1 — Before vs After Fine-tuning: Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def graph1_before_after_accuracy():
    print("Graph 1: Before vs After Fine-tuning Accuracy ...")
    models = ["AraBERT", "MARBERT"]
    files  = {
        "AraBERT":  (os.path.join(RESULTS_DIR, "arabert_before_finetune_metrics.txt"),
                     os.path.join(RESULTS_DIR, "arabert_metrics.txt")),
        "MARBERT":  (os.path.join(RESULTS_DIR, "marbert_before_finetune_metrics.txt"),
                     os.path.join(RESULTS_DIR, "marbert_metrics.txt")),
    }
    before_acc = [load_metrics_value(files[m][0], "Accuracy") for m in models]
    after_acc  = [load_metrics_value(files[m][1], "Accuracy") for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    b1 = ax.bar(x - w/2, before_acc, w, label="Before Fine-tuning", color="#95a5a6", edgecolor="white")
    b2 = ax.bar(x + w/2, after_acc,  w, label="After Fine-tuning",  color="#3498db", edgecolor="white")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy: Before vs After Fine-tuning", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    savefig("01_before_after_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 2 — All Models Accuracy Comparison
# ─────────────────────────────────────────────────────────────────────────────
def graph2_all_models_accuracy():
    print("Graph 2: All Models Accuracy Comparison ...")
    model_files = {
        "SVM":           os.path.join(RESULTS_DIR, "svm_metrics.txt"),
        "LR":            os.path.join(RESULTS_DIR, "lr_claim_metrics.txt"),
        "BiLSTM":        os.path.join(RESULTS_DIR, "bilstm_metrics.txt"),
        "Std RAG":       os.path.join(RESULTS_DIR, "standard_rag_metrics.txt"),
        "FAAV-Light":    os.path.join(RESULTS_DIR, "faithfulness_aware_rag_metrics.txt"),
        "AraBERT":       os.path.join(RESULTS_DIR, "arabert_metrics.txt"),
        "MARBERT":       os.path.join(RESULTS_DIR, "marbert_metrics.txt"),
        "MARBERT+FAAV":  os.path.join(RESULTS_DIR, "faav_full_metrics.txt"),
    }
    names = list(model_files.keys())
    accs  = [load_metrics_value(f, "Accuracy") for f in model_files.values()]

    colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#e67e22", "#e67e22",
              "#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(names, accs, color=colors, edgecolor="white", width=0.6)
    for bar, acc in zip(bars, accs):
        if not np.isnan(acc):
            ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f"{acc:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison — All Models", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.33, color="gray", linestyle="--", alpha=0.5, label="Random baseline (33%)")
    ax.legend(fontsize=9)
    savefig("02_all_models_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 3 — All Models Macro F1 Comparison
# ─────────────────────────────────────────────────────────────────────────────
def graph3_all_models_macro_f1():
    print("Graph 3: All Models Macro F1 ...")
    model_files = {
        "SVM":          os.path.join(RESULTS_DIR, "svm_metrics.txt"),
        "LR":           os.path.join(RESULTS_DIR, "lr_claim_metrics.txt"),
        "BiLSTM":       os.path.join(RESULTS_DIR, "bilstm_metrics.txt"),
        "Std RAG":      os.path.join(RESULTS_DIR, "standard_rag_metrics.txt"),
        "FAAV-Light":   os.path.join(RESULTS_DIR, "faithfulness_aware_rag_metrics.txt"),
        "AraBERT":      os.path.join(RESULTS_DIR, "arabert_metrics.txt"),
        "MARBERT":      os.path.join(RESULTS_DIR, "marbert_metrics.txt"),
        "MARBERT+FAAV": os.path.join(RESULTS_DIR, "faav_full_metrics.txt"),
    }
    names  = list(model_files.keys())
    values = [load_metrics_value(f, "Macro F1") for f in model_files.values()]

    colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#e67e22", "#e67e22",
              "#3498db", "#2ecc71", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1-score", fontsize=12)
    ax.set_title("Macro F1 Comparison — All Models", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    savefig("03_all_models_macro_f1.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 4 — Per-class F1 (Real / Fake / Unverified)
# ─────────────────────────────────────────────────────────────────────────────
def graph4_per_class_f1():
    print("Graph 4: Per-class F1 ...")
    results_csv = {
        "SVM":          os.path.join(RESULTS_DIR, "svm_claim_results.csv"),
        "LR":           os.path.join(RESULTS_DIR, "lr_claim_results.csv"),
        "AraBERT":      os.path.join(RESULTS_DIR, "arabert_claim_results.csv"),
        "MARBERT":      os.path.join(RESULTS_DIR, "marbert_claim_results.csv"),
        "MARBERT+FAAV": os.path.join(RESULTS_DIR, "faav_full_results.csv"),
    }
    pred_cols = {
        "SVM":          "svm_prediction",
        "LR":           "lr_prediction",
        "AraBERT":      "arabert_prediction",
        "MARBERT":      "marbert_prediction",
        "MARBERT+FAAV": "faav_prediction",
    }
    from sklearn.metrics import f1_score as skf1
    data = {lbl: [] for lbl in LABELS}
    model_names = []
    for model, csv_path in results_csv.items():
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        pred_col = pred_cols[model]
        if pred_col not in df.columns or "gold_label" not in df.columns: continue
        f1s = skf1(df["gold_label"], df[pred_col], labels=LABELS,
                   average=None, zero_division=0)
        for i, lbl in enumerate(LABELS):
            data[lbl].append(f1s[i] if i < len(f1s) else 0)
        model_names.append(model)

    if not model_names: print("  Skipping — no results yet"); return

    x   = np.arange(len(model_names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for i, lbl in enumerate(LABELS):
        ax.bar(x + (i-1)*w, data[lbl], w, label=lbl,
               color=list(PALETTE.values())[i], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title("Per-class F1 Score by Model", fontsize=14, fontweight="bold")
    ax.legend(title="Class", fontsize=10); ax.grid(axis="y", alpha=0.3)
    savefig("04_per_class_f1.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 5 — Confusion Matrices (best 3 models)
# ─────────────────────────────────────────────────────────────────────────────
def graph5_confusion_matrices():
    print("Graph 5: Confusion Matrices ...")
    from sklearn.metrics import confusion_matrix
    configs = [
        ("AraBERT (before)",  os.path.join(RESULTS_DIR, "arabert_before_finetune_results.csv"), "arabert_prediction"),
        ("MARBERT",           os.path.join(RESULTS_DIR, "marbert_claim_results.csv"),           "marbert_prediction"),
        ("FAAV-Full",         os.path.join(RESULTS_DIR, "faav_full_results.csv"),               "faav_prediction"),
    ]
    valid = [(n, p, c) for n, p, c in configs if os.path.exists(p)]
    if not valid: print("  Skipping — no results yet"); return

    fig, axes = plt.subplots(1, len(valid), figsize=(6*len(valid), 5), facecolor=BG_COLOR)
    if len(valid) == 1: axes = [axes]
    for ax, (name, path, col) in zip(axes, valid):
        df = pd.read_csv(path)
        if col not in df.columns: continue
        # skip stale files with old label names
        if not any(lbl in df["gold_label"].values for lbl in LABELS): continue
        df = df[df["gold_label"].isin(LABELS)]
        cm = confusion_matrix(df["gold_label"], df[col], labels=LABELS)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS,
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig("05_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 6 — Cross-validation Box Plot
# ─────────────────────────────────────────────────────────────────────────────
def graph6_cross_validation():
    print("Graph 6: Cross-validation ...")
    cv_csv = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
    if not os.path.exists(cv_csv): print("  Skipping — run cross_validation.py first"); return
    df = pd.read_csv(cv_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG_COLOR)
    for ax, (col, ylabel, title) in zip(axes, [
        ("acc_mean",      "Accuracy",  "5-Fold CV Accuracy"),
        ("macro_f1_mean", "Macro F1",  "5-Fold CV Macro F1"),
    ]):
        vals  = df[col].values
        stds  = df[col.replace("mean", "std")].values
        names = df["model"].values
        ax.set_facecolor(BG_COLOR)
        bars = ax.bar(names, vals, color=["#3498db", "#2ecc71"],
                      edgecolor="white", width=0.5)
        ax.errorbar(names, vals, yerr=stds, fmt="none", color="black",
                    capsize=6, linewidth=1.5)
        for bar, v, s in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width()/2, v + s + 0.01,
                    f"{v:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.0); ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("06_cross_validation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 7 — Ablation Study
# ─────────────────────────────────────────────────────────────────────────────
def graph7_ablation():
    print("Graph 7: Ablation Study ...")
    abl_csv = os.path.join(RESULTS_DIR, "ablation_results.csv")
    if not os.path.exists(abl_csv): print("  Skipping — run ablation_study.py first"); return
    df = pd.read_csv(abl_csv)

    full_f1 = df[df["config"] == "Full MARBERT"]["macro_f1"].values[0]
    df["f1_drop"] = full_f1 - df["macro_f1"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    colors = ["#2ecc71" if r == "Full MARBERT" else "#e74c3c" for r in df["config"]]
    bars = ax.barh(df["config"], df["macro_f1"], color=colors, edgecolor="white")
    for bar, v in zip(bars, df["macro_f1"]):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Macro F1-score", fontsize=12)
    ax.set_title("Ablation Study — MARBERT Component Contribution", fontsize=14, fontweight="bold")
    ax.axvline(x=full_f1, color="#2ecc71", linestyle="--", alpha=0.5, label="Full MARBERT")
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    savefig("07_ablation_study.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 8 — Training Loss Curves
# ─────────────────────────────────────────────────────────────────────────────
def graph8_training_loss():
    print("Graph 8: Training Loss Curves ...")
    loss_files = {
        "AraBERT": os.path.join(RESULTS_DIR, "arabert_training_loss.csv"),
        "MARBERT": os.path.join(RESULTS_DIR, "marbert_training_loss.csv"),
    }
    available = {k: v for k, v in loss_files.items() if os.path.exists(v)}
    if not available: print("  Skipping — run model training first"); return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    colors = {"AraBERT": "#3498db", "MARBERT": "#2ecc71"}
    for name, path in available.items():
        df = pd.read_csv(path)
        ax.plot(df["epoch"], df["loss"], marker="o", linewidth=2,
                color=colors.get(name, "#e74c3c"), label=name)
        for _, row in df.iterrows():
            ax.annotate(f"{row['loss']:.4f}",
                        (row["epoch"], row["loss"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss per Epoch", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    savefig("08_training_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 9 — Verification Metrics (Evidence + Faithfulness)
# ─────────────────────────────────────────────────────────────────────────────
def graph9_verification_metrics():
    print("Graph 9: Verification Metrics ...")
    faav_csv = os.path.join(RESULTS_DIR, "faav_full_results.csv")
    if not os.path.exists(faav_csv): print("  Skipping — run faav_full_metrics.py first"); return

    df = pd.read_csv(faav_csv)
    # skip if old-format file (pre-3-class run)
    if "retrieval_quality" not in df.columns or not any(lbl in df["gold_label"].values for lbl in LABELS):
        print("  Skipping — faav_full_results.csv is from old run; re-run after Kaggle session")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)

    # Left: retrieval quality pie
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    rq = df["retrieval_quality"].value_counts()
    colors_pie = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.pie(rq.values, labels=rq.index, autopct="%1.1f%%",
           colors=colors_pie[:len(rq)], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Evidence Retrieval Quality", fontsize=13, fontweight="bold")

    # Right: faithfulness and citation metrics
    ax = axes[1]
    ax.set_facecolor(BG_COLOR)
    faith_acc  = (df["faithfulness_label"] == df["predicted_faithfulness"]).mean()
    halluc     = (df["predicted_faithfulness"] == "Low").mean()
    cite_acc   = df["citation_supported"].mean() if "citation_supported" in df.columns else float("nan")
    metrics    = ["Faithfulness\nAccuracy", "Hallucination\nRate", "Citation Support\nAccuracy"]
    values     = [faith_acc, halluc, cite_acc]
    bar_colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(metrics, values, color=bar_colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("FAAV Verification Metrics", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Retrieval and Faithfulness Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    savefig("09_verification_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 10 — Original Dataset Composition (stacked bar per source)
# ─────────────────────────────────────────────────────────────────────────────
def graph10_dataset_composition():
    print("Graph 10: Dataset Composition ...")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TRAIN_FILE

    if not os.path.exists(TRAIN_FILE):
        print("  Skipping — four_dataset_training.csv not found")
        return

    df = pd.read_csv(TRAIN_FILE, usecols=["source", "label"], low_memory=False)

    # Original full-dataset sizes (before training cap / sampling)
    ORIGINAL_SIZES = {
        "AFND":           434_440,   # total articles across all credibility tiers
        "ArCOV19-tweets": 3_302,
        "ArCOV19-claims": 420,
        "VERA-ARAB":      20_076,
        "AraStance":      4_071,     # all 3 splits combined
    }

    # Compute what is actually in training CSV per source x label
    pivot = (df.groupby(["source", "label"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=LABELS, fill_value=0))

    source_order = ["AFND", "ArCOV19-tweets", "ArCOV19-claims", "VERA-ARAB", "AraStance"]
    pivot = pivot.reindex([s for s in source_order if s in pivot.index])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_COLOR)

    # ── Left: stacked bar — rows used in training ────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    bottoms = np.zeros(len(pivot))
    for lbl in LABELS:
        vals = pivot[lbl].values if lbl in pivot.columns else np.zeros(len(pivot))
        bars = ax.bar(pivot.index, vals, bottom=bottoms,
                      color=PALETTE[lbl], edgecolor="white", label=lbl, width=0.6)
        for bar, v, b in zip(bars, vals, bottoms):
            if v > 200:
                ax.text(bar.get_x() + bar.get_width()/2, b + v/2,
                        f"{v:,}", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
        bottoms += vals

    ax.set_title("Training Rows per Dataset (by Class)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Rows", fontsize=11)
    ax.set_xlabel("Dataset Source", fontsize=11)
    ax.tick_params(axis="x", labelsize=9, rotation=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Class", fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── Right: horizontal bar — original vs used ─────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG_COLOR)
    sources  = [s for s in source_order if s in pivot.index]
    originals = [ORIGINAL_SIZES.get(s, 0) for s in sources]
    used_totals = pivot.sum(axis=1).values

    y = np.arange(len(sources))
    h = 0.35
    b1 = ax.barh(y + h/2, originals,    h, label="Original size",   color="#95a5a6", edgecolor="white")
    b2 = ax.barh(y - h/2, used_totals,  h, label="Used in training", color="#3498db", edgecolor="white")

    for bar in list(b1) + list(b2):
        w = bar.get_width()
        ax.text(w + 500, bar.get_y() + bar.get_height()/2,
                f"{int(w):,}", va="center", fontsize=8)

    ax.set_yticks(y); ax.set_yticklabels(sources, fontsize=10)
    ax.set_xlabel("Number of Rows", fontsize=11)
    ax.set_title("Original Dataset Size vs. Used in Training", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    savefig("10_dataset_composition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 11 — Train / Validation (CV) / Test Split Pie Chart
# ─────────────────────────────────────────────────────────────────────────────
def graph11_train_test_split():
    print("Graph 11: Train / Validation / Test Split ...")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import TRAIN_FILE, TEST_FILE, TEST_SHEET

    # ── Known split numbers ────────────────────────────────────────────────────
    TRAIN_TOTAL = 120_000
    TEST_TOTAL  = 30_015

    # Class breakdown — training
    TRAIN_CLASSES = {"Real": 44_078, "Fake": 45_535, "Unverified": 30_387}
    # Class breakdown — test (UAE-MVD)
    TEST_CLASSES  = {"Real": 18_922, "Fake": 7_143,  "Unverified": 3_950}

    # 5-fold CV: each fold holds out 20% of training as validation
    CV_FOLDS        = 5
    CV_VAL_PER_FOLD = TRAIN_TOTAL // CV_FOLDS   # 24,000 per fold

    TOTAL = TRAIN_TOTAL + TEST_TOTAL  # 150,015

    fig = plt.figure(figsize=(16, 7), facecolor=BG_COLOR)
    fig.suptitle("Data Split Overview — FAAV Experiment", fontsize=16, fontweight="bold", y=1.01)

    # ── Panel 1: outer/inner donut (train+test outer, classes inner) ──────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor(BG_COLOR)

    outer_sizes  = [TRAIN_TOTAL, TEST_TOTAL]
    outer_labels = [f"Training\n{TRAIN_TOTAL:,}\n(80%)", f"Test\n{TEST_TOTAL:,}\n(20%)"]
    outer_colors = ["#3498db", "#e74c3c"]

    inner_sizes  = [TRAIN_CLASSES[l] for l in LABELS] + [TEST_CLASSES[l] for l in LABELS]
    inner_colors = [PALETTE["Real"], PALETTE["Fake"], PALETTE["Unverified"],
                    "#a8e6a3", "#f5a9a9", "#fde8bb"]

    wedge_kw = {"edgecolor": "white", "linewidth": 2}
    ax1.pie(outer_sizes, labels=outer_labels, colors=outer_colors,
            radius=1.0, startangle=90, wedgeprops={**wedge_kw, "width": 0.4},
            textprops={"fontsize": 10, "fontweight": "bold"})
    ax1.pie(inner_sizes, colors=inner_colors,
            radius=0.6, startangle=90, wedgeprops={**wedge_kw, "width": 0.4})

    # Inner legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[l], label=f"{l} (train: {TRAIN_CLASSES[l]:,} / test: {TEST_CLASSES[l]:,})")
        for l in LABELS
    ]
    ax1.legend(handles=legend_elements, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), fontsize=8, title="Class")
    ax1.set_title("Train / Test Split\n(outer: split | inner: class)", fontsize=11, fontweight="bold")

    # ── Panel 2: class distribution comparison (training vs test) ─────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor(BG_COLOR)

    x   = np.arange(len(LABELS))
    w   = 0.35
    tr  = [TRAIN_CLASSES[l] for l in LABELS]
    te  = [TEST_CLASSES[l]  for l in LABELS]
    tr_pct = [v / TRAIN_TOTAL * 100 for v in tr]
    te_pct = [v / TEST_TOTAL  * 100 for v in te]

    b1 = ax2.bar(x - w/2, tr_pct, w, label=f"Training (n={TRAIN_TOTAL:,})",
                 color="#3498db", edgecolor="white")
    b2 = ax2.bar(x + w/2, te_pct, w, label=f"Test (n={TEST_TOTAL:,})",
                 color="#e74c3c", edgecolor="white")
    for bar, pct, cnt in zip(list(b1) + list(b2),
                              tr_pct + te_pct,
                              tr + te):
        ax2.text(bar.get_x() + bar.get_width()/2, pct + 0.5,
                 f"{pct:.1f}%\n({cnt:,})",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2.set_xticks(x); ax2.set_xticklabels(LABELS, fontsize=11)
    ax2.set_ylabel("% of split", fontsize=11)
    ax2.set_ylim(0, 55)
    ax2.set_title("Class Distribution\nTraining vs Test", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: 5-fold CV breakdown ──────────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor(BG_COLOR)

    fold_train = TRAIN_TOTAL - CV_VAL_PER_FOLD   # 96,000
    fold_val   = CV_VAL_PER_FOLD                  # 24,000

    cv_labels  = [f"CV Train\n{fold_train:,}\n(80%)",
                  f"CV Validation\n{fold_val:,}\n(20%) × {CV_FOLDS} folds"]
    cv_sizes   = [fold_train, fold_val]
    cv_colors  = ["#2980b9", "#85c1e9"]
    explode    = (0, 0.06)

    ax3.pie(cv_sizes, labels=cv_labels, colors=cv_colors,
            autopct="%1.1f%%", startangle=90, explode=explode,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 10})
    ax3.set_title(f"5-Fold Cross-Validation\n(within training set, per fold)",
                  fontsize=11, fontweight="bold")

    # Summary text
    summary = (f"Total corpus: {TOTAL:,} posts\n"
               f"  Training : {TRAIN_TOTAL:,}  (80.0%)\n"
               f"  Test     : {TEST_TOTAL:,}  (20.0%)\n"
               f"  CV val   : {CV_VAL_PER_FOLD:,} / fold × {CV_FOLDS}")
    ax3.text(0, -1.45, summary, ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.tight_layout()
    savefig("11_train_test_split.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating graphs -> {GRAPHS_DIR}\n")
    graph1_before_after_accuracy()
    graph2_all_models_accuracy()
    graph3_all_models_macro_f1()
    graph4_per_class_f1()
    graph5_confusion_matrices()
    graph6_cross_validation()
    graph7_ablation()
    graph8_training_loss()
    graph9_verification_metrics()
    graph10_dataset_composition()
    graph11_train_test_split()
    print(f"\nAll graphs saved to: {GRAPHS_DIR}")
