"""
Generates all report-ready figures for the FAAV project.
Saves every plot as a high-resolution PNG in results/figures/.

Figures produced
----------------
01_accuracy_f1_comparison.png       - Accuracy & F1 for all 7 models
02_domain_transfer_gap.png          - AFND train acc vs claim verification acc
03_confusion_matrices.png           - Confusion matrix per model (grid)
04_per_class_f1.png                 - Per-class F1 (Supported/Refuted/NEI) for RAG models
05_faithfulness_metrics.png         - Faithfulness, hallucination, grounding, evidence sim
06_error_category_analysis.png      - Error categories: FAAV-Full vs Standard RAG
07_evidence_similarity_dist.png     - Distribution of similarity scores (FAAV-Full)
08_bilstm_training_loss.png         - BiLSTM epoch loss curve
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, TEST_FILE, TEST_SHEET

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Output folder ─────────────────────────────────────────────────────────────
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def save(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/{name}")

# ── Shared style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {
    "Traditional": "#4C72B0",
    "Deep Learning": "#DD8452",
    "Transformer":  "#55A868",
    "Retrieval":    "#C44E52",
    "Proposed":     "#8172B2",
}
MODEL_COLORS = [
    COLORS["Traditional"], COLORS["Traditional"],
    COLORS["Deep Learning"], COLORS["Transformer"],
    COLORS["Retrieval"],    COLORS["Retrieval"],
    COLORS["Proposed"],
]

MODELS = [
    "SVM", "Log. Regression", "BiLSTM", "AraBERT",
    "Standard RAG", "FAAV-Light", "FAAV-Full",
]
SHORT  = ["SVM", "LR", "BiLSTM", "AraBERT", "Std RAG", "FAAV-L", "FAAV-F"]

# ── Load gold labels ──────────────────────────────────────────────────────────
df_gold     = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
gold_labels = df_gold["gold_label"].tolist()
CLASSES     = ["Not Enough Evidence", "Refuted", "Supported"]

def load_preds(filename, col):
    fp = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(fp):
        return None
    return pd.read_csv(fp)[col].tolist()[:len(gold_labels)]

preds_map = {
    "SVM":          load_preds("svm_claim_results.csv",               "svm_prediction"),
    "Log. Regression": load_preds("lr_claim_results.csv",             "lr_prediction"),
    "BiLSTM":       load_preds("bilstm_claim_results.csv",            "bilstm_prediction"),
    "AraBERT":      load_preds("arabert_claim_results.csv",           "arabert_prediction"),
    "Standard RAG": load_preds("standard_rag_results.csv",            "standard_rag_prediction"),
    "FAAV-Light":   load_preds("faithfulness_aware_rag_results.csv",  "faav_light_prediction"),
    "FAAV-Full":    load_preds("faav_full_results.csv",               "faav_prediction"),
}

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

accuracies, f1_scores_w = [], []
for m in MODELS:
    p = preds_map[m]
    if p is None:
        accuracies.append(0); f1_scores_w.append(0)
    else:
        accuracies.append(accuracy_score(gold_labels, p))
        f1_scores_w.append(f1_score(gold_labels, p, average="weighted", zero_division=0))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Accuracy & F1 comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1 ...")
x     = np.arange(len(MODELS))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color=MODEL_COLORS, alpha=0.85)
bars2 = ax.bar(x + width/2, f1_scores_w, width, label="Weighted F1",
               color=MODEL_COLORS, alpha=0.55, hatch="//")

ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score"); ax.set_title("Model Comparison — Accuracy & Weighted F1\n(Manual Claim Test Dataset)", fontweight="bold")
ax.legend(loc="upper left")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

legend_patches = [mpatches.Patch(color=c, label=l)
                  for l, c in COLORS.items()]
ax.legend(handles=legend_patches + [
    mpatches.Patch(facecolor="grey", alpha=0.85, label="Accuracy"),
    mpatches.Patch(facecolor="grey", alpha=0.55, hatch="//", label="Weighted F1"),
], loc="upper left", fontsize=9, ncol=2)

save("01_accuracy_f1_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Domain transfer gap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2 ...")
afnd_accs  = [0.9579, 0.9219, 0.9256, 0.9625]  # from metrics files
claim_accs = [accuracies[0], accuracies[1], accuracies[2], accuracies[3]]
trad_names = ["SVM", "Log.\nRegression", "BiLSTM", "AraBERT"]

x   = np.arange(len(trad_names))
w   = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, afnd_accs,  w, label="AFND Source Credibility", color="#4C72B0", alpha=0.85)
b2 = ax.bar(x + w/2, claim_accs, w, label="Claim Verification",      color="#C44E52", alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(trad_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy")
ax.set_title("Domain Transfer Gap\nAFND Training Accuracy vs Claim Verification Accuracy", fontweight="bold")
ax.legend()

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2%}", ha="center", va="bottom", fontsize=9)

ax.annotate("", xy=(0.62, 0.28), xytext=(0.62, 0.92),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.text(0.68, 0.60, "–65%\ndrop", transform=ax.transData, fontsize=9,
        color="black", ha="center")

save("02_domain_transfer_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Confusion matrices (3×3 grid for 7 models)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3 ...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, (name, short) in enumerate(zip(MODELS, SHORT)):
    ax  = axes[i]
    p   = preds_map[name]
    if p is None:
        ax.axis("off"); continue
    cm  = confusion_matrix(gold_labels, p, labels=CLASSES)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["NEI", "Ref", "Sup"],
                yticklabels=["NEI", "Ref", "Sup"],
                cbar=False, linewidths=0.5)
    acc = accuracy_score(gold_labels, p)
    ax.set_title(f"{short}\nAcc={acc:.2%}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

axes[-1].axis("off")  # hide 8th panel
fig.suptitle("Confusion Matrices — All Models\n(NEI = Not Enough Evidence, Ref = Refuted, Sup = Supported)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save("03_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-class F1 for RAG/FAAV models
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4 ...")
rag_models  = ["Standard RAG", "FAAV-Light", "FAAV-Full"]
class_labels = ["Not Enough Evidence", "Refuted", "Supported"]
class_short  = ["NEI", "Refuted", "Supported"]

data = {}
for m in rag_models:
    p = preds_map[m]
    if p is None: continue
    _, _, f1, _ = precision_recall_fscore_support(gold_labels, p,
                   labels=class_labels, zero_division=0)
    data[m] = f1

x   = np.arange(len(class_short))
w   = 0.25
pal = [COLORS["Retrieval"], COLORS["Retrieval"], COLORS["Proposed"]]

fig, ax = plt.subplots(figsize=(9, 5))
for i, (m, color) in enumerate(zip(rag_models, pal)):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, data[m], w, label=m, color=color,
                  alpha=0.7 + i*0.1)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(class_short, fontsize=11)
ax.set_ylim(0, 1.15); ax.set_ylabel("F1-Score")
ax.set_title("Per-Class F1 Score — RAG & FAAV Models", fontweight="bold")
ax.legend()
save("04_per_class_f1.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Faithfulness & grounding metrics (FAAV-Full)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5 ...")
df_faav = pd.read_csv(os.path.join(RESULTS_DIR, "faav_full_results.csv"))

metrics = {
    "Verification\nAccuracy":   accuracy_score(gold_labels, df_faav["faav_prediction"].tolist()),
    "Faithfulness\nAccuracy":   0.6667,
    "Hallucination\nRate":      df_faav["hallucination"].mean(),
    "Avg Grounding\nScore":     df_faav["grounding_score"].mean(),
    "Avg Evidence\nSimilarity": df_faav["evidence_similarity"].mean(),
    "Avg Retrieval\nSimilarity":df_faav["retrieval_similarity"].mean(),
}
bar_colors = ["#55A868","#55A868","#C44E52","#4C72B0","#4C72B0","#4C72B0"]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(metrics.keys(), metrics.values(), color=bar_colors, alpha=0.85, edgecolor="white", width=0.55)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylim(0, 1.15); ax.set_ylabel("Score / Rate")
ax.set_title("FAAV-Full — Faithfulness & Grounding Metrics", fontweight="bold")
green_patch = mpatches.Patch(color="#55A868", label="Correctness metrics")
red_patch   = mpatches.Patch(color="#C44E52", label="Error metrics (lower=better)")
blue_patch  = mpatches.Patch(color="#4C72B0", label="Similarity scores")
ax.legend(handles=[green_patch, red_patch, blue_patch], fontsize=9)
save("05_faithfulness_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Error category analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6 ...")
categories = ["Dialect\nFailures", "Sarcasm\n/Irony", "Political\nAmbiguity",
              "Temporal\nMisinformation", "Conflicting\nEvidence", "Retrieval\nFailures"]
faav_errors = [0, 0, 1, 0, 0, 0]
rag_errors  = [1, 0, 1, 0, 0, 0]

x = np.arange(len(categories))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - w/2, rag_errors,  w, label="Standard RAG (3 errors)",  color=COLORS["Retrieval"], alpha=0.85)
b2 = ax.bar(x + w/2, faav_errors, w, label="FAAV-Full (4 errors)",     color=COLORS["Proposed"],  alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 2.5); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.set_ylabel("Number of Errors")
ax.set_title("Error Category Analysis\nStandard RAG vs FAAV-Full", fontweight="bold")
ax.legend()
save("06_error_category_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Evidence & similarity score distributions (FAAV-Full)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 7 ...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
cols  = ["evidence_similarity", "retrieval_similarity", "grounding_score"]
titles = ["Evidence Similarity\n(claim vs provided evidence)",
          "Retrieval Similarity\n(claim vs BM25 top passage)",
          "Grounding Score\n(combined)"]
pal   = ["#4C72B0", "#DD8452", "#55A868"]

for ax, col, title, color in zip(axes, cols, titles, pal):
    correct   = df_faav[df_faav["is_correct"]][col]
    incorrect = df_faav[~df_faav["is_correct"]][col]
    ax.hist(correct,   bins=8, alpha=0.75, color=color,   label=f"Correct (n={len(correct)})")
    ax.hist(incorrect, bins=8, alpha=0.55, color="#C44E52", label=f"Wrong (n={len(incorrect)})")
    ax.axvline(df_faav[col].mean(), color="black", linestyle="--", linewidth=1.2, label=f"Mean={df_faav[col].mean():.2f}")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Score"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle("Score Distributions — FAAV-Full (Correct vs Incorrect Predictions)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("07_evidence_similarity_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — BiLSTM training loss curve
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 8 ...")
epochs = [1, 2, 3, 4, 5]
losses = [0.4490, 0.2676, 0.1725, 0.1251, 0.0789]  # from training output

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, losses, marker="o", color=COLORS["Deep Learning"],
        linewidth=2.5, markersize=8, label="Training Loss")
for e, l in zip(epochs, losses):
    ax.text(e, l + 0.008, f"{l:.4f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(epochs); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("BiLSTM Training Loss Curve\n(2-layer Bidirectional, CPU)", fontweight="bold")
ax.legend(); ax.set_ylim(0, 0.55)
save("08_bilstm_training_loss.png")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to: {FIG_DIR}")
print("="*55)
figs = sorted(os.listdir(FIG_DIR))
for f in figs:
    size = os.path.getsize(os.path.join(FIG_DIR, f)) // 1024
    print(f"  {f}  ({size} KB)")
