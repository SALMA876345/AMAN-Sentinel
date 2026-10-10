"""Generate the figures embedded in the report (saved to graphs/)."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); G = os.path.join(ROOT, "graphs")
os.makedirs(G, exist_ok=True)

ind = pd.read_csv(os.path.join(RES, "comparison_2class_indist_3source.csv"))
crd = pd.read_csv(os.path.join(RES, "comparison_2class.csv"))
def mf1(df, model):
    r = df[df["Model"] == model]
    return float(r["Macro F1"].iloc[0]) if len(r) and pd.notna(r["Macro F1"].iloc[0]) else None

# ---- Fig 10: in-distribution macro F1 bar ----
d = ind.dropna(subset=["Macro F1"]).copy()
d["Macro F1"] = d["Macro F1"].astype(float)
d = d.sort_values("Macro F1", ascending=True)
plt.figure(figsize=(8, 5))
colors = ["#2c7fb8" if v >= 0.8 else "#c0c0c0" for v in d["Macro F1"]]
plt.barh(d["Model"], d["Macro F1"], color=colors)
for i, v in enumerate(d["Macro F1"]):
    plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
plt.xlabel("Macro F1"); plt.title("In-distribution macro F1 by model (held-out 20% test)")
plt.xlim(0, 1.0); plt.tight_layout()
plt.savefig(os.path.join(G, "10_indist_macro_f1.png"), dpi=150); plt.close()

# ---- Fig 11: fine-tuning effect ----
pairs = [("MARBERT", mf1(ind, "MARBERT (before)"), mf1(ind, "MARBERT (after)")),
         ("AraBERT", mf1(ind, "AraBERT (before)"), mf1(ind, "AraBERT (after FT)"))]
import numpy as np
x = np.arange(len(pairs)); w = 0.35
plt.figure(figsize=(6, 5))
plt.bar(x - w/2, [p[1] for p in pairs], w, label="Before fine-tuning", color="#c0c0c0")
plt.bar(x + w/2, [p[2] for p in pairs], w, label="After fine-tuning", color="#2c7fb8")
for i, p in enumerate(pairs):
    plt.text(i - w/2, p[1] + 0.01, f"{p[1]:.2f}", ha="center", fontsize=9)
    plt.text(i + w/2, p[2] + 0.01, f"{p[2]:.2f}", ha="center", fontsize=9)
plt.xticks(x, [p[0] for p in pairs]); plt.ylabel("Macro F1"); plt.ylim(0, 1.0)
plt.title("Effect of fine-tuning (in-distribution)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(G, "11_finetune_effect.png"), dpi=150); plt.close()

# ---- Fig 12: in-distribution vs cross-domain ----
models = ["MARBERT (after)", "SVM", "Logistic Regression", "BiLSTM"]
ind_v = [mf1(ind, m) for m in models]
crd_v = [mf1(crd, m) for m in models]
x = np.arange(len(models)); w = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - w/2, ind_v, w, label="In-distribution", color="#2c7fb8")
plt.bar(x + w/2, crd_v, w, label="Cross-domain (UAE-MVD)", color="#e34a33")
for i in range(len(models)):
    plt.text(i - w/2, ind_v[i] + 0.01, f"{ind_v[i]:.2f}", ha="center", fontsize=8)
    plt.text(i + w/2, crd_v[i] + 0.01, f"{crd_v[i]:.2f}", ha="center", fontsize=8)
plt.xticks(x, ["MARBERT", "SVM", "LR", "BiLSTM"]); plt.ylabel("Macro F1"); plt.ylim(0, 1.0)
plt.title("The generalisation gap: in-distribution vs cross-domain")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(G, "12_indist_vs_crossdomain.png"), dpi=150); plt.close()

print("Saved figures 10, 11, 12 to graphs/")
