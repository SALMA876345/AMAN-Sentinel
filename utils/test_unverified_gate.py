"""
Test Hamza's cascade idea PROPERLY on MARBERT confidence.

Idea: "if the model is unsure between Real/Fake, route to Unverified."
Implemented as a confidence-MARGIN gate on MARBERT's softmax:
  - Look only at Real vs Fake probability.
  - margin = |p_real - p_fake| / (p_real + p_fake)   (how decisive the R/F call is)
  - If margin < T  -> override prediction to "Unverified"

Crucially: threshold T is tuned on a HELD-OUT validation split (50%),
then reported on the untouched TEST split. No peeking at test for tuning.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "marbert_claim_results_3class.csv")
LABELS = ["Real", "Fake", "Unverified"]

df = pd.read_csv(CSV, low_memory=False)
need = ["gold_label", "marbert_prediction",
        "marbert_proba_real", "marbert_proba_fake", "marbert_proba_unverified"]
df = df.dropna(subset=need)
df = df[df["gold_label"].isin(LABELS)].reset_index(drop=True)
print(f"Rows usable: {len(df)}")
print("Gold dist:", df["gold_label"].value_counts().to_dict())

pr = df["marbert_proba_real"].to_numpy(float)
pf = df["marbert_proba_fake"].to_numpy(float)
margin = np.abs(pr - pf) / (pr + pf + 1e-9)
base_pred = df["marbert_prediction"].to_numpy()
gold = df["gold_label"].to_numpy()


def scored(pred):
    return dict(
        acc=accuracy_score(gold_sub, pred),
        macroF1=f1_score(gold_sub, pred, average="macro", labels=LABELS, zero_division=0),
        unvF1=f1_score(gold_sub, pred, labels=["Unverified"], average="macro", zero_division=0),
        realF1=f1_score(gold_sub, pred, labels=["Real"], average="macro", zero_division=0),
        fakeF1=f1_score(gold_sub, pred, labels=["Fake"], average="macro", zero_division=0),
        mcc=matthews_corrcoef(gold_sub, pred),
    )


def apply_gate(idx, T):
    p = base_pred[idx].copy()
    # only re-route confident-looking Real/Fake calls that are actually borderline
    mask = (margin[idx] < T) & np.isin(p, ["Real", "Fake"])
    p[mask] = "Unverified"
    return p


# 50/50 split: tune on val, report on test
rng = np.random.RandomState(42)
perm = rng.permutation(len(df))
val_idx, test_idx = perm[: len(df) // 2], perm[len(df) // 2:]

# --- tune T on validation ---
gold_sub = gold[val_idx]
best_T, best_val = 0.0, -1
for T in np.linspace(0, 0.6, 61):
    m = scored(apply_gate(val_idx, T))
    if m["macroF1"] > best_val:
        best_val, best_T = m["macroF1"], T
print(f"\nBest T on validation = {best_T:.3f} (val macroF1={best_val:.4f})")

# --- report on test ---
gold_sub = gold[test_idx]
base = scored(base_pred[test_idx])
gated = scored(apply_gate(test_idx, best_T))

print("\n=== TEST SPLIT (untouched) ===")
print(f"{'metric':<10}{'baseline':>10}{'gated':>10}")
for k in ["acc", "macroF1", "unvF1", "realF1", "fakeF1", "mcc"]:
    print(f"{k:<10}{base[k]:>10.4f}{gated[k]:>10.4f}")

# also show full-data threshold sweep for intuition
print("\n=== Full-data sweep (intuition only, NOT honest tuning) ===")
gold_sub = gold
print(f"{'T':>5}{'acc':>9}{'macroF1':>9}{'unvF1':>9}")
for T in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    m = scored(apply_gate(np.arange(len(df)), T))
    print(f"{T:>5.2f}{m['acc']:>9.4f}{m['macroF1']:>9.4f}{m['unvF1']:>9.4f}")
