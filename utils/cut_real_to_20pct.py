"""
Reduce the TEST set so that the 2-class (Real/Fake) test = 20% of the 2-class
training set, by downsampling the Real class ONLY (Fake kept in full).

  2-class train = 25,263  ->  20% = 5,053 test
  Fake kept     = 2,418
  Real target   = 5,053 - 2,418 = 2,635   (downsampled from 6,406)

The Real claims to keep are chosen once (seed=42) from the existing test Real
rows, so the predictions already in every result CSV remain valid and only need
filtering. Originals are backed up first.
"""
import os, shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES  = os.path.join(ROOT, "results")
TEST = os.path.join(DATA, "uae_test_dataset_final.xlsx")

REAL_TARGET = 2635

# --- 1. choose which Real claims to keep (deterministic) ---
te = pd.read_excel(TEST, sheet_name="Tabelle1")
real = te[te["gold_label"] == "Real"]
keep_real = set(real.sample(n=REAL_TARGET, random_state=42)["claim_ar"])
print(f"Real in test: {len(real)} -> keeping {len(keep_real)}")

with open(os.path.join(RES, "keep_real_2class.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(map(str, keep_real)))

def keep_mask(df):
    return (df["gold_label"] != "Real") | (df["claim_ar"].isin(keep_real))

# --- 2. rewrite the test file (Real cut; Fake + Unverified untouched) ---
shutil.copy(TEST, os.path.join(DATA, "uae_test_dataset_final_pre_realcut.xlsx"))
te_new = te[keep_mask(te)].reset_index(drop=True)
with pd.ExcelWriter(TEST, engine="openpyxl") as w:
    te_new.to_excel(w, sheet_name="Tabelle1", index=False)
print("New test file dist:", te_new["gold_label"].value_counts().to_dict(),
      "total", len(te_new))

# --- 3. filter every 2-class result CSV (backup originals first) ---
bdir = os.path.join(RES, "_pre_realcut_backup")
os.makedirs(bdir, exist_ok=True)
files = ["svm_claim_results_2class", "lr_claim_results_2class",
         "bilstm_claim_results_2class", "arabert_before_finetune_results_2class",
         "standard_rag_results_2class", "sbert_rag_results_2class",
         "faithfulness_aware_rag_results_2class"]
for stem in files:
    p = os.path.join(RES, stem + ".csv")
    if not os.path.exists(p):
        print("  skip (missing):", stem); continue
    df = pd.read_csv(p, low_memory=False)
    shutil.copy(p, os.path.join(bdir, stem + ".csv"))
    out = df[keep_mask(df)].reset_index(drop=True)
    out.to_csv(p, index=False)
    print(f"  {stem}: {len(df)} -> {len(out)} "
          f"({out['gold_label'].value_counts().to_dict()})")

print("\nDone. 2-class test is now", len(te_new) - 1337, "Real/Fake claims",
      "(=", REAL_TARGET, "Real +", 2418, "Fake).")
