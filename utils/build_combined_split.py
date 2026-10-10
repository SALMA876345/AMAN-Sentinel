"""
Build a COMBINED in-distribution benchmark:
  VERA-ARAB + AraStance + ArCOV19  (claim_training_2class.csv, with evidence)
  + ANS (data/ANS/{train,dev,test}.csv, claim_s + fake_flag, no evidence)
Dedup on claim text, stratified 80/20 split, zero leakage.
Overwrites data/indist_train.csv and data/indist_test.xlsx (now the 4-source set).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Real", "Fake"]

base = pd.read_csv("data/claim_training_2class.csv",
                   usecols=["text", "label", "evidence_text", "source"], low_memory=False)
base = base[base["label"].isin(LABELS)].dropna(subset=["text", "label"])
base["evidence_text"] = base["evidence_text"].fillna("")
print("base (VERA+AraStance+ArCOV):", len(base), base["source"].value_counts().to_dict())

ans_parts = [pd.read_csv(f"data/ANS/{s}.csv") for s in ["train", "dev", "test"]]
ans = pd.concat(ans_parts, ignore_index=True)
ans = pd.DataFrame({
    "text": ans["claim_s"].astype(str),
    "label": ans["fake_flag"].map(lambda f: "Fake" if int(f) == 1 else "Real"),
    "evidence_text": "",
    "source": "ANS",
})
print("ANS:", len(ans), ans["label"].value_counts().to_dict())

df = pd.concat([base, ans], ignore_index=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"\nCOMBINED (deduped): {len(df)}  {df['label'].value_counts().to_dict()}")
print("by source:", df["source"].value_counts().to_dict())

tr, te = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
print(f"\ntrain {len(tr)}  test {len(te)}  leakage={len(set(tr['text'])&set(te['text']))}")

tr[["text", "label", "evidence_text", "source"]].to_csv("data/indist_train.csv", index=False)
test_out = pd.DataFrame({
    "claim_ar":           te["text"].values,
    "gold_label":         te["label"].values,
    "evidence_text":      te["evidence_text"].values,
    "source":             te["source"].values,
    "faithfulness_label": "High",
    "retrieval_quality":  "Relevant",
})
with pd.ExcelWriter("data/indist_test.xlsx", engine="openpyxl") as w:
    test_out.to_excel(w, sheet_name="Tabelle1", index=False)
print("\nWrote data/indist_train.csv + data/indist_test.xlsx (4-source combined)")
print("test by source:", test_out["source"].value_counts().to_dict())
