"""
Build a clean in-distribution 80/20 split from the framework's own claim data.
  - dedup on claim text (no claim in both train and test)
  - stratified by label, seed 42
Outputs:
  data/indist_train.csv   (cols: text, label, evidence_text)  -> TRAIN_FILE
  data/indist_test.xlsx   (cols: claim_ar, gold_label, evidence_text,
                                  faithfulness_label, retrieval_quality)  -> TEST_FILE
"""
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Real", "Fake"]
df = pd.read_csv("data/claim_training_2class.csv",
                 usecols=["text", "label", "evidence_text"], low_memory=False)
df = df.dropna(subset=["text", "label"])
df = df[df["label"].isin(LABELS)].copy()
df["evidence_text"] = df["evidence_text"].fillna("")
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Deduped claims: {len(df)}  {df['label'].value_counts().to_dict()}")

tr, te = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
print(f"train {len(tr)}  test {len(te)}")

# sanity: zero overlap
overlap = set(tr["text"]) & set(te["text"])
print(f"leakage (claims in both): {len(overlap)}  (must be 0)")

tr[["text", "label", "evidence_text"]].to_csv("data/indist_train.csv", index=False)

test_out = pd.DataFrame({
    "claim_ar":          te["text"].values,
    "gold_label":        te["label"].values,
    "evidence_text":     te["evidence_text"].values,
    "faithfulness_label": "High",          # placeholder (faithfulness metrics N/A in-distribution)
    "retrieval_quality":  "Relevant",      # placeholder
})
with pd.ExcelWriter("data/indist_test.xlsx", engine="openpyxl") as w:
    test_out.to_excel(w, sheet_name="Tabelle1", index=False)

print("Wrote data/indist_train.csv and data/indist_test.xlsx")
print("test dist:", test_out["gold_label"].value_counts().to_dict())
