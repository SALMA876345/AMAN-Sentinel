"""
Standardises the UAE-MVD test CSV for use by all pipeline scripts.

Input  : c:\\Users\\Hamza\\social_dataset_agent\\datasets\\UAE-MVD-test-final.csv
Output : data/uae_test_dataset_final.xlsx  (sheet "Tabelle1")

Column mapping
  clean_text_final  -> claim_ar
  label             -> gold_label
  retrieved_evidence-> evidence_text
  faithfulness_score (string) -> faithfulness_label (High / Low)
    "Supported"          -> High
    "Partially Supported"-> Low
    "Unsupported"        -> Low
    nan                  -> Low

All other columns are kept as-is.
"""

import os, sys
import pandas as pd

SRC = r"c:\Users\Hamza\social_dataset_agent\datasets\UAE-MVD-test-final.csv"

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT     = os.path.join(ROOT, "data", "uae_test_dataset_final.xlsx")
SHEET   = "Tabelle1"

VALID_LABELS = {"Real", "Fake", "Unverified"}

FAITH_MAP = {
    "Supported":           "High",
    "Partially Supported": "Low",
    "Unsupported":         "Low",
}


def main():
    print(f"Reading: {SRC}")
    df = pd.read_csv(SRC, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── Column renames ──────────────────────────────────────────────────────────
    df = df.rename(columns={
        "clean_text_final":  "claim_ar",
        "label":             "gold_label",
        "retrieved_evidence":"evidence_text",
    })

    # ── faithfulness_score -> faithfulness_label ────────────────────────────────
    df["faithfulness_label"] = (
        df["faithfulness_score"]
        .map(lambda v: FAITH_MAP.get(str(v).strip(), "Low"))
    )

    # ── Keep only rows with valid labels ────────────────────────────────────────
    before = len(df)
    df = df[df["gold_label"].isin(VALID_LABELS)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with invalid gold_label")

    # ── Ensure required columns exist (fill blanks if missing) ──────────────────
    for col in ["retrieval_quality", "confidence", "label_source"]:
        if col not in df.columns:
            df[col] = ""

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\nFinal rows  : {len(df):,}")
    print(f"gold_label  : {df['gold_label'].value_counts().to_dict()}")
    print(f"faith_label : {df['faithfulness_label'].value_counts().to_dict()}")
    print(f"columns     : {list(df.columns)}")

    # ── Save ────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_excel(OUT, index=False, sheet_name=SHEET)
    print(f"\nSaved -> {OUT}")
    print(f"\nIn config.py, TEST_FILE is already updated to point to this file.")
    print("Run this script ONCE before running any model scripts.")


if __name__ == "__main__":
    main()
