"""
Prepares UAE-MVD-merged.csv as the pipeline test dataset.

UAE-MVD (UAE Misinformation Verification Dataset) contains 5,594 Arabic
social-media posts (Twitter / RSS) labeled by qwen3 + gemma4 LLMs with
Real / Fake / Unverified labels plus retrieved evidence, faithfulness
scores, and confidence values.

This script:
  1. Reads UAE-MVD-merged.csv from its original location
  2. Filters to Arabic-only rows
  3. Maps columns to the pipeline-expected schema:
       UAE column              -> Pipeline column
       -------------------------------------------------
       clean_text_final        -> claim_ar
       label (Real/Fake/Unv.)  -> gold_label (Supported/Refuted/NEI)
       retrieved_evidence      -> evidence_text
       faithfulness_score      -> faithfulness_label (High/Low/Unknown)
  4. Saves data/uae_test_dataset.xlsx  (sheet "Tabelle1")
     — same format as the old 15-claim manual_claim_test_dataset.xlsx
     so all pipeline scripts work without modification.

Run once before running any evaluation scripts:
    python utils/prepare_uae_test_dataset.py
"""

import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import DATA_DIR

import pandas as pd

SOURCE_CSV = r"c:\Users\Hamza\social_dataset_agent\datasets\UAE-MVD-merged.csv"
OUTPUT_XLS = os.path.join(DATA_DIR, "uae_test_dataset.xlsx")
SHEET_NAME = "Tabelle1"

# ── Label mappings ─────────────────────────────────────────────────────────────
GOLD_MAP = {
    "Real":       "Real",
    "Fake":       "Fake",
    "Unverified": "Unverified",
}

FAITH_MAP = {
    "Supported":   "High",
    "Unsupported": "Low",
}


def main():
    print(f"Reading {SOURCE_CSV} ...")
    df = pd.read_csv(SOURCE_CSV, low_memory=False)
    print(f"  Raw rows: {len(df)}")

    # ── Keep Arabic rows only ──────────────────────────────────────────────────
    arabic_mask = (df["is_arabic"] == 1) | (df["language"] == "ar")
    df = df[arabic_mask].copy()
    print(f"  Arabic rows: {len(df)}")

    # ── Keep all three classes: Real / Fake / Unverified ─────────────────────
    valid_labels = {"Real", "Fake", "Unverified"}
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    print(f"  Valid-label rows: {len(df)}")

    # ── Map labels ─────────────────────────────────────────────────────────────
    df["gold_label"]       = df["label"].map(GOLD_MAP)
    df["faithfulness_label"] = df["faithfulness_score"].map(FAITH_MAP).fillna("Unknown")

    # ── Rename / select columns ────────────────────────────────────────────────
    df["claim_ar"]      = df["clean_text_final"].fillna(df["raw_text"]).fillna("")
    df["evidence_text"] = df["retrieved_evidence"].fillna("")

    # Preserve extra columns useful for error analysis
    keep = [
        "post_id", "claim_ar", "gold_label", "evidence_text",
        "faithfulness_label", "confidence", "label", "label_source",
        "candidate_category", "platform", "date", "author",
        "retrieval_quality", "source", "evidence_url", "reasoning",
        "likes", "shares", "comments", "views",
        "word_count", "hashtag_count", "has_url",
    ]
    # Only keep columns that exist in the dataframe
    keep = [c for c in keep if c in df.columns]
    df_out = df[keep].reset_index(drop=True)

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("\nLabel distribution (gold_label):")
    print(df_out["gold_label"].value_counts())
    print("\nFaithfulness label distribution:")
    print(df_out["faithfulness_label"].value_counts())
    print("\nLabel source distribution:")
    print(df_out["label_source"].value_counts())
    print("\nRetrieval quality distribution:")
    print(df_out["retrieval_quality"].value_counts())
    print(f"\nMean confidence : {df_out['confidence'].mean():.3f}")
    print(f"Rows with evidence (len>20): "
          f"{(df_out['evidence_text'].str.len() > 20).sum()}")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    df_out.to_excel(OUTPUT_XLS, index=False, sheet_name=SHEET_NAME)
    print(f"\nSaved {len(df_out)} rows -> {OUTPUT_XLS}  (sheet: {SHEET_NAME})")
    print("Update config.py TEST_FILE to point to this file, then run all pipelines.")


if __name__ == "__main__":
    main()
