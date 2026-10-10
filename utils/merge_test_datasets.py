"""
Merges the existing UAE-MVD test set (5,538 rows from Excel)
with the new labeled test data (60k rows from CSV/Excel).

Required columns in new test file:
  claim_ar, gold_label, evidence_text,
  retrieval_quality, faithfulness_label, confidence, label_source

Usage:
  python utils/merge_test_datasets.py --new_file path/to/new_test.csv
  python utils/merge_test_datasets.py --new_file path/to/new_test.xlsx
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_FILE, TEST_SHEET, RESULTS_DIR

import pandas as pd

REQUIRED_COLS = [
    "claim_ar", "gold_label", "evidence_text",
    "retrieval_quality", "faithfulness_label",
]
VALID_LABELS = {"Real", "Fake", "Unverified"}
# Saved as .xlsx with sheet "Tabelle1" so every model script can read it
# via pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET) without changes.
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "merged_test_dataset.xlsx")


def load_file(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def validate(df, name):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING [{name}]: missing columns: {missing}")
    bad_labels = df[~df["gold_label"].isin(VALID_LABELS)]
    if len(bad_labels):
        print(f"  WARNING [{name}]: {len(bad_labels)} rows with unknown labels — will be dropped")
    return df[df["gold_label"].isin(VALID_LABELS)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_file", required=True,
                        help="Path to new labeled test file (CSV or XLSX)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading existing test set ...")
    df_existing = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
    df_existing = validate(df_existing, "existing")
    print(f"  Existing: {len(df_existing)} rows | {df_existing['gold_label'].value_counts().to_dict()}")

    print(f"Loading new test file: {args.new_file} ...")
    df_new = load_file(args.new_file)
    df_new = validate(df_new, "new")
    print(f"  New     : {len(df_new)} rows | {df_new['gold_label'].value_counts().to_dict()}")

    # Align columns — add missing optional columns with empty strings
    all_cols = list(dict.fromkeys(df_existing.columns.tolist() + df_new.columns.tolist()))
    for col in all_cols:
        if col not in df_existing.columns:
            df_existing[col] = ""
        if col not in df_new.columns:
            df_new[col] = ""

    df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["claim_ar"]).reset_index(drop=True)

    print(f"\nMerged: {len(df_merged)} rows")
    print(f"Label split: {df_merged['gold_label'].value_counts().to_dict()}")

    total = len(df_merged)
    test_pct  = total / (total + 120000) * 100
    print(f"Test split: {total} / {total + 120000} total = {test_pct:.1f}%")

    df_merged.to_excel(OUTPUT_FILE, index=False, sheet_name="Tabelle1")
    print(f"\nSaved -> {OUTPUT_FILE}")
    print("\nNext: in config.py set:")
    print('  TEST_FILE = os.path.join(DATA_DIR, "merged_test_dataset.xlsx")')
    print("then run all models.")


if __name__ == "__main__":
    main()
