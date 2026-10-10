import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET
import pandas as pd



df = pd.read_excel(INPUT_FILE, sheet_name="Tabelle1")

print("Manual dataset loaded successfully")
print("Total claims:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nGold label distribution:")
print(df["gold_label"].value_counts())

print("\nFaithfulness label distribution:")
print(df["faithfulness_label"].value_counts())

print("\nFirst 3 rows:")
print(df[["claim_id", "claim_ar", "gold_label", "source_name"]].head(3))