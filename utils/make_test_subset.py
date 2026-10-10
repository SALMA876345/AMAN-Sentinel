"""
Reduce the UAE-MVD test set to ~20% of (train + test) for a standard 80/20 split.

Stratified sample: keeps the exact Real/Fake/Unverified proportions of the full
test set. The original 30,015-row file is preserved as a backup.

    train = N rows (claim_training.csv)
    test  = 0.25 * N   →   test / (train+test) = 20%

Output: data/uae_test_dataset_final.xlsx  (reduced, sheet 'Tabelle1')
Backup: data/uae_test_dataset_full30k.xlsx (original, kept once)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, CLAIM_TRAIN_FILE

import pandas as pd

TEST_FILE   = os.path.join(DATA_DIR, "uae_test_dataset_final.xlsx")
BACKUP_FILE = os.path.join(DATA_DIR, "uae_test_dataset_full30k.xlsx")
SHEET       = "Tabelle1"
TARGET_FRAC = 0.20   # test share of (train + test)


def main():
    # Preserve the full original once
    df_full = pd.read_excel(TEST_FILE, sheet_name=SHEET)
    if not os.path.exists(BACKUP_FILE):
        df_full.to_excel(BACKUP_FILE, index=False, sheet_name=SHEET)
        print(f"Backed up full {len(df_full):,}-row test -> {os.path.basename(BACKUP_FILE)}")
    else:
        # always start from the full backup so we never sample a sample
        df_full = pd.read_excel(BACKUP_FILE, sheet_name=SHEET)
        print(f"Loaded full test from backup ({len(df_full):,} rows)")

    n_train = len(pd.read_csv(CLAIM_TRAIN_FILE, usecols=["label"], low_memory=False))
    n_test  = round(TARGET_FRAC / (1 - TARGET_FRAC) * n_train)   # = 0.25 * n_train
    n_test  = min(n_test, len(df_full))
    frac    = n_test / len(df_full)
    print(f"Train rows : {n_train:,}")
    print(f"Target test: {n_test:,}  ({n_test/(n_train+n_test)*100:.1f}% of total, frac={frac:.4f})")

    # Stratified by gold_label — sample each class, concat, shuffle
    parts = [g.sample(frac=frac, random_state=42) for _, g in df_full.groupby("gold_label")]
    sample = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    sample.to_excel(TEST_FILE, index=False, sheet_name=SHEET)
    print(f"\nSaved reduced test -> {os.path.basename(TEST_FILE)}")
    print(f"Test rows  : {len(sample):,}")
    print(f"Class dist : {sample['gold_label'].value_counts().to_dict()}")
    print(f"Proportions: {(sample['gold_label'].value_counts(normalize=True)*100).round(1).to_dict()}")
    print(f"(full original preserved as {os.path.basename(BACKUP_FILE)})")


if __name__ == "__main__":
    main()
