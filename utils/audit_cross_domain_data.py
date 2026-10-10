"""Audit FAAV datasets for labels, duplicates, leakage, and domain columns."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLAIM_TRAIN_2CLASS_FILE, RESULTS_DIR, TEST_2CLASS_FILE, TEST_SHEET
from utils.faav_common import (
    REAL_FAKE_LABELS,
    deduplicate_claims,
    leakage_report,
    load_ans_frame,
    load_test_frame,
    load_training_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FAAV train/test datasets.")
    parser.add_argument("--train-file", default=CLAIM_TRAIN_2CLASS_FILE)
    parser.add_argument("--test-file", default=TEST_2CLASS_FILE)
    parser.add_argument("--ans-dir", default=os.path.join("data", "ANS"))
    parser.add_argument("--near-threshold", type=float, default=0.92)
    parser.add_argument("--near-cap", type=int, default=15000)
    parser.add_argument("--out-json", default=os.path.join(RESULTS_DIR, "data_audit.json"))
    parser.add_argument("--out-csv", default=os.path.join(RESULTS_DIR, "data_audit_near_duplicates.csv"))
    return parser.parse_args()


def exact_duplicate_summary(df: pd.DataFrame) -> dict:
    counts = df["claim_id"].value_counts()
    dup_counts = counts[counts > 1]
    return {
        "rows": int(len(df)),
        "unique_claims": int(df["claim_id"].nunique()),
        "exact_duplicate_rows": int((counts - 1).clip(lower=0).sum()),
        "duplicate_groups": int(len(dup_counts)),
    }


def near_duplicate_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    *,
    threshold: float,
    cap: int,
    label: str,
) -> pd.DataFrame:
    left_small = left.head(cap).copy()
    if right is None:
        right_small = left_small
    else:
        right_small = right.head(cap).copy()
    if len(left_small) < 2 or len(right_small) < 2:
        return pd.DataFrame()

    texts = pd.concat([left_small["claim_norm"], right_small["claim_norm"]], ignore_index=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    matrix = vectorizer.fit_transform(texts)
    left_matrix = matrix[: len(left_small)]
    right_matrix = matrix[len(left_small) :]

    if right is None:
        nn = NearestNeighbors(n_neighbors=2, metric="cosine")
        nn.fit(left_matrix)
        distances, indices = nn.kneighbors(left_matrix)
        rows = []
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            sim = 1.0 - float(dist_row[1])
            j = int(idx_row[1])
            if i < j and sim >= threshold:
                rows.append(
                    {
                        "audit": label,
                        "similarity": round(sim, 4),
                        "left_index": int(left_small.index[i]),
                        "right_index": int(left_small.index[j]),
                        "left_label": left_small.iloc[i]["label"],
                        "right_label": left_small.iloc[j]["label"],
                        "left_source": left_small.iloc[i]["source_dataset"],
                        "right_source": left_small.iloc[j]["source_dataset"],
                    }
                )
        return pd.DataFrame(rows)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(right_matrix)
    distances, indices = nn.kneighbors(left_matrix)
    rows = []
    for i, (dist, idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
        sim = 1.0 - float(dist)
        if sim >= threshold:
            rows.append(
                {
                    "audit": label,
                    "similarity": round(sim, 4),
                    "left_index": int(left_small.index[i]),
                    "right_index": int(right_small.index[int(idx)]),
                    "left_label": left_small.iloc[i]["label"],
                    "right_label": right_small.iloc[int(idx)]["label"],
                    "left_source": left_small.iloc[i]["source_dataset"],
                    "right_source": right_small.iloc[int(idx)]["source_dataset"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    train = load_training_frame(args.train_file, REAL_FAKE_LABELS)
    test = load_test_frame(args.test_file, REAL_FAKE_LABELS, sheet_name=TEST_SHEET)
    ans = load_ans_frame(args.ans_dir, "test", REAL_FAKE_LABELS)

    train_dedup = deduplicate_claims(train)
    test_dedup = deduplicate_claims(test)
    ans_dedup = deduplicate_claims(ans)

    audit = {
        "train": exact_duplicate_summary(train),
        "test": exact_duplicate_summary(test),
        "ans_test": exact_duplicate_summary(ans),
        "train_after_exact_dedup_rows": int(len(train_dedup)),
        "test_after_exact_dedup_rows": int(len(test_dedup)),
        "ans_after_exact_dedup_rows": int(len(ans_dedup)),
        "train_label_distribution": train["label"].value_counts().to_dict(),
        "test_label_distribution": test["label"].value_counts().to_dict(),
        "ans_label_distribution": ans["label"].value_counts().to_dict(),
        "train_source_distribution": train["source_dataset"].value_counts().head(20).to_dict(),
        "test_domain_distribution": test["domain"].value_counts().head(20).to_dict(),
        "train_test_leakage": leakage_report(train_dedup, test_dedup),
        "train_ans_leakage": leakage_report(train_dedup, ans_dedup),
    }

    near_frames = [
        near_duplicate_pairs(
            train_dedup,
            None,
            threshold=args.near_threshold,
            cap=args.near_cap,
            label="within_train",
        ),
        near_duplicate_pairs(
            train_dedup,
            test_dedup,
            threshold=args.near_threshold,
            cap=args.near_cap,
            label="train_vs_uae_test",
        ),
        near_duplicate_pairs(
            train_dedup,
            ans_dedup,
            threshold=args.near_threshold,
            cap=args.near_cap,
            label="train_vs_ans_test",
        ),
    ]
    near = pd.concat([f for f in near_frames if not f.empty], ignore_index=True) if near_frames else pd.DataFrame()
    audit["near_duplicate_rows_saved"] = int(len(near))
    audit["near_duplicate_threshold"] = float(args.near_threshold)
    audit["near_duplicate_cap"] = int(args.near_cap)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    write_json(args.out_json, audit)
    near.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("Dataset audit")
    print(f"  train rows={len(train):,} exact_duplicate_rows={audit['train']['exact_duplicate_rows']:,}")
    print(f"  test rows={len(test):,} exact_duplicate_rows={audit['test']['exact_duplicate_rows']:,}")
    print(f"  train-test exact leakage={audit['train_test_leakage']['exact_overlap_count']}")
    print(f"  near duplicate rows saved={len(near):,}")
    print(f"Saved -> {args.out_json}")
    print(f"Saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
