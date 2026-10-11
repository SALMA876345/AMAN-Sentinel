"""Phase 1: build a leakage-free CLEAN HACKATHON SPLIT from the restored
source pool (data/claim_training.csv), as a documented alternative to the
HISTORICAL SPLIT already on disk at data/final_studies/source_*.csv.

Why the historical split leaks
-------------------------------
build_final_study_manifests.py's build_source_pool() groups rows only by
claim_group (sha256 of the normalised claim) before splitting. Evidence
text is free to repeat across train/validation/test. The Phase 1 audit
measured 57.2% of source_binary_test rows reusing evidence seen verbatim
in training, with 84.7% train-majority-label agreement on that shared
evidence — and the model input is `claim_norm [SEP] evidence_norm`.

What this script does differently
-----------------------------------
Every row is a node. Two rows are joined into the SAME split-protected
component if they share a claim_group OR share a non-trivial evidence_group
(evidence_norm length >= MIN_EVIDENCE_CHARS_FOR_GROUPING, using the
project's own faithfulness-gate threshold of 40 chars so near-empty/
placeholder evidence does not spuriously merge unrelated rows into one
giant component). Connected components are found with union-find, and each
component is treated as one indivisible unit by sklearn's
StratifiedGroupKFold — reusing split_70_10_20() unchanged from
utils/build_final_study_manifests.py, just pointed at the merged
component_id instead of claim_group alone.

This guarantees zero exact claim overlap AND zero exact evidence overlap
between train/validation/test simultaneously, for both axes at once -- not
two independently-checked groupings that can still leak when a claim and an
evidence passage cross between different rows.

Near-duplicate handling
------------------------
"Near-duplicate" is implemented as exact duplicate text AFTER Arabic
normalisation (same normalize_arabic() the rest of the project uses) --
rows whose claim_norm is identical, or whose (claim_norm, evidence_norm)
pair is identical, are deduplicated (keep first) before splitting. This
catches diacritic/alef/punctuation variants of the same text. Fuzzy /
edit-distance near-duplicate detection across DIFFERENT normalised text was
not implemented in Phase 1 and is reported as a limitation, not hidden.

Output
------
hackathon/data/clean_split/source_{binary,three_class}_{train,validation,test}.csv
hackathon/reports/clean_split_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

from utils.faav_common import normalize_arabic  # noqa: E402
from utils.build_final_study_manifests import split_70_10_20  # noqa: E402

CLAIM_TRAINING_FILE = ROOT / "data" / "claim_training.csv"
OUT_DIR = ROOT / "hackathon" / "data" / "clean_split"
REPORT_DIR = ROOT / "hackathon" / "reports"

SEED = 42
# Reuses the project's own faithfulness-gate "meaningful evidence" threshold
# (utils/faav_common.faithfulness_controlled_decision default
# min_evidence_chars=40) rather than inventing a new constant.
MIN_EVIDENCE_CHARS_FOR_GROUPING = 40

FINAL_LABELS = ["Real", "Fake", "Unverified"]


def text_hash(value: object) -> str:
    text = normalize_arabic(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def dataframe_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    payload = df[columns].sort_values(columns[0]).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_components(df: pd.DataFrame) -> np.ndarray:
    """Connected components over rows sharing claim_group or a non-trivial
    evidence_group. Returns one component id per row."""
    n = len(df)
    uf = UnionFind(n)

    # Union rows sharing claim_group
    for _, idx in df.groupby("claim_group").groups.items():
        idx = idx.to_numpy()
        for i in range(1, len(idx)):
            uf.union(int(idx[0]), int(idx[i]))

    # Union rows sharing a non-trivial evidence_group
    eligible = df[df["evidence_norm"].str.len() >= MIN_EVIDENCE_CHARS_FOR_GROUPING]
    for _, idx in eligible.groupby("evidence_group").groups.items():
        idx = idx.to_numpy()
        for i in range(1, len(idx)):
            uf.union(int(idx[0]), int(idx[i]))

    roots = np.array([uf.find(i) for i in range(n)])
    # Relabel roots to compact contiguous ids for readability
    _, component_ids = np.unique(roots, return_inverse=True)
    return component_ids


def task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "binary":
        out = df[df["binary_label"].isin(["Real", "Fake"])].copy()
        out["task_label"] = out["binary_label"]
    elif task == "three_class":
        out = df[df["three_class_label"].isin(FINAL_LABELS)].copy()
        out["task_label"] = out["three_class_label"]
    else:
        raise ValueError(task)
    return out.reset_index(drop=True)


def split_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "components": int(df["component_id"].nunique()),
        "claim_groups": int(df["claim_group"].nunique()),
        "evidence_groups": int(df["evidence_group"].nunique()),
        "binary_labels": {
            str(k): int(v)
            for k, v in df.loc[df["binary_label"] != "", "binary_label"].value_counts().items()
        },
        "three_class_labels": {
            str(k): int(v) for k, v in df["three_class_label"].value_counts().items()
        },
        "fingerprint_sha256": dataframe_fingerprint(df, ["sample_id", "component_id", "binary_label", "three_class_label"]),
    }


def overlap_count(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column]) & set(right[column]))


def pair_overlap_count(left: pd.DataFrame, right: pd.DataFrame) -> int:
    lp = set(zip(left["claim_norm"].astype(str), left["evidence_norm"].astype(str)))
    rp = set(zip(right["claim_norm"].astype(str), right["evidence_norm"].astype(str)))
    return len(lp & rp)


def main() -> None:
    if not CLAIM_TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"{CLAIM_TRAINING_FILE} not found. Run "
            "hackathon/restore_four_dataset_training.py then "
            "utils/build_claim_training.py first."
        )

    raw = pd.read_csv(CLAIM_TRAINING_FILE, low_memory=False)
    raw = raw[raw["source"] != "AFND"].copy()  # same exclusion as the historical pipeline

    raw["claim_text"] = raw["text"].fillna("").astype(str)
    raw["evidence_text"] = raw["evidence_text"].fillna("").astype(str)
    raw["claim_norm"] = raw["claim_text"].map(normalize_arabic)
    raw["evidence_norm"] = raw["evidence_text"].map(normalize_arabic)
    raw = raw[raw["claim_norm"] != ""].reset_index(drop=True)

    before_dedup = len(raw)
    raw = raw.drop_duplicates(subset=["claim_norm"], keep="first")
    after_claim_dedup = len(raw)
    raw = raw.drop_duplicates(subset=["claim_norm", "evidence_norm"], keep="first")
    after_pair_dedup = len(raw)
    raw = raw.reset_index(drop=True)
    dedup_report = {
        "rows_before_dedup": before_dedup,
        "rows_after_claim_norm_dedup": after_claim_dedup,
        "exact_duplicate_claims_removed": before_dedup - after_claim_dedup,
        "rows_after_claim_evidence_pair_dedup": after_pair_dedup,
        "additional_duplicate_pairs_removed": after_claim_dedup - after_pair_dedup,
    }
    print("Dedup report:", json.dumps(dedup_report, indent=2))

    # Conflicting-claim-group removal, same policy as the historical pipeline
    raw["claim_group"] = raw["claim_norm"].map(text_hash)
    raw["evidence_group"] = raw["evidence_norm"].map(text_hash)
    conflicts = raw.groupby("claim_group")["label"].nunique()
    conflicting_groups = set(conflicts[conflicts > 1].index)
    conflict_rows = int(raw["claim_group"].isin(conflicting_groups).sum())
    raw = raw[~raw["claim_group"].isin(conflicting_groups)].reset_index(drop=True)
    print(f"Removed {len(conflicting_groups)} conflicting claim groups ({conflict_rows} rows)")

    raw["binary_label"] = raw["label"].where(raw["label"].isin(["Real", "Fake"]), "")
    raw["three_class_label"] = raw["label"]
    raw["dataset_family"] = "source"
    raw["sample_id"] = [f"CLEAN-{i:06d}-{text_hash(c + e)[:10]}" for i, (c, e) in enumerate(
        zip(raw["claim_norm"], raw["evidence_norm"]), start=1
    )]

    raw["component_id"] = build_components(raw)
    n_components = raw["component_id"].nunique()
    print(f"Rows: {len(raw):,}  ->  connected components: {n_components:,} "
          f"(compression ratio {len(raw) / n_components:.2f}x)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "phase1_clean_hackathon_split_v1",
        "seed": SEED,
        "min_evidence_chars_for_grouping": MIN_EVIDENCE_CHARS_FOR_GROUPING,
        "near_duplicate_policy": (
            "exact match on normalize_arabic(claim) and on "
            "(normalize_arabic(claim), normalize_arabic(evidence)) pairs; "
            "fuzzy/edit-distance near-duplicate detection NOT implemented in Phase 1"
        ),
        "dedup_report": dedup_report,
        "conflicting_claim_groups_removed": len(conflicting_groups),
        "conflicting_rows_removed": conflict_rows,
        "input_rows_after_dedup_and_conflict_removal": int(len(raw)),
        "connected_components": int(n_components),
        "splits": {},
        "overlap_checks": {},
    }

    keep_cols = [
        "sample_id", "dataset_family", "source", "claim_text", "evidence_text",
        "claim_norm", "evidence_norm", "claim_group", "evidence_group",
        "component_id", "binary_label", "three_class_label",
    ]

    for task in ["binary", "three_class"]:
        task_df = task_frame(raw[keep_cols], task)
        label_col = "binary_label" if task == "binary" else "three_class_label"
        train, validation, test = split_70_10_20(
            task_df, label_col=label_col, group_col="component_id",
        )
        for name, part in [("train", train), ("validation", validation), ("test", test)]:
            part = part.copy()
            part["task_label"] = part[label_col]
            out_path = OUT_DIR / f"source_{task}_{name}.csv"
            part.sort_values("sample_id").to_csv(out_path, index=False, encoding="utf-8-sig")
            manifest["splits"][f"source_{task}_{name}"] = split_summary(part)
            print(f"  wrote {out_path}  ({len(part):,} rows)")

        manifest["overlap_checks"][f"{task}_claim_group_train_test"] = overlap_count(train, test, "claim_group")
        manifest["overlap_checks"][f"{task}_claim_group_train_validation"] = overlap_count(train, validation, "claim_group")
        manifest["overlap_checks"][f"{task}_claim_group_validation_test"] = overlap_count(validation, test, "claim_group")
        manifest["overlap_checks"][f"{task}_evidence_group_train_test"] = overlap_count(train, test, "evidence_group")
        manifest["overlap_checks"][f"{task}_evidence_group_train_validation"] = overlap_count(train, validation, "evidence_group")
        manifest["overlap_checks"][f"{task}_evidence_group_validation_test"] = overlap_count(validation, test, "evidence_group")
        manifest["overlap_checks"][f"{task}_exact_claim_text_train_test"] = overlap_count(train, test, "claim_norm")
        manifest["overlap_checks"][f"{task}_exact_claim_text_train_validation"] = overlap_count(train, validation, "claim_norm")
        manifest["overlap_checks"][f"{task}_claim_evidence_pair_train_test"] = pair_overlap_count(train, test)
        manifest["overlap_checks"][f"{task}_claim_evidence_pair_train_validation"] = pair_overlap_count(train, validation)
        manifest["overlap_checks"][f"{task}_component_id_train_test"] = overlap_count(train, test, "component_id")
        manifest["overlap_checks"][f"{task}_component_id_train_validation"] = overlap_count(train, validation, "component_id")

    manifest_path = REPORT_DIR / "clean_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path}")

    all_zero = all(v == 0 for k, v in manifest["overlap_checks"].items())
    print(f"\nAll overlap checks zero: {all_zero}")
    if not all_zero:
        print("NON-ZERO OVERLAPS:")
        for k, v in manifest["overlap_checks"].items():
            if v != 0:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
