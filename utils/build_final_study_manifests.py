"""Build the frozen manifests used by the final three-study evaluation.

The binary task predicts claim veracity (Real/Fake). The three-class task
predicts the faithfulness-aware final decision (Real/Fake/Unverified).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import normalize_arabic


DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "final_studies"
RESULTS_DIR = ROOT / "results" / "final_studies"
SEED = 42
FINAL_LABELS = ["Real", "Fake", "Unverified"]
VERIFICATION_TO_FINAL = {
    "Supported": "Real",
    "Refuted": "Fake",
    "Not Enough Evidence": "Unverified",
}


def text_hash(value: object) -> str:
    text = normalize_arabic(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    columns = ["sample_id", "claim_group", "binary_label", "three_class_label"]
    payload = df[columns].sort_values("sample_id").to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose_fold(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str,
    n_splits: int,
    seed: int,
    target_fraction: float,
    reference_distribution: pd.Series,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    labels = reference_distribution.index.tolist()
    best_score = None
    best_indices = None
    for _, indices in splitter.split(df, df[label_col], df[group_col]):
        fold = df.iloc[indices]
        distribution = (
            fold[label_col]
            .value_counts(normalize=True)
            .reindex(labels, fill_value=0.0)
        )
        size_error = abs(len(fold) / len(df) - target_fraction)
        label_error = float(
            np.abs(distribution.values - reference_distribution.values).sum()
        )
        score = size_error + label_error
        if best_score is None or score < best_score:
            best_score = score
            best_indices = indices
    if best_indices is None:
        raise RuntimeError("Unable to construct a grouped split.")
    return np.asarray(best_indices)


def adjust_grouped_selection_size(
    df: pd.DataFrame,
    selected_indices: np.ndarray,
    *,
    label_col: str,
    group_col: str,
    target_count: int,
) -> np.ndarray:
    """Move whole groups until the selected row count reaches target_count."""
    selected_groups = set(df.iloc[selected_indices][group_col])
    labels = sorted(df[label_col].unique().tolist())
    reference = (
        df[label_col].value_counts(normalize=True).reindex(labels, fill_value=0.0)
    )
    grouped = {
        group: indices.to_numpy()
        for group, indices in df.groupby(group_col).groups.items()
    }

    def selected_row_indices() -> np.ndarray:
        parts = [grouped[group] for group in selected_groups]
        if not parts:
            return np.asarray([], dtype=int)
        return np.sort(np.concatenate(parts))

    while True:
        current_indices = selected_row_indices()
        difference = target_count - len(current_indices)
        if difference == 0:
            return current_indices

        if difference > 0:
            candidates = [
                group
                for group in grouped
                if group not in selected_groups
                and len(grouped[group]) <= difference
            ]
            direction = 1
        else:
            candidates = [
                group
                for group in selected_groups
                if len(grouped[group]) <= -difference
            ]
            direction = -1
        if not candidates:
            return current_indices

        best_group = None
        best_score = None
        for group in candidates:
            candidate_groups = set(selected_groups)
            if direction > 0:
                candidate_groups.add(group)
            else:
                candidate_groups.remove(group)
            candidate_parts = [grouped[item] for item in candidate_groups]
            candidate_indices = np.concatenate(candidate_parts)
            candidate = df.iloc[candidate_indices]
            distribution = (
                candidate[label_col]
                .value_counts(normalize=True)
                .reindex(labels, fill_value=0.0)
            )
            size_error = abs(len(candidate) - target_count)
            label_error = float(
                np.abs(distribution.values - reference.values).sum()
            )
            score = (size_error, label_error, str(group))
            if best_score is None or score < best_score:
                best_score = score
                best_group = group
        if best_group is None:
            return current_indices
        if direction > 0:
            selected_groups.add(best_group)
        else:
            selected_groups.remove(best_group)


def split_70_10_20(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.reset_index(drop=True)
    reference = df[label_col].value_counts(normalize=True).sort_index()
    test_indices = choose_fold(
        df,
        label_col=label_col,
        group_col=group_col,
        n_splits=5,
        seed=SEED,
        target_fraction=0.20,
        reference_distribution=reference,
    )
    test_indices = adjust_grouped_selection_size(
        df,
        test_indices,
        label_col=label_col,
        group_col=group_col,
        target_count=round(len(df) * 0.20),
    )
    test = df.iloc[test_indices].copy()
    remaining = df.drop(df.index[test_indices]).reset_index(drop=True)
    val_indices = choose_fold(
        remaining,
        label_col=label_col,
        group_col=group_col,
        n_splits=8,
        seed=SEED + 1,
        target_fraction=0.125,
        reference_distribution=reference,
    )
    val_indices = adjust_grouped_selection_size(
        remaining,
        val_indices,
        label_col=label_col,
        group_col=group_col,
        target_count=round(len(df) * 0.10),
    )
    validation = remaining.iloc[val_indices].copy()
    train = remaining.drop(remaining.index[val_indices]).copy()
    return train, validation, test


def task_frame(
    df: pd.DataFrame,
    *,
    task: str,
) -> tuple[pd.DataFrame, str]:
    if task == "binary":
        out = df[df["binary_label"].isin(["Real", "Fake"])].copy()
        out["task_label"] = out["binary_label"]
        return out.reset_index(drop=True), "task_label"
    if task == "three_class":
        out = df[
            df["three_class_label"].isin(["Real", "Fake", "Unverified"])
        ].copy()
        out["task_label"] = out["three_class_label"]
        return out.reset_index(drop=True), "task_label"
    raise ValueError(f"Unknown task: {task}")


def build_source_pool() -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(DATA_DIR / "claim_training.csv", low_memory=False)
    source = source[source["source"] != "AFND"].copy()
    source["claim_text"] = source["text"].fillna("").astype(str)
    source["evidence_text"] = source["evidence_text"].fillna("").astype(str)
    source["claim_norm"] = source["claim_text"].map(normalize_arabic)
    source["evidence_norm"] = source["evidence_text"].map(normalize_arabic)
    source = source[source["claim_norm"] != ""].copy()
    source["claim_group"] = source["claim_norm"].map(text_hash)

    conflicts = source.groupby("claim_group")["label"].nunique()
    conflicting_groups = set(conflicts[conflicts > 1].index)
    conflict_rows = int(source["claim_group"].isin(conflicting_groups).sum())
    source = source[~source["claim_group"].isin(conflicting_groups)].copy()

    source["binary_label"] = source["label"].where(
        source["label"].isin(["Real", "Fake"]), ""
    )
    source["three_class_label"] = source["label"]
    source["dataset_family"] = "source"
    source["sample_id"] = [
        f"SRC-{i:06d}-{text_hash(c + e)[:10]}"
        for i, (c, e) in enumerate(
            zip(source["claim_norm"], source["evidence_norm"]), start=1
        )
    ]
    source["evidence_group"] = source["evidence_norm"].map(text_hash)

    keep = [
        "sample_id",
        "dataset_family",
        "source",
        "claim_text",
        "evidence_text",
        "claim_norm",
        "evidence_norm",
        "claim_group",
        "evidence_group",
        "binary_label",
        "three_class_label",
    ]
    audit = {
        "input_rows_without_afnd": int(len(source) + conflict_rows),
        "conflicting_claim_groups_removed": int(len(conflicting_groups)),
        "conflicting_rows_removed": conflict_rows,
    }
    return source[keep].reset_index(drop=True), audit


def build_uae_pool() -> pd.DataFrame:
    uae = pd.read_excel(DATA_DIR / "uae_test_verification.xlsx")
    uae["claim_text"] = uae["claim_ar"].fillna("").astype(str)
    uae["evidence_text"] = uae["evidence_text"].fillna("").astype(str)
    uae["claim_norm"] = uae["claim_text"].map(normalize_arabic)
    uae["evidence_norm"] = uae["evidence_text"].map(normalize_arabic)
    uae = uae[uae["claim_norm"] != ""].copy()
    uae["claim_group"] = uae["claim_norm"].map(text_hash)
    uae["evidence_group"] = [
        text_hash(evidence) if evidence else f"no-evidence-{i:06d}"
        for i, evidence in enumerate(uae["evidence_norm"], start=1)
    ]
    uae["binary_label"] = uae["claim_truth_label"].where(
        uae["claim_truth_label"].isin(["Real", "Fake"]), ""
    )
    uae["three_class_label"] = uae["verification_label"].map(
        VERIFICATION_TO_FINAL
    )
    uae["dataset_family"] = "UAE"
    uae["sample_id"] = [
        f"UAE-{i:06d}-{text_hash(c)[:10]}"
        for i, c in enumerate(uae["claim_norm"], start=1)
    ]
    if "source" not in uae.columns:
        uae["source"] = "UAE-FND"

    keep = [
        "sample_id",
        "dataset_family",
        "source",
        "claim_text",
        "evidence_text",
        "claim_norm",
        "evidence_norm",
        "claim_group",
        "evidence_group",
        "binary_label",
        "three_class_label",
        "claim_truth_label",
        "verification_label",
        "retrieval_quality",
        "faithfulness_label",
        "evidence_url",
    ]
    return uae[keep].reset_index(drop=True)


def overlap_count(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column]) & set(right[column]))


def split_summary(df: pd.DataFrame, full_rows: int) -> dict:
    return {
        "rows": int(len(df)),
        "fraction_of_task_dataset": float(len(df) / full_rows),
        "claim_groups": int(df["claim_group"].nunique()),
        "evidence_groups": int(df["evidence_group"].nunique()),
        "binary_labels": {
            str(k): int(v)
            for k, v in df.loc[df["binary_label"] != "", "binary_label"]
            .value_counts()
            .items()
        },
        "three_class_labels": {
            str(k): int(v)
            for k, v in df["three_class_label"].value_counts().items()
        },
        "fingerprint_sha256": dataframe_fingerprint(df),
    }


def write_split(name: str, df: pd.DataFrame) -> None:
    df.sort_values("sample_id").to_csv(
        OUT_DIR / f"{name}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    source, source_audit = build_source_pool()
    uae = build_uae_pool()
    splits: dict[str, pd.DataFrame] = {}
    full_rows: dict[str, int] = {}
    overlap_checks: dict[str, int] = {}

    for task in ["binary", "three_class"]:
        source_task, label_col = task_frame(source, task=task)
        uae_task, _ = task_frame(uae, task=task)
        source_train, source_val, source_test = split_70_10_20(
            source_task,
            label_col=label_col,
            group_col="claim_group",
        )
        uae_train, uae_val, uae_test = split_70_10_20(
            uae_task,
            label_col=label_col,
            group_col="evidence_group",
        )
        task_splits = {
            f"source_{task}_train": source_train,
            f"source_{task}_validation": source_val,
            f"source_{task}_test": source_test,
            f"uae_{task}_full": uae_task,
            f"uae_{task}_train": uae_train,
            f"uae_{task}_validation": uae_val,
            f"uae_{task}_test": uae_test,
        }
        combined_train = pd.concat(
            [source_train, uae_train],
            ignore_index=True,
        )
        combined_val = pd.concat(
            [source_val, uae_val],
            ignore_index=True,
        )
        combined_test = pd.concat(
            [source_test, uae_test],
            ignore_index=True,
        )
        for frame in [combined_train, combined_val, combined_test]:
            frame["protected_group"] = np.where(
                frame["dataset_family"] == "source",
                "source-claim:" + frame["claim_group"].astype(str),
                "uae-evidence:" + frame["evidence_group"].astype(str),
            )
        task_splits.update(
            {
                f"combined_{task}_train": combined_train,
                f"combined_{task}_validation": combined_val,
                f"combined_{task}_test": combined_test,
            }
        )
        splits.update(task_splits)
        full_rows.update(
            {
                f"source_{task}_{part}": len(source_task)
                for part in ["train", "validation", "test"]
            }
        )
        full_rows.update(
            {
                f"uae_{task}_{part}": len(uae_task)
                for part in ["full", "train", "validation", "test"]
            }
        )
        combined_rows = len(source_task) + len(uae_task)
        full_rows.update(
            {
                f"combined_{task}_{part}": combined_rows
                for part in ["train", "validation", "test"]
            }
        )
        overlap_checks.update(
            {
                f"source_{task}_claim_train_validation": overlap_count(
                    source_train, source_val, "claim_group"
                ),
                f"source_{task}_claim_train_test": overlap_count(
                    source_train, source_test, "claim_group"
                ),
                f"source_{task}_claim_validation_test": overlap_count(
                    source_val, source_test, "claim_group"
                ),
                f"uae_{task}_evidence_train_validation": overlap_count(
                    uae_train, uae_val, "evidence_group"
                ),
                f"uae_{task}_evidence_train_test": overlap_count(
                    uae_train, uae_test, "evidence_group"
                ),
                f"uae_{task}_evidence_validation_test": overlap_count(
                    uae_val, uae_test, "evidence_group"
                ),
                f"source_uae_{task}_claim_overlap": overlap_count(
                    source_task, uae_task, "claim_group"
                ),
                f"combined_{task}_protected_train_validation": overlap_count(
                    combined_train, combined_val, "protected_group"
                ),
                f"combined_{task}_protected_train_test": overlap_count(
                    combined_train, combined_test, "protected_group"
                ),
                f"combined_{task}_protected_validation_test": overlap_count(
                    combined_val, combined_test, "protected_group"
                ),
            }
        )

    for name, frame in splits.items():
        write_split(name, frame)

    audit = {
        "protocol": "final_three_study_full_uae_zero_shot_v3",
        "seed": SEED,
        "target_split": {
            "train": 0.70,
            "validation": 0.10,
            "test": 0.20,
            "note": (
                "Grouped splitting can differ slightly from exact percentages "
                "because an entire claim/evidence group must stay together."
            ),
        },
        "task_definitions": {
            "binary": "Claim veracity: Real/Fake; Unverified rows excluded.",
            "three_class": (
                "Faithfulness-aware final decision: Real/Fake/Unverified. "
                "For UAE rows this is mapped from Supported/Refuted/"
                "Not Enough Evidence."
            ),
        },
        "study_definitions": {
            "study_1_zero_shot": (
                "Train/tune on source data only; evaluate on the complete UAE "
                "task dataset (5,053 binary / 6,390 three-class rows)."
            ),
            "study_2_source_domain": (
                "Train/tune on source data; evaluate on frozen source test."
            ),
            "study_3_combined_adaptation": (
                "Train on source plus UAE train; tune on combined validation; "
                "evaluate on the frozen 20% combined source-plus-UAE test."
            ),
        },
        "source_pool": source_audit,
        "splits": {
            name: split_summary(frame, full_rows[name])
            for name, frame in splits.items()
        },
        "overlap_checks": overlap_checks,
        "important_limitations": [
            (
                "VERA-ARAB and ArCOV19 provide claim-veracity labels, while "
                "AraStance supplies the native evidence-stance/NEI labels."
            ),
            (
                "The source three-class task is highly imbalanced because "
                "native Unverified supervision comes primarily from AraStance."
            ),
            (
                "UAE three-class labels are evidence-conditioned silver labels "
                "derived from claim truth and evidence-quality fields."
            ),
        ],
    }
    audit_path = RESULTS_DIR / "manifest_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(audit["splits"], ensure_ascii=False, indent=2))
    print("\nOverlap checks:")
    print(json.dumps(audit["overlap_checks"], indent=2))
    print(f"\nSaved manifests to {OUT_DIR}")
    print(f"Saved audit to {audit_path}")


if __name__ == "__main__":
    main()
