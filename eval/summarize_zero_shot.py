"""Summarize strict zero-shot FAAV experiments on the frozen UAE test set."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR
from utils.faav_common import normalize_arabic


RUN_DIR = Path(RESULTS_DIR) / "zero_shot"
OUT_CSV = RUN_DIR / "zero_shot_model_comparison.csv"
OUT_MD = RUN_DIR / "zero_shot_model_comparison.md"
OUT_JSON = RUN_DIR / "zero_shot_evaluation_audit.json"


def friendly_name(stem: str) -> str:
    if "claim_only_lr" in stem:
        return "Claim-only LR"
    if "claim_evidence_lr" in stem:
        return "Claim + provided evidence LR"
    if "fusion_lr" in stem:
        return "Fusion LR"
    if "two_stage_lr" in stem and "_tfidf_" in stem:
        return "Two-stage LR + full TF-IDF retrieval"
    if "two_stage_lr" in stem and "_bm25_" in stem:
        return "Two-stage LR + BM25 retrieval (1k corpus)"
    if "two_stage_lr" in stem:
        return "Two-stage LR + provided evidence"
    return stem


def token_overlap(a: object, b: object) -> float:
    a_tokens = set(normalize_arabic(a).split())
    b_tokens = set(normalize_arabic(b).split())
    if not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(b_tokens)


def prediction_path(metrics_path: str) -> str:
    return metrics_path.replace("_metrics.json", "_predictions.csv")


def test_fingerprint(df: pd.DataFrame) -> str:
    if "zero_shot_row_id" in df.columns:
        values = df["zero_shot_row_id"].astype(str)
    elif "claim_id" in df.columns:
        values = df["claim_id"].astype(str)
    else:
        values = df["claim_text"].astype(str).apply(normalize_arabic)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    rows = []
    fingerprints = {}
    retrieval_details = {}
    for metrics_path in sorted(glob.glob(str(RUN_DIR / "*_metrics.json"))):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        pred_path = prediction_path(metrics_path)
        if not os.path.exists(pred_path):
            continue

        usecols = [
            "zero_shot_row_id",
            "claim_id",
            "claim_text",
            "label",
            "prediction",
            "retrieved_evidence",
            "evidence_text",
            "effective_evidence",
            "retrieval_score",
            "evidence_usable_proba",
        ]
        header = pd.read_csv(pred_path, nrows=0).columns
        cols = [c for c in usecols if c in header]
        pred = pd.read_csv(pred_path, usecols=cols, low_memory=False)
        stem = Path(metrics_path).name.replace("_metrics.json", "")
        name = friendly_name(stem)
        fingerprints[name] = test_fingerprint(pred)

        report = metrics["classification_report"]
        abstention_rate = float((pred["prediction"] == "Not Enough Evidence").mean())
        supported_f1 = float(report.get("Supported", {}).get("f1-score", 0.0))
        refuted_f1 = float(report.get("Refuted", {}).get("f1-score", 0.0))
        nei_f1 = float(report.get("Not Enough Evidence", {}).get("f1-score", 0.0))

        retrieval_overlap = None
        retrieval_hit_rate = None
        mean_retrieval_score = None
        if "retrieved_evidence" in pred.columns and pred["retrieved_evidence"].fillna("").str.strip().ne("").any():
            gold_evidence = pred.get("evidence_text", pd.Series("", index=pred.index))
            overlaps = [
                token_overlap(retrieved, gold)
                for retrieved, gold in zip(pred["retrieved_evidence"], gold_evidence)
            ]
            retrieval_overlap = float(pd.Series(overlaps).mean())
            retrieval_hit_rate = float((pd.Series(overlaps) >= 0.10).mean())
            if "retrieval_score" in pred.columns:
                mean_retrieval_score = float(
                    pd.to_numeric(pred["retrieval_score"], errors="coerce").fillna(0).mean()
                )
            retrieval_details[name] = {
                "provided_evidence_token_overlap_mean": retrieval_overlap,
                "provided_evidence_token_overlap_at_0_10": retrieval_hit_rate,
                "mean_retrieval_score": mean_retrieval_score,
                "note": (
                    "This is a lexical proxy against the provided evidence, not "
                    "gold document Recall@K."
                ),
            }

        rows.append(
            {
                "Model": name,
                "N": int(metrics["n"]),
                "Accuracy": float(metrics["accuracy"]),
                "Macro F1": float(metrics["macro_f1"]),
                "MCC": float(metrics["mcc"]),
                "Supported F1": supported_f1,
                "Refuted F1": refuted_f1,
                "NEI F1": nei_f1,
                "Abstention Rate": abstention_rate,
                "Retrieval Evidence Overlap": retrieval_overlap,
                "Retrieval Hit Proxy @0.10": retrieval_hit_rate,
                "Leakage Count": int(metrics["leakage"]["exact_overlap_count"]),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["Macro F1", "MCC"], ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    unique_fingerprints = sorted(set(fingerprints.values()))
    same_rows = len(unique_fingerprints) == 1 and len(comparison) > 0
    audit = {
        "protocol": "strict_zero_shot_cross_domain",
        "uae_training_rows": 0,
        "models_evaluated": int(len(comparison)),
        "all_models_same_test_rows": same_rows,
        "test_fingerprints": fingerprints,
        "unique_test_fingerprint_count": len(unique_fingerprints),
        "all_leakage_counts_zero": bool(
            len(comparison) and (comparison["Leakage Count"] == 0).all()
        ),
        "best_model_by_macro_f1": (
            comparison.iloc[0]["Model"] if len(comparison) else None
        ),
        "best_macro_f1": (
            float(comparison.iloc[0]["Macro F1"]) if len(comparison) else None
        ),
        "retrieval_metrics": retrieval_details,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    display = comparison.copy()
    numeric_cols = [
        "Accuracy",
        "Macro F1",
        "MCC",
        "Supported F1",
        "Refuted F1",
        "NEI F1",
        "Abstention Rate",
        "Retrieval Evidence Overlap",
        "Retrieval Hit Proxy @0.10",
    ]
    for col in numeric_cols:
        display[col] = display[col].map(
            lambda value: "-" if pd.isna(value) else f"{value:.4f}"
        )
    headers = display.columns.tolist()
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        markdown_rows.append(
            "| " + " | ".join(str(row[col]) for col in headers) + " |"
        )
    lines = [
        "# Strict Zero-Shot UAE Evaluation",
        "",
        "- UAE claims used for training: **0**",
        f"- All models used identical frozen test rows: **{same_rows}**",
        f"- Exact train/test leakage for every run: **{audit['all_leakage_counts_zero']}**",
        "",
        "\n".join(markdown_rows),
        "",
        "## Interpretation",
        "",
        (
            "The comparison is a zero-shot cross-domain evaluation. Retrieval "
            "overlap is a lexical proxy against the provided UAE evidence and is "
            "not a gold-document Recall@K measurement."
        ),
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(comparison.to_string(index=False))
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Saved -> {OUT_MD}")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
