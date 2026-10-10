"""Build supervised evidence-relevance and stance training files for FAAV.

FAAV is an evidence-grounded verifier, so the model needs supervision for two
separate research questions:

1. Is the retrieved evidence usable for this claim?
2. If usable, does the evidence support or refute the claim?

This script converts an evidence-annotated verification file, such as
``data/uae_test_verification.xlsx``, into:

  - relevance_training.csv: claim/evidence pairs labelled usable/not_usable
  - stance_training.csv: usable pairs labelled Supported/Refuted

Hard negatives are included for the relevance gate by pairing claims with
evidence from other claims. The output keeps claim_id and evidence_claim_id so
experiment scripts can remove rows that overlap the evaluation set.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEST_SHEET
from utils.faav_common import (
    SEP_TOKEN,
    claim_hash,
    infer_source_credibility,
    lexical_overlap,
    normalize_arabic,
    parse_date,
    recency_gap_days,
    recency_score,
)

VERIFICATION_LABELS = {"Supported", "Refuted", "Not Enough Evidence"}
STANCE_LABELS = {"Supported", "Refuted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAAV relevance and stance training data.")
    parser.add_argument(
        "--input",
        default=os.path.join(DATA_DIR, "uae_test_verification.xlsx"),
        help="Input CSV/XLSX containing claim, evidence, verification_label, and optional retrieval_quality.",
    )
    parser.add_argument("--sheet", default=TEST_SHEET)
    parser.add_argument(
        "--output-relevance",
        default=os.path.join(DATA_DIR, "relevance_training.csv"),
        help="Output CSV for the evidence relevance gate.",
    )
    parser.add_argument(
        "--output-stance",
        default=os.path.join(DATA_DIR, "stance_training.csv"),
        help="Output CSV for the Supported/Refuted stance verifier.",
    )
    parser.add_argument("--claim-col", default=None)
    parser.add_argument("--evidence-col", default="evidence_text")
    parser.add_argument("--label-col", default="verification_label")
    parser.add_argument("--quality-col", default="retrieval_quality")
    parser.add_argument(
        "--quality-policy",
        choices=["relevant_only", "relevant_or_partial", "any_non_irrelevant"],
        default="relevant_only",
    )
    parser.add_argument("--min-evidence-chars", type=int, default=40)
    parser.add_argument("--max-evidence-chars", type=int, default=1600)
    parser.add_argument("--hard-negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-hard-negatives", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_input(path: str, sheet: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path, low_memory=False)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of these columns exist: {candidates}")


def evidence_quality_ok(value: object, policy: str) -> bool:
    quality = "" if pd.isna(value) else str(value).strip().lower()
    relevant = {"relevant", "relevant evidence", "high"}
    partial = {"partially relevant", "partial", "medium"}
    irrelevant = {"irrelevant", "low", "not relevant"}
    if quality in relevant:
        return True
    if policy == "relevant_or_partial" and quality in partial:
        return True
    if policy == "any_non_irrelevant" and quality not in irrelevant and quality != "":
        return True
    return False


def safe_str(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def prepare_observed_pairs(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    claim_col = args.claim_col or first_existing_column(
        df,
        ["claim_text", "claim_ar", "text", "raw_text", "claim"],
    )
    if args.evidence_col not in df.columns:
        raise ValueError(f"Missing evidence column: {args.evidence_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing verification label column: {args.label_col}")

    out = pd.DataFrame()
    out["claim_text"] = df[claim_col].map(safe_str)
    out["evidence_text"] = df[args.evidence_col].map(safe_str).str.slice(0, args.max_evidence_chars)
    out["claim_norm"] = out["claim_text"].apply(normalize_arabic)
    out["evidence_norm"] = out["evidence_text"].apply(normalize_arabic)
    out["claim_id"] = out["claim_norm"].apply(claim_hash)
    out["evidence_claim_id"] = out["claim_id"]
    out["verification_label"] = df[args.label_col].map(lambda v: safe_str(v).strip())
    out.loc[~out["verification_label"].isin(VERIFICATION_LABELS), "verification_label"] = "Not Enough Evidence"

    if "claim_truth_label" in df.columns:
        out["claim_truth_label"] = df["claim_truth_label"].map(safe_str)
    elif "gold_label" in df.columns:
        out["claim_truth_label"] = df["gold_label"].map(safe_str)
    else:
        out["claim_truth_label"] = ""

    if args.quality_col in df.columns:
        out["retrieval_quality"] = df[args.quality_col].map(safe_str)
        quality_ok = df[args.quality_col].map(lambda v: evidence_quality_ok(v, args.quality_policy))
    else:
        out["retrieval_quality"] = ""
        quality_ok = pd.Series(True, index=df.index)

    has_evidence = out["evidence_text"].str.strip().str.len() >= args.min_evidence_chars
    stance_label = out["verification_label"].isin(STANCE_LABELS)
    out["relevance_label"] = np.where(stance_label & has_evidence & quality_ok, "usable", "not_usable")
    out["relevance_reason"] = np.select(
        [
            ~has_evidence,
            ~stance_label,
            ~quality_ok,
        ],
        [
            "evidence_too_short",
            "verification_label_not_supported_or_refuted",
            f"retrieval_quality_not_sufficient:{args.quality_policy}",
        ],
        default="usable_verified_evidence",
    )

    source_col = first_existing_column(df, ["source_dataset", "source", "label_source"]) if any(
        c in df.columns for c in ["source_dataset", "source", "label_source"]
    ) else None
    domain_col = first_existing_column(df, ["domain", "topics", "topic", "candidate_category"]) if any(
        c in df.columns for c in ["domain", "topics", "topic", "candidate_category"]
    ) else None
    source_name_col = first_existing_column(df, ["source_name", "source", "evidence_url"]) if any(
        c in df.columns for c in ["source_name", "source", "evidence_url"]
    ) else None

    out["source_dataset"] = df[source_col].map(safe_str) if source_col else "relevance_input"
    out["domain"] = df[domain_col].map(safe_str) if domain_col else out["source_dataset"]
    out["source_name"] = df[source_name_col].map(safe_str) if source_name_col else out["source_dataset"]
    out["evidence_url"] = df["evidence_url"].map(safe_str) if "evidence_url" in df.columns else ""
    out["source_credibility"] = [
        infer_source_credibility(src, ds)
        for src, ds in zip(out["source_name"], out["source_dataset"])
    ]

    claim_date_col = next((c for c in ["claim_date", "date", "published_date"] if c in df.columns), None)
    evidence_date_col = next(
        (c for c in ["evidence_date", "labeled_at", "published_date", "date"] if c in df.columns),
        None,
    )
    out["claim_date"] = df[claim_date_col].map(safe_str) if claim_date_col else ""
    out["evidence_date"] = df[evidence_date_col].map(safe_str) if evidence_date_col else ""
    out["recency_gap"] = [
        recency_gap_days(c, e) for c, e in zip(out["claim_date"], out["evidence_date"])
    ]
    out["recency_score"] = out["recency_gap"].apply(recency_score)
    out["faithfulness_score"] = [
        lexical_overlap(c, e) for c, e in zip(out["claim_text"], out["evidence_text"])
    ]
    out["gate_text"] = out["claim_norm"] + SEP_TOKEN + out["evidence_norm"]
    out["row_origin"] = "observed"
    out["pair_id"] = [f"observed_{i}" for i in range(len(out))]

    # Force date parsing once so malformed dates surface during data build, not training.
    _ = [parse_date(v) for v in out["claim_date"].head(5)]
    return out


def build_hard_negatives(observed: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    positives = observed[
        (observed["relevance_label"] == "usable")
        & (observed["evidence_norm"].str.strip() != "")
    ].copy()
    evidence_pool = observed[
        observed["evidence_text"].str.strip().str.len() >= args.min_evidence_chars
    ].reset_index(drop=True)
    if (
        args.hard_negatives_per_positive <= 0
        or positives.empty
        or len(evidence_pool) < 2
    ):
        return observed.iloc[0:0].copy()

    max_positives = max(1, args.max_hard_negatives // args.hard_negatives_per_positive)
    positives = positives.sample(min(len(positives), max_positives), random_state=args.seed)
    rng = np.random.default_rng(args.seed)
    rows = []
    for _, row in positives.iterrows():
        candidates = evidence_pool[evidence_pool["claim_id"] != row["claim_id"]]
        opposite = candidates[candidates["verification_label"] != row["verification_label"]]
        if not opposite.empty:
            candidates = opposite
        if candidates.empty:
            continue
        sample_n = min(args.hard_negatives_per_positive, len(candidates))
        sampled_idx = rng.choice(candidates.index.to_numpy(), size=sample_n, replace=False)
        for _, neg in candidates.loc[sampled_idx].iterrows():
            new_row = row.copy()
            new_row["evidence_text"] = neg["evidence_text"]
            new_row["evidence_norm"] = neg["evidence_norm"]
            new_row["evidence_claim_id"] = neg["claim_id"]
            new_row["verification_label"] = "Not Enough Evidence"
            new_row["relevance_label"] = "not_usable"
            new_row["relevance_reason"] = "hard_negative_other_claim_evidence"
            new_row["retrieval_quality"] = "Hard Negative"
            new_row["gate_text"] = new_row["claim_norm"] + SEP_TOKEN + new_row["evidence_norm"]
            new_row["row_origin"] = "hard_negative"
            new_row["pair_id"] = f"hard_negative_{len(rows)}"
            rows.append(new_row)
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    df = read_input(args.input, args.sheet)
    observed = prepare_observed_pairs(df, args)
    hard_negatives = build_hard_negatives(observed, args)
    relevance = pd.concat([observed, hard_negatives], ignore_index=True)
    relevance = relevance.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    stance = observed[
        (observed["relevance_label"] == "usable")
        & (observed["verification_label"].isin(STANCE_LABELS))
    ].copy()

    write_csv(relevance, args.output_relevance)
    write_csv(stance, args.output_stance)

    print(f"Saved relevance training -> {args.output_relevance}")
    print(relevance["relevance_label"].value_counts(dropna=False).to_string())
    print("\nRelevance row origin:")
    print(relevance["row_origin"].value_counts(dropna=False).to_string())
    print(f"\nSaved stance training -> {args.output_stance}")
    print(stance["verification_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
