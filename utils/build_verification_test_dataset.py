"""Build an evidence-grounded UAE verification test set.

The original UAE files keep claim-truth labels in ``gold_label``:
Real / Fake / Unverified.  FAAV, however, is an evidence-based verifier, so
the evaluation target should answer a different question:

    Does the available evidence support, refute, or fail to verify the claim?

This script preserves the original ``gold_label`` and adds:
  - ``claim_truth_label``: immutable copy of the original Real/Fake/Unverified
  - ``verification_label``: Supported / Refuted / Not Enough Evidence
  - ``verification_decision_reason``: auditable mapping reason

By default the mapping is conservative: only rows marked Relevant are converted
to Supported/Refuted; partial/irrelevant/missing evidence becomes NEI. The
original ``gold_label`` is left unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEST_2CLASS_FILE, TEST_FILE, TEST_SHEET
from utils.faav_common import lexical_overlap

VERIFICATION_LABELS = {"Supported", "Refuted", "Not Enough Evidence"}
TRUTH_TO_VERIFICATION = {
    "Real": "Supported",
    "Fake": "Refuted",
    "Unverified": "Not Enough Evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create evidence-grounded verification labels for UAE-MVD.")
    parser.add_argument("--input", default=TEST_FILE, help="Input UAE XLSX/CSV with gold_label and evidence_text.")
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_DIR, "uae_test_verification.xlsx"),
        help="Output XLSX/CSV path.",
    )
    parser.add_argument("--sheet", default=TEST_SHEET)
    parser.add_argument(
        "--quality-policy",
        choices=["relevant_only", "relevant_or_partial", "any_non_irrelevant"],
        default="relevant_only",
        help="How strict evidence quality must be before mapping Real/Fake to Supported/Refuted.",
    )
    parser.add_argument("--min-evidence-chars", type=int, default=40)
    parser.add_argument(
        "--min-lexical-overlap",
        type=float,
        default=0.0,
        help="Optional extra guard; 0 disables overlap filtering.",
    )
    parser.add_argument(
        "--legacy-gold-label",
        action="store_true",
        help="Also replace gold_label with verification_label for older scripts that cannot choose label columns.",
    )
    return parser.parse_args()


def read_input(path: str, sheet: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path, low_memory=False)


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


def assign_verification_label(row: pd.Series, args: argparse.Namespace) -> tuple[str, str, float]:
    truth = "" if pd.isna(row.get("gold_label")) else str(row.get("gold_label")).strip()
    evidence = "" if pd.isna(row.get("evidence_text")) else str(row.get("evidence_text")).strip()
    claim = "" if pd.isna(row.get("claim_ar")) else str(row.get("claim_ar")).strip()
    overlap = lexical_overlap(claim, evidence)

    if truth not in TRUTH_TO_VERIFICATION:
        return "Not Enough Evidence", "unknown_original_label", overlap
    if truth == "Unverified":
        return "Not Enough Evidence", "original_unverified", overlap
    if len(evidence) < args.min_evidence_chars:
        return "Not Enough Evidence", "evidence_too_short", overlap
    if not evidence_quality_ok(row.get("retrieval_quality", ""), args.quality_policy):
        return "Not Enough Evidence", f"retrieval_quality_not_sufficient:{row.get('retrieval_quality', '')}", overlap
    if args.min_lexical_overlap > 0 and overlap < args.min_lexical_overlap:
        return "Not Enough Evidence", f"low_lexical_overlap:{overlap:.4f}", overlap
    return TRUTH_TO_VERIFICATION[truth], f"mapped_from_truth_with_{args.quality_policy}", overlap


def write_output(df: pd.DataFrame, path: str, sheet: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith((".xlsx", ".xls")):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    df = read_input(args.input, args.sheet)
    if "gold_label" not in df.columns:
        raise ValueError("Input must contain gold_label.")
    if "claim_ar" not in df.columns:
        raise ValueError("Input must contain claim_ar.")
    if "evidence_text" not in df.columns:
        df["evidence_text"] = ""
    if "retrieval_quality" not in df.columns:
        df["retrieval_quality"] = ""

    df = df.copy()
    df["claim_truth_label"] = df["gold_label"]
    assigned = df.apply(lambda row: assign_verification_label(row, args), axis=1)
    df["verification_label"] = [x[0] for x in assigned]
    df["verification_decision_reason"] = [x[1] for x in assigned]
    df["claim_evidence_overlap"] = [x[2] for x in assigned]

    if args.legacy_gold_label:
        # Compatibility mode only. Prefer leaving gold_label as claim truth and
        # passing --label-col verification_label to experiment scripts.
        df["gold_label"] = df["verification_label"]

    write_output(df, args.output, args.sheet)

    print(f"Saved -> {args.output}")
    print("Original claim-truth distribution:")
    print(df["claim_truth_label"].value_counts(dropna=False).to_string())
    print("\nEvidence-grounded verification distribution:")
    print(df["verification_label"].value_counts(dropna=False).to_string())
    print("\nMapping reasons:")
    print(df["verification_decision_reason"].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()
