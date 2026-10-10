"""Export false positives/negatives from a FAAV prediction CSV."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR
from utils.faav_common import likely_failure_reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CSV of model errors for report analysis.")
    parser.add_argument("--predictions", required=True, help="Prediction CSV from an experiment/model run.")
    parser.add_argument("--true-col", default="label")
    parser.add_argument("--pred-col", default="prediction")
    parser.add_argument("--claim-col", default="claim_text")
    parser.add_argument("--confidence-col", default="confidence")
    parser.add_argument("--evidence-col", default="effective_evidence")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions, low_memory=False)
    for col in [args.true_col, args.pred_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    errors = df[df[args.true_col] != df[args.pred_col]].copy()
    errors["error_type"] = np.where(
        (errors[args.true_col] == "Real") & (errors[args.pred_col] == "Fake"),
        "false_positive",
        np.where(
            (errors[args.true_col] == "Fake") & (errors[args.pred_col] == "Real"),
            "false_negative",
            np.where(
                (errors[args.true_col] == "Supported") & (errors[args.pred_col] == "Refuted"),
                "false_refutation",
                np.where(
                    (errors[args.true_col] == "Refuted") & (errors[args.pred_col] == "Supported"),
                    "false_support",
                    "other_error",
                ),
            ),
        ),
    )
    errors["likely_failure"] = errors.apply(likely_failure_reason, axis=1)

    out = pd.DataFrame(
        {
            "error_type": errors["error_type"],
            "claim_text": errors.get(args.claim_col, ""),
            "true_label": errors[args.true_col],
            "predicted_label": errors[args.pred_col],
            "prediction_confidence": errors.get(args.confidence_col, ""),
            "retrieved_evidence": errors.get(args.evidence_col, errors.get("retrieved_evidence", "")),
            "source_dataset": errors.get("source_dataset", ""),
            "topic_domain": errors.get("domain", ""),
            "retrieval_score": errors.get("retrieval_score", ""),
            "source_name": errors.get("source_name", errors.get("retrieved_source_name", "")),
            "source_credibility": errors.get("source_credibility", ""),
            "claim_date": errors.get("claim_date", ""),
            "evidence_date": errors.get("evidence_date", ""),
            "recency_gap": errors.get("recency_gap", ""),
            "likely_failure": errors["likely_failure"],
        }
    )
    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(args.predictions))[0]
        out_path = os.path.join(RESULTS_DIR, f"{base}_error_analysis.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} error rows -> {out_path}")


if __name__ == "__main__":
    main()
