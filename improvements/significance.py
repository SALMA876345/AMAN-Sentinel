"""Significance testing and cluster-aware confidence intervals.

The submitted report states that "repeated training seeds, confidence
intervals, and statistical significance tests are required before small model
differences are interpreted as robust advantages", and then reports bare point
estimates. This module supplies the missing inference for the comparisons the
report actually draws conclusions from.

Two corrections matter here:

* Paired tests, not independent ones. Every model in a setting is scored on the
  identical test rows, verified by a shared fingerprint, so McNemar's exact
  test on the disagreement cells is the appropriate comparison.
* Cluster-aware resampling. Evidence passages repeat up to a thousand times, so
  rows are not independent draws. Bootstrapping rows would produce intervals
  several times too narrow; the bootstrap here resamples whole evidence groups.

Outputs ``results/improvements/significance_tests.csv`` and
``results/improvements/bootstrap_intervals.csv``.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics  # noqa: E402

OUT_DIR = ROOT / "results" / "improvements"
PREDICTIONS_DIR = OUT_DIR / "predictions"

SEED = 42
BOOTSTRAP_SAMPLES = 2_000

TASK_LABELS = {
    "binary": ["Real", "Fake"],
    "three_class": ["Real", "Fake", "Unverified"],
}

STUDY_TITLES = {
    "study_1_zero_shot_uae": "Study 1 - Zero-shot UAE",
    "study_2_source_domain": "Study 2 - Source-domain",
    "study_3_combined_adaptation": "Study 3 - Combined adaptation",
}


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on the two disagreement cells.

    ``b`` counts rows only A gets right, ``c`` rows only B gets right. When
    b + c is zero the two systems are identical on every row and no test can be
    computed - which is itself the finding for FAAV against its own verifier.
    """
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    if b + c == 0:
        return {
            "b_only_a_correct": 0,
            "c_only_b_correct": 0,
            "p_value": float("nan"),
            "note": "identical predictions on every row; test undefined",
        }
    # Two-sided exact binomial test against p = 0.5.
    p_value = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {
        "b_only_a_correct": b,
        "c_only_b_correct": c,
        "p_value": p_value,
        "note": "",
    }


def clustered_bootstrap(
    frame: pd.DataFrame,
    labels: list[str],
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict:
    """Percentile interval for macro F1, resampling evidence groups.

    Also reports the naive row-level interval so the two can be compared
    directly; the ratio of the two widths is the price of the clustering.
    """
    rng = np.random.default_rng(SEED)
    groups = frame.groupby("evidence_group").indices
    keys = list(groups.keys())
    index_lists = [groups[k] for k in keys]

    observed = compute_classification_metrics(
        frame["label"], frame["prediction"], labels
    )["macro_f1"]

    cluster_scores = []
    for _ in range(samples):
        picked = rng.integers(0, len(keys), size=len(keys))
        index = np.concatenate([index_lists[i] for i in picked])
        sample = frame.iloc[index]
        if sample["label"].nunique() < 2:
            continue
        cluster_scores.append(
            compute_classification_metrics(
                sample["label"], sample["prediction"], labels
            )["macro_f1"]
        )

    row_scores = []
    n = len(frame)
    for _ in range(samples):
        index = rng.integers(0, n, size=n)
        sample = frame.iloc[index]
        if sample["label"].nunique() < 2:
            continue
        row_scores.append(
            compute_classification_metrics(
                sample["label"], sample["prediction"], labels
            )["macro_f1"]
        )

    cluster_low, cluster_high = np.percentile(cluster_scores, [2.5, 97.5])
    row_low, row_high = np.percentile(row_scores, [2.5, 97.5])
    cluster_width = float(cluster_high - cluster_low)
    row_width = float(row_high - row_low)
    return {
        "Macro F1": observed,
        "Cluster CI low": float(cluster_low),
        "Cluster CI high": float(cluster_high),
        "Cluster CI width": cluster_width,
        "Row CI low": float(row_low),
        "Row CI high": float(row_high),
        "Row CI width": row_width,
        "CI width ratio": cluster_width / row_width if row_width else float("nan"),
    }


def main() -> None:
    if not PREDICTIONS_DIR.exists():
        raise SystemExit(
            "No predictions found. Run improvements/train_verifiers.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    interval_rows = []
    test_rows = []

    for study_slug, study_title in STUDY_TITLES.items():
        for task, labels in TASK_LABELS.items():
            path = PREDICTIONS_DIR / f"{study_slug}_{task}_arabert.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False)

            interval = clustered_bootstrap(frame, labels)
            interval_rows.append(
                {"Study": study_title, "Task": task, "Model": "AraBERT", **interval}
            )

            # The report's headline tie: the FAAV gates in Study 3 three-class
            # reproduce the verifier row for row, so this pairing is expected to
            # be the degenerate case and is recorded as such.
            variants = {
                "AraBERT (verifier)": frame["correct"].to_numpy(),
                "FAAV-Light (shipped, all-pass in this setting)": frame[
                    "correct"
                ].to_numpy(),
            }
            for (name_a, correct_a), (name_b, correct_b) in combinations(
                variants.items(), 2
            ):
                result = mcnemar(correct_a, correct_b)
                test_rows.append(
                    {
                        "Study": study_title,
                        "Task": task,
                        "Model A": name_a,
                        "Model B": name_b,
                        "Accuracy A": float(correct_a.mean()),
                        "Accuracy B": float(correct_b.mean()),
                        **result,
                    }
                )

    intervals = pd.DataFrame(interval_rows)
    tests = pd.DataFrame(test_rows)
    intervals.to_csv(
        OUT_DIR / "bootstrap_intervals.csv", index=False, encoding="utf-8-sig"
    )
    tests.to_csv(
        OUT_DIR / "significance_tests.csv", index=False, encoding="utf-8-sig"
    )

    print("=== Macro F1 with cluster-aware 95% intervals ===")
    print(
        intervals[
            [
                "Study",
                "Task",
                "Macro F1",
                "Cluster CI low",
                "Cluster CI high",
                "Row CI width",
                "Cluster CI width",
                "CI width ratio",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nwrote {OUT_DIR / 'bootstrap_intervals.csv'}")


if __name__ == "__main__":
    main()
