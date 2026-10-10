"""Effective sample size and evidence-reuse structure of the final splits.

Grouped splitting keeps a repeated evidence passage inside a single partition,
so there is no leakage across train/validation/test. It does not make the test
rows independent: one passage is reused up to four figures of times, so rows
sharing it are near-duplicates and the usual sqrt(N) intuition overstates the
precision of every reported score.

This module reports, per split, the number of rows, the number of distinct
evidence and claim groups, the reuse distribution, and Kish's effective sample
size, which is the standard correction for cluster-structured samples:

    N_eff = (sum_g n_g)^2 / sum_g n_g^2

Outputs ``results/improvements/evidence_structure.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final_studies"
OUT_DIR = ROOT / "results" / "improvements"

TASKS = {
    "binary": ("binary_label", ["Real", "Fake"]),
    "three_class": ("three_class_label", ["Real", "Fake", "Unverified"]),
}

SPLITS = [
    "source_{task}_train",
    "source_{task}_validation",
    "source_{task}_test",
    "combined_{task}_train",
    "combined_{task}_validation",
    "combined_{task}_test",
    "uae_{task}_full",
]


def effective_n(sizes: np.ndarray) -> float:
    """Kish effective sample size for a set of cluster sizes."""
    total = sizes.sum()
    return float(total * total / np.square(sizes).sum()) if total else 0.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, (label_col, labels) in TASKS.items():
        for template in SPLITS:
            name = template.format(task=task)
            path = DATA_DIR / f"{name}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False).fillna("")
            frame = frame[frame[label_col].isin(labels)]
            if frame.empty:
                continue
            evidence_sizes = frame.groupby("evidence_group").size().to_numpy()
            claim_sizes = frame.groupby("claim_group").size().to_numpy()
            n_eff = effective_n(evidence_sizes)
            rows.append(
                {
                    "Split": name,
                    "Task": task,
                    "Rows": len(frame),
                    "Distinct claim groups": frame["claim_group"].nunique(),
                    "Distinct evidence groups": frame["evidence_group"].nunique(),
                    "Max evidence reuse": int(evidence_sizes.max()),
                    "Mean evidence reuse": float(evidence_sizes.mean()),
                    "Median evidence reuse": float(np.median(evidence_sizes)),
                    "Max claim reuse": int(claim_sizes.max()),
                    "Effective N (evidence clusters)": n_eff,
                    "Effective N ratio": n_eff / len(frame),
                    # A 95% interval scales with 1/sqrt(N); this is how much
                    # wider it becomes once clustering is accounted for.
                    "CI inflation factor": float(np.sqrt(len(frame) / n_eff)),
                }
            )

    frame = pd.DataFrame(rows)
    path = OUT_DIR / "evidence_structure.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")

    show = frame[
        [
            "Split",
            "Rows",
            "Distinct evidence groups",
            "Max evidence reuse",
            "Effective N (evidence clusters)",
            "Effective N ratio",
            "CI inflation factor",
        ]
    ]
    print(show.round(3).to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
