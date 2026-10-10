"""Does an evidence gate rank errors better than softmax confidence?

This is the comparison the framework lives or dies by, and the submitted report
never runs it. A faithfulness gate that cannot outrank the verifier's own
confidence is not buying anything: the verifier already knows which of its
answers are shaky, for free.

For each setting the AURC advantage of a gate over the MaxProb baseline is
reported with a cluster bootstrap interval, resampling evidence groups rather
than rows because repeated passages make rows non-independent. A positive
advantage means the gate ranks its errors better.

Note on binary tasks: with two classes, max-probability, top-two margin, and
negative entropy are monotone transformations of one another, so they induce
identical rankings and identical AURC. Only the three-class settings
distinguish them.

Outputs ``results/improvements/gate_vs_confidence.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from improvements.selective_evaluation import (  # noqa: E402
    STUDY_FAMILY,
    STUDY_TITLES,
    TASK_LABELS,
    add_features,
    apply_rich_gate,
    apply_tfidf_gate,
    build_rich_gate,
    build_tfidf_gate,
    evidence_reuse_counts,
    faav_light_score,
    risk_coverage,
)

OUT_DIR = ROOT / "results" / "improvements"
PREDICTIONS_DIR = OUT_DIR / "predictions"

SEED = 42
BOOTSTRAP = 400


def bootstrap_advantage(
    baseline: np.ndarray,
    gate: np.ndarray,
    correct: np.ndarray,
    groups: dict,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    keys = list(groups)
    index_lists = [groups[k] for k in keys]
    differences = []
    for _ in range(BOOTSTRAP):
        picked = rng.integers(0, len(keys), size=len(keys))
        index = np.concatenate([index_lists[i] for i in picked])
        sample_correct = correct[index]
        if sample_correct.min() == sample_correct.max():
            continue
        differences.append(
            risk_coverage(baseline[index], sample_correct)["aurc"]
            - risk_coverage(gate[index], sample_correct)["aurc"]
        )
    differences = np.asarray(differences)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high), float((differences > 0).mean())


def main() -> None:
    reuse = evidence_reuse_counts()
    rows = []
    for study_slug, family in STUDY_FAMILY.items():
        for task in TASK_LABELS:
            test_path = PREDICTIONS_DIR / f"{study_slug}_{task}_arabert.csv"
            validation_path = (
                PREDICTIONS_DIR / f"validation_{family}_{task}_arabert.csv"
            )
            if not test_path.exists() or not validation_path.exists():
                continue
            test = add_features(pd.read_csv(test_path, low_memory=False), reuse)
            validation = add_features(
                pd.read_csv(validation_path, low_memory=False), reuse
            )
            sources = sorted(validation["source"].dropna().unique())
            source_index = {name: i for i, name in enumerate(sources)}
            n_sources = len(sources)

            tfidf_fitted, _ = build_tfidf_gate(validation)
            rich = build_rich_gate(validation, source_index, n_sources, True)
            rich_linear = build_rich_gate(
                validation, source_index, n_sources, True, learner="linear"
            )

            correct = test["correct"].to_numpy()
            groups = test.groupby("evidence_group").indices
            baseline = test["confidence"].to_numpy(float)
            baseline_aurc = risk_coverage(baseline, correct)["aurc"]

            gates = {
                "FAAV-Light (shipped rule gate)": faav_light_score(test),
                "FAAV-Full (shipped TF-IDF gate)": apply_tfidf_gate(
                    tfidf_fitted, test
                ),
                "FAAV-Rich (corrected, trees)": apply_rich_gate(
                    rich, test, source_index, n_sources, True
                ),
                "FAAV-Rich (corrected, linear)": apply_rich_gate(
                    rich_linear, test, source_index, n_sources, True
                ),
            }
            for name, scores in gates.items():
                gate_aurc = risk_coverage(scores, correct)["aurc"]
                advantage = baseline_aurc - gate_aurc
                low, high, win_rate = bootstrap_advantage(
                    baseline, scores, correct, groups
                )
                rows.append(
                    {
                        "Study": STUDY_TITLES[study_slug],
                        "Task": task,
                        "Gate": name,
                        "Test N": len(test),
                        "Evidence clusters": len(groups),
                        "MaxProb AURC": baseline_aurc,
                        "Gate AURC": gate_aurc,
                        "AURC advantage": advantage,
                        "CI low": low,
                        "CI high": high,
                        "Resamples gate wins": win_rate,
                        "Beats confidence": bool(low > 0),
                    }
                )
                print(
                    f"{STUDY_TITLES[study_slug]:30s} {task:12s}"
                    f" {name:34s} advantage={advantage:+.4f}"
                    f" CI [{low:+.4f},{high:+.4f}]"
                    f" {'BEATS CONFIDENCE' if low > 0 else ''}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    path = OUT_DIR / "gate_vs_confidence.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nwrote {path}")
    wins = frame[frame["Beats confidence"]]
    if wins.empty:
        print("No gate beats softmax confidence in any setting.")
    else:
        print("Gates that beat confidence with the interval excluding zero:")
        print(
            wins[["Study", "Task", "Gate", "AURC advantage", "CI low", "CI high"]]
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
