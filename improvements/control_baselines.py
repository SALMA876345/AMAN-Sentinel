"""Provenance and majority control baselines for every final study setting.

The submitted report compares each model against a majority-class floor only.
That floor is weak: because Unverified is drawn almost entirely from UAE
official and news feeds while Real/Fake come from VERA-ARAB, ArCOV19 and
AraStance, a model could score well by recognising publisher style instead of
verifying claims against evidence. This module quantifies exactly how much of
the label is recoverable from provenance alone, so the reported gains can be
stated net of that confound.

Three controls, none of which read the claim or the evidence text:

* Majority          - always predict the most frequent training label.
* Dataset-family    - predict the majority label of the row's corpus (source/UAE).
* Source-lookup     - predict the majority label of the row's publisher.

Outputs ``results/improvements/control_baselines.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics  # noqa: E402

DATA_DIR = ROOT / "data" / "final_studies"
OUT_DIR = ROOT / "results" / "improvements"

TASKS = {
    "binary": ("binary_label", ["Real", "Fake"]),
    "three_class": ("three_class_label", ["Real", "Fake", "Unverified"]),
}

# Each study pairs the training frame that defines the lookup with its test
# frame, mirroring the protocol in experiments/final_studies_neural.py.
STUDIES = {
    "Study 1 - Zero-shot UAE": ("source_{task}_train", "uae_{task}_full"),
    "Study 2 - Source-domain": ("source_{task}_train", "source_{task}_test"),
    "Study 3 - Combined adaptation": (
        "combined_{task}_train",
        "combined_{task}_test",
    ),
}

# Best reported macro F1 per setting, taken from the submitted
# results/final_studies/final_model_comparison.md, for a like-for-like column.
REPORTED_BEST = {
    ("Study 1 - Zero-shot UAE", "binary"): ("AraBERT / FAAV", 0.5637),
    ("Study 1 - Zero-shot UAE", "three_class"): ("BiLSTM", 0.1722),
    ("Study 2 - Source-domain", "binary"): ("AraBERT / FAAV", 0.9016),
    ("Study 2 - Source-domain", "three_class"): ("MARBERT", 0.7610),
    ("Study 3 - Combined adaptation", "binary"): ("MARBERT", 0.8669),
    ("Study 3 - Combined adaptation", "three_class"): ("AraBERT / FAAV", 0.8763),
}


def load(name: str, task: str) -> pd.DataFrame:
    label_col, labels = TASKS[task]
    frame = pd.read_csv(DATA_DIR / f"{name}.csv", low_memory=False).fillna("")
    frame = frame[frame[label_col].isin(labels)].copy()
    frame["label"] = frame[label_col]
    return frame.reset_index(drop=True)


def lookup_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    key: str,
) -> pd.Series:
    """Majority training label per ``key`` value, backing off to the global mode.

    The back-off matters for Study 1, where UAE publishers are unseen at
    training time; without it the control would be undefined rather than weak.
    """
    majority = train.groupby(key)["label"].agg(lambda s: s.value_counts().idxmax())
    fallback = train["label"].value_counts().idxmax()
    return test[key].map(majority).fillna(fallback)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, (_, labels) in TASKS.items():
        for study, (train_name, test_name) in STUDIES.items():
            train = load(train_name.format(task=task), task)
            test = load(test_name.format(task=task), task)
            fallback = train["label"].value_counts().idxmax()

            controls = {
                "Majority class": pd.Series([fallback] * len(test)),
                "Dataset-family lookup": lookup_predictions(
                    train, test, "dataset_family"
                ),
                "Source lookup (provenance)": lookup_predictions(
                    train, test, "source"
                ),
            }
            best_name, best_f1 = REPORTED_BEST[(study, task)]
            for control, predicted in controls.items():
                metrics = compute_classification_metrics(
                    test["label"], predicted, labels
                )
                rows.append(
                    {
                        "Study": study,
                        "Task": task,
                        "Control": control,
                        "Test N": len(test),
                        "Accuracy": metrics["accuracy"],
                        "Macro F1": metrics["macro_f1"],
                        "MCC": metrics["mcc"],
                        "Reported best model": best_name,
                        "Reported best macro F1": best_f1,
                        "Headroom over control": best_f1 - metrics["macro_f1"],
                    }
                )

    frame = pd.DataFrame(rows)
    path = OUT_DIR / "control_baselines.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")

    show = frame[
        [
            "Study",
            "Task",
            "Control",
            "Macro F1",
            "Reported best macro F1",
            "Headroom over control",
        ]
    ]
    print(show.round(4).to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
