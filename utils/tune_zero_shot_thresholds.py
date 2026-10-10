"""Tune zero-shot gate settings on non-UAE grouped validation data only."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR


RELEVANCE_FILE = os.path.join(DATA_DIR, "relevance_training_zeroshot.csv")
STANCE_FILE = os.path.join(DATA_DIR, "stance_training_zeroshot.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "zero_shot_thresholds.json")


def text_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def grouped_split(df: pd.DataFrame, seed: int = 42):
    fallback_groups = pd.Series(
        df.index.astype(str),
        index=df.index,
    )
    groups = df["claim_id"].where(df["claim_id"].notna(), fallback_groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(df, groups=groups)
    )
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def tune_gate() -> dict:
    df = pd.read_csv(RELEVANCE_FILE, low_memory=False)
    df = df[df["relevance_label"].isin(["usable", "not_usable"])].copy()
    train, val = grouped_split(df)
    model = text_model()
    model.fit(train["gate_text"], train["relevance_label"])
    proba = model.predict_proba(val["gate_text"])
    usable_idx = model.classes_.tolist().index("usable")
    usable_proba = proba[:, usable_idx]

    rows = []
    best = None
    for threshold in np.linspace(0.10, 0.90, 81):
        pred = np.where(usable_proba >= threshold, "usable", "not_usable")
        metrics = classification_metrics(val["relevance_label"], pred)
        row = {"threshold": float(round(threshold, 2)), **metrics}
        rows.append(row)
        if best is None or (metrics["macro_f1"], metrics["mcc"]) > (
            best["macro_f1"],
            best["mcc"],
        ):
            best = row
    return {
        "training_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "training_claim_groups": int(train["claim_id"].nunique()),
        "validation_claim_groups": int(val["claim_id"].nunique()),
        "best": best,
        "sweep": rows,
    }


def validate_stance() -> dict:
    df = pd.read_csv(STANCE_FILE, low_memory=False)
    df = df[df["verification_label"].isin(["Supported", "Refuted"])].copy()
    train, val = grouped_split(df)
    model = text_model()
    model.fit(train["gate_text"], train["verification_label"])
    pred = model.predict(val["gate_text"])
    return {
        "training_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "training_claim_groups": int(train["claim_id"].nunique()),
        "validation_claim_groups": int(val["claim_id"].nunique()),
        **classification_metrics(val["verification_label"], pred),
        "validation_label_distribution": val[
            "verification_label"
        ].value_counts().to_dict(),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "protocol": "non_UAE_grouped_validation_only",
        "uae_rows_used": 0,
        "gate": tune_gate(),
        "stance": validate_stance(),
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
