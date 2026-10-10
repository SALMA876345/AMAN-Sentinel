"""Selective-prediction evaluation of the FAAV gates against proper baselines.

The submitted protocol scores an abstaining system against forced-choice models
on full-test macro F1. That comparison is decided before it starts: any gate
that abstains loses coverage without being credited for the errors it removed,
so the validation sweep always drifts towards the all-accept corner. Study 3's
three-class gate lands exactly there, which is why FAAV-Light, FAAV-Full and
AraBERT report identical scores to sixteen decimal places.

Selective prediction has its own standard evaluation, used here:

* risk-coverage curves - error rate over the accepted fraction, swept over all
  thresholds;
* AURC - area under that curve, lower is better, and E-AURC, its excess over
  the oracle ranking, which removes the base-error-rate term so settings with
  different difficulty stay comparable;
* selective accuracy and macro F1 at matched coverage, the only way to compare
  two gates without one of them buying its score with abstention.

Six selectors are compared. The first three are confidence baselines the
submitted report never includes, and they are the ones a reviewer asks for
first, since a learned gate has to beat plain softmax confidence to justify
itself:

* MaxProb        - threshold the top softmax probability;
* Margin         - threshold the gap between the top two probabilities;
* NegEntropy     - threshold the negated predictive entropy;
* FAAV-Light     - the shipped rule gate (confidence, lexical overlap, length);
* FAAV-Full      - the shipped learned TF-IDF gate over claim + evidence;
* FAAV-Rich      - a corrected learned gate, described below.

FAAV-Rich changes two things about FAAV-Full. It is fitted out-of-fold on the
held-out validation split rather than on the verifier's own training rows, so
its target is a real error signal instead of a near-constant 1; and its
features are ones the verifier cannot already see - confidence geometry,
retrieval and evidence structure, provenance - instead of a TF-IDF view of the
same claim and evidence text the verifier was given.

Outputs land in ``results/improvements``; nothing under ``results/final_studies``
is read for writing or modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import compute_classification_metrics, lexical_overlap  # noqa: E402

OUT_DIR = ROOT / "results" / "improvements"
PREDICTIONS_DIR = OUT_DIR / "predictions"

SEED = 42
COVERAGE_TARGETS = [0.70, 0.80, 0.90]

TASK_LABELS = {
    "binary": ["Real", "Fake"],
    "three_class": ["Real", "Fake", "Unverified"],
}

# Study -> which trained family produced it, matching train_verifiers.py.
STUDY_FAMILY = {
    "study_1_zero_shot_uae": "source",
    "study_2_source_domain": "source",
    "study_3_combined_adaptation": "combined",
}

STUDY_TITLES = {
    "study_1_zero_shot_uae": "Study 1 - Zero-shot UAE",
    "study_2_source_domain": "Study 2 - Source-domain",
    "study_3_combined_adaptation": "Study 3 - Combined adaptation",
}


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------
def add_features(frame: pd.DataFrame, reuse: dict[str, int]) -> pd.DataFrame:
    frame = frame.copy()
    frame["claim_norm"] = frame["claim_norm"].fillna("").astype(str)
    frame["evidence_norm"] = frame["evidence_norm"].fillna("").astype(str)
    frame["evidence_chars"] = frame["evidence_norm"].str.len()
    frame["claim_chars"] = frame["claim_norm"].str.len()
    frame["lexical_overlap"] = [
        lexical_overlap(claim, evidence)
        for claim, evidence in zip(frame["claim_norm"], frame["evidence_norm"])
    ]
    # How often this evidence passage is reused anywhere in the corpus. A
    # passage that answers a thousand different claims is generic, and generic
    # evidence is exactly where a grounded verdict should not be trusted.
    frame["evidence_reuse"] = (
        frame["evidence_group"].map(reuse).fillna(1).astype(float)
    )
    frame["log_evidence_reuse"] = np.log1p(frame["evidence_reuse"])
    frame["has_url"] = (
        frame.get("evidence_url", pd.Series([""] * len(frame))).fillna("").astype(str).str.len() > 0
    ).astype(int)
    return frame


RICH_NUMERIC = [
    "confidence",
    "margin",
    "entropy",
    "lexical_overlap",
    "evidence_chars",
    "claim_chars",
    "log_evidence_reuse",
    "has_url",
]


def rich_matrix(
    frame: pd.DataFrame,
    source_index: dict[str, int],
    n_sources: int,
    use_provenance: bool,
) -> np.ndarray:
    numeric = frame[RICH_NUMERIC].to_numpy(dtype=float)
    if not use_provenance:
        return numeric
    codes = frame["source"].map(source_index).fillna(n_sources).astype(int)
    onehot = np.zeros((len(frame), n_sources + 1), dtype=float)
    onehot[np.arange(len(frame)), codes.to_numpy()] = 1.0
    return np.hstack([numeric, onehot])


# --------------------------------------------------------------------------
# selective-prediction machinery
# --------------------------------------------------------------------------
def risk_coverage(scores: np.ndarray, correct: np.ndarray) -> dict:
    """Risk-coverage curve, AURC and E-AURC for a confidence score.

    Rows are accepted in descending score order, so coverage k/n corresponds to
    keeping the k rows the selector trusts most.
    """
    order = np.argsort(-scores, kind="mergesort")
    ordered_correct = correct[order].astype(float)
    n = len(ordered_correct)
    ranks = np.arange(1, n + 1)
    # Risk at coverage k is the error rate among the k most-trusted rows.
    risk = 1.0 - np.cumsum(ordered_correct) / ranks
    coverage = ranks / n
    aurc = float(risk.mean())

    # Oracle: every correct row ranked above every incorrect one. Subtracting
    # its AURC leaves only the part attributable to the selector's ranking.
    oracle_correct = np.sort(correct.astype(float))[::-1]
    oracle_risk = 1.0 - np.cumsum(oracle_correct) / ranks
    excess = aurc - float(oracle_risk.mean())
    return {
        "coverage": coverage,
        "risk": risk,
        "aurc": aurc,
        "eaurc": excess,
    }


def metrics_at_coverage(
    scores: np.ndarray,
    frame: pd.DataFrame,
    labels: list[str],
    target: float,
) -> dict:
    """Accept the highest-scoring ``target`` fraction and score what remains."""
    n = len(scores)
    keep = max(int(round(target * n)), 1)
    order = np.argsort(-scores, kind="mergesort")[:keep]
    accepted = frame.iloc[order]
    metrics = compute_classification_metrics(
        accepted["label"], accepted["prediction"], labels
    )
    return {
        "coverage": keep / n,
        "selective_accuracy": metrics["accuracy"],
        "selective_macro_f1": metrics["macro_f1"],
        "selective_mcc": metrics["mcc"],
        "accepted_rows": int(keep),
    }


def faav_light_score(frame: pd.DataFrame) -> np.ndarray:
    """Continuous stand-in for the shipped three-threshold rule gate.

    The shipped gate is a hard AND of three thresholds, which yields a single
    operating point rather than a curve. Ranking by the minimum of the three
    min-max normalised signals reproduces its accept ordering while sweeping
    every coverage level, so it can be placed on the same axes as the others.
    """
    def normalise(values: np.ndarray) -> np.ndarray:
        values = values.astype(float)
        low, high = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.ones_like(values)
        return (values - low) / (high - low)

    return np.minimum.reduce(
        [
            normalise(frame["confidence"].to_numpy()),
            normalise(frame["lexical_overlap"].to_numpy()),
            normalise(np.minimum(frame["evidence_chars"].to_numpy(), 200)),
        ]
    )


def build_tfidf_gate(validation: pd.DataFrame) -> tuple[object, np.ndarray]:
    """The shipped FAAV-Full gate, but fitted out-of-fold on held-out rows.

    Refitting on validation instead of the verifier's own training rows is the
    correction under test; keeping the TF-IDF feature space identical isolates
    that one change from the feature change made by FAAV-Rich.
    """
    text = (
        "verdict_"
        + validation["prediction"].astype(str)
        + " [SEP] "
        + validation["claim_norm"].astype(str)
        + " [SEP] "
        + validation["evidence_norm"].astype(str)
    )
    target = validation["correct"].to_numpy()
    scores = np.zeros(len(validation), dtype=float)
    if len(np.unique(target)) < 2:
        return None, scores

    def make_vectorizer() -> FeatureUnion:
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_index, test_index in folds.split(text, target):
        vectorizer = make_vectorizer()
        matrix = vectorizer.fit_transform(text.iloc[train_index])
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
        )
        model.fit(matrix, target[train_index])
        scores[test_index] = model.predict_proba(
            vectorizer.transform(text.iloc[test_index])
        )[:, 1]

    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform(text)
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2_000,
        random_state=SEED,
    )
    model.fit(matrix, target)
    return (vectorizer, model), scores


def apply_tfidf_gate(fitted, frame: pd.DataFrame) -> np.ndarray:
    if fitted is None:
        return np.ones(len(frame), dtype=float)
    vectorizer, model = fitted
    text = (
        "verdict_"
        + frame["prediction"].astype(str)
        + " [SEP] "
        + frame["claim_norm"].astype(str)
        + " [SEP] "
        + frame["evidence_norm"].astype(str)
    )
    return model.predict_proba(vectorizer.transform(text))[:, 1]


def build_rich_gate(
    validation: pd.DataFrame,
    source_index: dict[str, int],
    n_sources: int,
    use_provenance: bool,
    learner: str = "trees",
):
    """Fit the corrected gate on held-out validation rows.

    Two learners are offered because a single one cannot separate "these
    features carry no extra signal" from "this model class handles them badly".
    Trees discretise a smooth confidence score, which can lose to raw
    confidence for reasons that have nothing to do with the features; a
    standardised linear model keeps confidence monotone and is the control for
    that.
    """
    target = validation["correct"].to_numpy()
    if len(np.unique(target)) < 2:
        return None
    matrix = rich_matrix(validation, source_index, n_sources, use_provenance)
    if learner == "linear":
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale == 0] = 1.0
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
        )
        model.fit((matrix - mean) / scale, target)
        return ("linear", model, mean, scale)
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED,
    )
    model.fit(matrix, target)
    return ("trees", model)


def apply_rich_gate(
    fitted,
    frame: pd.DataFrame,
    source_index: dict[str, int],
    n_sources: int,
    use_provenance: bool,
) -> np.ndarray:
    if fitted is None:
        return np.ones(len(frame), dtype=float)
    matrix = rich_matrix(frame, source_index, n_sources, use_provenance)
    if fitted[0] == "linear":
        _, model, mean, scale = fitted
        matrix = (matrix - mean) / scale
    else:
        model = fitted[1]
    return model.predict_proba(matrix)[:, 1]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def evidence_reuse_counts() -> dict[str, int]:
    """Reuse counts pooled over every manifest, so the feature is split-agnostic."""
    counts: dict[str, int] = {}
    for path in (ROOT / "data" / "final_studies").glob("*.csv"):
        frame = pd.read_csv(path, low_memory=False, usecols=["evidence_group"])
        for group, size in frame["evidence_group"].value_counts().items():
            counts[group] = max(counts.get(group, 0), int(size))
    return counts


def main() -> None:
    if not PREDICTIONS_DIR.exists():
        raise SystemExit(
            "No predictions found. Run improvements/train_verifiers.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reuse = evidence_reuse_counts()

    summary_rows = []
    coverage_rows = []
    curve_rows = []

    for study_slug, family in STUDY_FAMILY.items():
        for task, labels in TASK_LABELS.items():
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
            rich_model = build_rich_gate(
                validation, source_index, n_sources, use_provenance=True
            )
            rich_model_no_source = build_rich_gate(
                validation, source_index, n_sources, use_provenance=False
            )
            rich_model_linear = build_rich_gate(
                validation,
                source_index,
                n_sources,
                use_provenance=True,
                learner="linear",
            )

            selectors = {
                "MaxProb (confidence baseline)": test["confidence"].to_numpy(float),
                "Margin (confidence baseline)": test["margin"].to_numpy(float),
                "NegEntropy (confidence baseline)": -test["entropy"].to_numpy(float),
                "FAAV-Light (shipped rule gate)": faav_light_score(test),
                "FAAV-Full (shipped TF-IDF gate)": apply_tfidf_gate(
                    tfidf_fitted, test
                ),
                "FAAV-Rich (corrected gate)": apply_rich_gate(
                    rich_model, test, source_index, n_sources, True
                ),
                "FAAV-Rich (no provenance)": apply_rich_gate(
                    rich_model_no_source, test, source_index, n_sources, False
                ),
                "FAAV-Rich (linear)": apply_rich_gate(
                    rich_model_linear, test, source_index, n_sources, True
                ),
            }

            correct = test["correct"].to_numpy()
            full_metrics = compute_classification_metrics(
                test["label"], test["prediction"], labels
            )

            for name, scores in selectors.items():
                curve = risk_coverage(scores, correct)
                summary_rows.append(
                    {
                        "Study": STUDY_TITLES[study_slug],
                        "Task": task,
                        "Selector": name,
                        "Test N": len(test),
                        "Full-coverage accuracy": full_metrics["accuracy"],
                        "Full-coverage macro F1": full_metrics["macro_f1"],
                        "AURC": curve["aurc"],
                        "E-AURC": curve["eaurc"],
                    }
                )
                # Thin the curve for plotting; the metrics above use every point.
                step = max(len(curve["coverage"]) // 400, 1)
                for cov, rk in zip(
                    curve["coverage"][::step], curve["risk"][::step]
                ):
                    curve_rows.append(
                        {
                            "Study": STUDY_TITLES[study_slug],
                            "Task": task,
                            "Selector": name,
                            "Coverage": float(cov),
                            "Risk": float(rk),
                        }
                    )
                for target in COVERAGE_TARGETS:
                    point = metrics_at_coverage(scores, test, labels, target)
                    coverage_rows.append(
                        {
                            "Study": STUDY_TITLES[study_slug],
                            "Task": task,
                            "Selector": name,
                            "Target coverage": target,
                            **point,
                            "Macro F1 gain over full coverage": (
                                point["selective_macro_f1"]
                                - full_metrics["macro_f1"]
                            ),
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    coverage = pd.DataFrame(coverage_rows)
    curves = pd.DataFrame(curve_rows)

    summary.to_csv(OUT_DIR / "selective_aurc.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(
        OUT_DIR / "selective_at_matched_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curves.to_csv(
        OUT_DIR / "risk_coverage_curves.csv", index=False, encoding="utf-8-sig"
    )

    print("=== AURC (lower is better) ===")
    print(
        summary.pivot_table(
            index=["Study", "Task"], columns="Selector", values="AURC"
        )
        .round(4)
        .to_string()
    )
    print("\n=== Selective macro F1 at 80% coverage ===")
    at80 = coverage[coverage["Target coverage"] == 0.80]
    print(
        at80.pivot_table(
            index=["Study", "Task"],
            columns="Selector",
            values="selective_macro_f1",
        )
        .round(4)
        .to_string()
    )
    print(f"\nwrote {OUT_DIR / 'selective_aurc.csv'}")


if __name__ == "__main__":
    main()
