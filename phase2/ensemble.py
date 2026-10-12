"""Phase 2 Step 9: ensemble experiments on the TEST split.

Works with whichever of {svm, bilstm, transformer} calibrated/raw
prediction files are present under phase2/reports/. Tests only the two
simple, defensible strategies the brief asks for:
  - majority voting (unweighted, ties broken by highest mean confidence)
  - calibrated-confidence-weighted voting (each model's vote weighted by
    its own calibrated confidence in that prediction; falls back to raw
    confidence for a model with no calibration file)

Compares both against the single strongest individual model on the same
rows. If neither ensemble beats the best individual model, that is
reported as the conclusion -- this script does not keep an ensemble
alive just because it exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]  # Code files/
REPORTS_DIR = ROOT / "phase2" / "reports"
LABELS = ["Real", "Fake"]

# Prefer the calibrated file if it exists (SVM, BiLSTM get one from
# phase2/calibration.py; transformer gets one once
# phase2/run_transformer_calibration.py has been run), else fall back to
# the raw prediction file.
MODEL_FILES = {
    "svm": (REPORTS_DIR / "svm_predictions_test_calibrated.csv", REPORTS_DIR / "svm_predictions_test.csv"),
    "bilstm": (REPORTS_DIR / "bilstm_predictions_test_calibrated.csv", REPORTS_DIR / "bilstm_predictions_test.csv"),
    "transformer": (REPORTS_DIR / "transformer_predictions_test_calibrated.csv", REPORTS_DIR / "transformer_predictions_test.csv"),
}


def load_available_models() -> dict[str, pd.DataFrame]:
    available = {}
    for name, (calibrated_path, raw_path) in MODEL_FILES.items():
        path = calibrated_path if calibrated_path.exists() else raw_path
        if path.exists():
            df = pd.read_csv(path, low_memory=False).set_index("sample_id")
            if "calibrated_confidence" not in df.columns:
                df["calibrated_confidence"] = df["confidence"]
            available[name] = df
    return available


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def majority_vote(frame: pd.DataFrame, model_names: list[str]) -> pd.Series:
    def vote(row):
        counts = row[model_names].value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            return top[0]
        # tie: break by mean calibrated confidence among tied models' voters
        conf_cols = [f"{m}_conf" for m in model_names]
        best_label, best_conf = top[0], -1.0
        for label in top:
            voters = [m for m in model_names if row[m] == label]
            mean_conf = sum(row[f"{m}_conf"] for m in voters) / len(voters)
            if mean_conf > best_conf:
                best_label, best_conf = label, mean_conf
        return best_label
    return frame.apply(vote, axis=1)


def confidence_weighted_vote(frame: pd.DataFrame, model_names: list[str]) -> pd.Series:
    def vote(row):
        scores = {label: 0.0 for label in LABELS}
        for m in model_names:
            scores[row[m]] += row[f"{m}_conf"]
        return max(scores, key=scores.get)
    return frame.apply(vote, axis=1)


def main() -> None:
    models = load_available_models()
    names = list(models.keys())
    print(f"Available models for ensembling: {names}")
    if len(names) < 2:
        print("Need at least 2 models; nothing to ensemble yet.")
        return

    common_idx = models[names[0]].index
    for n in names[1:]:
        common_idx = common_idx.intersection(models[n].index)

    frame = pd.DataFrame(index=common_idx)
    frame["true_label"] = models[names[0]].loc[common_idx, "true_label"]
    for n in names:
        frame[n] = models[n].loc[common_idx, "prediction"]
        frame[f"{n}_conf"] = models[n].loc[common_idx, "calibrated_confidence"]

    individual_results = {
        n: metrics(frame["true_label"], frame[n]) for n in names
    }
    best_individual_name = max(individual_results, key=lambda n: individual_results[n]["macro_f1"])
    best_individual = individual_results[best_individual_name]

    frame["majority_vote"] = majority_vote(frame, names)
    frame["weighted_vote"] = confidence_weighted_vote(frame, names)

    majority_result = metrics(frame["true_label"], frame["majority_vote"])
    weighted_result = metrics(frame["true_label"], frame["weighted_vote"])

    print(f"\nIndividual models (n={len(frame)}):")
    for n, r in individual_results.items():
        print(f"  {n:<12} accuracy={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}  mcc={r['mcc']:.4f}")
    print(f"\nBest individual model: {best_individual_name}  macro_f1={best_individual['macro_f1']:.4f}")
    print(f"\nMajority vote          accuracy={majority_result['accuracy']:.4f}  macro_f1={majority_result['macro_f1']:.4f}  mcc={majority_result['mcc']:.4f}")
    print(f"Confidence-weighted vote accuracy={weighted_result['accuracy']:.4f}  macro_f1={weighted_result['macro_f1']:.4f}  mcc={weighted_result['mcc']:.4f}")

    majority_beats_best = majority_result["macro_f1"] > best_individual["macro_f1"]
    weighted_beats_best = weighted_result["macro_f1"] > best_individual["macro_f1"]
    recommendation = None
    if not majority_beats_best and not weighted_beats_best:
        recommendation = f"NEITHER ensemble beats the best individual model ({best_individual_name}). Do not use an ensemble in production."
    elif weighted_beats_best and weighted_result["macro_f1"] >= majority_result["macro_f1"]:
        recommendation = "Confidence-weighted vote is the best option, beating the best individual model."
    elif majority_beats_best:
        recommendation = "Majority vote is the best option, beating the best individual model."
    print(f"\nRecommendation: {recommendation}")

    out = {
        "n": int(len(frame)), "models_used": names,
        "individual": individual_results, "best_individual_model": best_individual_name,
        "majority_vote": majority_result, "weighted_vote": weighted_result,
        "recommendation": recommendation,
    }
    out_path = REPORTS_DIR / "ensemble_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
