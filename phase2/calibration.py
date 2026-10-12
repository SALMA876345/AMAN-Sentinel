"""Phase 2 Step 4: confidence calibration for SVM, BiLSTM, and (once
available) the transformer.

- SVM: Platt / sigmoid scaling fit on VALIDATION decision_function values
  only (sklearn's `_SigmoidCalibration`), applied to TEST.
- BiLSTM / transformer: temperature scaling. For binary softmax, only the
  DIFFERENCE between the two logits affects the output (adding a constant
  to both logits leaves softmax unchanged), so the single scalar that
  matters is z = logit_real - logit_fake. For models that expose only
  class probabilities (not raw logits), z is recovered exactly via
  z = log(p_real) - log(p_fake) -- an exact identity for 2-class softmax,
  not an approximation. Temperature T is fit by minimizing validation NLL
  of sigmoid(z / T) against the Real/Fake label, then applied to TEST.

Metrics (both BEFORE and AFTER calibration, on TEST only; calibration
itself is fit on VALIDATION only, never on test):
  - ECE (top-label / confidence calibration, 10 equal-width bins)
  - Brier score (binary, w.r.t. indicator(true_label == "Fake"))

Run standalone for SVM + BiLSTM now; call `run_for_model(...)` again for
the transformer once its prediction CSVs exist (see
phase2/run_transformer_calibration.py, added after that subagent returns).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import _SigmoidCalibration

ROOT = Path(__file__).resolve().parents[1]  # Code files/
REPORTS_DIR = ROOT / "phase2" / "reports"
LABELS = ["Real", "Fake"]


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Top-label (confidence) ECE: bins by predicted-class confidence,
    compares mean confidence to empirical accuracy within each bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_brier(p_fake: np.ndarray, true_label: np.ndarray) -> float:
    """Binary Brier score w.r.t. P(label == 'Fake')."""
    y = (true_label == "Fake").astype(float)
    return float(np.mean((p_fake - y) ** 2))


def fit_platt(raw_score: np.ndarray, true_label: np.ndarray) -> _SigmoidCalibration:
    """raw_score: LinearSVC decision_function value. Empirically verified
    (model.classes_ == ['Fake', 'Real']) that positive values correlate
    with the 'Real' prediction, negative with 'Fake' -- sklearn's
    _SigmoidCalibration fits its own sign-correct slope `a` regardless, so
    this direction does not need to be hardcoded, only noted for clarity
    when reading the persisted calibrator_params."""
    y = (true_label == "Fake").astype(int)
    calibrator = _SigmoidCalibration()
    calibrator.fit(raw_score, y)
    return calibrator


def fit_temperature(z: np.ndarray, true_label: np.ndarray) -> float:
    """z = logit_real - logit_fake (or log(p_real) - log(p_fake)).
    Fits T minimizing validation NLL of sigmoid(z / T) as P(Real)."""
    y_real = (true_label == "Real").astype(float)

    def nll(log_t: float) -> float:
        t = np.exp(log_t)  # optimize in log-space to keep T > 0
        p_real = 1.0 / (1.0 + np.exp(-z / t))
        p_real = np.clip(p_real, 1e-7, 1 - 1e-7)
        return float(-np.mean(y_real * np.log(p_real) + (1 - y_real) * np.log(1 - p_real)))

    result = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def apply_platt_to_split(calibrator: _SigmoidCalibration, df: pd.DataFrame, out_name: str) -> np.ndarray:
    p_fake_calibrated = np.clip(calibrator.predict(df["raw_score"].to_numpy()), 1e-7, 1 - 1e-7)
    pred_is_fake = df["prediction"].to_numpy() == "Fake"
    confidence_calibrated = np.where(pred_is_fake, p_fake_calibrated, 1 - p_fake_calibrated)
    out = df.copy()
    out["calibrated_confidence"] = confidence_calibrated
    out["calibrated_p_fake"] = p_fake_calibrated
    out.to_csv(REPORTS_DIR / out_name, index=False, encoding="utf-8-sig")
    return confidence_calibrated, p_fake_calibrated


def run_svm() -> dict:
    val = pd.read_csv(REPORTS_DIR / "svm_predictions_validation.csv", low_memory=False)
    test = pd.read_csv(REPORTS_DIR / "svm_predictions_test.csv", low_memory=False)

    calibrator = fit_platt(val["raw_score"].to_numpy(), val["true_label"].to_numpy())

    # Apply to validation too (for router threshold-setting consistency —
    # NOT re-fit on validation's own calibrated output, just scored with it).
    apply_platt_to_split(calibrator, val, "svm_predictions_validation_calibrated.csv")
    confidence_calibrated, p_fake_calibrated = apply_platt_to_split(
        calibrator, test, "svm_predictions_test_calibrated.csv"
    )

    pred_is_fake = test["prediction"].to_numpy() == "Fake"
    correct = (test["prediction"] == test["true_label"]).to_numpy()

    before_ece = compute_ece(test["confidence"].to_numpy(), correct)
    before_brier = compute_brier(
        np.where(pred_is_fake, test["confidence"].to_numpy(), 1 - test["confidence"].to_numpy()),
        test["true_label"].to_numpy(),
    )
    after_ece = compute_ece(confidence_calibrated, correct)
    after_brier = compute_brier(p_fake_calibrated, test["true_label"].to_numpy())

    return {
        "model": "svm", "method": "platt_sigmoid_scaling",
        "fit_on": "validation (2000 rows)", "evaluated_on": "test (4000 rows)",
        "ece_before": before_ece, "ece_after": after_ece,
        "brier_before": before_brier, "brier_after": after_brier,
        "calibrator_params": {"a": float(calibrator.a_), "b": float(calibrator.b_)},
    }


def run_bilstm() -> dict:
    val = pd.read_csv(REPORTS_DIR / "bilstm_predictions_validation.csv", low_memory=False)
    test = pd.read_csv(REPORTS_DIR / "bilstm_predictions_test.csv", low_memory=False)

    z_val = val["logit_real"].to_numpy() - val["logit_fake"].to_numpy()
    z_test = test["logit_real"].to_numpy() - test["logit_fake"].to_numpy()

    temperature = fit_temperature(z_val, val["true_label"].to_numpy())
    p_real_calibrated = 1.0 / (1.0 + np.exp(-z_test / temperature))
    p_real_calibrated = np.clip(p_real_calibrated, 1e-7, 1 - 1e-7)
    p_fake_calibrated = 1 - p_real_calibrated

    pred_is_fake = test["prediction"].to_numpy() == "Fake"
    confidence_calibrated = np.where(pred_is_fake, p_fake_calibrated, p_real_calibrated)
    correct = (test["prediction"] == test["true_label"]).to_numpy()

    before_ece = compute_ece(test["confidence"].to_numpy(), correct)
    before_brier = compute_brier(
        np.where(pred_is_fake, test["confidence"].to_numpy(), 1 - test["confidence"].to_numpy()),
        test["true_label"].to_numpy(),
    )
    after_ece = compute_ece(confidence_calibrated, correct)
    after_brier = compute_brier(p_fake_calibrated, test["true_label"].to_numpy())

    out = test.copy()
    out["calibrated_confidence"] = confidence_calibrated
    out["calibrated_p_fake"] = p_fake_calibrated
    out.to_csv(REPORTS_DIR / "bilstm_predictions_test_calibrated.csv", index=False, encoding="utf-8-sig")

    # Validation-calibrated output too (router threshold-setting consistency)
    p_real_val = 1.0 / (1.0 + np.exp(-z_val / temperature))
    p_real_val = np.clip(p_real_val, 1e-7, 1 - 1e-7)
    p_fake_val = 1 - p_real_val
    val_pred_is_fake = val["prediction"].to_numpy() == "Fake"
    val_out = val.copy()
    val_out["calibrated_confidence"] = np.where(val_pred_is_fake, p_fake_val, p_real_val)
    val_out["calibrated_p_fake"] = p_fake_val
    val_out.to_csv(REPORTS_DIR / "bilstm_predictions_validation_calibrated.csv", index=False, encoding="utf-8-sig")

    return {
        "model": "bilstm", "method": "temperature_scaling",
        "fit_on": "validation (2000 rows)", "evaluated_on": "test (4000 rows)",
        "ece_before": before_ece, "ece_after": after_ece,
        "brier_before": before_brier, "brier_after": after_brier,
        "temperature": temperature,
    }


def run_transformer() -> dict:
    """Same temperature-scaling approach as BiLSTM, but the transformer
    exposes only softmax probabilities (prob_real, prob_fake), not raw
    logits. For 2-class softmax this is not a limitation: only the
    DIFFERENCE between the two logits affects the output, and
    z = log(p_real) - log(p_fake) recovers that difference exactly
    (softmax is invariant to adding the same constant to both logits, so
    this recovers z up to that irrelevant constant)."""
    val = pd.read_csv(REPORTS_DIR / "transformer_predictions_validation.csv", low_memory=False)
    test = pd.read_csv(REPORTS_DIR / "transformer_predictions_test.csv", low_memory=False)

    eps = 1e-12
    z_val = np.log(val["prob_real"].to_numpy() + eps) - np.log(val["prob_fake"].to_numpy() + eps)
    z_test = np.log(test["prob_real"].to_numpy() + eps) - np.log(test["prob_fake"].to_numpy() + eps)

    temperature = fit_temperature(z_val, val["true_label"].to_numpy())

    def calibrated_p_real(z: np.ndarray) -> np.ndarray:
        return np.clip(1.0 / (1.0 + np.exp(-z / temperature)), 1e-7, 1 - 1e-7)

    p_real_val, p_real_test = calibrated_p_real(z_val), calibrated_p_real(z_test)

    for df, p_real, out_name in [(val, p_real_val, "transformer_predictions_validation_calibrated.csv"),
                                   (test, p_real_test, "transformer_predictions_test_calibrated.csv")]:
        pred_is_fake = df["prediction"].to_numpy() == "Fake"
        out = df.copy()
        out["calibrated_confidence"] = np.where(pred_is_fake, 1 - p_real, p_real)
        out["calibrated_p_fake"] = 1 - p_real
        out.to_csv(REPORTS_DIR / out_name, index=False, encoding="utf-8-sig")

    pred_is_fake = test["prediction"].to_numpy() == "Fake"
    p_fake_calibrated = 1 - p_real_test
    confidence_calibrated = np.where(pred_is_fake, p_fake_calibrated, p_real_test)
    correct = (test["prediction"] == test["true_label"]).to_numpy()

    before_ece = compute_ece(test["confidence"].to_numpy(), correct)
    before_brier = compute_brier(test["prob_fake"].to_numpy(), test["true_label"].to_numpy())
    after_ece = compute_ece(confidence_calibrated, correct)
    after_brier = compute_brier(p_fake_calibrated, test["true_label"].to_numpy())

    return {
        "model": "transformer", "method": "temperature_scaling (z recovered from log-prob difference)",
        "fit_on": "validation", "evaluated_on": "test",
        "ece_before": before_ece, "ece_after": after_ece,
        "brier_before": before_brier, "brier_after": after_brier,
        "temperature": temperature,
    }


def main() -> None:
    results = [run_svm(), run_bilstm(), run_transformer()]
    for r in results:
        print(f"[{r['model']}] method={r['method']}  ECE {r['ece_before']:.4f} -> {r['ece_after']:.4f}  "
              f"Brier {r['brier_before']:.4f} -> {r['brier_after']:.4f}")
    out_path = REPORTS_DIR / "calibration_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
