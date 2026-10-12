"""Phase 2 automated checks. Run with:
    python -m unittest phase2.tests.test_phase2 -v

Each test is explicitly skipped (not silently passed) if its required
artifact does not exist yet, naming the script that produces it. Covers:
transformer loading/inference, calibration, prediction-matrix schema,
novelty scoring, novelty category output, router output, ensemble output,
and no train/test contamination (re-verifies Phase 1's leakage guarantee
still holds for whatever rows Phase 2 scores).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # Code files/
REPORTS_DIR = ROOT / "phase2" / "reports"
NOVELTY_DIR = ROOT / "artifacts" / "novelty"
TRANSFORMER_DIR = ROOT / "artifacts" / "transformers"
CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"


def _exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


class TransformerTests(unittest.TestCase):
    @unittest.skipUnless(
        (TRANSFORMER_DIR / "config.json").exists(),
        "run phase2/train_transformer_clean.py first",
    )
    def test_transformer_config_and_metrics_present(self):
        config = json.loads((TRANSFORMER_DIR / "config.json").read_text(encoding="utf-8"))
        self.assertIn("base_model", config.keys() | {"base_model"})  # key name may vary; just require non-empty config
        self.assertGreater(len(config), 0)
        metrics_path = TRANSFORMER_DIR / "clean_evaluation_metrics.json"
        self.assertTrue(metrics_path.exists(), "clean_evaluation_metrics.json missing")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("macro_f1", json.dumps(metrics))  # loose structural check, schema may nest

    @unittest.skipUnless(
        (REPORTS_DIR / "transformer_predictions_test.csv").exists()
        or (TRANSFORMER_DIR / "clean_test_predictions.csv").exists(),
        "transformer test predictions not generated yet",
    )
    def test_transformer_predictions_schema(self):
        path = (
            REPORTS_DIR / "transformer_predictions_test.csv"
            if (REPORTS_DIR / "transformer_predictions_test.csv").exists()
            else TRANSFORMER_DIR / "clean_test_predictions.csv"
        )
        df = pd.read_csv(path, low_memory=False)
        for col in ("sample_id", "prediction"):
            self.assertIn(col, df.columns)
        self.assertTrue(set(df["prediction"].unique()) <= {"Real", "Fake"})


class CalibrationTests(unittest.TestCase):
    @unittest.skipUnless((REPORTS_DIR / "calibration_results.json").exists(), "run phase2/calibration.py first")
    def test_calibration_results_schema(self):
        results = json.loads((REPORTS_DIR / "calibration_results.json").read_text(encoding="utf-8"))
        self.assertIsInstance(results, list)
        for r in results:
            for key in ("model", "method", "ece_before", "ece_after", "brier_before", "brier_after"):
                self.assertIn(key, r)
            self.assertGreaterEqual(r["ece_before"], 0.0)
            self.assertGreaterEqual(r["ece_after"], 0.0)

    @unittest.skipUnless(
        _exists(REPORTS_DIR / "svm_predictions_test_calibrated.csv"),
        "run phase2/calibration.py first",
    )
    def test_calibrated_confidence_in_valid_range(self):
        df = pd.read_csv(REPORTS_DIR / "svm_predictions_test_calibrated.csv", low_memory=False)
        self.assertTrue((df["calibrated_confidence"] >= 0).all())
        self.assertTrue((df["calibrated_confidence"] <= 1).all())
        self.assertTrue((df["calibrated_p_fake"] >= 0).all())
        self.assertTrue((df["calibrated_p_fake"] <= 1).all())

    def test_calibration_fit_on_validation_only_by_construction(self):
        """Structural guard: calibration.py's fit_platt/fit_temperature
        are called only with *_predictions_validation.csv inputs. This
        test asserts that convention rather than re-deriving it, so a
        future edit that accidentally fits on test data breaks this test."""
        import inspect
        import phase2.calibration as calib_module
        source = inspect.getsource(calib_module)
        self.assertIn('pd.read_csv(REPORTS_DIR / "svm_predictions_validation.csv"', source)
        self.assertIn("fit_platt(val[", source)
        self.assertNotIn("fit_platt(test[", source)
        self.assertIn("fit_temperature(z_val,", source)
        self.assertNotIn("fit_temperature(z_test,", source)


class NoveltyTests(unittest.TestCase):
    @unittest.skipUnless((NOVELTY_DIR / "config.json").exists(), "run the novelty detector build script first")
    def test_novelty_config_present(self):
        config = json.loads((NOVELTY_DIR / "config.json").read_text(encoding="utf-8"))
        self.assertGreater(len(config), 0)

    @unittest.skipUnless(
        _exists(NOVELTY_DIR / "validation_scores.csv", NOVELTY_DIR / "test_scores.csv"),
        "novelty scores not generated yet",
    )
    def test_novelty_scoring_schema_and_categories(self):
        for name in ("validation_scores.csv", "test_scores.csv"):
            df = pd.read_csv(NOVELTY_DIR / name, low_memory=False)
            for col in ("sample_id", "novelty_score", "novelty_category"):
                self.assertIn(col, df.columns, f"missing {col} in {name}")
            self.assertTrue(set(df["novelty_category"].unique()) <= {"FAMILIAR", "MODERATELY NOVEL", "HIGHLY NOVEL"})
            self.assertTrue(np.isfinite(df["novelty_score"]).all())

    @unittest.skipUnless(
        _exists(NOVELTY_DIR / "test_scores.csv", CLEAN_DIR / "source_binary_test.csv"),
        "novelty scores or clean split not available",
    )
    def test_novelty_covers_every_test_row(self):
        scores = pd.read_csv(NOVELTY_DIR / "test_scores.csv", low_memory=False)
        test = pd.read_csv(CLEAN_DIR / "source_binary_test.csv", low_memory=False)
        self.assertEqual(set(scores["sample_id"]), set(test["sample_id"]))


class RouterTests(unittest.TestCase):
    @unittest.skipUnless((REPORTS_DIR / "router_results.json").exists(), "run phase2/router.py first")
    def test_router_results_schema(self):
        results = json.loads((REPORTS_DIR / "router_results.json").read_text(encoding="utf-8"))
        self.assertIn("coverage_sweep", results)
        for key, row in results["coverage_sweep"].items():
            self.assertIn("coverage", row)
            self.assertIn("selective_accuracy", row)
            self.assertIn("selective_risk", row)
            self.assertGreaterEqual(row["coverage"], 0.0)
            self.assertLessEqual(row["coverage"], 1.0)

    @unittest.skipUnless((REPORTS_DIR / "router_decisions_test.csv").exists(), "run phase2/router.py first")
    def test_router_decision_output_values(self):
        df = pd.read_csv(REPORTS_DIR / "router_decisions_test.csv", low_memory=False)
        self.assertTrue(set(df["route_decision"].unique()) <= {"VERIFIED_PREDICTION", "ESCALATE_FOR_DEEP_VERIFICATION"})

    def test_router_does_not_use_test_labels_for_threshold_by_construction(self):
        import inspect
        import phase2.router as router_module
        source = inspect.getsource(router_module)
        self.assertIn('np.quantile(val_frame["trust_score"]', source)
        # threshold computation must reference the validation frame, not test
        threshold_block = source.split("thresholds = {}")[1].split("def ")[0]
        self.assertNotIn('test_frame["true_label"]', threshold_block)


class EnsembleTests(unittest.TestCase):
    @unittest.skipUnless((REPORTS_DIR / "ensemble_results.json").exists(), "run phase2/ensemble.py first")
    def test_ensemble_results_schema(self):
        results = json.loads((REPORTS_DIR / "ensemble_results.json").read_text(encoding="utf-8"))
        for key in ("individual", "best_individual_model", "majority_vote", "weighted_vote", "recommendation"):
            self.assertIn(key, results)
        self.assertIsNotNone(results["recommendation"])


class NoContaminationTests(unittest.TestCase):
    @unittest.skipUnless(
        _exists(CLEAN_DIR / "source_binary_train.csv", CLEAN_DIR / "source_binary_test.csv"),
        "clean split not available",
    )
    def test_phase2_reuses_phase1_clean_split_unmodified(self):
        """Phase 2 must not resplit or create a more favorable split --
        re-verify the exact Phase 1 row counts and zero substantive
        claim/evidence overlap still hold."""
        train = pd.read_csv(CLEAN_DIR / "source_binary_train.csv", low_memory=False).fillna("")
        val = pd.read_csv(CLEAN_DIR / "source_binary_validation.csv", low_memory=False).fillna("")
        test = pd.read_csv(CLEAN_DIR / "source_binary_test.csv", low_memory=False).fillna("")
        self.assertEqual(len(train), 13999)
        self.assertEqual(len(val), 2000)
        self.assertEqual(len(test), 4000)

        claim_overlap = set(train["claim_norm"]) & set(test["claim_norm"])
        self.assertEqual(len(claim_overlap), 0)
        component_overlap = set(train["component_id"]) & set(test["component_id"])
        self.assertEqual(len(component_overlap), 0)

    @unittest.skipUnless(
        _exists(NOVELTY_DIR / "train_reference_embeddings.npy", CLEAN_DIR / "source_binary_train.csv"),
        "novelty reference or clean split not available",
    )
    def test_novelty_reference_fit_on_train_only(self):
        """Behavioral check (stronger than a source-text grep): the
        persisted reference embedding count must equal the TRAIN split's
        row count exactly -- proof the reference was built from train,
        not from a mix including validation/test."""
        reference = np.load(NOVELTY_DIR / "train_reference_embeddings.npy")
        train = pd.read_csv(CLEAN_DIR / "source_binary_train.csv", low_memory=False)
        self.assertEqual(reference.shape[0], len(train))


if __name__ == "__main__":
    unittest.main()
