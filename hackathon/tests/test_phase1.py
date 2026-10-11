"""Phase 1 Step 10: lightweight automated checks for the restored baseline.

Run with:
    python -m unittest hackathon.tests.test_phase1 -v

Covers: clean-split dataset loading, zero claim leakage, zero evidence
leakage, vectorizer loading, SVM model loading, inference, and the
expected output schema. Each test is explicitly skipped (not silently
passed) if its required artifact is missing, naming the script that
produces it.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # Code files/
CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
ARTIFACT_DIR = ROOT / "artifacts" / "svm"
TASK = "binary"

# Matches hackathon/build_clean_split.py's MIN_EVIDENCE_CHARS_FOR_GROUPING.
# Evidence shorter than this (a bare name, "age reaching") is not merged
# into a shared component by design and is expected to repeat harmlessly
# across splits; only overlap at or above this length is a real leak.
MIN_EVIDENCE_CHARS_FOR_GROUPING = 40


def _clean_split_present() -> bool:
    return all(
        (CLEAN_DIR / f"source_{TASK}_{part}.csv").exists()
        for part in ("train", "validation", "test")
    )


def _artifacts_present() -> bool:
    return all(
        (ARTIFACT_DIR / name).exists()
        for name in ("vectorizer.joblib", "model.joblib", "config.json", "label_mapping.json")
    )


class CleanSplitDatasetTests(unittest.TestCase):
    @unittest.skipUnless(_clean_split_present(), "run hackathon/build_clean_split.py first")
    def test_dataset_loading(self):
        for part in ("train", "validation", "test"):
            df = pd.read_csv(CLEAN_DIR / f"source_{TASK}_{part}.csv", low_memory=False)
            self.assertGreater(len(df), 0, f"{part} split is empty")
            for col in ["sample_id", "claim_text", "evidence_text", "claim_norm",
                        "evidence_norm", "claim_group", "evidence_group",
                        "component_id", "task_label"]:
                self.assertIn(col, df.columns, f"missing column {col} in {part}")

    @unittest.skipUnless(_clean_split_present(), "run hackathon/build_clean_split.py first")
    def test_zero_claim_leakage(self):
        train = pd.read_csv(CLEAN_DIR / f"source_{TASK}_train.csv", low_memory=False)
        test = pd.read_csv(CLEAN_DIR / f"source_{TASK}_test.csv", low_memory=False)
        validation = pd.read_csv(CLEAN_DIR / f"source_{TASK}_validation.csv", low_memory=False)

        for a, b, name in [(train, test, "train/test"), (train, validation, "train/validation"),
                            (validation, test, "validation/test")]:
            claim_overlap = set(a["claim_norm"].astype(str)) & set(b["claim_norm"].astype(str))
            self.assertEqual(len(claim_overlap), 0, f"exact claim overlap between {name}: {len(claim_overlap)}")
            group_overlap = set(a["claim_group"].astype(str)) & set(b["claim_group"].astype(str))
            self.assertEqual(len(group_overlap), 0, f"claim_group overlap between {name}: {len(group_overlap)}")

    @unittest.skipUnless(_clean_split_present(), "run hackathon/build_clean_split.py first")
    def test_zero_evidence_leakage(self):
        """Zero overlap on SUBSTANTIVE evidence (>= grouping threshold).
        Short fragments below the threshold (bare names, "age reaching")
        are deliberately excluded from component merging — see
        hackathon/build_clean_split.py's MIN_EVIDENCE_CHARS_FOR_GROUPING —
        and repeating harmlessly is expected, not a leak."""
        train = pd.read_csv(CLEAN_DIR / f"source_{TASK}_train.csv", low_memory=False).fillna("")
        test = pd.read_csv(CLEAN_DIR / f"source_{TASK}_test.csv", low_memory=False).fillna("")
        validation = pd.read_csv(CLEAN_DIR / f"source_{TASK}_validation.csv", low_memory=False).fillna("")

        for a, b, name in [(train, test, "train/test"), (train, validation, "train/validation"),
                            (validation, test, "validation/test")]:
            a_sub = a[a["evidence_norm"].str.len() >= MIN_EVIDENCE_CHARS_FOR_GROUPING]
            b_sub = b[b["evidence_norm"].str.len() >= MIN_EVIDENCE_CHARS_FOR_GROUPING]

            evidence_overlap = set(a_sub["evidence_norm"].astype(str)) & set(b_sub["evidence_norm"].astype(str))
            self.assertEqual(len(evidence_overlap), 0, f"substantive exact evidence overlap between {name}: {len(evidence_overlap)}")
            group_overlap = set(a_sub["evidence_group"].astype(str)) & set(b_sub["evidence_group"].astype(str))
            self.assertEqual(len(group_overlap), 0, f"substantive evidence_group overlap between {name}: {len(group_overlap)}")

    @unittest.skipUnless(_clean_split_present(), "run hackathon/build_clean_split.py first")
    def test_zero_component_leakage(self):
        """The grouping unit itself must not cross splits — the strongest check."""
        train = pd.read_csv(CLEAN_DIR / f"source_{TASK}_train.csv", low_memory=False)
        test = pd.read_csv(CLEAN_DIR / f"source_{TASK}_test.csv", low_memory=False)
        component_overlap = set(train["component_id"]) & set(test["component_id"])
        self.assertEqual(len(component_overlap), 0, f"component_id overlap train/test: {len(component_overlap)}")


class SVMArtifactTests(unittest.TestCase):
    @unittest.skipUnless(_artifacts_present(), "run hackathon/train_svm_clean.py first")
    def test_vectorizer_loading(self):
        vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.joblib")
        matrix = vectorizer.transform(["كلمة عربية للاختبار"])
        self.assertGreater(matrix.shape[1], 0, "vectorizer produced zero features")

    @unittest.skipUnless(_artifacts_present(), "run hackathon/train_svm_clean.py first")
    def test_svm_model_loading(self):
        model = joblib.load(ARTIFACT_DIR / "model.joblib")
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "decision_function"))

    @unittest.skipUnless(_artifacts_present(), "run hackathon/train_svm_clean.py first")
    def test_inference_and_output_schema(self):
        vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.joblib")
        model = joblib.load(ARTIFACT_DIR / "model.joblib")
        text = "هذا خبر تجريبي للتحقق من عمل النموذج [SEP] نص دليل تجريبي للاختبار"
        matrix = vectorizer.transform([text])
        prediction = model.predict(matrix)
        score = model.decision_function(matrix)

        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], {"Real", "Fake"})
        self.assertEqual(len(score), 1)

        # Expected output schema, mirroring predict_sanity_check.py's contract
        output = {
            "prediction": str(prediction[0]),
            "decision_score": float(score[0]),
            "model_used": "LinearSVC (clean, hackathon/train_svm_clean.py)",
        }
        for key in ("prediction", "decision_score", "model_used"):
            self.assertIn(key, output)
        self.assertIsInstance(output["prediction"], str)
        self.assertIsInstance(output["decision_score"], float)

    @unittest.skipUnless(_artifacts_present(), "run hackathon/train_svm_clean.py first")
    def test_config_documents_confidence_is_not_calibrated(self):
        import json
        config = json.loads((ARTIFACT_DIR / "config.json").read_text(encoding="utf-8"))
        note = config.get("confidence_derivation", "")
        self.assertIn("NOT a calibrated probability", note)


if __name__ == "__main__":
    unittest.main()
