"""Tests for the FAAV faithfulness-controlled final decision."""
import unittest

from utils.faav_common import faithfulness_controlled_decision


GOOD_EVIDENCE = (
    "This evidence passage is long enough and directly discusses the claim "
    "with sufficient detail for the verification checks."
)


class FaithfulnessControlledDecisionTests(unittest.TestCase):
    def decide(self, label="Real", **overrides):
        values = {
            "verifier_confidence": 0.85,
            "semantic_similarity": 0.70,
            "grounding_score": 0.65,
        }
        values.update(overrides)
        return faithfulness_controlled_decision(
            label,
            GOOD_EVIDENCE,
            **values,
        )

    def test_accepts_supported_verifier_decision(self):
        self.assertEqual(self.decide("Real"), ("Real", "accepted"))
        self.assertEqual(self.decide("Fake"), ("Fake", "accepted"))

    def test_preserves_verifier_abstention(self):
        self.assertEqual(
            self.decide("Unverified"),
            ("Unverified", "verifier_abstained"),
        )

    def test_abstains_for_short_evidence(self):
        result = faithfulness_controlled_decision(
            "Real",
            "too short",
            verifier_confidence=0.90,
            semantic_similarity=0.80,
            grounding_score=0.80,
        )
        self.assertEqual(result, ("Unverified", "insufficient_evidence"))

    def test_abstains_for_low_verifier_confidence(self):
        self.assertEqual(
            self.decide(verifier_confidence=0.49),
            ("Unverified", "low_verifier_confidence"),
        )

    def test_abstains_for_low_claim_evidence_similarity(self):
        self.assertEqual(
            self.decide(semantic_similarity=0.19),
            ("Unverified", "low_claim_evidence_similarity"),
        )

    def test_abstains_for_low_explanation_grounding(self):
        self.assertEqual(
            self.decide(grounding_score=0.29),
            ("Unverified", "low_explanation_grounding"),
        )

    def test_reports_all_failed_checks(self):
        result = faithfulness_controlled_decision(
            "Fake",
            "short",
            verifier_confidence=0.20,
            semantic_similarity=0.10,
            grounding_score=0.10,
        )
        self.assertEqual(
            result,
            (
                "Unverified",
                "insufficient_evidence|low_verifier_confidence|"
                "low_claim_evidence_similarity|low_explanation_grounding",
            ),
        )


if __name__ == "__main__":
    unittest.main()
