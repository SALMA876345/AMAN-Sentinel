import os
import re
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (
    DATA_DIR, RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET,
    CLASS_MODE, CLASS_LABELS, result_tagged,
)
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


OUTPUT_FILE = result_tagged("faithfulness_aware_rag_results.csv")
RESULTS_FILE = result_tagged("faithfulness_aware_rag_metrics.txt")

print(f"=== FAAV-Light — {CLASS_MODE}-class {CLASS_LABELS} ===")
df = pd.read_excel(TEST_FILE, sheet_name="Tabelle1")
df = df[df["gold_label"].isin(CLASS_LABELS)].reset_index(drop=True)

# Country names that signal a non-UAE claim
_FOREIGN_COUNTRIES = [
    "الجزائر", "جزائر", "لبنان", "لبناني", "مصر", "مصري", "سوريا", "سوري",
    "العراق", "عراقي", "اليمن", "يمني", "السودان", "سوداني", "ليبيا", "ليبي",
    "تونس", "تونسي", "المغرب", "مغربي", "موريتانيا", "الأردن", "أردني",
    "فلسطين", "فلسطيني", "إيران", "إيراني", "تركيا", "تركي",
    "السعودية", "سعودي", "قطر", "قطري", "البحرين", "الكويت",
    "فرنسا", "أمريكا", "بريطانيا", "روسيا", "الصين",
]
_UAE_WORDS = [
    "الإمارات", "إماراتي", "إماراتية", "أبوظبي", "دبي", "الشارقة",
    "عجمان", "الفجيرة", "رأس الخيمة", "أم القيوين",
    "وزارة التربية", "وزارة الصحة", "وزارة الموارد",
]

def _lexical_overlap(claim: str, evidence: str) -> float:
    claim_words = set(re.findall(r'[؀-ۿ]{3,}', claim))
    if not claim_words:
        return 1.0
    ev_words = set(re.findall(r'[؀-ۿ]{3,}', evidence))
    return len(claim_words & ev_words) / len(claim_words)


def predict_label_with_evidence(claim, evidence):
    """
    Faithfulness-aware rule-based verification.
    It uses claim + evidence and gives priority to explicit refutation,
    insufficient evidence, and support signals.
    """

    claim = str(claim)
    evidence = str(evidence)

    refute_patterns = [
        "غير صحيح", "غير صحيحة", "كاذب", "مضلل", "مضللة",
        "زائف", "مزيف", "مفبرك", "لا صحة", "لا أساس له",
        "نفى", "نفت", "ينفي", "تنفي", "لم يحدث", "لم يثبت",
        "لا يوجد دليل", "لا توجد أدلة", "ادعاء غير صحيح",
        "الادعاء خاطئ", "الخبر غير صحيح",
        "معلومة مضللة", "خبر كاذب", "معلومة خاطئة",
    ]

    support_patterns = [
        "أكد", "أكدت", "تؤكد", "أعلنت", "أعلن", "صرح", "صرحت",
        "نشرت", "نشر", "أفاد", "أفادت", "أوضحت", "أوضح",
        "بحسب", "وفق", "جاء في", "ورد في", "أثبت", "أثبتت", "تم تأكيد",
    ]

    # 1. Explicit refutation keywords
    for pattern in refute_patterns:
        if pattern in evidence:
            return "Fake"

    # 2. Country mismatch: claim about non-UAE country but evidence is UAE-only
    claim_has_foreign = any(c in claim for c in _FOREIGN_COUNTRIES)
    ev_has_uae        = any(u in evidence for u in _UAE_WORDS)
    ev_has_foreign    = any(c in evidence for c in _FOREIGN_COUNTRIES)
    if claim_has_foreign and ev_has_uae and not ev_has_foreign:
        return "Fake"

    # 3. Support keywords
    for pattern in support_patterns:
        if pattern in evidence:
            return "Real"

    # 4. Insufficient-evidence cases → Unverified (3-class only)
    if "Unverified" in CLASS_LABELS:
        nei_patterns = [
            "غير مؤكد", "غير واضح", "لا يمكن التحقق", "لا يوجد دليل كافٍ",
            "لا تتوفر معلومات كافية", "يحتاج إلى تحقق", "لم يتم التحقق",
        ]
        for p in nei_patterns:
            if p in evidence:
                return "Unverified"
        if len(str(evidence).strip()) < 30:
            return "Unverified"

    return "Real"


def generate_explanation(claim, evidence, predicted_label):
    """
    Generate a simple evidence-grounded explanation.
    This keeps the explanation conservative and based only on the evidence.
    """

    if predicted_label == "Supported":
        return "The claim is supported because the retrieved evidence confirms the main information stated in the claim."

    if predicted_label == "Refuted":
        return "The claim is refuted because the retrieved evidence contradicts the claim or states that the claim is false or unsupported."

    return "There is not enough evidence because the retrieved evidence does not provide sufficient confirmation or denial of the claim."


def faithfulness_check(explanation, evidence):
    """
    Simple faithfulness check.
    The explanation is considered highly faithful if it does not introduce
    unsupported extra details and only refers to support/refutation/insufficient evidence.
    """

    explanation = str(explanation)
    evidence = str(evidence)

    unsupported_terms = [
        "وزير",
        "استقال",
        "وفاة",
        "اعتقال",
        "انفجار",
        "حرب",
        "مؤامرة",
        "تمويل",
    ]

    # If explanation introduces specific extra details not present in evidence,
    # mark it lower. Our generated explanations are intentionally generic,
    # so they should normally be High.
    for term in unsupported_terms:
        if term in explanation and term not in evidence:
            return "Low"

    return "High"


predicted_labels = []
generated_explanations = []
predicted_faithfulness = []

for _, row in df.iterrows():
    claim = row["claim_ar"]
    evidence = row["evidence_text"]

    label = predict_label_with_evidence(claim, evidence)
    explanation = generate_explanation(claim, evidence, label)
    faithfulness = faithfulness_check(explanation, evidence)

    predicted_labels.append(label)
    generated_explanations.append(explanation)
    predicted_faithfulness.append(faithfulness)


df["faav_light_prediction"] = predicted_labels
df["faav_light_explanation"] = generated_explanations
df["predicted_faithfulness"] = predicted_faithfulness
df["is_correct"] = df["gold_label"] == df["faav_light_prediction"]
df["faithfulness_correct"] = df["faithfulness_label"] == df["predicted_faithfulness"]

accuracy = accuracy_score(df["gold_label"], df["faav_light_prediction"])
report = classification_report(df["gold_label"], df["faav_light_prediction"], labels=CLASS_LABELS, zero_division=0)
matrix = confusion_matrix(df["gold_label"], df["faav_light_prediction"], labels=CLASS_LABELS)

faithfulness_accuracy = accuracy_score(
    df["faithfulness_label"],
    df["predicted_faithfulness"]
)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("FAAV-Light Faithfulness-Aware RAG Results\n")
    f.write("========================================\n\n")
    f.write(f"Total claims: {len(df)}\n")
    f.write(f"Verification Accuracy: {accuracy:.4f}\n")
    f.write(f"Faithfulness Accuracy: {faithfulness_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))

print("FAAV-Light experiment completed.")
print("Verification Accuracy:", round(accuracy, 4))
print("Faithfulness Accuracy:", round(faithfulness_accuracy, 4))

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(matrix)

print(f"\nResults saved to: {OUTPUT_FILE}")
print(f"Metrics saved to: {RESULTS_FILE}")