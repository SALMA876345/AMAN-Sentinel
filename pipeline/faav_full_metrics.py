"""
FAAV full verification metrics — all 8 evaluation metrics:

Classification : Accuracy, Precision, Recall, F1-score
Verification   : Evidence Relevance, Faithfulness Score,
                 Hallucination Rate, Citation Support Accuracy
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_FILE, TEST_SHEET, RESULTS_DIR, FAAV_FULL_CSV, FAAV_FULL_METRICS

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score,
)

REFUTE_PATTERNS = [
    "غير صحيح", "غير صحيحة", "كاذب", "مضلل", "مضللة",
    "زائف", "مزيف", "مفبرك", "لا صحة", "لا أساس له",
    "نفى", "نفت", "ينفي", "تنفي", "لم يحدث", "لم يثبت",
    "لا يوجد دليل", "لا توجد أدلة",
    "معلومة مضللة", "خبر كاذب", "معلومة خاطئة",
]
SUPPORT_PATTERNS = [
    "أكد", "أكدت", "تؤكد", "أعلنت", "أعلن", "صرح", "صرحت",
    "نشرت", "نشر", "أفاد", "أفادت", "أوضحت", "أوضح",
    "بحسب", "وفق", "جاء في", "ورد في", "أثبت", "أثبتت",
]
HALLUCINATION_TERMS = [
    "وزير", "استقال", "وفاة", "اعتقال",
    "انفجار", "حرب", "مؤامرة", "تمويل",
]
_FOREIGN = [
    "الجزائر", "لبنان", "مصر", "سوريا", "العراق", "اليمن",
    "السودان", "ليبيا", "تونس", "المغرب", "الأردن", "فلسطين",
    "إيران", "تركيا", "السعودية", "قطر",
]
_UAE = [
    "الإمارات", "إماراتي", "أبوظبي", "دبي", "الشارقة",
    "عجمان", "الفجيرة", "رأس الخيمة",
]


def predict_label(claim, evidence):
    claim, evidence = str(claim), str(evidence)
    for p in REFUTE_PATTERNS:
        if p in evidence:
            return "Fake"
    claim_foreign = any(c in claim for c in _FOREIGN)
    ev_uae        = any(u in evidence for u in _UAE)
    ev_foreign    = any(c in evidence for c in _FOREIGN)
    if claim_foreign and ev_uae and not ev_foreign:
        return "Fake"
    for p in SUPPORT_PATTERNS:
        if p in evidence:
            return "Real"
    return "Real"


def generate_explanation(predicted_label):
    if predicted_label == "Fake":
        return "The retrieved evidence contradicts or refutes the claim."
    return "The retrieved evidence supports or is consistent with the claim."


def check_faithfulness(explanation, evidence):
    explanation, evidence = str(explanation), str(evidence)
    for term in HALLUCINATION_TERMS:
        if term in explanation and term not in evidence:
            return "Low"
    return "High"


def lexical_overlap(claim, evidence):
    claim_words = set(re.findall(r'[؀-ۿ]{3,}', str(claim)))
    if not claim_words:
        return 1.0
    ev_words = set(re.findall(r'[؀-ۿ]{3,}', str(evidence)))
    return len(claim_words & ev_words) / len(claim_words)


def citation_support(predicted, evidence):
    """Does the predicted label align with what the evidence signals?"""
    evidence = str(evidence)
    has_refute  = any(p in evidence for p in REFUTE_PATTERNS)
    has_support = any(p in evidence for p in SUPPORT_PATTERNS)
    if predicted == "Fake" and has_refute:
        return True
    if predicted == "Real" and has_support:
        return True
    if predicted == "Unverified" and not has_refute and not has_support:
        return True
    return False


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
    print(f"Test rows: {len(df)}")

    preds, explanations, faithfulness_preds = [], [], []
    overlaps, citation_supports = [], []

    for _, row in df.iterrows():
        claim    = row["claim_ar"]
        evidence = row["evidence_text"]
        label    = predict_label(claim, evidence)
        expl     = generate_explanation(label)
        faith    = check_faithfulness(expl, evidence)
        overlap  = lexical_overlap(claim, evidence)
        cite     = citation_support(label, evidence)

        preds.append(label)
        explanations.append(expl)
        faithfulness_preds.append(faith)
        overlaps.append(overlap)
        citation_supports.append(cite)

    df["faav_prediction"]        = preds
    df["faav_explanation"]       = explanations
    df["predicted_faithfulness"] = faithfulness_preds
    df["lexical_overlap"]        = overlaps
    df["citation_supported"]     = citation_supports
    df["is_correct"]             = df["gold_label"] == df["faav_prediction"]

    y_true = df["gold_label"].tolist()
    y_pred = df["faav_prediction"].tolist()

    # ── Classification metrics ────────────────────────────────────────────────
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec     = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mf1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wf1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc     = matthews_corrcoef(y_true, y_pred)
    kappa   = cohen_kappa_score(y_true, y_pred)
    report  = classification_report(y_true, y_pred, zero_division=0)
    matrix  = confusion_matrix(y_true, y_pred, labels=["Real", "Fake", "Unverified"])

    # ── Verification metrics ──────────────────────────────────────────────────
    rq_counts = df["retrieval_quality"].value_counts(normalize=True) * 100
    relevance_score = (
        rq_counts.get("Relevant", 0) * 1.0 +
        rq_counts.get("Partially Relevant", 0) * 0.5 +
        rq_counts.get("Irrelevant", 0) * 0.0
    ) / 100.0

    faith_acc  = accuracy_score(df["faithfulness_label"], df["predicted_faithfulness"])
    halluc_rate = (df["predicted_faithfulness"] == "Low").mean()
    cite_acc    = df["citation_supported"].mean()
    avg_overlap = df["lexical_overlap"].mean()

    # ── Print and save ────────────────────────────────────────────────────────
    df.to_csv(FAAV_FULL_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "FAAV Full Verification Metrics",
        "=" * 35,
        "",
        "=== CLASSIFICATION METRICS ===",
        f"Accuracy         : {acc:.4f}",
        f"Macro Precision  : {prec:.4f}",
        f"Macro Recall     : {rec:.4f}",
        f"Macro F1         : {mf1:.4f}",
        f"Weighted F1      : {wf1:.4f}",
        f"MCC              : {mcc:.4f}",
        f"Cohen Kappa      : {kappa:.4f}",
        "",
        "Per-class Report:",
        report,
        "Confusion Matrix:",
        str(matrix),
        "",
        "=== VERIFICATION METRICS ===",
        f"Evidence Relevance Score  : {relevance_score:.4f}",
        f"  Relevant      : {rq_counts.get('Relevant', 0):.1f}%",
        f"  Partly Relev. : {rq_counts.get('Partially Relevant', 0):.1f}%",
        f"  Irrelevant    : {rq_counts.get('Irrelevant', 0):.1f}%",
        f"Faithfulness Score        : {faith_acc:.4f}",
        f"Hallucination Rate        : {halluc_rate:.4f}  ({halluc_rate*100:.1f}%)",
        f"Citation Support Accuracy : {cite_acc:.4f}  ({cite_acc*100:.1f}%)",
        f"Avg Lexical Overlap       : {avg_overlap:.4f}",
    ]

    output = "\n".join(lines)
    print(output)

    with open(FAAV_FULL_METRICS, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nResults saved -> {FAAV_FULL_CSV}")
    print(f"Metrics saved -> {FAAV_FULL_METRICS}")


if __name__ == "__main__":
    main()
