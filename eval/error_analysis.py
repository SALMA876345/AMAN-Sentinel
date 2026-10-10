"""
Error Analysis — FAAV Pipeline (all models)

Categorises every mis-classified claim into one or more of six failure modes:
  1. Dialect failures
  2. Sarcasm / irony
  3. Political ambiguity
  4. Temporal misinformation
  5. Conflicting evidence
  6. Retrieval failures (low token overlap or short evidence)

Runs across ALL model result files so the assignment can compare failure
profiles side-by-side.  Models without a result file are silently skipped.

Also includes a SentenceTransformer semantic error analysis section that
computes, for each model, the mean SBERT similarity between correctly
classified vs incorrectly classified claims and their evidence — revealing
whether harder claims have systematically lower semantic similarity.

Run after all model scripts have been executed.
"""
import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, TEST_FILE, TEST_SHEET

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding="utf-8")

REPORT_FILE = os.path.join(RESULTS_DIR, "error_analysis_report.txt")

# ── All models to analyse ─────────────────────────────────────────────────────
# (name, results_filename, pred_col)
MODEL_REGISTRY = [
    ("SVM",               "svm_claim_results.csv",              "svm_prediction"),
    ("Logistic Regression","lr_claim_results.csv",              "lr_prediction"),
    ("BiLSTM",            "bilstm_claim_results.csv",           "bilstm_prediction"),
    ("AraBERT",           "arabert_claim_results.csv",          "arabert_prediction"),
    ("MARBERT",           "marbert_claim_results.csv",          "marbert_prediction"),
    ("Standard RAG",      "standard_rag_results.csv",           "standard_rag_prediction"),
    ("SBERT-RAG",         "sbert_rag_results.csv",              "sbert_rag_prediction"),
    ("FAAV-Light (rule)", "faithfulness_aware_rag_results.csv", "faav_light_prediction"),
    ("FAAV-Full (proposed)", "faav_full_results.csv",           "faav_prediction"),
]


# ── Error-type detectors ───────────────────────────────────────────────────────

_DIALECT_MARKERS = [
    "مو", "شو", "هيك", "ما في", "ما عندي", "وين", "ليش", "بدي",
    "هلق", "هلا", "زين", "ايش", "عاد", "طيب", "يعني", "لازم",
    "شايف", "قاعد", "راح", "حدا", "صاير", "بيصير", "ولك", "يلا",
]

_SARCASM_MARKERS = [
    "بالطبع", "طبعاً", "أكيد", "بكل تأكيد", "من دون شك",
    "كم هو رائع", "يا له من", "لا شك", "حتماً", "من المؤكد",
    "بالتأكيد التام", "عجيب أمر",
]

_POLITICAL_TERMS = [
    "حكومة", "وزير", "رئيس", "برلمان", "انتخابات", "حزب",
    "سياسة", "سياسي", "نواب", "مجلس", "دولة", "معارضة",
    "ائتلاف", "ديمقراطية", "تحالف", "وزارة", "نيابي",
    "حكومي", "تشريعي", "رئاسي", "انقلاب",
]

_TEMPORAL_MARKERS = [
    "أمس", "البارحة", "اليوم", "هذا الأسبوع", "هذا الشهر",
    "الآن", "مؤخراً", "حديثاً", "للتو", "قبل ساعات", "منذ",
    "في الساعة", "صباح اليوم", "مساء اليوم", "قبل أيام",
    "في الأيام الأخيرة", "خلال الساعات الماضية",
]

_CONFLICT_MARKERS = [
    "من جهة", "من جهة أخرى", "بينما", "في المقابل", "رغم",
    "على الرغم", "إلا أن", "غير أن", "لكن في", "ومع ذلك",
    "في حين", "بيد أن", "إلا أنه", "رغم ذلك",
]


def has_dialect(text: str) -> bool:
    return any(m in str(text) for m in _DIALECT_MARKERS)


def has_sarcasm(text: str) -> bool:
    return any(m in str(text) for m in _SARCASM_MARKERS)


def is_political(text: str) -> bool:
    return any(t in str(text) for t in _POLITICAL_TERMS)


def has_temporal(text: str) -> bool:
    return any(m in str(text) for m in _TEMPORAL_MARKERS)


def has_conflicting_evidence(evidence: str) -> bool:
    return sum(1 for m in _CONFLICT_MARKERS if m in str(evidence)) >= 2


def has_retrieval_failure(evidence: str, claim: str) -> bool:
    if len(str(evidence).strip()) < 20:
        return True
    claim_toks    = set(str(claim).split())
    evidence_toks = set(str(evidence).split())
    return len(claim_toks & evidence_toks) < 2


# ── Helpers ────────────────────────────────────────────────────────────────────

def tag_errors(df_errors: pd.DataFrame) -> pd.DataFrame:
    df = df_errors.copy()
    df["dialect_failure"]      = df["claim_ar"].apply(has_dialect)
    df["sarcasm_failure"]      = df["claim_ar"].apply(has_sarcasm)
    df["political_ambiguity"]  = df["claim_ar"].apply(is_political)
    df["temporal_mismatch"]    = df["claim_ar"].apply(has_temporal)
    df["conflicting_evidence"] = df.apply(
        lambda r: has_conflicting_evidence(r.get("evidence_text", "")), axis=1
    )
    df["retrieval_failure"]    = df.apply(
        lambda r: has_retrieval_failure(
            r.get("evidence_text", ""), r["claim_ar"]
        ), axis=1,
    )
    return df


def summarise(df_model: pd.DataFrame, model_label: str, pred_col: str,
              gold_labels: list, evidences: list) -> dict:
    preds = df_model[pred_col].tolist()

    # Use gold labels from the model's own CSV if available (handles sampled subsets)
    if "gold_label" in df_model.columns:
        gold = df_model["gold_label"].tolist()
    else:
        n    = min(len(preds), len(gold_labels))
        gold = gold_labels[:n]
        preds = preds[:n]

    n   = len(preds)
    acc = accuracy_score(gold, preds)

    model_claims = df_model["claim_ar"].tolist()[:n] if "claim_ar" in df_model.columns else ([""] * n)
    df_eval = pd.DataFrame({
        "claim_ar":      model_claims,
        "gold_label":    gold,
        "prediction":    preds,
        "evidence_text": evidences[:n],
    })
    df_eval["is_correct"] = df_eval["gold_label"] == df_eval["prediction"]

    df_errors = df_eval[~df_eval["is_correct"]].copy()
    df_errors = tag_errors(df_errors)

    total_err = len(df_errors)
    cats = {
        "Dialect failures":        int(df_errors["dialect_failure"].sum()),
        "Sarcasm / irony":         int(df_errors["sarcasm_failure"].sum()),
        "Political ambiguity":     int(df_errors["political_ambiguity"].sum()),
        "Temporal misinformation": int(df_errors["temporal_mismatch"].sum()),
        "Conflicting evidence":    int(df_errors["conflicting_evidence"].sum()),
        "Retrieval failures":      int(df_errors["retrieval_failure"].sum()),
    }

    per_label = {}
    for label in sorted(set(gold)):
        sub   = df_eval[df_eval["gold_label"] == label]
        wrong = sub[~sub["is_correct"]]
        per_label[label] = (len(wrong), len(sub))

    # SBERT similarity for correct vs error rows (if column present)
    sbert_correct = sbert_error = None
    if "evidence_similarity" in df_model.columns:
        ev_sims = df_model["evidence_similarity"].tolist()[:n]
        df_eval["ev_sim"] = ev_sims
        sbert_correct = round(float(df_eval.loc[df_eval["is_correct"],  "ev_sim"].mean()), 4)
        sbert_error   = round(float(df_eval.loc[~df_eval["is_correct"], "ev_sim"].mean()), 4)

    return {
        "model":         model_label,
        "total":         n,
        "correct":       n - total_err,
        "errors":        total_err,
        "accuracy":      round(acc, 4),
        "categories":    cats,
        "per_label":     per_label,
        "sbert_correct": sbert_correct,
        "sbert_error":   sbert_error,
    }


def print_summary(s: dict, f=None):
    lines = [
        f"\n{'='*62}",
        f"Model: {s['model']}",
        f"{'='*62}",
        f"Total claims : {s['total']}",
        f"Correct      : {s['correct']}  ({s['correct']/s['total']:.1%})",
        f"Errors       : {s['errors']}  ({s['errors']/s['total']:.1%})",
        f"Accuracy     : {s['accuracy']}",
        "",
        "Error breakdown by failure category:",
    ]
    total_err = s["errors"]
    for cat, count in s["categories"].items():
        pct = (count / total_err * 100) if total_err else 0
        lines.append(f"  {cat:<34}: {count:>3}  ({pct:.1f}% of errors)")

    lines += ["", "Error breakdown by gold label:"]
    for label, (err, tot) in s["per_label"].items():
        lines.append(f"  {label:<26}: {err}/{tot}  ({err/tot:.1%})")

    if s["sbert_correct"] is not None:
        lines += [
            "",
            "SBERT evidence similarity (semantic error analysis):",
            f"  Correctly classified : mean sim = {s['sbert_correct']}",
            f"  Mis-classified       : mean sim = {s['sbert_error']}",
            f"  Gap (correct - error): {round(s['sbert_correct'] - s['sbert_error'], 4)}"
            + "  (positive => harder claims have weaker evidence)",
        ]

    text = "\n".join(lines)
    print(text)
    if f:
        f.write(text + "\n")


# ── Comparative summary table ──────────────────────────────────────────────────

def print_comparison_table(summaries: list, f=None):
    W = 80
    header = f"\n{'='*W}\nCOMPARATIVE FAILURE-MODE TABLE\n{'='*W}"
    cols   = [
        "Model", "Err%", "Dialect", "Sarcasm", "Political",
        "Temporal", "Conflict", "Retrieval",
    ]
    fmt_h  = f"{'Model':<26} {'Err%':>5} {'Dialect':>8} {'Sarcasm':>8} {'Polit.':>7} {'Temporal':>9} {'Conflict':>9} {'Retrieval':>10}"
    sep    = "-" * W

    lines = [header, fmt_h, sep]
    for s in summaries:
        n    = s["errors"]
        pct  = lambda x: f"{x/n*100:.0f}%" if n else "–"
        c    = s["categories"]
        line = (
            f"{s['model']:<26} {s['errors']/s['total']*100:>4.1f}%"
            f" {pct(c['Dialect failures']):>8}"
            f" {pct(c['Sarcasm / irony']):>8}"
            f" {pct(c['Political ambiguity']):>7}"
            f" {pct(c['Temporal misinformation']):>9}"
            f" {pct(c['Conflicting evidence']):>9}"
            f" {pct(c['Retrieval failures']):>10}"
        )
        lines.append(line)
    lines.append("=" * W)
    text = "\n".join(lines)
    print(text)
    if f:
        f.write(text + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

df_gold     = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
gold_labels = df_gold["gold_label"].tolist()
evidences   = df_gold["evidence_text"].fillna("").tolist()

summaries = []

for model_name, filename, pred_col in MODEL_REGISTRY:
    fpath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(fpath):
        print(f"[SKIP] {model_name}: {filename} not found")
        continue

    df_model = pd.read_csv(fpath)
    if pred_col not in df_model.columns:
        print(f"[SKIP] {model_name}: column '{pred_col}' missing in {filename}")
        continue

    # Merge claim text from test sheet if not in CSV
    if "claim_ar" not in df_model.columns:
        df_model["claim_ar"] = df_gold["claim_ar"].values[:len(df_model)]

    # Use provided evidence from model CSV if available, else gold sheet
    ev_col = None
    for c in ("retrieved_evidence", "evidence_text", "combined_evidence"):
        if c in df_model.columns:
            ev_col = c
            break
    model_evidences = df_model[ev_col].fillna("").tolist() if ev_col else evidences

    s = summarise(df_model, model_name, pred_col, gold_labels, model_evidences)
    summaries.append(s)
    print(f"[OK] {model_name}")

if not summaries:
    print("No results found. Run model scripts first.")
else:
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("ERROR ANALYSIS REPORT — FAAV PROJECT\n")
        f.write("=====================================\n")
        f.write("Failure mode analysis across all models\n")
        f.write("Test set: manual_claim_test_dataset.xlsx\n\n")

        for s in summaries:
            print_summary(s, f)

        print_comparison_table(summaries, f)

        # ── Narrative insights ─────────────────────────────────────────────────
        insight = """

FAILURE MODE INSIGHTS
=====================

1. DIALECT FAILURES
   AFND corpus and BM25/TF-IDF indices are dominated by Modern Standard
   Arabic (MSA).  Dialectal tokens (Gulf, Levantine, Egyptian) reduce
   retrieval precision and lower SBERT similarity scores.  MARBERT, being
   pre-trained on 1B dialectal tweets, should show fewer dialect failures
   than AraBERT — the gap between their dialect failure rates quantifies
   MARBERT's advantage on informal Arabic.

2. SARCASM / IRONY
   Sarcasm markers ("بالطبع", "طبعاً") overlap with support vocabulary,
   causing retrieval models to incorrectly label sarcastic false claims as
   Supported.  MARBERT's social-media pre-training may capture some
   pragmatic context but cannot reliably detect irony without explicit
   sarcasm labels in training data.

3. POLITICAL AMBIGUITY
   Political claims produce retrievable passages from both supporting and
   opposing sources.  FAAV-Full's faithfulness check helps filter
   hallucinated justifications; SBERT-RAG benefits from semantic matching
   that avoids keyword surface noise.  Both still under-perform on Refuted
   political claims where retrieved evidence is equivocal.

4. TEMPORAL MISINFORMATION
   Static corpora cannot verify time-sensitive claims.  All retrieval
   models are equally affected; the error rate in this category is an
   upper bound on how much temporal filtering could improve recall.

5. CONFLICTING EVIDENCE
   When provided evidence contains contradiction markers (رغم, بينما etc.),
   keyword-based models (Standard RAG, FAAV-Light) pick verdicts from
   whichever keyword appears first.  SBERT-RAG and FAAV-Full use similarity
   aggregation which is more robust but can still straddle the threshold,
   defaulting to Not Enough Evidence.

6. RETRIEVAL FAILURES
   Claims with rare named entities or technical vocabulary produce low
   retrieval scores across all methods.  The SBERT similarity gap table
   above (correct vs. error mean similarity) quantifies this: a large gap
   means mis-classified claims consistently have weaker evidence — a signal
   that retrieval quality, not the classification head, is the bottleneck.
   Mitigation: add external retrieval (Google News, Wikimedia) or a
   dialectal Arabic corpus.

MARBERT vs AraBERT (Transformer Comparison)
-------------------------------------------
MARBERT was pre-trained on 1B dialectal Arabic tweets.  If Arabic social-
media claims (ArCOV19, VERA-ARAB) dominate the test set, MARBERT should
show lower dialect + retrieval failure rates and higher overall accuracy.
If the test set is primarily MSA news (AFND, AraNEWS), the gap will narrow.

SBERT-RAG vs Standard RAG (Retrieval Comparison)
-------------------------------------------------
Standard RAG uses TF-IDF/BM25 keyword matching.  SBERT-RAG uses dense
cosine similarity.  SBERT-RAG should reduce retrieval failures on dialectal
or paraphrased claims where keyword overlap is low but semantic similarity
is high.  The evidence_similarity mean for correctly-classified vs.
mis-classified claims (shown in each model section above) confirms whether
denser retrieval reduces the systematic evidence gap.
"""
        print(insight)
        f.write(insight)

    print(f"\nError analysis report saved -> {REPORT_FILE}")
