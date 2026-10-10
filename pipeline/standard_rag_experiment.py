"""
Standard RAG Experiment — BM25 Retrieval + Rule-based Verification

Pipeline:
  1. BM25 Retriever  — indexes training corpus, retrieves top-K passages per claim
  2. Verdict logic   — refutation / support keywords on retrieved evidence
  3. Fallback        — uses pre-existing evidence_text if BM25 score is low

Retriever: rank_bm25 (BM25Okapi)
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_FILE, TEST_FILE, TEST_SHEET, RESULTS_DIR,
    CLASS_MODE, CLASS_LABELS, result_tagged,
)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, matthews_corrcoef,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Install rank_bm25:  pip install rank-bm25")

OUTPUT_FILE  = result_tagged("standard_rag_results.csv")
RESULTS_FILE = result_tagged("standard_rag_metrics.txt")

TOP_K         = 5        # passages to retrieve per claim
CORPUS_LIMIT  = 10_000   # 10k docs keeps RAM low and BM25 fast on laptop
BM25_MIN_SCORE = 0.5     # below this, fall back to pre-existing evidence_text

_FOREIGN = [
    "الجزائر", "جزائر", "لبنان", "لبناني", "مصر", "مصري", "سوريا", "سوري",
    "العراق", "عراقي", "اليمن", "يمني", "السودان", "سوداني", "ليبيا", "ليبي",
    "تونس", "تونسي", "المغرب", "مغربي", "الأردن", "أردني",
    "فلسطين", "فلسطيني", "إيران", "إيراني", "تركيا", "تركي",
    "السعودية", "سعودي", "قطر", "قطري", "البحرين", "الكويت",
    "فرنسا", "أمريكا", "بريطانيا", "روسيا", "الصين",
]
_UAE = [
    "الإمارات", "إماراتي", "إماراتية", "أبوظبي", "دبي", "الشارقة",
    "عجمان", "الفجيرة", "رأس الخيمة", "أم القيوين",
    "وزارة التربية", "وزارة الصحة", "وزارة الموارد",
]
REFUTE_WORDS = [
    "غير صحيح", "كاذب", "مضلل", "نفى", "نفت", "لا صحة",
    "لم يحدث", "مزيف", "غير دقيق", "لا يوجد دليل", "ينفي",
    "معلومة مضللة", "خبر كاذب", "معلومة خاطئة",
]
SUPPORT_WORDS = [
    "أكد", "أعلنت", "صرح", "نشرت", "وافق", "أثبت",
    "أفاد", "جاء في", "بحسب", "أوضحت", "تؤكد",
]


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return normalize_arabic(text).split()


def verdict(claim, evidence):
    claim, evidence = str(claim), str(evidence)
    # 1. Refutation keywords
    for w in REFUTE_WORDS:
        if w in evidence:
            return "Fake"
    # 2. Country mismatch
    if (any(c in claim for c in _FOREIGN) and
            any(u in evidence for u in _UAE) and
            not any(c in evidence for c in _FOREIGN)):
        return "Fake"
    # 3. Support keywords
    for w in SUPPORT_WORDS:
        if w in evidence:
            return "Real"
    # 4. No clear signal → Unverified (3-class) or default Real (2-class)
    if "Unverified" in CLASS_LABELS and len(evidence.strip()) < 30:
        return "Unverified"
    return "Real"


# ── Build BM25 index from training corpus ─────────────────────────────────────
print(f"Loading training corpus (cap={CORPUS_LIMIT:,}) ...")
df_corpus = pd.read_csv(TRAIN_FILE, usecols=["text"], low_memory=False).dropna()
df_corpus = df_corpus.head(CORPUS_LIMIT)
corpus_texts = df_corpus["text"].tolist()

print("Tokenizing corpus for BM25 ...")
tokenized_corpus = [tokenize(t) for t in corpus_texts]

print("Building BM25 index ...")
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 index ready — {len(corpus_texts):,} documents")

# ── Load test set ─────────────────────────────────────────────────────────────
print(f"=== Standard RAG — {CLASS_MODE}-class {CLASS_LABELS} ===")
df = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
df = df[df["gold_label"].isin(CLASS_LABELS)].reset_index(drop=True)
print(f"Evaluating {len(df)} test claims ...")

predictions, bm25_scores, retrieval_sources = [], [], []
total = len(df)
for idx, (_, row) in enumerate(df.iterrows()):
    if idx % 300 == 0:
        print(f"  {idx}/{total} done", flush=True)
    claim        = str(row["claim_ar"])
    existing_ev  = str(row.get("evidence_text", "") or "")

    # BM25 retrieval
    query_tokens  = tokenize(claim)
    scores        = bm25.get_scores(query_tokens)
    top_idx       = np.argsort(scores)[::-1][:TOP_K]
    top_score     = float(scores[top_idx[0]]) if len(top_idx) > 0 else 0.0
    retrieved_ev  = " ".join(corpus_texts[i] for i in top_idx)

    # Use BM25 evidence if score is strong enough, else fall back to existing
    if top_score >= BM25_MIN_SCORE:
        evidence_used  = retrieved_ev
        source         = "bm25"
    else:
        evidence_used  = existing_ev if existing_ev.strip() else retrieved_ev
        source         = "existing" if existing_ev.strip() else "bm25_fallback"

    pred = verdict(claim, evidence_used)
    predictions.append(pred)
    bm25_scores.append(round(top_score, 4))
    retrieval_sources.append(source)

df["standard_rag_prediction"] = predictions
df["bm25_top_score"]          = bm25_scores
df["retrieval_source"]        = retrieval_sources
df["is_correct"]              = df["gold_label"] == df["standard_rag_prediction"]

# ── Metrics ───────────────────────────────────────────────────────────────────
y_true = df["gold_label"].tolist()
y_pred = df["standard_rag_prediction"].tolist()

acc    = accuracy_score(y_true, y_pred)
prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)
mf1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
wf1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
mcc    = matthews_corrcoef(y_true, y_pred)
report = classification_report(y_true, y_pred, labels=CLASS_LABELS, zero_division=0)
matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

bm25_hit_rate = (df["retrieval_source"] == "bm25").mean()

print(f"\nStandard RAG (BM25)  Accuracy={acc:.4f}  Macro F1={mf1:.4f}  MCC={mcc:.4f}")
print(f"BM25 hit rate        : {bm25_hit_rate:.2%}")
print(report)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("Standard RAG — BM25 Retrieval Results\n")
    f.write("======================================\n\n")
    f.write(f"Retriever     : BM25Okapi (rank_bm25)\n")
    f.write(f"Corpus size   : {len(corpus_texts):,}\n")
    f.write(f"Top-K         : {TOP_K}\n")
    f.write(f"BM25 hit rate : {bm25_hit_rate:.2%}\n\n")
    f.write(f"Accuracy        : {acc:.4f}\n")
    f.write(f"Macro Precision : {prec:.4f}\n")
    f.write(f"Macro Recall    : {rec:.4f}\n")
    f.write(f"Macro F1        : {mf1:.4f}\n")
    f.write(f"Weighted F1     : {wf1:.4f}\n")
    f.write(f"MCC             : {mcc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix (Real / Fake / Unverified):\n")
    f.write(str(matrix))

print(f"\nResults saved -> {OUTPUT_FILE}")
print(f"Metrics saved -> {RESULTS_FILE}")
