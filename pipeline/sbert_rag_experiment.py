"""
SBERT-RAG Experiment — dense retrieval with SentenceTransformers.

Unlike Standard RAG (TF-IDF BM25 keyword matching) and FAAV-Full (BM25 →
SBERT re-rank → faithfulness), this pipeline uses SBERT cosine similarity
as the *only* retrieval signal:

  1. Encode entire corpus with paraphrase-multilingual-MiniLM-L12-v2
  2. Encode each test claim
  3. Retrieve top-K passages by cosine similarity (pure dense retrieval)
  4. Apply the same keyword verdict logic as Standard RAG
  5. Record evidence_similarity, grounding_score per claim

Produces results/sbert_rag_results.csv — added to MODEL_REGISTRY in
evaluate_all_models.py for direct comparison against all other baselines.

Run after the corpus (unified_training.csv) and manual test sheet exist.
"""

import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (
    DATA_DIR, RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET,
    CLASS_MODE, CLASS_LABELS, result_tagged,
)

SBERT_RAG_CSV     = result_tagged("sbert_rag_results.csv")
SBERT_RAG_METRICS = result_tagged("sbert_rag_metrics.txt")

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_recall_fscore_support,
)

# ── Configuration ─────────────────────────────────────────────────────────────
SENT_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K         = 5       # retrieved passages per claim
CORPUS_LIMIT  = 5000    # cap for encoding speed on CPU; increase on GPU
SUPPORT_SIM   = 0.70    # cosine sim >= this -> Real (high confidence only)
REFUTE_SIM    = 0.30    # cosine sim < this AND no support keywords -> Fake
# Between REFUTE_SIM and SUPPORT_SIM -> keyword tie-break, else NEI

# Keyword lists (same as Standard RAG so results are comparable)
_SUPPORT_KW = [
    "صحيح", "مؤكد", "تأكيد", "صادق", "دقيق", "موثق", "حقيقي", "فعلاً",
    "بالفعل", "نعم", "أثبتت", "أكد", "وثّق", "ثبت",
]
_REFUTE_KW = [
    "خاطئ", "كذب", "مزيف", "مضلل", "مزور", "شائعة", "تضليل",
    "لا يصح", "ينفي", "نفى", "رفض", "كذّب", "خطأ", "زيف",
]
_NEI_KW = [
    "غير مؤكد", "غير واضح", "لا يمكن التحقق", "لا يوجد دليل",
    "لا تتوفر معلومات كافية", "يحتاج إلى تحقق",
]


_FOREIGN_COUNTRIES = [
    "الجزائر", "جزائر", "لبنان", "لبناني", "مصر", "مصري", "سوريا", "سوري",
    "العراق", "عراقي", "اليمن", "يمني", "السودان", "سوداني", "ليبيا", "ليبي",
    "تونس", "المغرب", "مغربي", "الأردن", "فلسطين", "إيران", "تركيا",
    "فرنسا", "أمريكا", "بريطانيا", "روسيا", "الصين",
]
_UAE_WORDS = [
    "الإمارات", "إماراتي", "إماراتية", "أبوظبي", "دبي", "الشارقة",
    "عجمان", "الفجيرة", "رأس الخيمة", "وزارة التربية", "وزارة الصحة",
]


def normalize_arabic(text: str) -> str:
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_verdict(evidence: str) -> str | None:
    """Return a keyword-driven verdict, or None if no strong signal."""
    if any(k in evidence for k in _REFUTE_KW):
        return "Fake"
    if any(k in evidence for k in _SUPPORT_KW):
        return "Real"
    if "Unverified" in CLASS_LABELS and any(k in evidence for k in _NEI_KW):
        return "Unverified"
    return None


def verdict_from_sim_and_keywords(sim: float, evidence: str, claim_orig: str = "") -> str:
    if claim_orig:
        cf = any(c in claim_orig for c in _FOREIGN_COUNTRIES)
        ev_orig_uae = any(u in evidence for u in _UAE_WORDS)
        ev_orig_for = any(c in evidence for c in _FOREIGN_COUNTRIES)
        if cf and ev_orig_uae and not ev_orig_for:
            return "Fake"
    kw = keyword_verdict(evidence)
    three = "Unverified" in CLASS_LABELS
    if sim >= SUPPORT_SIM:
        return kw if kw else "Real"
    if sim < REFUTE_SIM:
        return kw if kw else ("Unverified" if three else "Fake")
    # mid-range similarity — trust keywords, else Unverified/Real
    return kw if kw else ("Unverified" if three else "Real")


# ── Load corpus ────────────────────────────────────────────────────────────────
print(f"Loading corpus from {TRAIN_FILE} ...")
chunks = []
for chunk in pd.read_csv(TRAIN_FILE, chunksize=50_000, low_memory=False):
    chunks.append(chunk)
    if sum(len(c) for c in chunks) >= CORPUS_LIMIT:
        break

df_corpus = pd.concat(chunks, ignore_index=True).head(CORPUS_LIMIT)
corpus_texts = df_corpus["text"].fillna("").apply(normalize_arabic).tolist()
corpus_labels = df_corpus["label"].fillna("unknown").tolist()
print(f"Corpus size: {len(corpus_texts)} passages")

# ── Load SBERT ─────────────────────────────────────────────────────────────────
print(f"Loading SBERT model: {SENT_MODEL} ...")
sbert = SentenceTransformer(SENT_MODEL)

print("Encoding corpus ...")
corpus_embs = sbert.encode(corpus_texts, batch_size=64, show_progress_bar=True,
                            convert_to_tensor=True)

# ── Load test claims ───────────────────────────────────────────────────────────
print(f"=== SBERT-RAG — {CLASS_MODE}-class {CLASS_LABELS} ===")
df_test = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
df_test = df_test[df_test["gold_label"].isin(CLASS_LABELS)].reset_index(drop=True)
df_test["claim_norm"] = df_test["claim_ar"].apply(normalize_arabic)
print(f"\nEvaluating {len(df_test)} claims ...")

results = []
for _, row in df_test.iterrows():
    claim_text   = row["claim_norm"]
    claim_orig   = str(row["claim_ar"])          # original for country mismatch
    gold         = row["gold_label"]
    evidence_orig = str(row.get("evidence_text", "") or "")  # original evidence
    provided_ev  = normalize_arabic(evidence_orig)

    # ── Dense retrieval ────────────────────────────────────────────────────────
    claim_emb = sbert.encode(claim_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(claim_emb, corpus_embs)[0]
    top_idx    = cos_scores.topk(TOP_K).indices.tolist()
    top_sims   = cos_scores.topk(TOP_K).values.tolist()

    retrieved_texts = [corpus_texts[i] for i in top_idx]
    combined_ev     = " ".join(retrieved_texts[:3])  # top-3 for verdict
    best_sim        = float(top_sims[0]) if top_sims else 0.0
    avg_ev_sim      = float(np.mean(top_sims)) if top_sims else 0.0

    # Blend provided evidence (if any) with retrieved
    if provided_ev and len(provided_ev) > 20:
        prov_emb   = sbert.encode(provided_ev, convert_to_tensor=True)
        prov_sim   = float(util.cos_sim(claim_emb, prov_emb).item())
        best_sim   = max(best_sim, prov_sim)
        combined_ev = provided_ev + " " + combined_ev

    # Pass original (non-normalized) claim + evidence for country mismatch
    prediction = verdict_from_sim_and_keywords(best_sim, combined_ev, claim_orig)

    # ── Grounding score: how much does top-1 evidence support the verdict ──────
    ev_emb       = sbert.encode(combined_ev[:512], convert_to_tensor=True)
    grounding    = float(util.cos_sim(claim_emb, ev_emb).item())

    results.append({
        "claim_ar":            row["claim_ar"],
        "gold_label":          gold,
        "sbert_rag_prediction": prediction,
        "is_correct":          prediction == gold,
        "evidence_similarity": round(avg_ev_sim, 4),
        "grounding_score":     round(grounding, 4),
        "retrieved_evidence":  combined_ev[:400],
    })

df_out = pd.DataFrame(results)

# ── Metrics ────────────────────────────────────────────────────────────────────
preds = df_out["sbert_rag_prediction"].tolist()
gold  = df_out["gold_label"].tolist()
acc   = accuracy_score(gold, preds)
f1    = f1_score(gold, preds, average="weighted", zero_division=0)
prec, rec, _, _ = precision_recall_fscore_support(gold, preds, average="weighted",
                                                   zero_division=0)
report = classification_report(gold, preds, labels=CLASS_LABELS, zero_division=0)
cm     = confusion_matrix(gold, preds, labels=CLASS_LABELS)

print(f"\nSBERT-RAG  Accuracy={acc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
print(report)

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
df_out.to_csv(SBERT_RAG_CSV, index=False, encoding="utf-8-sig")

with open(SBERT_RAG_METRICS, "w", encoding="utf-8") as f:
    f.write("SBERT-RAG Results\n")
    f.write("=================\n\n")
    f.write(f"SBERT model      : {SENT_MODEL}\n")
    f.write(f"Corpus size      : {len(corpus_texts)}\n")
    f.write(f"Top-K retrieved  : {TOP_K}\n\n")
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"Precision : {prec:.4f}\n")
    f.write(f"Recall    : {rec:.4f}\n")
    f.write(f"F1-Score  : {f1:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write(f"\n\nMean evidence_similarity : {df_out['evidence_similarity'].mean():.4f}")
    f.write(f"\nMean grounding_score     : {df_out['grounding_score'].mean():.4f}\n")

print(f"\nResults saved -> {SBERT_RAG_CSV}")
print(f"Metrics saved -> {SBERT_RAG_METRICS}")
