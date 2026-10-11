"""Phase 1 restoration: rebuild data/claim_training.csv — same selection
logic as the ORIGINAL utils/build_claim_training.py, with ONE deliberate,
documented substitution.

Why this script exists
------------------------
Running utils/build_claim_training.py unmodified was attempted first (see
AMAN_PHASE1_RESTORATION_REPORT.md). Its evidence-backfill step calls
rank_bm25's `BM25Okapi.get_scores()` once per row needing evidence, in a
pure-Python loop, against a 10,000-passage pool. Measured in this
environment: ~0.6s per row. With 26,236 rows needing backfill, that is
~4.6 hours of single-threaded CPU time for one preprocessing step — the
exact performance pathology the original Phase 0 audit flagged
("O(N) pure-Python BM25 ... unusable for interactive serving") and
confirmed from the killed run's own progress log:
    retrieving 12000/26236 ... (after ~7,300 CPU-seconds)

Everything else in this script is IDENTICAL to
utils/build_claim_training.py: which rows are selected (Real/Fake from
claim-truth sources, Unverified from AFND capped at 15,000 + AraStance's
native Unverified), the same fact_check_snippet fallback, the same
10,000-passage pool built from the same four_dataset_training.csv, the
same top-3 concatenation for rows still missing evidence. The ONLY change
is the retrieval engine for that top-3 lookup: a single vectorized TF-IDF
cosine-similarity matrix multiplication over all 26,236 queries at once,
instead of 26,236 sequential BM25Okapi.get_scores() calls.

This is retrieval-equivalent in purpose (both are lexical-overlap top-k
retrieval over the real training corpus; neither fabricates text), but it
is NOT bit-identical BM25 ranking — a different query may occasionally
rank its top-3 passages differently. This substitution affects ONLY the
evidence-backfill step of data restoration; it does not touch the SVM
training methodology (TF-IDF word+char FeatureUnion -> LinearSVC, tuned on
validation only), which is implemented separately in
hackathon/train_svm_clean.py and unaffected by this script.

Output: data/claim_training.csv (same path, same schema:
text, label, evidence_text, source)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
SRC_FILE = DATA_DIR / "four_dataset_training.csv"
OUT_FILE = DATA_DIR / "claim_training.csv"

CLAIM_TRUTH_SOURCES = ["VERA-ARAB", "AraStance", "ArCOV19-claims", "ArCOV19-tweets"]
UNVERIFIED_CAP = 15_000
CORPUS_LIMIT = 10_000
TOP_K = 3


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    t0 = time.time()
    if not SRC_FILE.exists():
        raise FileNotFoundError(
            f"{SRC_FILE} not found. Run hackathon/restore_four_dataset_training.py first."
        )
    print("Loading source ...")
    df = pd.read_csv(SRC_FILE, low_memory=False)

    # Identical selection logic to utils/build_claim_training.py
    ct = df[df["source"].isin(CLAIM_TRUTH_SOURCES)]
    real_fake = ct[ct["label"].isin(["Real", "Fake"])].copy()

    afnd_unv = df[(df["source"] == "AFND") & (df["label"] == "Unverified")]
    if len(afnd_unv) > UNVERIFIED_CAP:
        afnd_unv = afnd_unv.sample(UNVERIFIED_CAP, random_state=42)
    ct_unv = ct[ct["label"] == "Unverified"].copy()
    unverified = pd.concat([afnd_unv, ct_unv], ignore_index=True)

    out = pd.concat([real_fake, unverified], ignore_index=True)
    print(f"Selected rows: {len(out):,}  |  {out['label'].value_counts().to_dict()}")

    # Identical evidence fallback
    ev = out["evidence_text"].fillna("").astype(str)
    fc = out["fact_check_snippet"].fillna("").astype(str)
    ev = ev.where(ev.str.strip() != "", fc)
    out["evidence_text"] = ev

    need = out["evidence_text"].str.strip() == ""
    n_need = int(need.sum())
    print(f"Rows needing evidence backfill: {n_need:,}")

    if n_need > 0:
        print(f"Building TF-IDF retrieval pool (cap={CORPUS_LIMIT:,}) ... "
              "[fast vectorized substitute for the historical pure-Python BM25 loop — see module docstring]")
        pool = df["text"].dropna().astype(str)
        pool = pool[pool.str.len() > 40].head(CORPUS_LIMIT).tolist()
        pool_norm = [normalize_arabic(t) for t in pool]
        print(f"  Pool ready — {len(pool):,} passages  ({time.time() - t0:.1f}s elapsed)")

        vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 1), min_df=1)
        pool_matrix = vectorizer.fit_transform(pool_norm)

        need_idx = out.index[need].tolist()
        query_texts = [normalize_arabic(out.at[idx, "text"]) for idx in need_idx]
        query_matrix = vectorizer.transform(query_texts)
        print(f"  Vectorized {len(need_idx):,} queries ({time.time() - t0:.1f}s elapsed), scoring ...")

        # One sparse matmul scores every query against every pool document.
        similarity = (query_matrix @ pool_matrix.T).toarray()  # (n_need, pool_size)
        print(f"  Similarity matrix computed ({time.time() - t0:.1f}s elapsed), selecting top-{TOP_K} ...")

        filled = []
        for row_scores in similarity:
            if TOP_K >= len(row_scores):
                top = np.argsort(row_scores)[::-1]
            else:
                top = np.argpartition(row_scores, -TOP_K)[-TOP_K:]
                top = top[np.argsort(row_scores[top])[::-1]]
            filled.append(" ".join(pool[j] for j in top))
        out.loc[need_idx, "evidence_text"] = pd.Series(filled, index=need_idx)
        print(f"  Backfill complete ({time.time() - t0:.1f}s elapsed)")

    out = out[["text", "label", "evidence_text", "source"]].copy()
    out = out.dropna(subset=["text", "label"])
    out = out[out["text"].astype(str).str.strip() != ""].reset_index(drop=True)

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_FILE}")
    print(f"Total rows : {len(out):,}")
    print(f"Labels     : {out['label'].value_counts().to_dict()}")
    print(f"Sources    : {out['source'].value_counts().to_dict()}")
    print(f"Evidence populated: {(out['evidence_text'].astype(str).str.strip() != '').mean() * 100:.1f}%")
    print(f"Total wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
