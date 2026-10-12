"""Evidence retrieval for AMAN Sentinel Phase 3.

Claims that the Phase 2 SVM router escalates to ESCALATE_FOR_DEEP_VERIFICATION
need an "Evidence Pack" of supporting passages. This module serves that need
by wrapping the EXISTING `HybridRetriever` class from
`pipeline/hybrid_retrieval.py` (BM25 via rank_bm25.BM25Okapi, dense via
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 -- the same model
already used by `phase2/novelty_detector.py`). It does NOT reimplement BM25 or
the embedding call; it only adds:

  * a corpus builder restricted to TRAIN + VALIDATION evidence (TEST evidence
    is never indexed -- see phase3/evaluate_retrieval.py for why),
  * on-disk caching of the (expensive) dense corpus embeddings so the index
    does not need to be re-encoded on every process start, and
  * Reciprocal Rank Fusion (RRF) as a second hybrid-fusion option alongside
    the weighted min-max fusion already built into HybridRetriever.

Index artifacts persisted under phase3/data/retrieval_index/:
  - corpus.csv            deduped evidence passages (doc_id, text, source, sample_id)
  - dense_embeddings.npy  cached dense embeddings for corpus.csv, in row order
  - index_meta.json       corpus hash + model name + RRF k, used to validate the cache
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.hybrid_retrieval import HybridRetriever, _minmax  # noqa: E402
from utils.faav_common import normalize_arabic  # noqa: E402

CLEAN_SPLIT_DIR = os.path.join(_REPO_ROOT, "hackathon", "data", "clean_split")
TRAIN_CSV = os.path.join(CLEAN_SPLIT_DIR, "source_binary_train.csv")
VAL_CSV = os.path.join(CLEAN_SPLIT_DIR, "source_binary_validation.csv")

_PHASE3_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(_PHASE3_DIR, "data", "retrieval_index")
CORPUS_CSV = os.path.join(INDEX_DIR, "corpus.csv")
EMBED_NPY = os.path.join(INDEX_DIR, "dense_embeddings.npy")
META_JSON = os.path.join(INDEX_DIR, "index_meta.json")

DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60  # standard literature default (Cormack et al. 2009); not tuned here
TEXT_COL = "evidence_text"


def build_corpus(train_path: str = TRAIN_CSV, val_path: str = VAL_CSV) -> pd.DataFrame:
    """Evidence corpus = TRAIN + VALIDATION evidence passages only, deduped.

    TEST evidence is intentionally excluded -- indexing it would leak the
    answer into retrieval for the held-out evaluation in
    evaluate_retrieval.py. Dedup key is `evidence_norm` (identical normalized
    text); first occurrence's raw text/source/sample_id is kept as the
    representative row.
    """
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    both = pd.concat([train, val], ignore_index=True)
    both["evidence_text"] = both["evidence_text"].fillna("").astype(str)
    both["evidence_norm"] = both["evidence_norm"].fillna("").astype(str)
    both = both[both["evidence_norm"].str.strip() != ""]
    both = both.drop_duplicates(subset=["evidence_norm"], keep="first").reset_index(drop=True)
    corpus = both[["evidence_text", "evidence_norm", "source", "sample_id"]].copy()
    corpus.insert(0, "doc_id", [f"ev_{i:05d}" for i in range(len(corpus))])
    return corpus


def _corpus_hash(corpus: pd.DataFrame) -> str:
    joined = "||".join(corpus["evidence_norm"].tolist())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _build_and_cache_retriever(top_k: int = 10) -> HybridRetriever:
    os.makedirs(INDEX_DIR, exist_ok=True)
    corpus = build_corpus()
    corpus_hash = _corpus_hash(corpus)

    cache_ok = os.path.exists(EMBED_NPY) and os.path.exists(META_JSON)
    meta = None
    if cache_ok:
        with open(META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cache_ok = (
            meta.get("corpus_hash") == corpus_hash
            and meta.get("dense_model_name") == DENSE_MODEL_NAME
            and meta.get("n_docs") == len(corpus)
        )

    if cache_ok:
        # Fast path: build the (cheap) BM25 index fresh, but skip re-encoding
        # the corpus with the sentence-transformer -- load the cached array
        # and only load the model itself (needed to encode incoming queries).
        retriever = HybridRetriever(corpus, text_col=TEXT_COL, mode="bm25", top_k=top_k)
        from sentence_transformers import SentenceTransformer

        retriever.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        retriever.dense_embeddings = np.load(EMBED_NPY)
        retriever.dense_model_name = DENSE_MODEL_NAME
        retriever.mode = "hybrid"
    else:
        retriever = HybridRetriever(
            corpus, text_col=TEXT_COL, mode="hybrid", top_k=top_k, dense_model_name=DENSE_MODEL_NAME
        )
        corpus.to_csv(CORPUS_CSV, index=False, encoding="utf-8-sig")
        np.save(EMBED_NPY, retriever.dense_embeddings)
        with open(META_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "corpus_hash": corpus_hash,
                    "dense_model_name": DENSE_MODEL_NAME,
                    "n_docs": len(corpus),
                    "rrf_k": RRF_K,
                },
                f,
                indent=2,
            )
    return retriever


_RETRIEVER: HybridRetriever | None = None


def get_retriever(top_k: int = 10) -> HybridRetriever:
    """Singleton accessor: builds/loads the index once per process."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = _build_and_cache_retriever(top_k=top_k)
    return _RETRIEVER


def raw_scores(retriever: HybridRetriever, query_text: str) -> tuple[np.ndarray, np.ndarray]:
    """BM25 and dense similarity scores for `query_text` against every corpus
    document, using the retriever's already-built BM25 index / dense model /
    cached dense embeddings (no reimplementation of either algorithm)."""
    query_norm = normalize_arabic(query_text)
    n = len(retriever.df)
    bm25_scores = np.zeros(n, dtype=float)
    dense_scores = np.zeros(n, dtype=float)
    if retriever.bm25 is not None:
        bm25_scores = np.asarray(retriever.bm25.get_scores(query_norm.split()), dtype=float)
    if retriever.dense_model is not None and retriever.dense_embeddings is not None:
        q_emb = retriever.dense_model.encode(
            [query_norm], normalize_embeddings=True, show_progress_bar=False
        )[0]
        dense_scores = np.asarray(retriever.dense_embeddings @ q_emb, dtype=float)
    return bm25_scores, dense_scores


def rrf_fuse(bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)), rank 1-indexed."""
    n = len(bm25_scores)
    bm25_rank = np.empty(n, dtype=int)
    bm25_rank[np.argsort(-bm25_scores)] = np.arange(1, n + 1)
    dense_rank = np.empty(n, dtype=int)
    dense_rank[np.argsort(-dense_scores)] = np.arange(1, n + 1)
    return 1.0 / (k + bm25_rank) + 1.0 / (k + dense_rank)


def weighted_fuse(retriever: HybridRetriever, bm25_scores: np.ndarray, dense_scores: np.ndarray) -> np.ndarray:
    """The EXISTING min-max weighted fusion from HybridRetriever(mode='hybrid')."""
    return retriever.bm25_weight * _minmax(bm25_scores) + retriever.dense_weight * _minmax(dense_scores)


def _topk_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if top_k >= len(scores):
        return list(np.argsort(scores)[::-1])
    cand = np.argpartition(scores, -top_k)[-top_k:]
    return list(cand[np.argsort(scores[cand])[::-1]])


_METHOD_TAGS = {
    "bm25": "bm25",
    "dense": "dense",
    "hybrid": "hybrid_weighted",
    "hybrid_rrf": "hybrid_rrf",
}


def retrieve_evidence(claim_text: str, top_k: int = 5, method: str = "hybrid") -> dict:
    """Retrieve an Evidence Pack for one claim.

    method: "bm25" | "dense" | "hybrid" (existing weighted min-max fusion)
            | "hybrid_rrf" (Reciprocal Rank Fusion, k=RRF_K)

    Returns:
        {"claim": claim_text,
         "retrieved_evidence": [
             {"id", "text", "source", "retrieval_method", "retrieval_score"}, ...
         ]}
    """
    if method not in _METHOD_TAGS:
        raise ValueError(f"method must be one of {sorted(_METHOD_TAGS)}")

    retriever = get_retriever(top_k=top_k)
    bm25_scores, dense_scores = raw_scores(retriever, claim_text)

    if method == "bm25":
        combined = bm25_scores
    elif method == "dense":
        combined = dense_scores
    elif method == "hybrid":
        combined = weighted_fuse(retriever, bm25_scores, dense_scores)
    else:
        combined = rrf_fuse(bm25_scores, dense_scores, k=RRF_K)

    tag = _METHOD_TAGS[method]
    top_idx = _topk_indices(combined, top_k)
    results = []
    for idx in top_idx:
        row = retriever.df.iloc[int(idx)]
        results.append(
            {
                "id": str(row.get("doc_id", idx)),
                "text": str(row.get(TEXT_COL, "")),
                "source": str(row.get("source", "")),
                "retrieval_method": tag,
                "retrieval_score": float(combined[idx]),
            }
        )
    return {"claim": claim_text, "retrieved_evidence": results}


if __name__ == "__main__":
    demo = retrieve_evidence("الحكومة تعلن عن قرار جديد", top_k=3, method="hybrid_rrf")
    print(json.dumps(demo, ensure_ascii=False, indent=2))
