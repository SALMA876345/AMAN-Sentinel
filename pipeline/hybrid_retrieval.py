"""Hybrid evidence retrieval for FAAV experiments."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.faav_common import (
    lexical_overlap,
    normalize_arabic,
    recency_gap_days,
    recency_score,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class RetrievalHit:
    text: str
    score: float
    bm25_score: float
    dense_score: float
    row_index: int
    claim_id: str
    source_dataset: str
    source_name: str
    source_credibility: float
    evidence_date: object


def _minmax(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    mn = float(np.min(scores))
    mx = float(np.max(scores))
    if mx <= mn:
        return np.zeros_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


class HybridRetriever:
    """Sparse/dense evidence retrieval over an external or training corpus."""

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        *,
        text_col: str = "retrieval_document",
        mode: str = "hybrid",
        top_k: int = 5,
        corpus_limit: int | None = None,
        bm25_weight: float = 0.50,
        dense_weight: float = 0.50,
        dense_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        if mode not in {"none", "bm25", "tfidf", "dense", "hybrid"}:
            raise ValueError(
                "retrieval mode must be one of: none, bm25, tfidf, dense, hybrid"
            )
        if mode in {"bm25", "hybrid"} and BM25Okapi is None:
            raise ImportError("rank-bm25 is required for BM25 retrieval")
        if mode in {"dense", "hybrid"} and SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for dense retrieval")

        self.mode = mode
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.dense_model_name = dense_model_name
        self.text_col = text_col

        df = corpus_df.copy()
        if corpus_limit:
            df = df.head(corpus_limit).copy()
        if text_col not in df.columns:
            raise ValueError(f"Missing retrieval text column: {text_col}")
        df[text_col] = df[text_col].fillna("").astype(str)
        df = df[df[text_col].str.strip() != ""].reset_index(drop=True)
        self.df = df
        self.texts = df[text_col].tolist()
        self.norm_texts = [normalize_arabic(t) for t in self.texts]
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.dense_model = None
        self.dense_embeddings = None

        if self.mode in {"bm25", "hybrid"}:
            self.bm25 = BM25Okapi([t.split() for t in self.norm_texts])
        if self.mode == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100000,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                lowercase=False,
                norm="l2",
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.norm_texts
            )
        if self.mode in {"dense", "hybrid"}:
            self.dense_model = SentenceTransformer(dense_model_name)
            self.dense_embeddings = self.dense_model.encode(
                self.norm_texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

    def retrieve(self, query: str, *, exclude_claim_ids: Iterable[str] | None = None) -> list[RetrievalHit]:
        if self.mode == "none" or len(self.df) == 0:
            return []
        exclude_claim_ids = set(exclude_claim_ids or [])
        query_norm = normalize_arabic(query)
        bm25_scores = np.zeros(len(self.df), dtype=float)
        dense_scores = np.zeros(len(self.df), dtype=float)
        tfidf_scores = np.zeros(len(self.df), dtype=float)

        if self.bm25 is not None:
            bm25_scores = np.asarray(self.bm25.get_scores(query_norm.split()), dtype=float)
        if self.dense_model is not None and self.dense_embeddings is not None:
            q_emb = self.dense_model.encode(
                [query_norm], normalize_embeddings=True, show_progress_bar=False
            )[0]
            dense_scores = np.asarray(self.dense_embeddings @ q_emb, dtype=float)
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            q_vec = self.tfidf_vectorizer.transform([query_norm])
            tfidf_scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()

        if self.mode == "bm25":
            combined = _minmax(bm25_scores)
        elif self.mode == "tfidf":
            combined = tfidf_scores
        elif self.mode == "dense":
            combined = _minmax(dense_scores)
        else:
            combined = self.bm25_weight * _minmax(bm25_scores) + self.dense_weight * _minmax(dense_scores)

        if exclude_claim_ids and "claim_id" in self.df.columns:
            mask = self.df["claim_id"].isin(exclude_claim_ids).to_numpy()
            combined[mask] = -np.inf

        if self.top_k >= len(combined):
            top_idx = np.argsort(combined)[::-1]
        else:
            candidate_idx = np.argpartition(
                combined,
                -self.top_k,
            )[-self.top_k :]
            top_idx = candidate_idx[
                np.argsort(combined[candidate_idx])[::-1]
            ]
        hits = []
        for idx in top_idx:
            score = float(combined[idx])
            if not np.isfinite(score):
                continue
            row = self.df.iloc[int(idx)]
            hits.append(
                RetrievalHit(
                    text=str(row.get(self.text_col, "")),
                    score=score,
                    bm25_score=float(bm25_scores[idx]),
                    dense_score=float(dense_scores[idx]),
                    row_index=int(idx),
                    claim_id=str(row.get("claim_id", "")),
                    source_dataset=str(row.get("source_dataset", "")),
                    source_name=str(row.get("source_name", "")),
                    source_credibility=float(row.get("source_credibility", 0.5) or 0.5),
                    evidence_date=row.get("evidence_date", pd.NaT),
                )
            )
        return hits


def build_retrieval_corpus(train_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer gold evidence where available; otherwise use the claim as a passage."""
    corpus = train_df.copy()
    evidence = corpus["evidence_text"].fillna("").astype(str)
    fallback = corpus["claim_text"].fillna("").astype(str)
    corpus["retrieval_document"] = evidence.where(evidence.str.strip() != "", fallback)
    return corpus[corpus["retrieval_document"].str.strip() != ""].reset_index(drop=True)


def attach_retrieved_evidence(
    df: pd.DataFrame,
    retriever: HybridRetriever | None,
    *,
    top_k: int = 5,
    exclude_self: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    if retriever is None:
        out["retrieved_evidence"] = ""
        out["retrieval_score"] = 0.0
        out["bm25_score"] = 0.0
        out["dense_score"] = 0.0
        out["retrieved_source_name"] = ""
        out["retrieved_source_dataset"] = ""
        return out

    rows = []
    for _, row in out.iterrows():
        exclude = [row["claim_id"]] if exclude_self else None
        hits = retriever.retrieve(row["claim_text"], exclude_claim_ids=exclude)
        selected = hits[:top_k]
        evidence = " ".join(hit.text for hit in selected)
        best = selected[0] if selected else None
        recency_gap = recency_gap_days(row.get("claim_date", pd.NaT), best.evidence_date if best else pd.NaT)
        rows.append(
            {
                "retrieved_evidence": evidence,
                "retrieval_score": best.score if best else 0.0,
                "bm25_score": best.bm25_score if best else 0.0,
                "dense_score": best.dense_score if best else 0.0,
                "retrieved_source_name": best.source_name if best else "",
                "retrieved_source_dataset": best.source_dataset if best else "",
                "source_credibility": best.source_credibility if best else row.get("source_credibility", 0.5),
                "evidence_date": best.evidence_date if best else row.get("evidence_date", pd.NaT),
                "recency_gap": recency_gap,
                "recency_score": recency_score(recency_gap),
                "faithfulness_score": lexical_overlap(row["claim_text"], evidence),
            }
        )
    retrieved = pd.DataFrame(rows, index=out.index)
    for col in retrieved.columns:
        out[col] = retrieved[col]
    return out


def build_hard_negative_pairs(
    train_df: pd.DataFrame,
    retriever: HybridRetriever,
    *,
    max_examples: int = 2000,
) -> pd.DataFrame:
    """Return high-scoring opposite-label evidence candidates for analysis.

    These rows are not automatically injected into supervised training because
    a retrieved opposite-label article is a hard negative candidate, not a
    guaranteed contradiction. They are saved so a supervised NLI/contrastive
    experiment can opt in after manual inspection.
    """
    rows = []
    sample = train_df.sample(min(len(train_df), max_examples), random_state=42)
    for _, row in sample.iterrows():
        hits = retriever.retrieve(row["claim_text"], exclude_claim_ids=[row["claim_id"]])
        for hit in hits:
            hit_label = train_df.loc[train_df["claim_id"] == hit.claim_id, "label"]
            if len(hit_label) and hit_label.iloc[0] != row["label"]:
                rows.append(
                    {
                        "claim_text": row["claim_text"],
                        "label": row["label"],
                        "hard_negative_evidence": hit.text,
                        "hard_negative_label": hit_label.iloc[0],
                        "retrieval_score": hit.score,
                        "source_dataset": row["source_dataset"],
                    }
                )
                break
    return pd.DataFrame(rows)
