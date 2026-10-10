"""Run final-study RAG baselines on the frozen Assignment 2 splits.

This runner evaluates only the retrieval-grounded baselines requested for the
final report:

* Standard BM25 RAG: BM25 top-k retrieval over the training evidence corpus,
  followed by weighted label voting from retrieved training evidence.
* SBERT-RAG: dense SentenceTransformer retrieval, followed by the same weighted
  evidence-label voting.
* FAAV-Light RAG: BM25 retrieval plus a validation-tuned transparent
  faithfulness gate using retrieval confidence, claim/evidence overlap, and
  evidence length.

The visible study names intentionally follow the final report numbering:
Study 1 = source-domain, Study 2 = combined source-UAE adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import lexical_overlap, normalize_arabic


DATA_DIR = ROOT / "data" / "final_studies"
OUTPUT_DIR = ROOT / "results" / "final_studies" / "rag_baselines"
CACHE_DIR = OUTPUT_DIR / "cache"
SEED = 42
DEFAULT_TOP_K = 5
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LABELS_BY_TASK = {
    "binary": ["Real", "Fake"],
    "three_class": ["Real", "Fake", "Unverified"],
}

SPLITS = [
    {
        "study": "Study 1 - Source-domain",
        "task": "binary",
        "stem": "source_binary",
    },
    {
        "study": "Study 1 - Source-domain",
        "task": "three_class",
        "stem": "source_three_class",
    },
    {
        "study": "Study 2 - Combined adaptation",
        "task": "binary",
        "stem": "combined_binary",
    },
    {
        "study": "Study 2 - Combined adaptation",
        "task": "three_class",
        "stem": "combined_three_class",
    },
]


@dataclass
class RetrievalPack:
    frame: pd.DataFrame
    labels: list[str]
    predictions: np.ndarray
    confidences: np.ndarray
    retrieval_scores: np.ndarray
    retrieval_overlaps: np.ndarray
    evidence_chars: np.ndarray
    retrieved_evidence: list[str]
    retrieved_sources: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Standard BM25 RAG, SBERT-RAG, and FAAV-Light RAG on final studies."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["standard_bm25", "sbert", "faav_light"],
        choices=["standard_bm25", "sbert", "faav_light"],
        help="RAG methods to run.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=0,
        help="Optional cap for retrieval corpus rows. 0 means full train split.",
    )
    parser.add_argument("--sbert-model", default=SBERT_MODEL)
    parser.add_argument("--sbert-batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument(
        "--allow-sbert-download",
        action="store_true",
        help="Allow SentenceTransformer to fetch the model if not cached.",
    )
    return parser.parse_args()


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def read_split(stem: str, part: str, labels: list[str]) -> pd.DataFrame:
    path = DATA_DIR / f"{stem}_{part}.csv"
    frame = pd.read_csv(path, low_memory=False).fillna("")
    frame = frame[frame["task_label"].isin(labels)].copy()
    frame["label"] = frame["task_label"]
    frame["claim_norm_final"] = frame["claim_norm"].where(
        frame["claim_norm"].astype(str).str.strip() != "",
        frame["claim_text"].map(normalize_arabic),
    )
    frame["evidence_norm_final"] = frame["evidence_norm"].where(
        frame["evidence_norm"].astype(str).str.strip() != "",
        frame["evidence_text"].map(normalize_arabic),
    )
    return frame.reset_index(drop=True)


def fingerprint(frame: pd.DataFrame) -> str:
    payload = "\n".join(sorted(frame["sample_id"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_corpus(train: pd.DataFrame, corpus_limit: int) -> pd.DataFrame:
    corpus = train.copy()
    claim = corpus["claim_norm_final"].astype(str)
    evidence = corpus["evidence_norm_final"].astype(str)
    evidence_raw = corpus["evidence_text"].astype(str)
    claim_raw = corpus["claim_text"].astype(str)
    corpus["search_text"] = (claim + " " + evidence).str.strip()
    corpus["retrieval_document"] = evidence_raw.where(
        evidence_raw.str.strip() != "", claim_raw
    )
    corpus = corpus[corpus["search_text"].str.strip() != ""].copy()
    corpus = corpus.drop_duplicates(
        subset=["search_text", "retrieval_document", "label"]
    ).reset_index(drop=True)
    if corpus_limit and corpus_limit > 0:
        corpus = corpus.head(corpus_limit).copy()
    return corpus.reset_index(drop=True)


def tokenize(text: object) -> list[str]:
    return normalize_arabic(text).split()


class FastBM25Index:
    """Sparse BM25 index with batched query scoring.

    The earlier project script used rank_bm25, which is simple but slow because
    each query loops over the corpus in Python. This keeps the same BM25 formula
    but moves scoring into sparse matrix multiplication.
    """

    def __init__(
        self,
        texts: pd.Series,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.vectorizer = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            min_df=1,
        )
        counts = self.vectorizer.fit_transform(texts.astype(str).tolist()).tocsr()
        n_docs = counts.shape[0]
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        avg_doc_len = float(doc_len.mean()) if len(doc_len) else 1.0
        avg_doc_len = avg_doc_len if avg_doc_len > 0 else 1.0
        df = np.diff(counts.tocsc().indptr)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        coo = counts.tocoo(copy=True)
        denom = coo.data + k1 * (1.0 - b + b * doc_len[coo.row] / avg_doc_len)
        data = idf[coo.col] * ((coo.data * (k1 + 1.0)) / denom)
        self.matrix = sparse.csr_matrix(
            (data.astype(np.float32), (coo.row, coo.col)),
            shape=counts.shape,
        )

    def score_batch(self, queries: list[str]):
        query_matrix = self.vectorizer.transform(queries)
        query_matrix.data = np.ones_like(query_matrix.data)
        return query_matrix @ self.matrix.T


def top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=int)
    k = min(top_k, len(scores))
    if k == len(scores):
        idx = np.argsort(scores)[::-1]
    else:
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(int)


def weighted_vote(
    labels: np.ndarray,
    scores: np.ndarray,
    class_labels: list[str],
) -> tuple[str, float]:
    weights = np.maximum(scores.astype(float), 0.0)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights, dtype=float)
    totals = {label: 0.0 for label in class_labels}
    for label, weight in zip(labels, weights):
        if label in totals:
            totals[str(label)] += float(weight)
    best_label = max(class_labels, key=lambda label: (totals[label], -class_labels.index(label)))
    total_weight = sum(totals.values())
    confidence = totals[best_label] / total_weight if total_weight else 0.0
    return best_label, float(confidence)


def pack_outputs(
    frame: pd.DataFrame,
    labels: list[str],
    predictions: list[str],
    confidences: list[float],
    retrieval_scores: list[float],
    retrieval_overlaps: list[float],
    evidence_chars: list[int],
    retrieved_evidence: list[str],
    retrieved_sources: list[str],
) -> RetrievalPack:
    return RetrievalPack(
        frame=frame.copy(),
        labels=labels,
        predictions=np.asarray(predictions, dtype=object),
        confidences=np.asarray(confidences, dtype=float),
        retrieval_scores=np.asarray(retrieval_scores, dtype=float),
        retrieval_overlaps=np.asarray(retrieval_overlaps, dtype=float),
        evidence_chars=np.asarray(evidence_chars, dtype=int),
        retrieved_evidence=retrieved_evidence,
        retrieved_sources=retrieved_sources,
    )


def retrieve_bm25(
    corpus: pd.DataFrame,
    frame: pd.DataFrame,
    labels: list[str],
    *,
    top_k: int,
    run_label: str,
) -> RetrievalPack:
    bm25 = FastBM25Index(corpus["search_text"])
    corpus_labels = corpus["label"].astype(str).to_numpy()
    docs = corpus["retrieval_document"].astype(str).to_numpy()
    sources = corpus["source"].astype(str).to_numpy() if "source" in corpus.columns else np.array([""] * len(corpus))

    predictions: list[str] = []
    confidences: list[float] = []
    retrieval_scores: list[float] = []
    retrieval_overlaps: list[float] = []
    evidence_chars: list[int] = []
    retrieved_evidence: list[str] = []
    retrieved_sources: list[str] = []

    total = len(frame)
    start = time.time()
    query_batch_size = 256
    queries = frame["claim_norm_final"].astype(str).tolist()
    for batch_start in range(0, total, query_batch_size):
        batch_end = min(batch_start + query_batch_size, total)
        if batch_start:
            elapsed = time.time() - start
            print(
                f"  {run_label}: {batch_start:,}/{total:,} rows ({elapsed/60:.1f} min)",
                flush=True,
            )
        score_matrix = bm25.score_batch(queries[batch_start:batch_end]).toarray()
        for row_offset in range(score_matrix.shape[0]):
            global_idx = batch_start + row_offset
            row = frame.iloc[global_idx]
            scores = score_matrix[row_offset]
            top_idx = top_k_indices(np.asarray(scores, dtype=float), top_k)
            top_scores = np.asarray(scores, dtype=float)[top_idx] if len(top_idx) else np.array([])
            top_labels = corpus_labels[top_idx] if len(top_idx) else np.array([], dtype=object)
            if len(top_labels):
                pred, confidence = weighted_vote(top_labels, top_scores, labels)
                evidence = " ".join(docs[top_idx].tolist())
                source = " | ".join(dict.fromkeys(sources[top_idx].tolist()))
                top_score = float(top_scores[0])
            else:
                pred = labels[0]
                confidence = 0.0
                evidence = ""
                source = ""
                top_score = 0.0
            predictions.append(pred)
            confidences.append(confidence)
            retrieval_scores.append(top_score)
            retrieval_overlaps.append(lexical_overlap(row["claim_text"], evidence))
            evidence_chars.append(len(evidence))
            retrieved_evidence.append(evidence[:1200])
            retrieved_sources.append(source[:250])

    return pack_outputs(
        frame,
        labels,
        predictions,
        confidences,
        retrieval_scores,
        retrieval_overlaps,
        evidence_chars,
        retrieved_evidence,
        retrieved_sources,
    )


def corpus_hash(corpus: pd.DataFrame, model_name: str) -> str:
    key_frame = corpus[["sample_id", "search_text", "label"]].copy()
    hashes = pd.util.hash_pandas_object(key_frame, index=False).values.tobytes()
    return hashlib.sha256(model_name.encode("utf-8") + hashes).hexdigest()[:20]


def load_sentence_model(model_name: str, allow_download: bool):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=not allow_download)
    except TypeError:
        return SentenceTransformer(model_name)


def corpus_embeddings(
    corpus: pd.DataFrame,
    *,
    model,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"sbert_corpus_{corpus_hash(corpus, model_name)}.npy"
    if cache_path.exists():
        print(f"  Loading cached SBERT corpus embeddings: {cache_path.name}", flush=True)
        return np.load(cache_path)
    print(f"  Encoding SBERT corpus ({len(corpus):,} rows) ...", flush=True)
    embeddings = model.encode(
        corpus["search_text"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(cache_path, embeddings)
    return embeddings


def retrieve_sbert(
    corpus: pd.DataFrame,
    frame: pd.DataFrame,
    labels: list[str],
    *,
    top_k: int,
    model_name: str,
    batch_size: int,
    query_batch_size: int,
    allow_download: bool,
    run_label: str,
) -> RetrievalPack:
    print(f"  Loading SBERT model: {model_name}", flush=True)
    model = load_sentence_model(model_name, allow_download)
    corpus_emb = corpus_embeddings(
        corpus,
        model=model,
        model_name=model_name,
        batch_size=batch_size,
    )
    corpus_labels = corpus["label"].astype(str).to_numpy()
    docs = corpus["retrieval_document"].astype(str).to_numpy()
    sources = corpus["source"].astype(str).to_numpy() if "source" in corpus.columns else np.array([""] * len(corpus))

    predictions: list[str] = []
    confidences: list[float] = []
    retrieval_scores: list[float] = []
    retrieval_overlaps: list[float] = []
    evidence_chars: list[int] = []
    retrieved_evidence: list[str] = []
    retrieved_sources: list[str] = []

    queries = frame["claim_norm_final"].astype(str).tolist()
    total = len(frame)
    start = time.time()
    for batch_start in range(0, total, query_batch_size):
        batch_end = min(batch_start + query_batch_size, total)
        if batch_start:
            elapsed = time.time() - start
            print(f"  {run_label}: {batch_start:,}/{total:,} rows ({elapsed/60:.1f} min)", flush=True)
        q_emb = model.encode(
            queries[batch_start:batch_end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sim = q_emb @ corpus_emb.T
        for row_offset in range(sim.shape[0]):
            global_idx = batch_start + row_offset
            row_scores = sim[row_offset]
            top_idx = top_k_indices(row_scores, top_k)
            top_scores = row_scores[top_idx]
            top_labels = corpus_labels[top_idx]
            pred, confidence = weighted_vote(top_labels, top_scores, labels)
            evidence = " ".join(docs[top_idx].tolist())
            source = " | ".join(dict.fromkeys(sources[top_idx].tolist()))
            predictions.append(pred)
            confidences.append(confidence)
            retrieval_scores.append(float(top_scores[0]) if len(top_scores) else 0.0)
            retrieval_overlaps.append(
                lexical_overlap(frame.iloc[global_idx]["claim_text"], evidence)
            )
            evidence_chars.append(len(evidence))
            retrieved_evidence.append(evidence[:1200])
            retrieved_sources.append(source[:250])

    return pack_outputs(
        frame,
        labels,
        predictions,
        confidences,
        retrieval_scores,
        retrieval_overlaps,
        evidence_chars,
        retrieved_evidence,
        retrieved_sources,
    )


def apply_faav_light(
    pack: RetrievalPack,
    *,
    task: str,
    confidence_threshold: float,
    overlap_threshold: float,
    evidence_char_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    accepted = (
        (pack.confidences >= confidence_threshold)
        & (pack.retrieval_overlaps >= overlap_threshold)
        & (pack.evidence_chars >= evidence_char_threshold)
    )
    if task == "binary":
        predictions = pack.predictions.copy()
    else:
        predictions = np.where(
            pack.predictions == "Unverified",
            "Unverified",
            np.where(accepted, pack.predictions, "Unverified"),
        )
    return predictions.astype(object), accepted


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro Precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "Macro Recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "Macro F1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "Weighted F1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    for label in ["Real", "Fake", "Unverified"]:
        if label in labels:
            metrics[f"{label} Precision"] = float(
                precision_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
            )
            metrics[f"{label} Recall"] = float(
                recall_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
            )
            metrics[f"{label} F1"] = float(
                f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
            )
        else:
            metrics[f"{label} Precision"] = math.nan
            metrics[f"{label} Recall"] = math.nan
            metrics[f"{label} F1"] = math.nan
    return metrics


def annotated_support_accuracy(frame: pd.DataFrame, predicted_supported: np.ndarray) -> float:
    if "retrieval_quality" not in frame.columns:
        return math.nan
    quality = frame["retrieval_quality"].astype(str).str.strip()
    mask = quality.ne("") & quality.ne("nan")
    if not bool(mask.any()):
        return math.nan
    gold_supported = quality.isin(["Relevant", "Partially Relevant"])
    return float((predicted_supported[mask.to_numpy()] == gold_supported[mask]).mean())


def write_predictions(
    *,
    pack: RetrievalPack,
    study: str,
    task: str,
    model_name: str,
    predictions: np.ndarray,
    accepted: np.ndarray | None,
    settings: dict | None,
) -> Path:
    slug = (
        f"{study}_{task}_{model_name}"
        .lower()
        .replace(" - ", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    path = OUTPUT_DIR / f"{slug}_predictions.csv"
    out = pack.frame[
        [
            "sample_id",
            "dataset_family",
            "source",
            "claim_text",
            "evidence_text",
            "label",
        ]
    ].copy()
    for optional in ["retrieval_quality", "faithfulness_label", "evidence_url"]:
        if optional in pack.frame.columns:
            out[optional] = pack.frame[optional]
    out["prediction"] = predictions
    out["is_correct"] = out["label"].values == predictions
    out["retrieval_confidence"] = pack.confidences
    out["retrieval_top_score"] = pack.retrieval_scores
    out["retrieval_overlap"] = pack.retrieval_overlaps
    out["retrieved_evidence_chars"] = pack.evidence_chars
    out["retrieved_sources"] = pack.retrieved_sources
    out["retrieved_evidence"] = pack.retrieved_evidence
    if accepted is not None:
        out["faithfulness_status"] = np.where(accepted, "Verified", "Unverified")
    if settings:
        out["settings_json"] = json.dumps(settings, ensure_ascii=True, sort_keys=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def result_row(
    *,
    study: str,
    task: str,
    model_name: str,
    train_n: int,
    validation_n: int,
    test_n: int,
    corpus_n: int,
    top_k: int,
    pack: RetrievalPack,
    predictions: np.ndarray,
    accepted: np.ndarray | None = None,
    settings: dict | None = None,
) -> dict:
    metrics = classification_metrics(pack.frame["label"], predictions, pack.labels)
    predicted_supported = (
        (pack.retrieval_overlaps >= 0.10)
        & (pack.evidence_chars >= 40)
        & (pack.confidences >= 0.50)
    )
    if accepted is None:
        if task == "three_class":
            abstention = float(np.mean(predictions == "Unverified"))
        else:
            abstention = 0.0
        faithfulness_score = float(np.mean(predicted_supported))
        coverage = 1.0 - abstention
    else:
        abstention = float(1.0 - np.mean(accepted))
        faithfulness_score = float(np.mean(accepted))
        coverage = float(np.mean(accepted))
    row = {
        "Study": study,
        "Task": task,
        "Model": model_name,
        "Train N": train_n,
        "Validation N": validation_n,
        "Test N": test_n,
        "Corpus N": corpus_n,
        "Top K": top_k,
        **metrics,
        "Evidence Relevance": float(np.mean(pack.retrieval_overlaps)),
        "Faithfulness Score": faithfulness_score,
        "Hallucination Rate": float(1.0 - faithfulness_score),
        "Citation Support Accuracy": annotated_support_accuracy(pack.frame, predicted_supported),
        "Abstention Rate": abstention,
        "Faithfulness Coverage": coverage,
        "Mean Retrieval Confidence": float(np.mean(pack.confidences)),
        "Mean Retrieval Score": float(np.mean(pack.retrieval_scores)),
        "Test Fingerprint": fingerprint(pack.frame),
    }
    if settings:
        row["Settings"] = json.dumps(settings, ensure_ascii=True, sort_keys=True)
    else:
        row["Settings"] = ""
    return row


def tune_faav_light(
    validation_pack: RetrievalPack,
    *,
    task: str,
) -> dict:
    best: dict | None = None
    sweep: list[dict] = []
    for confidence_threshold in [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        for overlap_threshold in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            for evidence_char_threshold in [0, 20, 40, 80, 120]:
                predictions, accepted = apply_faav_light(
                    validation_pack,
                    task=task,
                    confidence_threshold=confidence_threshold,
                    overlap_threshold=overlap_threshold,
                    evidence_char_threshold=evidence_char_threshold,
                )
                if task == "binary":
                    coverage = float(np.mean(accepted))
                    if accepted.any():
                        macro_f1 = float(
                            f1_score(
                                validation_pack.frame.loc[accepted, "label"],
                                validation_pack.predictions[accepted],
                                labels=validation_pack.labels,
                                average="macro",
                                zero_division=0,
                            )
                        )
                        mcc = float(
                            matthews_corrcoef(
                                validation_pack.frame.loc[accepted, "label"],
                                validation_pack.predictions[accepted],
                            )
                        )
                    else:
                        macro_f1 = 0.0
                        mcc = 0.0
                    score = (
                        macro_f1 if coverage >= 0.80 else -1.0,
                        mcc,
                        coverage,
                    )
                    accuracy = float(accuracy_score(validation_pack.frame["label"], predictions))
                else:
                    metrics = classification_metrics(
                        validation_pack.frame["label"], predictions, validation_pack.labels
                    )
                    coverage = float(np.mean(accepted))
                    macro_f1 = metrics["Macro F1"]
                    mcc = metrics["MCC"]
                    accuracy = metrics["Accuracy"]
                    score = (macro_f1, mcc, accuracy)
                row = {
                    "confidence_threshold": confidence_threshold,
                    "overlap_threshold": overlap_threshold,
                    "evidence_char_threshold": evidence_char_threshold,
                    "validation_accuracy": accuracy,
                    "validation_macro_f1": macro_f1,
                    "validation_mcc": mcc,
                    "validation_coverage": coverage,
                    "score": score,
                }
                sweep.append(row)
                if best is None or row["score"] > best["score"]:
                    best = row
    if best is None:
        raise RuntimeError("No FAAV-Light threshold candidate completed.")
    selected = {k: v for k, v in best.items() if k != "score"}
    selected["sweep_size"] = len(sweep)
    return selected


def save_metrics_json(row: dict, output_path: Path) -> None:
    metrics_path = output_path.with_name(output_path.name.replace("_predictions.csv", "_metrics.json"))
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)


def run_split(config: dict, args: argparse.Namespace) -> list[dict]:
    labels = LABELS_BY_TASK[config["task"]]
    train = read_split(config["stem"], "train", labels)
    validation = read_split(config["stem"], "validation", labels)
    test = read_split(config["stem"], "test", labels)
    corpus = build_corpus(train, args.corpus_limit)
    print(
        f"\n{config['study']} | {config['task']}: train={len(train):,}, "
        f"validation={len(validation):,}, test={len(test):,}, corpus={len(corpus):,}",
        flush=True,
    )

    rows: list[dict] = []
    bm25_validation: RetrievalPack | None = None
    bm25_test: RetrievalPack | None = None

    if "standard_bm25" in args.methods or "faav_light" in args.methods:
        print("Building/running BM25 retrieval ...", flush=True)
        bm25_validation = retrieve_bm25(
            corpus,
            validation,
            labels,
            top_k=args.top_k,
            run_label=f"{config['study']} {config['task']} BM25 validation",
        )
        bm25_test = retrieve_bm25(
            corpus,
            test,
            labels,
            top_k=args.top_k,
            run_label=f"{config['study']} {config['task']} BM25 test",
        )

    if "standard_bm25" in args.methods and bm25_test is not None:
        model_name = "Standard BM25 RAG"
        path = write_predictions(
            pack=bm25_test,
            study=config["study"],
            task=config["task"],
            model_name=model_name,
            predictions=bm25_test.predictions,
            accepted=None,
            settings={"retriever": "BM25Okapi", "top_k": args.top_k},
        )
        row = result_row(
            study=config["study"],
            task=config["task"],
            model_name=model_name,
            train_n=len(train),
            validation_n=0,
            test_n=len(test),
            corpus_n=len(corpus),
            top_k=args.top_k,
            pack=bm25_test,
            predictions=bm25_test.predictions,
            settings={"retriever": "BM25Okapi", "top_k": args.top_k},
        )
        save_metrics_json(row, path)
        rows.append(row)
        print(
            f"  {model_name}: Accuracy={row['Accuracy']:.4f}, Macro F1={row['Macro F1']:.4f}",
            flush=True,
        )

    if "faav_light" in args.methods and bm25_validation is not None and bm25_test is not None:
        model_name = "FAAV-Light RAG"
        settings = tune_faav_light(bm25_validation, task=config["task"])
        predictions, accepted = apply_faav_light(
            bm25_test,
            task=config["task"],
            confidence_threshold=float(settings["confidence_threshold"]),
            overlap_threshold=float(settings["overlap_threshold"]),
            evidence_char_threshold=int(settings["evidence_char_threshold"]),
        )
        settings = {
            **settings,
            "retriever": "BM25Okapi",
            "top_k": args.top_k,
            "gate": "confidence+overlap+evidence_length",
        }
        path = write_predictions(
            pack=bm25_test,
            study=config["study"],
            task=config["task"],
            model_name=model_name,
            predictions=predictions,
            accepted=accepted,
            settings=settings,
        )
        row = result_row(
            study=config["study"],
            task=config["task"],
            model_name=model_name,
            train_n=len(train),
            validation_n=len(validation),
            test_n=len(test),
            corpus_n=len(corpus),
            top_k=args.top_k,
            pack=bm25_test,
            predictions=predictions,
            accepted=accepted,
            settings=settings,
        )
        save_metrics_json(row, path)
        rows.append(row)
        print(
            f"  {model_name}: Accuracy={row['Accuracy']:.4f}, Macro F1={row['Macro F1']:.4f}, "
            f"Coverage={row['Faithfulness Coverage']:.4f}",
            flush=True,
        )

    if "sbert" in args.methods:
        model_name = "SBERT-RAG"
        try:
            sbert_test = retrieve_sbert(
                corpus,
                test,
                labels,
                top_k=args.top_k,
                model_name=args.sbert_model,
                batch_size=args.sbert_batch_size,
                query_batch_size=args.query_batch_size,
                allow_download=args.allow_sbert_download,
                run_label=f"{config['study']} {config['task']} SBERT test",
            )
            path = write_predictions(
                pack=sbert_test,
                study=config["study"],
                task=config["task"],
                model_name=model_name,
                predictions=sbert_test.predictions,
                accepted=None,
                settings={
                    "retriever": "SentenceTransformer",
                    "model": args.sbert_model,
                    "top_k": args.top_k,
                },
            )
            row = result_row(
                study=config["study"],
                task=config["task"],
                model_name=model_name,
                train_n=len(train),
                validation_n=0,
                test_n=len(test),
                corpus_n=len(corpus),
                top_k=args.top_k,
                pack=sbert_test,
                predictions=sbert_test.predictions,
                settings={
                    "retriever": "SentenceTransformer",
                    "model": args.sbert_model,
                    "top_k": args.top_k,
                },
            )
            save_metrics_json(row, path)
            rows.append(row)
            print(
                f"  {model_name}: Accuracy={row['Accuracy']:.4f}, Macro F1={row['Macro F1']:.4f}",
                flush=True,
            )
        except Exception as exc:
            error_row = {
                "Study": config["study"],
                "Task": config["task"],
                "Model": model_name,
                "Train N": len(train),
                "Validation N": 0,
                "Test N": len(test),
                "Corpus N": len(corpus),
                "Top K": args.top_k,
                "Error": repr(exc),
            }
            rows.append(error_row)
            print(f"  {model_name} failed: {exc!r}", flush=True)

    return rows


def main() -> None:
    np.random.seed(SEED)
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    started = time.time()
    for config in SPLITS:
        all_rows.extend(run_split(config, args))
        comparison = pd.DataFrame(all_rows)
        comparison.to_csv(
            OUTPUT_DIR / "final_rag_baseline_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
    elapsed = time.time() - started
    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(
        OUTPUT_DIR / "final_rag_baseline_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"\nSaved RAG comparison to {OUTPUT_DIR / 'final_rag_baseline_comparison.csv'}",
        flush=True,
    )
    print(f"Total elapsed: {elapsed/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
