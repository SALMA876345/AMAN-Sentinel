"""Cross-domain FAAV experiments.

This runner is intentionally lightweight: it lets the project test domain
splits, evidence formatting, hybrid retrieval, few-shot UAE adaptation, and
fusion features without retraining a transformer for every ablation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLAIM_TRAIN_2CLASS_FILE,
    CLAIM_TRAIN_FILE,
    DATA_DIR,
    RESULTS_DIR,
    TEST_2CLASS_FILE,
    TEST_FILE,
    TEST_SHEET,
)
from pipeline.hybrid_retrieval import (
    HybridRetriever,
    attach_retrieved_evidence,
    build_hard_negative_pairs,
    build_retrieval_corpus,
)
from utils.faav_common import (
    REAL_FAKE_LABELS,
    SEP_TOKEN,
    THREE_CLASS_LABELS,
    VERIFICATION_LABELS,
    claim_hash,
    compute_classification_metrics,
    compute_sample_weights,
    deduplicate_claims,
    experiment_settings,
    group_metrics,
    leakage_report,
    load_ans_frame,
    load_test_frame,
    load_training_frame,
    make_model_text,
    likely_failure_reason,
    normalize_arabic,
    read_table,
    set_seed,
    write_json,
)

NUMERIC_FEATURES = [
    "retrieval_score",
    "source_credibility",
    "recency_score",
    "faithfulness_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FAAV cross-domain experiments.")
    parser.add_argument(
        "--experiment",
        choices=["random_split", "cross_domain_uae", "ans", "leave_one_domain_out", "fewshot_uae"],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["claim_only_lr", "claim_evidence_lr", "hybrid_rag_lr", "fusion_lr", "two_stage_lr"],
        default="claim_only_lr",
    )
    parser.add_argument("--class-mode", choices=["2", "3"], default="2")
    parser.add_argument(
        "--label-space",
        choices=["final", "verification"],
        default="final",
        help="final = Real/Fake(/Unverified), verification = Supported/Refuted/Not Enough Evidence.",
    )
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument(
        "--retrieval-corpus-file",
        default=None,
        help="Optional non-test passage corpus used only for evidence retrieval.",
    )
    parser.add_argument("--train-label-col", default=None)
    parser.add_argument("--test-label-col", default=None)
    parser.add_argument(
        "--training-dedup",
        choices=["claim", "pair", "none"],
        default="claim",
        help="Deduplicate supervised rows by claim, claim-evidence pair, or not at all.",
    )
    parser.add_argument("--ans-dir", default=os.path.join(DATA_DIR, "ANS"))
    parser.add_argument("--held-out-domain", default=None)
    parser.add_argument("--fewshot-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--input-mode", choices=["claim_only", "claim_plus_evidence"], default=None)
    parser.add_argument(
        "--retrieval",
        choices=["none", "bm25", "tfidf", "dense", "hybrid"],
        default="none",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corpus-limit", type=int, default=20000)
    parser.add_argument("--bm25-weight", type=float, default=0.50)
    parser.add_argument("--dense-weight", type=float, default=0.50)
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--domain-balanced", action="store_true")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--max-text-chars", type=int, default=2500)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--gate-source", choices=["learned", "metadata"], default="learned")
    parser.add_argument(
        "--gate-model",
        choices=["lr", "transformer"],
        default="lr",
        help="Model used for learned evidence relevance: sklearn LR or Arabic transformer cross-encoder.",
    )
    parser.add_argument(
        "--gate-transformer-model",
        default="UBC-NLP/MARBERTv2",
        help="HF model/path for transformer gate, e.g. UBC-NLP/MARBERTv2 or results/marbert_finetuned.",
    )
    parser.add_argument("--gate-transformer-epochs", type=int, default=1)
    parser.add_argument("--gate-transformer-batch-size", type=int, default=16)
    parser.add_argument("--gate-transformer-lr", type=float, default=2e-5)
    parser.add_argument("--gate-transformer-max-len", type=int, default=128)
    parser.add_argument(
        "--gate-transformer-train-limit",
        type=int,
        default=8000,
        help="Stratified cap for transformer gate rows; 0 uses all rows.",
    )
    parser.add_argument(
        "--relevance-train-file",
        default=None,
        help=(
            "Optional CSV/XLSX with gate_text/gate_label or "
            "claim_text/evidence_text/relevance_label for the learned evidence gate."
        ),
    )
    parser.add_argument(
        "--allow-relevance-test-overlap",
        action="store_true",
        help="Diagnostic only: allow relevance rows whose claim/evidence IDs overlap the test set.",
    )
    parser.add_argument(
        "--gate-quality-policy",
        choices=["relevant_only", "relevant_or_partial", "any_non_irrelevant"],
        default="relevant_only",
    )
    parser.add_argument("--retrieval-score-threshold", type=float, default=0.50)
    parser.add_argument("--min-evidence-chars", type=int, default=40)
    parser.add_argument("--hard-negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-hard-negatives", type=int, default=20000)
    parser.add_argument("--save-hard-negatives", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "cross_domain"))
    return parser.parse_args()


def class_labels(args: argparse.Namespace) -> list[str]:
    if args.label_space == "verification":
        return VERIFICATION_LABELS
    return REAL_FAKE_LABELS if args.class_mode == "2" else THREE_CLASS_LABELS


def default_train_file(args: argparse.Namespace) -> str:
    if args.train_file:
        return args.train_file
    if args.label_space == "verification":
        return CLAIM_TRAIN_FILE
    return CLAIM_TRAIN_2CLASS_FILE if args.class_mode == "2" else CLAIM_TRAIN_FILE


def default_test_file(args: argparse.Namespace) -> str:
    if args.test_file:
        return args.test_file
    if args.label_space == "verification":
        verification_file = os.path.join(DATA_DIR, "uae_test_verification.xlsx")
        if os.path.exists(verification_file):
            return verification_file
    return TEST_2CLASS_FILE if args.class_mode == "2" else TEST_FILE


def default_test_label_col(args: argparse.Namespace) -> str | None:
    if args.test_label_col:
        return args.test_label_col
    if args.label_space == "verification":
        return "verification_label"
    return None


def load_base_training(args: argparse.Namespace) -> pd.DataFrame:
    labels = class_labels(args)
    train = load_training_frame(
        default_train_file(args),
        labels,
        label_space=args.label_space,
        label_col=args.train_label_col,
    )
    train = train[train["claim_norm"].str.strip() != ""].reset_index(drop=True)
    if args.training_dedup == "claim":
        train = deduplicate_claims(train)
    elif args.training_dedup == "pair":
        train = train.drop_duplicates(
            subset=["claim_id", "evidence_norm", "label"],
            keep="first",
        ).reset_index(drop=True)
    return train


def split_fewshot(test_df: pd.DataFrame, k: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if k <= 0:
        return test_df.iloc[0:0].copy(), test_df.copy()
    if k >= len(test_df):
        raise ValueError("fewshot-k must be smaller than the target set")
    parts = []
    remaining = []
    per_label = max(1, k // max(test_df["label"].nunique(), 1))
    for _, group in test_df.groupby("label"):
        n = min(len(group) - 1, per_label)
        if n <= 0:
            remaining.append(group)
            continue
        sampled = group.sample(n=n, random_state=seed)
        parts.append(sampled)
        remaining.append(group.drop(index=sampled.index))
    adapt = pd.concat(parts, ignore_index=False) if parts else test_df.iloc[0:0]
    if len(adapt) < k:
        pool = test_df.drop(index=adapt.index)
        extra = pool.sample(n=min(k - len(adapt), len(pool) - 1), random_state=seed)
        adapt = pd.concat([adapt, extra], ignore_index=False)
    eval_df = test_df.drop(index=adapt.index).reset_index(drop=True)
    return adapt.reset_index(drop=True), eval_df


def build_splits(args: argparse.Namespace) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    labels = class_labels(args)
    base = load_base_training(args)

    if args.experiment == "random_split":
        train, test = train_test_split(
            base,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=base["label"],
        )
        return [("random_split", train.reset_index(drop=True), test.reset_index(drop=True))]

    if args.experiment == "cross_domain_uae":
        test = load_test_frame(
            default_test_file(args),
            labels,
            sheet_name=TEST_SHEET,
            label_space=args.label_space,
            label_col=default_test_label_col(args),
        )
        test = deduplicate_claims(test)
        return [("cross_domain_uae", base, test)]

    if args.experiment == "fewshot_uae":
        target = load_test_frame(
            default_test_file(args),
            labels,
            sheet_name=TEST_SHEET,
            label_space=args.label_space,
            label_col=default_test_label_col(args),
        )
        target = deduplicate_claims(target)
        adapt, test = split_fewshot(target, args.fewshot_k, args.seed)
        if len(adapt):
            adapt = adapt.copy()
            adapt["source_dataset"] = "UAE-MVD-fewshot"
            adapt["domain"] = adapt.get("domain", "UAE-MVD-fewshot")
            train = pd.concat([base, adapt], ignore_index=True)
        else:
            train = base
        return [(f"fewshot_uae_k{args.fewshot_k}", train.reset_index(drop=True), test)]

    if args.experiment == "ans":
        test = load_ans_frame(args.ans_dir, "test", labels, label_space=args.label_space)
        return [("cross_domain_ans", base, test)]

    if args.experiment == "leave_one_domain_out":
        domains = sorted(base["source_dataset"].dropna().unique().tolist())
        if args.held_out_domain and args.held_out_domain != "all":
            domains = [args.held_out_domain]
        splits = []
        for domain in domains:
            test = base[base["source_dataset"] == domain].copy()
            train = base[base["source_dataset"] != domain].copy()
            if len(train) == 0 or len(test) == 0:
                continue
            splits.append((f"lodo_{domain}", train.reset_index(drop=True), test.reset_index(drop=True)))
        return splits

    raise ValueError(f"Unsupported experiment: {args.experiment}")


def choose_input_mode(args: argparse.Namespace) -> str:
    if args.input_mode:
        return args.input_mode
    if args.model in {"claim_evidence_lr", "hybrid_rag_lr", "fusion_lr", "two_stage_lr"}:
        return "claim_plus_evidence"
    return "claim_only"


def prepare_retrieval(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, HybridRetriever | None]:
    retrieval_mode = args.retrieval
    if args.model == "hybrid_rag_lr" and retrieval_mode == "none":
        retrieval_mode = "hybrid"

    if retrieval_mode == "none":
        for frame in (train, test):
            frame["effective_evidence"] = frame["evidence_text"].fillna("")
            frame["effective_evidence_norm"] = frame["evidence_norm"].fillna("")
        return train, test, None

    if args.retrieval_corpus_file:
        corpus = read_table(args.retrieval_corpus_file)
        if "retrieval_document" not in corpus.columns:
            text_col = next(
                (c for c in ["evidence_text", "passage_text", "text"] if c in corpus.columns),
                None,
            )
            if text_col is None:
                raise ValueError(
                    "--retrieval-corpus-file must contain retrieval_document, "
                    "evidence_text, passage_text, or text."
                )
            corpus["retrieval_document"] = corpus[text_col].fillna("").astype(str)
        if "claim_id" not in corpus.columns:
            corpus["claim_id"] = [
                f"retrieval_{i}" for i in range(len(corpus))
            ]
        if "source_dataset" not in corpus.columns:
            corpus["source_dataset"] = "external_retrieval_corpus"
        if "source_name" not in corpus.columns:
            corpus["source_name"] = corpus["source_dataset"]
        if "source_credibility" not in corpus.columns:
            corpus["source_credibility"] = 0.5
        if "evidence_date" not in corpus.columns:
            corpus["evidence_date"] = pd.NaT
        corpus = corpus[
            corpus["retrieval_document"].fillna("").astype(str).str.strip() != ""
        ].reset_index(drop=True)
    else:
        corpus = build_retrieval_corpus(train)
    try:
        retriever = HybridRetriever(
            corpus,
            mode=retrieval_mode,
            top_k=args.top_k,
            corpus_limit=args.corpus_limit,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            dense_model_name=args.dense_model,
        )
    except Exception as exc:
        if retrieval_mode != "hybrid":
            raise
        print(f"WARNING: hybrid dense retrieval unavailable ({exc}); falling back to BM25.")
        args.retrieval = "bm25"
        retriever = HybridRetriever(
            corpus,
            mode="bm25",
            top_k=args.top_k,
            corpus_limit=args.corpus_limit,
            bm25_weight=1.0,
            dense_weight=0.0,
            dense_model_name=args.dense_model,
        )
    train = train.copy()
    test = attach_retrieved_evidence(test, retriever, top_k=args.top_k, exclude_self=False)
    train["retrieved_evidence"] = ""
    train["bm25_score"] = 0.0
    train["dense_score"] = 0.0
    train["retrieved_source_name"] = ""
    train["retrieved_source_dataset"] = ""
    train["retrieval_score"] = (train["evidence_text"].fillna("").astype(str).str.strip() != "").astype(float)
    for frame in (train, test):
        existing = frame["evidence_text"].fillna("").astype(str)
        retrieved = frame["retrieved_evidence"].fillna("").astype(str)
        frame["effective_evidence"] = retrieved.where(retrieved.str.strip() != "", existing)
        frame["effective_evidence_norm"] = frame["effective_evidence"].apply(normalize_arabic)
    return train, test, retriever


def text_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def binary_text_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def feature_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=max_features,
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                                lowercase=False,
                            ),
                            "model_text",
                        ),
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            NUMERIC_FEATURES,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def proba_frame(model: Pipeline, values, prefix: str) -> pd.DataFrame:
    proba = model.predict_proba(values)
    return pd.DataFrame(
        proba,
        columns=[f"{prefix}_proba_{label.lower()}" for label in model.classes_],
    )


def truncate_series(values: pd.Series, max_chars: int) -> pd.Series:
    values = values.fillna("").astype(str)
    if max_chars <= 0:
        return values
    return values.str.slice(0, max_chars)


def fit_single_model(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    input_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train = train.copy()
    test = test.copy()
    train["model_text"] = truncate_series(make_model_text(train, input_mode), args.max_text_chars)
    test["model_text"] = truncate_series(make_model_text(test, input_mode), args.max_text_chars)
    weights = compute_sample_weights(train, args.domain_balanced)

    if args.model == "fusion_lr":
        model = feature_pipeline(args.max_features)
        fit_kwargs = {"clf__sample_weight": weights} if weights is not None else {}
        model.fit(train[["model_text"] + NUMERIC_FEATURES], train["label"], **fit_kwargs)
        proba = model.predict_proba(test[["model_text"] + NUMERIC_FEATURES])
        preds = model.classes_[np.argmax(proba, axis=1)]
        return preds, proba, model.classes_.tolist()

    model = text_pipeline(args.max_features)
    fit_kwargs = {"clf__sample_weight": weights} if weights is not None else {}
    model.fit(train["model_text"], train["label"], **fit_kwargs)
    proba = model.predict_proba(test["model_text"])
    preds = model.classes_[np.argmax(proba, axis=1)]
    return preds, proba, model.classes_.tolist()


def pair_text(claim_norm: pd.Series, evidence_norm: pd.Series) -> pd.Series:
    return claim_norm.fillna("").astype(str) + SEP_TOKEN + evidence_norm.fillna("").astype(str)


def capped_pair_text(claim_norm: pd.Series, evidence_norm: pd.Series, max_chars: int) -> pd.Series:
    return truncate_series(pair_text(claim_norm, evidence_norm), max_chars)


def usable_evidence_mask(df: pd.DataFrame, min_chars: int) -> pd.Series:
    evidence = df.get("effective_evidence", df.get("evidence_text", "")).fillna("").astype(str)
    return evidence.str.len() >= min_chars


def evidence_quality_ok_series(values: pd.Series, policy: str) -> pd.Series:
    quality = values.fillna("").astype(str).str.strip().str.lower()
    relevant = quality.isin({"relevant", "relevant evidence", "high"})
    partial = quality.isin({"partially relevant", "partial", "medium"})
    irrelevant = quality.isin({"irrelevant", "low", "not relevant"})
    if policy == "relevant_only":
        return relevant
    if policy == "relevant_or_partial":
        return relevant | partial
    return quality.ne("") & ~irrelevant


def metadata_gate_mask(args: argparse.Namespace, df: pd.DataFrame) -> pd.Series:
    has_text = usable_evidence_mask(df, args.min_evidence_chars)
    if "retrieval_quality" in df.columns:
        quality_ok = evidence_quality_ok_series(df["retrieval_quality"], args.gate_quality_policy)
    else:
        quality_ok = pd.Series(True, index=df.index)
    score = pd.to_numeric(df.get("retrieval_score", pd.Series(np.nan, index=df.index)), errors="coerce")
    if score.notna().any() and score.max(skipna=True) > 0:
        score_ok = score.fillna(0.0) >= args.retrieval_score_threshold
        if "retrieval_quality" not in df.columns:
            quality_ok = score_ok
    return has_text & quality_ok


def canonical_gate_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    usable = {"usable", "relevant", "supported", "refuted", "1", "true", "yes"}
    not_usable = {
        "not_usable",
        "not usable",
        "irrelevant",
        "partially relevant",
        "partial",
        "not enough evidence",
        "nei",
        "0",
        "false",
        "no",
        "hard negative",
    }
    if value in usable:
        return "usable"
    if value in not_usable:
        return "not_usable"
    return None


def external_relevance_gate_training(
    args: argparse.Namespace,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    meta = {
        "path": args.relevance_train_file,
        "rows_loaded": 0,
        "rows_after_overlap_filter": 0,
        "rows_used": 0,
        "overlap_rows_removed": 0,
    }
    if not args.relevance_train_file:
        return pd.DataFrame(columns=["gate_text", "gate_label"]), meta

    df = read_table(args.relevance_train_file)
    meta["rows_loaded"] = int(len(df))
    if "gate_label" in df.columns:
        label_col = "gate_label"
    elif "relevance_label" in df.columns:
        label_col = "relevance_label"
    else:
        raise ValueError("--relevance-train-file must contain gate_label or relevance_label.")

    out = df.copy()
    out["gate_label"] = out[label_col].apply(canonical_gate_label)
    out = out[out["gate_label"].isin(["usable", "not_usable"])].copy()

    if "claim_id" not in out.columns:
        claim_col = next((c for c in ["claim_norm", "claim_text", "claim_ar", "text"] if c in out.columns), None)
        if claim_col:
            out["claim_id"] = out[claim_col].apply(claim_hash)
        else:
            out["claim_id"] = ""
    if "evidence_claim_id" not in out.columns:
        out["evidence_claim_id"] = out["claim_id"]

    if not args.allow_relevance_test_overlap:
        test_ids = set(test["claim_id"].dropna().astype(str))
        before = len(out)
        claim_overlap = out["claim_id"].fillna("").astype(str).isin(test_ids)
        evidence_overlap = out["evidence_claim_id"].fillna("").astype(str).isin(test_ids)
        out = out[~(claim_overlap | evidence_overlap)].copy()
        meta["overlap_rows_removed"] = int(before - len(out))
    meta["rows_after_overlap_filter"] = int(len(out))

    if "gate_text" not in out.columns:
        claim_col = next((c for c in ["claim_norm", "claim_text", "claim_ar", "text"] if c in out.columns), None)
        evidence_col = next((c for c in ["evidence_norm", "evidence_text", "evidence"] if c in out.columns), None)
        if claim_col is None or evidence_col is None:
            raise ValueError(
                "--relevance-train-file needs gate_text, or claim/evidence text columns."
            )
        claim = out[claim_col].fillna("").astype(str).apply(normalize_arabic)
        evidence = out[evidence_col].fillna("").astype(str).apply(normalize_arabic)
        out["gate_text"] = claim + SEP_TOKEN + evidence
    out["gate_text"] = truncate_series(out["gate_text"], args.max_text_chars)
    out = out[out["gate_text"].str.strip() != ""]
    out = out[["gate_text", "gate_label"]].drop_duplicates().reset_index(drop=True)
    meta["rows_used"] = int(len(out))
    return out, meta


def split_gate_pair_texts(values: pd.Series | list[str]) -> tuple[list[str], list[str]]:
    claims: list[str] = []
    evidences: list[str] = []
    series = pd.Series(values).fillna("").astype(str)
    for text in series:
        if SEP_TOKEN in text:
            claim, evidence = text.split(SEP_TOKEN, 1)
        else:
            claim, evidence = text, ""
        claims.append(claim)
        evidences.append(evidence)
    return claims, evidences


def cap_gate_train_for_transformer(
    gate_train: pd.DataFrame,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    if limit <= 0 or len(gate_train) <= limit:
        return gate_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    labels = sorted(gate_train["gate_label"].dropna().unique().tolist())
    per_label = max(1, limit // max(len(labels), 1))
    parts = []
    for label in labels:
        group = gate_train[gate_train["gate_label"] == label]
        parts.append(group.sample(n=min(len(group), per_label), random_state=seed))
    capped = pd.concat(parts, ignore_index=False)
    if len(capped) < limit:
        remaining = gate_train.drop(index=capped.index)
        if len(remaining):
            capped = pd.concat(
                [
                    capped,
                    remaining.sample(
                        n=min(limit - len(capped), len(remaining)),
                        random_state=seed,
                    ),
                ],
                ignore_index=False,
            )
    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def transformer_gate_predict_proba(
    args: argparse.Namespace,
    gate_train: pd.DataFrame,
    test_pair_text: pd.Series,
) -> tuple[np.ndarray, dict]:
    try:
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
    except Exception as exc:
        raise RuntimeError(
            "Transformer gate requires torch and transformers. Use --gate-model lr "
            "or install the missing packages."
        ) from exc

    set_seed(args.seed)
    gate_train = cap_gate_train_for_transformer(
        gate_train[["gate_text", "gate_label"]].dropna().copy(),
        args.gate_transformer_train_limit,
        args.seed,
    )
    label_to_id = {"not_usable": 0, "usable": 1}
    gate_train = gate_train[gate_train["gate_label"].isin(label_to_id)].copy()
    if gate_train["gate_label"].nunique() < 2:
        raise ValueError("Transformer gate needs both usable and not_usable examples.")

    train_claims, train_evidences = split_gate_pair_texts(gate_train["gate_text"])
    train_labels = gate_train["gate_label"].map(label_to_id).astype(int).tolist()

    class GateDataset(Dataset):
        def __init__(self, claims, evidences, labels=None):
            self.claims = claims
            self.evidences = evidences
            self.labels = labels

        def __len__(self):
            return len(self.claims)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.claims[idx],
                self.evidences[idx],
                max_length=args.gate_transformer_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
            if self.labels is not None:
                item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.gate_transformer_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.gate_transformer_model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)

    train_dl = DataLoader(
        GateDataset(train_claims, train_evidences, train_labels),
        batch_size=args.gate_transformer_batch_size,
        shuffle=True,
        num_workers=0,
    )
    counts = pd.Series(train_labels).value_counts().to_dict()
    total = float(len(train_labels))
    class_weights = torch.tensor(
        [total / (2.0 * max(counts.get(i, 1), 1)) for i in range(2)],
        dtype=torch.float,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.gate_transformer_lr, weight_decay=0.01)
    total_steps = max(1, len(train_dl) * max(args.gate_transformer_epochs, 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    epoch_losses = []
    for epoch in range(args.gate_transformer_epochs):
        model.train()
        running = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
        avg_loss = running / max(len(train_dl), 1)
        epoch_losses.append(avg_loss)
        print(f"  Transformer gate epoch {epoch + 1}/{args.gate_transformer_epochs} loss={avg_loss:.4f}")

    test_claims, test_evidences = split_gate_pair_texts(test_pair_text)
    model.eval()
    usable_proba: list[float] = []
    with torch.no_grad():
        for start in range(0, len(test_claims), args.gate_transformer_batch_size):
            enc = tokenizer(
                test_claims[start : start + args.gate_transformer_batch_size],
                test_evidences[start : start + args.gate_transformer_batch_size],
                max_length=args.gate_transformer_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
            usable_proba.extend(proba[:, label_to_id["usable"]].tolist())

    meta = {
        "gate_model": "transformer",
        "transformer_model": args.gate_transformer_model,
        "device": str(device),
        "transformer_train_rows": int(len(gate_train)),
        "transformer_train_distribution": gate_train["gate_label"].value_counts().to_dict(),
        "transformer_epochs": int(args.gate_transformer_epochs),
        "transformer_batch_size": int(args.gate_transformer_batch_size),
        "transformer_max_len": int(args.gate_transformer_max_len),
        "transformer_loss_by_epoch": epoch_losses,
    }
    return np.asarray(usable_proba, dtype=float), meta


def build_relevance_gate_training(args: argparse.Namespace, train: pd.DataFrame) -> pd.DataFrame:
    base = train.copy()
    base["gate_text"] = capped_pair_text(base["claim_norm"], base["effective_evidence_norm"], args.max_text_chars)
    has_evidence = usable_evidence_mask(base, args.min_evidence_chars)
    if "retrieval_quality" in base.columns:
        raw_quality = base["retrieval_quality"].fillna("").astype(str).str.strip()
        has_quality_annotation = raw_quality.ne("")
        quality_ok = evidence_quality_ok_series(base["retrieval_quality"], args.gate_quality_policy)
    else:
        has_quality_annotation = pd.Series(False, index=base.index)
        quality_ok = pd.Series(True, index=base.index)
    usable = (
        base["label"].isin(["Supported", "Refuted"])
        & has_evidence
        & (~has_quality_annotation | quality_ok)
    )
    base["gate_label"] = np.where(
        usable,
        "usable",
        "not_usable",
    )
    rows = base[["gate_text", "gate_label"]].copy()

    positives = base[(base["gate_label"] == "usable") & (base["effective_evidence_norm"].str.strip() != "")]
    evidence_pool = base[
        (base["effective_evidence_norm"].str.strip() != "")
        & (base["claim_id"].notna())
    ][["claim_id", "label", "effective_evidence_norm"]].reset_index(drop=True)

    if (
        args.hard_negatives_per_positive <= 0
        or positives.empty
        or len(evidence_pool) < 2
    ):
        return rows.reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    hard_rows = []
    positives = positives.sample(
        min(len(positives), max(1, args.max_hard_negatives // args.hard_negatives_per_positive)),
        random_state=args.seed,
    )
    for _, row in positives.iterrows():
        candidates = evidence_pool[
            (evidence_pool["claim_id"] != row["claim_id"])
            & (evidence_pool["label"] != row["label"])
        ]
        if candidates.empty:
            candidates = evidence_pool[evidence_pool["claim_id"] != row["claim_id"]]
        if candidates.empty:
            continue
        sample_n = min(args.hard_negatives_per_positive, len(candidates))
        sampled_idx = rng.choice(candidates.index.to_numpy(), size=sample_n, replace=False)
        for ev in candidates.loc[sampled_idx, "effective_evidence_norm"]:
            hard_rows.append(
                {
                    "gate_text": str(f"{row['claim_norm']}{SEP_TOKEN}{ev}")[: args.max_text_chars],
                    "gate_label": "not_usable",
                }
            )

    if hard_rows:
        rows = pd.concat([rows, pd.DataFrame(hard_rows)], ignore_index=True)
    return rows.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)


def fit_two_stage_model(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    if args.label_space != "verification":
        raise ValueError("two_stage_lr requires --label-space verification.")

    train = train.copy()
    test = test.copy()
    train["effective_evidence_norm"] = train["effective_evidence_norm"].fillna("")
    test["effective_evidence_norm"] = test["effective_evidence_norm"].fillna("")
    test["pair_text"] = capped_pair_text(test["claim_norm"], test["effective_evidence_norm"], args.max_text_chars)

    stance_train = train[
        train["label"].isin(["Supported", "Refuted"])
        & usable_evidence_mask(train, args.min_evidence_chars)
    ].copy()
    if stance_train["label"].nunique() < 2:
        raise ValueError("Need both Supported and Refuted rows with usable evidence to train stance verifier.")
    stance_train["pair_text"] = capped_pair_text(
        stance_train["claim_norm"],
        stance_train["effective_evidence_norm"],
        args.max_text_chars,
    )
    stance_model = binary_text_pipeline(args.max_features)
    stance_model.fit(stance_train["pair_text"], stance_train["label"])

    if args.gate_source == "metadata":
        usable_mask = metadata_gate_mask(args, test).to_numpy()
        usable_proba = usable_mask.astype(float)
        gate_train_dist = {"metadata_gate": int(len(test))}
        gate_meta = {}
    else:
        gate_train = build_relevance_gate_training(args, train)
        external_gate, gate_meta = external_relevance_gate_training(args, test)
        if not external_gate.empty:
            gate_train = pd.concat([gate_train, external_gate], ignore_index=True)
            gate_train = gate_train.drop_duplicates().sample(frac=1.0, random_state=args.seed)
        if gate_train["gate_label"].nunique() < 2:
            raise ValueError(
                "Need both usable and not_usable rows to train the learned evidence gate."
            )
        if args.gate_model == "transformer":
            usable_proba, transformer_meta = transformer_gate_predict_proba(
                args,
                gate_train,
                test["pair_text"],
            )
            gate_meta.update(transformer_meta)
        else:
            gate_model = binary_text_pipeline(args.max_features)
            gate_model.fit(gate_train["gate_text"], gate_train["gate_label"])
            gate_proba = gate_model.predict_proba(test["pair_text"])
            gate_classes = gate_model.classes_.tolist()
            usable_idx = gate_classes.index("usable")
            usable_proba = gate_proba[:, usable_idx]
            gate_meta["gate_model"] = "lr"
        gate_train_dist = gate_train["gate_label"].value_counts().to_dict()

    stance_proba = stance_model.predict_proba(test["pair_text"])
    stance_classes = stance_model.classes_.tolist()
    stance_pred = stance_model.classes_[np.argmax(stance_proba, axis=1)]

    labels = VERIFICATION_LABELS
    proba = np.zeros((len(test), len(labels)), dtype=float)
    nei_idx = labels.index("Not Enough Evidence")
    proba[:, nei_idx] = 1.0 - usable_proba
    for idx, label in enumerate(stance_classes):
        proba[:, labels.index(label)] = usable_proba * stance_proba[:, idx]
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    proba = proba / row_sums

    threshold = 0.5 if args.gate_source == "metadata" else args.gate_threshold
    preds = np.where(
        usable_proba < threshold,
        "Not Enough Evidence",
        stance_pred,
    )
    extras = {
        "evidence_usable_proba": usable_proba,
        "gate_prediction": np.where(usable_proba < threshold, "not_usable", "usable"),
        "stance_prediction": stance_pred,
        "gate_train_distribution": gate_train_dist,
        "stance_train_distribution": stance_train["label"].value_counts().to_dict(),
        "gate_training_metadata": gate_meta,
    }
    return preds, proba, labels, extras


def fit_stacked_fusion(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = sorted(train["label"].unique().tolist())
    if len(train) < 50 or min(train["label"].value_counts()) < 2:
        return fit_single_model(args, train, test, "claim_plus_evidence")

    base_train, meta_train = train_test_split(
        train,
        test_size=0.20,
        random_state=args.seed,
        stratify=train["label"],
    )

    claim_model = text_pipeline(args.max_features)
    pair_model = text_pipeline(args.max_features)
    claim_model.fit(base_train["claim_norm"], base_train["label"])
    base_train = base_train.copy()
    meta_train = meta_train.copy()
    test = test.copy()
    base_train["pair_text"] = make_model_text(base_train, "claim_plus_evidence")
    meta_train["pair_text"] = make_model_text(meta_train, "claim_plus_evidence")
    test["pair_text"] = make_model_text(test, "claim_plus_evidence")
    pair_model.fit(base_train["pair_text"], base_train["label"])

    meta_X = pd.concat(
        [
            proba_frame(claim_model, meta_train["claim_norm"], "claim"),
            proba_frame(pair_model, meta_train["pair_text"], "pair"),
            meta_train[NUMERIC_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    meta.fit(meta_X, meta_train["label"])

    # Refit base models on all labelled training rows before predicting target.
    train = train.copy()
    train["pair_text"] = make_model_text(train, "claim_plus_evidence")
    claim_model.fit(train["claim_norm"], train["label"])
    pair_model.fit(train["pair_text"], train["label"])

    test_X = pd.concat(
        [
            proba_frame(claim_model, test["claim_norm"], "claim"),
            proba_frame(pair_model, test["pair_text"], "pair"),
            test[NUMERIC_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )
    for col in meta_X.columns:
        if col not in test_X.columns:
            test_X[col] = 0.0
    test_X = test_X[meta_X.columns]
    proba = meta.predict_proba(test_X)
    preds = meta.classes_[np.argmax(proba, axis=1)]
    return preds, proba, meta.classes_.tolist()


def save_errors(output_dir: str, run_name: str, df_out: pd.DataFrame) -> str:
    err = df_out[df_out["label"] != df_out["prediction"]].copy()
    if err.empty:
        out_path = os.path.join(output_dir, f"{run_name}_errors.csv")
        err.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
    err["error_type"] = np.where(
        (err["label"] == "Real") & (err["prediction"] == "Fake"),
        "false_positive",
        np.where(
            (err["label"] == "Fake") & (err["prediction"] == "Real"),
            "false_negative",
            np.where(
                (err["label"] == "Supported") & (err["prediction"] == "Refuted"),
                "false_refutation",
                np.where(
                    (err["label"] == "Refuted") & (err["prediction"] == "Supported"),
                    "false_support",
                    "other_error",
                ),
            ),
        ),
    )
    err["likely_failure"] = err.apply(likely_failure_reason, axis=1)
    keep = [
        "error_type",
        "claim_text",
        "label",
        "prediction",
        "confidence",
        "effective_evidence",
        "retrieved_evidence",
        "source_dataset",
        "domain",
        "retrieval_score",
        "source_credibility",
        "recency_gap",
        "faithfulness_score",
        "evidence_usable_proba",
        "gate_prediction",
        "stance_prediction",
        "likely_failure",
    ]
    keep = [c for c in keep if c in err.columns]
    out_path = os.path.join(output_dir, f"{run_name}_errors.csv")
    err[keep].to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def run_one(args: argparse.Namespace, split_name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train, test, retriever = prepare_retrieval(args, train.copy(), test.copy())
    input_mode = choose_input_mode(args)
    if args.model == "claim_only_lr":
        input_mode = "claim_only"
    extras = {}
    if args.model == "two_stage_lr":
        preds, proba, classes, extras = fit_two_stage_model(args, train, test)
    elif args.model == "fusion_lr":
        preds, proba, classes = fit_stacked_fusion(args, train, test)
    else:
        preds, proba, classes = fit_single_model(args, train, test, input_mode)

    df_out = test.copy()
    df_out["prediction"] = preds
    df_out["confidence"] = np.max(proba, axis=1)
    for idx, label in enumerate(classes):
        df_out[f"proba_{label.lower()}"] = proba[:, idx]
    for col, values in extras.items():
        if isinstance(values, (list, tuple, np.ndarray, pd.Series)) and len(values) == len(df_out):
            df_out[col] = values
    df_out["is_correct"] = df_out["label"] == df_out["prediction"]

    labels = class_labels(args)
    metrics = compute_classification_metrics(df_out["label"], df_out["prediction"], labels)
    metrics["leakage"] = leakage_report(train, test)
    metrics["train_rows"] = int(len(train))
    metrics["test_rows"] = int(len(test))
    metrics["train_label_distribution"] = train["label"].value_counts().to_dict()
    metrics["test_label_distribution"] = test["label"].value_counts().to_dict()
    metrics["train_domain_distribution"] = train["source_dataset"].value_counts().head(20).to_dict()
    metrics["test_domain_distribution"] = test["source_dataset"].value_counts().head(20).to_dict()
    metrics["input_mode"] = input_mode
    metrics["retrieval"] = args.retrieval
    metrics["label_space"] = args.label_space
    metrics["domain_balanced"] = bool(args.domain_balanced)
    metrics["gate_threshold"] = args.gate_threshold if args.model == "two_stage_lr" else None
    if args.model == "two_stage_lr":
        metrics["gate_source"] = args.gate_source
        metrics["gate_model"] = args.gate_model if args.gate_source == "learned" else "metadata"
        metrics["gate_quality_policy"] = args.gate_quality_policy
        metrics["retrieval_score_threshold"] = args.retrieval_score_threshold
        metrics["gate_train_distribution"] = extras.get("gate_train_distribution", {})
        metrics["stance_train_distribution"] = extras.get("stance_train_distribution", {})
        metrics["gate_training_metadata"] = extras.get("gate_training_metadata", {})

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    run_name = f"{split_name}_{args.label_space}_{args.model}_{input_mode}_{args.retrieval}"
    if args.model == "two_stage_lr":
        if args.gate_source == "learned":
            run_name += f"_{args.gate_source}_{args.gate_model}_gate"
        else:
            run_name += f"_{args.gate_source}_gate"
        run_name += f"_thr{args.gate_threshold:.2f}".replace(".", "p")
    if args.experiment == "fewshot_uae":
        run_name += f"_k{args.fewshot_k}"
    pred_path = os.path.join(output_dir, f"{run_name}_predictions.csv")
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.json")
    settings_path = os.path.join(output_dir, f"{run_name}_settings.json")
    domain_path = os.path.join(output_dir, f"{run_name}_per_domain.csv")
    dataset_path = os.path.join(output_dir, f"{run_name}_per_dataset.csv")

    df_out.to_csv(pred_path, index=False, encoding="utf-8-sig")
    write_json(metrics_path, metrics)
    write_json(settings_path, experiment_settings(args))
    group_metrics(df_out, "domain", "prediction", labels).to_csv(domain_path, index=False, encoding="utf-8-sig")
    group_metrics(df_out, "source_dataset", "prediction", labels).to_csv(
        dataset_path, index=False, encoding="utf-8-sig"
    )
    errors_path = save_errors(output_dir, run_name, df_out)

    hard_neg_path = None
    if args.save_hard_negatives and retriever is not None:
        hard_neg = build_hard_negative_pairs(train, retriever)
        hard_neg_path = os.path.join(output_dir, f"{run_name}_hard_negatives.csv")
        hard_neg.to_csv(hard_neg_path, index=False, encoding="utf-8-sig")

    print(
        f"{run_name}: n={metrics['test_rows']} "
        f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"mcc={metrics['mcc']:.4f} leakage={metrics['leakage']['exact_overlap_count']}"
    )
    return {
        "run_name": run_name,
        "predictions": pred_path,
        "metrics": metrics_path,
        "settings": settings_path,
        "per_domain": domain_path,
        "per_dataset": dataset_path,
        "errors": errors_path,
        "hard_negatives": hard_neg_path,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mcc": metrics["mcc"],
        "n": metrics["test_rows"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    runs = []
    for split_name, train, test in build_splits(args):
        runs.append(run_one(args, split_name, train, test))

    summary = pd.DataFrame(runs)
    summary_path = os.path.join(args.output_dir, f"{args.experiment}_{args.label_space}_{args.model}_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
