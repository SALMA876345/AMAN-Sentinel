"""Shared FAAV utilities for reproducible cross-domain experiments."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

REAL_FAKE_LABELS = ["Real", "Fake"]
THREE_CLASS_LABELS = ["Real", "Fake", "Unverified"]
VERIFICATION_LABELS = ["Supported", "Refuted", "Not Enough Evidence"]

# Research label space. Real/Fake should only be treated as final binary labels
# after each source dataset has been normalised into this verification schema.
FINAL_TO_VERIFICATION = {
    "Real": "Supported",
    "Fake": "Refuted",
    "Unverified": "Not Enough Evidence",
}
VERIFICATION_TO_FINAL = {
    "Supported": "Real",
    "Refuted": "Fake",
    "Not Enough Evidence": "Unverified",
}

# Raw labels observed in the included datasets. AFND credible/not credible are
# source-credibility labels, not claim-truth labels; use them as metadata unless
# an experiment explicitly opts into source-level supervision.
RAW_TO_VERIFICATION = {
    "Real": "Supported",
    "Fake": "Refuted",
    "Unverified": "Not Enough Evidence",
    "Supported": "Supported",
    "Refuted": "Refuted",
    "Not Enough Evidence": "Not Enough Evidence",
    "True": "Supported",
    "False": "Refuted",
    "true": "Supported",
    "false": "Refuted",
    "Agree": "Supported",
    "Disagree": "Refuted",
    "Discuss": "Not Enough Evidence",
    "NEI": "Not Enough Evidence",
}

SEP_TOKEN = " [SEP] "

_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_ALEF_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_WORD_RE = re.compile(r"[^\w\s\u0600-\u06ff]")
_SPACE_RE = re.compile(r"\s+")
_ARABIC_TOKEN_RE = re.compile(r"[\u0600-\u06ff]{3,}")

TRUSTED_SOURCE_MARKERS = [
    "gov",
    "wam",
    "mohap",
    "mohre",
    "cabinet",
    "official",
    "ministry",
    "\u0648\u0632\u0627\u0631\u0629",
    "\u0648\u0627\u0645",
    "\u062d\u0643\u0648\u0645\u0629",
    "\u0627\u0644\u0625\u0639\u0644\u0627\u0646\u0627\u062a \u0627\u0644\u0631\u0633\u0645\u064a\u0629",
]
FACT_CHECK_SOURCE_MARKERS = ["fatabyyano", "bbc", "sky news"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def normalize_arabic(text: object) -> str:
    """Apply the same Arabic normalisation to train, validation, and test text."""
    text = "" if pd.isna(text) else str(text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("\u0640", "")
    text = _DIACRITICS_RE.sub("", text)
    text = _ALEF_RE.sub("\u0627", text)
    text = text.replace("\u0649", "\u064a")
    text = text.replace("\u0629", "\u0647")
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def claim_hash(text: object) -> str:
    norm = normalize_arabic(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def canonical_verification_label(raw_label: object, source_dataset: str | None = None) -> str | None:
    if pd.isna(raw_label):
        return None
    if isinstance(raw_label, (bool, np.bool_)):
        return "Supported" if bool(raw_label) else "Refuted"
    raw = str(raw_label).strip()
    if raw in RAW_TO_VERIFICATION:
        return RAW_TO_VERIFICATION[raw]
    if raw in {"0", "1"} and source_dataset == "ANS":
        return "Refuted" if raw == "1" else "Supported"
    return None


def final_label_from_verification(label: object, class_labels: Iterable[str]) -> str | None:
    class_labels = list(class_labels)
    label = None if pd.isna(label) else str(label).strip()
    final = VERIFICATION_TO_FINAL.get(label)
    if final is None:
        return None
    if final == "Unverified" and "Unverified" not in class_labels:
        return None
    return final


def map_final_to_verification(label: object) -> str | None:
    if pd.isna(label):
        return None
    return FINAL_TO_VERIFICATION.get(str(label).strip())


def infer_source_credibility(source_name: object, source_dataset: object = "") -> float:
    text = f"{source_name or ''} {source_dataset or ''}".lower()
    if any(marker in text for marker in TRUSTED_SOURCE_MARKERS):
        return 0.95
    if any(marker in text for marker in FACT_CHECK_SOURCE_MARKERS):
        return 0.80
    if "afnd" in text:
        return 0.50
    if not text.strip():
        return 0.50
    return 0.60


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return parsed.tz_convert(None)


def recency_gap_days(claim_date: object, evidence_date: object) -> float:
    c = parse_date(claim_date)
    e = parse_date(evidence_date)
    if pd.isna(c) or pd.isna(e):
        return np.nan
    return float(abs((c - e).days))


def recency_score(days: object) -> float:
    if pd.isna(days):
        return 0.5
    days = max(float(days), 0.0)
    return float(1.0 / (1.0 + days / 30.0))


def lexical_overlap(claim: object, evidence: object) -> float:
    c_toks = set(_ARABIC_TOKEN_RE.findall(normalize_arabic(claim)))
    if not c_toks:
        return 0.0
    e_toks = set(_ARABIC_TOKEN_RE.findall(normalize_arabic(evidence)))
    return float(len(c_toks & e_toks) / len(c_toks))


def faithfulness_controlled_decision(
    verifier_label: object,
    evidence: object,
    *,
    verifier_confidence: float,
    semantic_similarity: float,
    grounding_score: float,
    min_evidence_chars: int = 40,
    min_verifier_confidence: float = 0.50,
    min_semantic_similarity: float = 0.20,
    min_grounding_score: float = 0.30,
    abstain_label: str = "Unverified",
) -> tuple[str, str]:
    """Fuse verifier and evidence-quality signals into the final verdict.

    A Real/Fake verifier output is accepted only when the available evidence,
    verifier confidence, claim-evidence similarity, and explanation grounding
    all pass their configured thresholds. Otherwise the system abstains.
    """
    label = "" if pd.isna(verifier_label) else str(verifier_label).strip()
    evidence_text = "" if pd.isna(evidence) else str(evidence).strip()

    if label == abstain_label:
        return abstain_label, "verifier_abstained"
    if label not in {"Real", "Fake"}:
        return abstain_label, "invalid_verifier_label"

    failed_checks = []
    if len(evidence_text) < min_evidence_chars:
        failed_checks.append("insufficient_evidence")
    if float(verifier_confidence) < min_verifier_confidence:
        failed_checks.append("low_verifier_confidence")
    if float(semantic_similarity) < min_semantic_similarity:
        failed_checks.append("low_claim_evidence_similarity")
    if float(grounding_score) < min_grounding_score:
        failed_checks.append("low_explanation_grounding")

    if failed_checks:
        return abstain_label, "|".join(failed_checks)
    return label, "accepted"


def ensure_research_columns(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    class_labels: Iterable[str],
    text_col: str | None = None,
    label_col: str | None = None,
    label_space: str = "final",
) -> pd.DataFrame:
    """Return a common schema used by the cross-domain experiment runner."""
    if label_space not in {"final", "verification"}:
        raise ValueError("label_space must be 'final' or 'verification'")
    out = df.copy()
    if text_col is None:
        text_col = "text" if "text" in out.columns else "claim_ar"
    if label_col is None:
        label_col = "label" if "label" in out.columns else "gold_label"
    if text_col not in out.columns:
        raise ValueError(f"Missing text column: {text_col}")
    if label_col not in out.columns:
        raise ValueError(f"Missing label column: {label_col}")

    out["claim_text"] = out[text_col].fillna("").astype(str)
    out["claim_norm"] = out["claim_text"].apply(normalize_arabic)
    out["claim_id"] = out["claim_norm"].apply(claim_hash)
    if "source" in out.columns:
        out["source_dataset"] = out["source"].fillna(dataset_name).astype(str)
    else:
        out["source_dataset"] = dataset_name
    out.loc[out["source_dataset"].str.strip() == "", "source_dataset"] = dataset_name

    out["verification_label"] = out[label_col].apply(
        lambda v: canonical_verification_label(v, dataset_name)
    )
    if label_space == "verification":
        out["label"] = out["verification_label"]
    else:
        out["label"] = out["verification_label"].apply(
            lambda v: final_label_from_verification(v, class_labels)
        )
    out = out[out["label"].isin(list(class_labels))]
    out = out.dropna(subset=["label"])

    if "evidence_text" not in out.columns:
        out["evidence_text"] = ""
    out["evidence_text"] = out["evidence_text"].fillna("").astype(str)
    out["evidence_norm"] = out["evidence_text"].apply(normalize_arabic)

    source_name = out["source_name"] if "source_name" in out.columns else out["source_dataset"]
    out["source_name"] = source_name.fillna("").astype(str)
    out["source_credibility"] = [
        infer_source_credibility(src, ds)
        for src, ds in zip(out["source_name"], out["source_dataset"])
    ]

    domain_candidates = [
        "domain",
        "topic",
        "topics",
        "category",
        "candidate_category",
        "source_dataset",
    ]
    domain_col = next((c for c in domain_candidates if c in out.columns), "source_dataset")
    out["domain"] = out[domain_col].fillna(out["source_dataset"]).astype(str)
    out.loc[out["domain"].str.strip() == "", "domain"] = out["source_dataset"]

    claim_date_col = next((c for c in ["claim_date", "date", "published_date"] if c in out.columns), None)
    evidence_date_col = next(
        (c for c in ["evidence_date", "labeled_at", "published_date", "date"] if c in out.columns),
        None,
    )
    out["claim_date"] = out[claim_date_col] if claim_date_col else pd.NaT
    out["evidence_date"] = out[evidence_date_col] if evidence_date_col else pd.NaT
    out["recency_gap"] = [
        recency_gap_days(c, e) for c, e in zip(out["claim_date"], out["evidence_date"])
    ]
    out["recency_score"] = out["recency_gap"].apply(recency_score)
    out["faithfulness_score"] = [
        lexical_overlap(c, e) for c, e in zip(out["claim_text"], out["evidence_text"])
    ]
    if "retrieval_score" not in out.columns:
        out["retrieval_score"] = 0.0
    return out.reset_index(drop=True)


def read_table(path: str, sheet_name: str = "Tabelle1") -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path, low_memory=False)


def load_training_frame(
    path: str,
    class_labels: Iterable[str],
    *,
    label_space: str = "final",
    label_col: str | None = None,
) -> pd.DataFrame:
    df = read_table(path)
    return ensure_research_columns(
        df,
        dataset_name=os.path.splitext(os.path.basename(path))[0],
        class_labels=class_labels,
        label_space=label_space,
        label_col=label_col,
    )


def load_test_frame(
    path: str,
    class_labels: Iterable[str],
    sheet_name: str = "Tabelle1",
    *,
    label_space: str = "final",
    label_col: str | None = None,
) -> pd.DataFrame:
    df = read_table(path, sheet_name=sheet_name)
    return ensure_research_columns(
        df,
        dataset_name=os.path.splitext(os.path.basename(path))[0],
        class_labels=class_labels,
        label_space=label_space,
        label_col=label_col,
    )


def load_ans_frame(
    ans_dir: str,
    split: str,
    class_labels: Iterable[str],
    *,
    label_space: str = "final",
    label_col: str | None = None,
) -> pd.DataFrame:
    path = os.path.join(ans_dir, f"{split}.csv")
    df = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "text": df["claim_s"].astype(str),
            "label": df["fake_flag"].map(lambda v: "Fake" if int(v) == 1 else "Real"),
            "evidence_text": "",
            "source": "ANS",
            "domain": "ANS",
        }
    )
    return ensure_research_columns(
        df,
        dataset_name="ANS",
        class_labels=class_labels,
        label_space=label_space,
        label_col=label_col,
    )


def deduplicate_claims(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates(subset=["claim_id"], keep=keep).reset_index(drop=True)
    out.attrs["duplicates_removed"] = before - len(out)
    return out


def leakage_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_ids = set(train_df["claim_id"])
    test_ids = set(test_df["claim_id"])
    overlap = sorted(train_ids & test_ids)
    return {
        "exact_overlap_count": len(overlap),
        "exact_overlap_claim_ids": overlap[:100],
    }


def make_model_text(df: pd.DataFrame, input_mode: str) -> pd.Series:
    if input_mode == "claim_plus_evidence":
        evidence = df.get("effective_evidence_norm", df.get("evidence_norm", "")).fillna("")
        return df["claim_norm"].fillna("") + SEP_TOKEN + evidence.astype(str)
    if input_mode != "claim_only":
        raise ValueError("input_mode must be 'claim_only' or 'claim_plus_evidence'")
    return df["claim_norm"].fillna("")


def compute_sample_weights(df: pd.DataFrame, domain_balanced: bool = False) -> np.ndarray | None:
    if not domain_balanced:
        return None
    label_counts = df["label"].value_counts().to_dict()
    domain_counts = df["domain"].value_counts().to_dict()
    weights = []
    for _, row in df.iterrows():
        label_w = 1.0 / max(label_counts.get(row["label"], 1), 1)
        domain_w = 1.0 / max(domain_counts.get(row["domain"], 1), 1)
        weights.append(np.sqrt(label_w * domain_w))
    weights = np.array(weights, dtype=float)
    return weights / weights.mean()


def compute_classification_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: Iterable[str],
) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = list(labels)
    all_seen_labels = set(y_true) | set(y_pred)
    mcc = 0.0 if len(all_seen_labels) < 2 else float(matthews_corrcoef(y_true, y_pred))
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "mcc": mcc,
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def group_metrics(df: pd.DataFrame, group_col: str, pred_col: str, labels: Iterable[str]) -> pd.DataFrame:
    rows = []
    if group_col not in df.columns:
        return pd.DataFrame(rows)
    for group, sub in df.groupby(group_col, dropna=False):
        if len(sub) == 0:
            continue
        m = compute_classification_metrics(sub["label"], sub[pred_col], labels)
        rows.append(
            {
                "group": group,
                "n": m["n"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "mcc": m["mcc"],
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "group"], ascending=[False, True])


def likely_failure_reason(row: pd.Series) -> str:
    reasons = []
    retrieval_score = float(row.get("retrieval_score", 0.0) or 0.0)
    confidence = float(row.get("confidence", 0.0) or 0.0)
    recency = row.get("recency_gap", np.nan)
    faithfulness = float(row.get("faithfulness_score", 0.0) or 0.0)
    source_cred = float(row.get("source_credibility", 0.5) or 0.5)

    if retrieval_score < 0.15 and not str(row.get("retrieved_evidence", row.get("evidence_text", ""))).strip():
        reasons.append("retrieval_failure")
    elif retrieval_score < 0.15:
        reasons.append("weak_retrieval_score")
    if confidence and confidence < 0.60:
        reasons.append("low_model_confidence")
    if not pd.isna(recency) and float(recency) > 180:
        reasons.append("temporal_mismatch")
    if faithfulness < 0.10:
        reasons.append("low_claim_evidence_overlap")
    if source_cred < 0.60:
        reasons.append("low_or_unknown_source_credibility")
    if str(row.get("domain", "")).strip() in {"", "ANS", "uae_test_2class", "uae_test_dataset_final"}:
        reasons.append("domain_shift_possible")
    return "; ".join(reasons) if reasons else "unclear_model_error"


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def experiment_settings(args: object) -> dict:
    settings = vars(args).copy() if hasattr(args, "__dict__") else dict(args)
    settings["created_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return settings
