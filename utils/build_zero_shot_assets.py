"""Build leakage-free assets for zero-shot cross-domain FAAV evaluation.

UAE-MVD is used only as a frozen external test set. Supervised verification
pairs come from AraStance. AFND is used only as an evidence retrieval corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, TEST_SHEET
from utils.faav_common import (
    SEP_TOKEN,
    claim_hash,
    infer_source_credibility,
    lexical_overlap,
    normalize_arabic,
)


ROOT = Path(__file__).resolve().parents[1]
ARASTANCE_DIR = Path(DATA_DIR) / "AraStance"
FOUR_DATASET_FILE = Path(DATA_DIR) / "four_dataset_training.csv"
UAE_VERIFICATION_FILE = Path(DATA_DIR) / "uae_test_verification.xlsx"

VERIFICATION_OUT = Path(DATA_DIR) / "verification_training_zeroshot.csv"
RELEVANCE_OUT = Path(DATA_DIR) / "relevance_training_zeroshot.csv"
STANCE_OUT = Path(DATA_DIR) / "stance_training_zeroshot.csv"
RETRIEVAL_OUT = Path(DATA_DIR) / "retrieval_corpus_zeroshot.csv"
TEST_OUT = Path(DATA_DIR) / "uae_zero_shot_test.xlsx"
ANNOTATION_OUT = Path(DATA_DIR) / "zero_shot_annotation_queue.csv"
AUDIT_OUT = Path(RESULTS_DIR) / "zero_shot_data_audit.json"

STANCE_MAP = {
    "Agree": "Supported",
    "Disagree": "Refuted",
    "Discuss": "Not Enough Evidence",
    "Unrelated": "Not Enough Evidence",
}
USABLE_STANCES = {"Agree", "Disagree"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict zero-shot FAAV datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passage-max-chars", type=int, default=650)
    parser.add_argument("--passage-min-chars", type=int, default=80)
    parser.add_argument("--afnd-article-limit", type=int, default=40000)
    parser.add_argument("--annotation-queue-size", type=int, default=1000)
    return parser.parse_args()


def split_passages(text: object, max_chars: int, min_chars: int) -> list[str]:
    clean = " ".join(str(text or "").split())
    if not clean:
        return []
    words = clean.split()
    passages: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            passage = " ".join(current).strip()
            if len(passage) >= min_chars:
                passages.append(passage)
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        passage = " ".join(current).strip()
        if len(passage) >= min_chars:
            passages.append(passage)
        elif passages:
            passages[-1] = f"{passages[-1]} {passage}".strip()
        elif passage:
            passages.append(passage)
    return passages


def best_passage(claim: object, article: object, max_chars: int, min_chars: int) -> str:
    passages = split_passages(article, max_chars, min_chars)
    if not passages:
        return ""
    scores = [lexical_overlap(claim, passage) for passage in passages]
    return passages[int(np.argmax(scores))]


def load_arastance() -> pd.DataFrame:
    files = [
        ARASTANCE_DIR / "arastance_train.csv",
        ARASTANCE_DIR / "arastance_valid.csv",
        ARASTANCE_DIR / "arastance_test.csv",
    ]
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["original_split"] = path.stem.replace("arastance_", "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_supervised_pairs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_arastance()
    raw = raw[raw["stance"].isin(STANCE_MAP)].copy()
    raw["claim_text"] = raw["claim"].fillna("").astype(str).str.strip()
    raw["evidence_text"] = [
        best_passage(c, a, args.passage_max_chars, args.passage_min_chars)
        for c, a in zip(raw["claim"], raw["article"])
    ]
    raw["claim_norm"] = raw["claim_text"].apply(normalize_arabic)
    raw["evidence_norm"] = raw["evidence_text"].apply(normalize_arabic)
    raw["claim_id"] = raw["claim_norm"].apply(claim_hash)
    raw["evidence_claim_id"] = raw["claim_id"]
    raw["verification_label"] = raw["stance"].map(STANCE_MAP)
    raw["claim_truth_label"] = ""
    raw["retrieval_quality"] = np.where(
        raw["stance"].isin(USABLE_STANCES),
        "Relevant",
        "Not Usable",
    )
    raw["relevance_label"] = np.where(
        raw["stance"].isin(USABLE_STANCES)
        & (raw["evidence_text"].str.len() >= args.passage_min_chars),
        "usable",
        "not_usable",
    )
    raw["relevance_reason"] = np.where(
        raw["relevance_label"] == "usable",
        "gold_arastance_support_or_refute",
        "gold_arastance_discuss_or_unrelated",
    )
    raw["source_dataset"] = "AraStance"
    raw["source"] = "AraStance"
    raw["source_name"] = "AraStance article"
    raw["source_credibility"] = 0.70
    raw["domain"] = "AraStance"
    raw["evidence_url"] = ""
    raw["faithfulness_score"] = [
        lexical_overlap(c, e) for c, e in zip(raw["claim_text"], raw["evidence_text"])
    ]
    raw["gate_text"] = raw["claim_norm"] + SEP_TOKEN + raw["evidence_norm"]
    raw["row_origin"] = "observed_arastance"
    raw["pair_id"] = [f"arastance_{i}" for i in range(len(raw))]

    verification = raw.rename(columns={"claim_text": "text"})[
        [
            "text",
            "verification_label",
            "evidence_text",
            "source",
            "source_name",
            "domain",
            "original_split",
            "claim_id",
            "faithfulness_score",
        ]
    ].copy()

    relevance_cols = [
        "claim_text",
        "evidence_text",
        "claim_norm",
        "evidence_norm",
        "claim_id",
        "evidence_claim_id",
        "verification_label",
        "claim_truth_label",
        "retrieval_quality",
        "relevance_label",
        "relevance_reason",
        "source_dataset",
        "domain",
        "source_name",
        "evidence_url",
        "source_credibility",
        "faithfulness_score",
        "gate_text",
        "row_origin",
        "pair_id",
    ]
    relevance = raw[relevance_cols].copy()

    positives = raw[raw["relevance_label"] == "usable"].copy()
    evidence_pool = raw[
        raw["evidence_text"].str.len() >= args.passage_min_chars
    ].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    hard_rows = []
    for _, row in positives.iterrows():
        candidates = evidence_pool[evidence_pool["claim_id"] != row["claim_id"]]
        opposite = candidates[
            candidates["verification_label"] != row["verification_label"]
        ]
        if not opposite.empty:
            candidates = opposite
        if candidates.empty:
            continue
        neg = candidates.iloc[int(rng.integers(0, len(candidates)))]
        new_row = {col: row.get(col, "") for col in relevance_cols}
        new_row["evidence_text"] = neg["evidence_text"]
        new_row["evidence_norm"] = neg["evidence_norm"]
        new_row["evidence_claim_id"] = neg["claim_id"]
        new_row["verification_label"] = "Not Enough Evidence"
        new_row["retrieval_quality"] = "Hard Negative"
        new_row["relevance_label"] = "not_usable"
        new_row["relevance_reason"] = "hard_negative_other_claim_evidence"
        new_row["faithfulness_score"] = lexical_overlap(
            row["claim_text"], neg["evidence_text"]
        )
        new_row["gate_text"] = row["claim_norm"] + SEP_TOKEN + neg["evidence_norm"]
        new_row["row_origin"] = "hard_negative"
        new_row["pair_id"] = f"hard_negative_{len(hard_rows)}"
        hard_rows.append(new_row)
    if hard_rows:
        relevance = pd.concat([relevance, pd.DataFrame(hard_rows)], ignore_index=True)
    relevance = relevance.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    stance = raw[
        (raw["relevance_label"] == "usable")
        & raw["verification_label"].isin(["Supported", "Refuted"])
    ][relevance_cols].copy()
    return verification, relevance, stance


def build_retrieval_corpus(args: argparse.Namespace) -> pd.DataFrame:
    source = pd.read_csv(FOUR_DATASET_FILE, low_memory=False)
    afnd = source[source["source"] == "AFND"].copy()
    if len(afnd) > args.afnd_article_limit:
        afnd = afnd.sample(args.afnd_article_limit, random_state=args.seed)

    rows = []
    for idx, row in afnd.reset_index(drop=True).iterrows():
        passages = split_passages(
            row["text"], args.passage_max_chars, args.passage_min_chars
        )
        if not passages:
            continue
        # Use the lead passage to keep the corpus compact and article-focused.
        passage = passages[0]
        rows.append(
            {
                "retrieval_document": passage,
                "passage_text": passage,
                "claim_id": f"afnd_{idx}",
                "source_dataset": "AFND",
                "source_name": str(row.get("source_name", "AFND") or "AFND"),
                "source_credibility": infer_source_credibility(
                    row.get("source_name", ""), "AFND"
                ),
                "evidence_date": "",
                "source_url": str(row.get("source_url", "") or ""),
            }
        )

    arastance = load_arastance()
    for idx, row in arastance.reset_index(drop=True).iterrows():
        for passage_idx, passage in enumerate(
            split_passages(
                row["article"], args.passage_max_chars, args.passage_min_chars
            )[:3]
        ):
            rows.append(
                {
                    "retrieval_document": passage,
                    "passage_text": passage,
                    "claim_id": f"arastance_article_{idx}_{passage_idx}",
                    "source_dataset": "AraStance",
                    "source_name": "AraStance article",
                    "source_credibility": 0.70,
                    "evidence_date": "",
                    "source_url": "",
                }
            )
    corpus = pd.DataFrame(rows)
    corpus = corpus.drop_duplicates(subset=["retrieval_document"]).reset_index(drop=True)
    return corpus


def freeze_uae_test() -> tuple[pd.DataFrame, str]:
    test = pd.read_excel(UAE_VERIFICATION_FILE, sheet_name=TEST_SHEET)
    test = test.copy()
    test["claim_id"] = test["claim_ar"].apply(claim_hash)
    test = test.drop_duplicates(subset=["claim_id"], keep="first").reset_index(drop=True)
    test["zero_shot_row_id"] = [f"uae_zs_{i:05d}" for i in range(len(test))]
    ordered_ids = "\n".join(test["claim_id"].astype(str))
    fingerprint = hashlib.sha256(ordered_ids.encode("utf-8")).hexdigest()
    return test, fingerprint


def build_annotation_queue(
    test: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    n = min(size, len(test))
    sample = test.sample(n=n, random_state=seed).copy()
    keep = [
        "zero_shot_row_id",
        "claim_ar",
        "evidence_text",
        "source",
        "evidence_url",
        "gold_label",
        "verification_label",
        "retrieval_quality",
        "faithfulness_label",
    ]
    keep = [c for c in keep if c in sample.columns]
    queue = sample[keep].copy()
    queue["human_verification_label"] = ""
    queue["human_retrieval_quality"] = ""
    queue["human_faithfulness_label"] = ""
    queue["annotator_id"] = ""
    queue["annotation_notes"] = ""
    return queue


def main() -> None:
    args = parse_args()
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    verification, relevance, stance = build_supervised_pairs(args)
    retrieval = build_retrieval_corpus(args)
    frozen_test, fingerprint = freeze_uae_test()
    annotation_queue = build_annotation_queue(
        frozen_test, args.annotation_queue_size, args.seed
    )

    verification.to_csv(VERIFICATION_OUT, index=False, encoding="utf-8-sig")
    relevance.to_csv(RELEVANCE_OUT, index=False, encoding="utf-8-sig")
    stance.to_csv(STANCE_OUT, index=False, encoding="utf-8-sig")
    retrieval.to_csv(RETRIEVAL_OUT, index=False, encoding="utf-8-sig")
    frozen_test.to_excel(TEST_OUT, index=False, sheet_name=TEST_SHEET)
    annotation_queue.to_csv(ANNOTATION_OUT, index=False, encoding="utf-8-sig")

    train_ids = set(verification["claim_id"].astype(str))
    test_ids = set(frozen_test["claim_id"].astype(str))
    overlap = sorted(train_ids & test_ids)
    audit = {
        "protocol": "zero_shot_cross_domain",
        "uae_used_for_training": False,
        "supervised_source": "AraStance only",
        "afnd_role": "retrieval corpus only",
        "verification_training_rows": int(len(verification)),
        "verification_label_distribution": verification[
            "verification_label"
        ].value_counts().to_dict(),
        "relevance_training_rows": int(len(relevance)),
        "relevance_label_distribution": relevance[
            "relevance_label"
        ].value_counts().to_dict(),
        "relevance_origin_distribution": relevance[
            "row_origin"
        ].value_counts().to_dict(),
        "stance_training_rows": int(len(stance)),
        "stance_label_distribution": stance[
            "verification_label"
        ].value_counts().to_dict(),
        "retrieval_corpus_rows": int(len(retrieval)),
        "retrieval_source_distribution": retrieval[
            "source_dataset"
        ].value_counts().to_dict(),
        "frozen_test_rows": int(len(frozen_test)),
        "frozen_test_label_distribution": frozen_test[
            "verification_label"
        ].value_counts().to_dict(),
        "frozen_test_fingerprint_sha256": fingerprint,
        "exact_claim_overlap_count": int(len(overlap)),
        "exact_claim_overlap_ids": overlap[:20],
        "evidence_length": {
            "training_median": float(
                verification["evidence_text"].str.len().median()
            ),
            "test_median": float(frozen_test["evidence_text"].fillna("").str.len().median()),
            "retrieval_passage_median": float(
                retrieval["retrieval_document"].str.len().median()
            ),
        },
        "annotation_queue_rows": int(len(annotation_queue)),
        "note": (
            "Human annotation columns are intentionally blank. They must not be "
            "filled automatically or inferred from claim-truth labels."
        ),
    }
    with AUDIT_OUT.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {VERIFICATION_OUT}")
    print(f"Saved -> {RELEVANCE_OUT}")
    print(f"Saved -> {STANCE_OUT}")
    print(f"Saved -> {RETRIEVAL_OUT}")
    print(f"Saved -> {TEST_OUT}")
    print(f"Saved -> {ANNOTATION_OUT}")
    print(f"Saved -> {AUDIT_OUT}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
