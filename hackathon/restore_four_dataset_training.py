"""Phase 1 restoration: rebuild data/four_dataset_training.csv.

This is the SAME transformation as utils/prepare_four_dataset_training.py
(label mapping, AFND undecided-cap, AraStance stance mapping — unchanged),
pointed at the raw corpora's real location on disk instead of the
`data/<corpus>` paths the original script assumed.

Why this script exists instead of editing the original
--------------------------------------------------------
The audit found data/four_dataset_training.csv missing, and traced its
source corpora (AFND, ArCOV19-Rumors, VERA-ARAB, AraStance) to
`Assignment 2/Dataset files/` — a sibling directory of this git repo
(`Assignment 2/Code files/`) that is present on disk but was never
vendored into `data/`. Nothing here is fabricated: every row below is
read from those raw files, with the identical label-mapping logic the
original script used. We do not touch
utils/prepare_four_dataset_training.py so the historical script stays
intact as a record of the original (non-portable) assumption.

UAE-MVD is NOT handled here. Its only known source
(c:\\Users\\Hamza\\social_dataset_agent\\datasets\\UAE-MVD-merged.csv) does
not exist anywhere in this project tree — confirmed by a recursive search
for UAE-MVD / uae_test_verification / uae_test_dataset before writing this
script. That pipeline is legitimately blocked and is reported as such,
not reconstructed.

Output: data/four_dataset_training.csv  (same path the historical
utils/build_claim_training.py expects as SRC_FILE, so that script can run
unmodified once this file exists).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # Code files/
PROJECT_ROOT = ROOT.parent                            # Assignment 2/
RAW_DIR = PROJECT_ROOT / "Dataset files"              # sibling dir, present on disk
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "four_dataset_training.csv"

COLS = [
    "source", "text", "label",
    "evidence_text", "source_url", "source_name",
    "source_type", "fact_check_snippet", "category",
]

AFND_LABEL_MAP = {"credible": "Real", "not credible": "Fake"}
AFND_UNVERIFIED_CAP = 30_000  # identical cap to the historical script


def _parse_list_field(val):
    if pd.isna(val):
        return ""
    try:
        import ast
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except Exception:
        pass
    s = str(val).strip()
    return s if s not in ("[]", "nan", "") else ""


def _join_list_field(val, sep=" | "):
    if pd.isna(val):
        return ""
    try:
        import ast
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return sep.join(str(x).strip() for x in parsed if str(x).strip())
    except Exception:
        pass
    s = str(val).strip()
    return s if s not in ("[]", "nan", "") else ""


def build_afnd() -> pd.DataFrame:
    sources_file = RAW_DIR / "AFND" / "sources.json"
    dataset_dir = RAW_DIR / "AFND" / "Dataset"
    if not sources_file.exists():
        raise FileNotFoundError(f"AFND sources.json not found at {sources_file}")

    with open(sources_file, encoding="utf-8") as f:
        source_labels = json.load(f)

    rows, unverified_buf = [], []
    for source_id, label in source_labels.items():
        mapped = AFND_LABEL_MAP.get(label)
        is_undecided = label == "undecided"
        if not mapped and not is_undecided:
            continue
        articles_file = dataset_dir / source_id / "scraped_articles.json"
        if not articles_file.exists():
            continue
        with open(articles_file, encoding="utf-8") as f:
            data = json.load(f)
        for article in data.get("articles", []):
            title = article.get("title", "").strip()
            body = article.get("text", "").strip()
            combined = (title + " " + body).strip()
            if not combined:
                continue
            row = {
                "source": "AFND",
                "text": combined,
                "label": mapped if mapped else "Unverified",
                "evidence_text": "",
                "source_url": "",
                "source_name": source_id,
                "source_type": "Arabic news website",
                "fact_check_snippet": "",
                "category": "",
            }
            (unverified_buf if is_undecided else rows).append(row)

    random.seed(42)
    if len(unverified_buf) > AFND_UNVERIFIED_CAP:
        unverified_buf = random.sample(unverified_buf, AFND_UNVERIFIED_CAP)
    rows.extend(unverified_buf)

    df = pd.DataFrame(rows)[COLS]
    print(f"AFND           : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
    print(f"  (undecided=Unverified: {len(unverified_buf):,} articles, cap={AFND_UNVERIFIED_CAP:,})")
    return df


def build_arcov19_tweets() -> pd.DataFrame:
    path = RAW_DIR / "ArCOV19-Rumors" / "tweet_verification" / "Tweets.txt"
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df = df.rename(columns={"tweetText": "text"})
    label_map = {True: "Real", False: "Fake", "True": "Real", "False": "Fake"}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].str.strip()
    df["source"] = "ArCOV19-tweets"
    df["evidence_text"] = ""
    df["source_url"] = "https://twitter.com/i/web/status/" + df["tweetID"].astype(str)
    df["source_name"] = "Twitter"
    df["source_type"] = "social media tweet"
    df["fact_check_snippet"] = ""
    df["category"] = ""
    df = df[COLS]
    print(f"ArCOV19 tweets : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
    return df


def build_arcov19_claims() -> pd.DataFrame:
    path = RAW_DIR / "ArCOV19-Rumors" / "claim_verification" / "claims"
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df = df.rename(columns={
        "Claim": "text", "ClaimLabel": "orig_label",
        "ClaimSource": "source_url", "Category": "category",
    })
    label_map = {"True": "Real", "False": "Fake", True: "Real", False: "Fake"}
    df["label"] = df["orig_label"].map(label_map)
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].str.strip()
    df["source"] = "ArCOV19-claims"
    df["evidence_text"] = ""
    df["source_name"] = "ArCOV19 fact-check"
    df["source_type"] = "fact-checked claim"
    df["fact_check_snippet"] = df["text"]
    df = df[COLS]
    print(f"ArCOV19 claims : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
    return df


def build_vera_arab() -> pd.DataFrame:
    path = RAW_DIR / "VERA-ARAB" / "peerj-cs-10-2432-s001.xlsx"
    df = pd.read_excel(path, usecols=[
        "text_arabic", "label", "urls", "domain", "domain_label",
        "annotations_text", "annotations_type",
        "context_domain", "context_domain_desc",
    ])
    df = df.rename(columns={"text_arabic": "text"})
    label_map = {0: "Real", 1: "Fake"}
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label", "text"])
    df["text"] = df["text"].str.strip()
    df["source"] = "VERA-ARAB"
    df["evidence_text"] = ""
    df["source_url"] = df["urls"].apply(_parse_list_field)
    df["source_name"] = df["context_domain"].apply(_parse_list_field)
    df["source_type"] = "social media tweet"
    df["fact_check_snippet"] = df["annotations_text"].apply(_join_list_field)
    df["category"] = df.apply(
        lambda r: _join_list_field(r["context_domain_desc"]) or str(r["domain"]).strip(),
        axis=1,
    )
    df = df[COLS]
    print(f"VERA-ARAB      : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
    return df


def build_arastance() -> pd.DataFrame:
    files = [
        RAW_DIR / "AraStance" / "arastance_train.csv",
        RAW_DIR / "AraStance" / "arastance_valid.csv",
        RAW_DIR / "AraStance" / "arastance_test.csv",
    ]
    label_map = {"Agree": "Real", "Disagree": "Fake", "Discuss": "Unverified"}
    parts = []
    for fp in files:
        tmp = pd.read_csv(fp)
        tmp["label"] = tmp["stance"].map(label_map)
        tmp = tmp.dropna(subset=["label"])
        tmp["text"] = (tmp["claim"].fillna("") + " " + tmp["article"].fillna("")).str.strip()
        tmp["evidence_text"] = tmp["article"].fillna("").str.strip()
        tmp["source_url"] = ""
        tmp["source_name"] = ""
        tmp["source_type"] = "news article"
        tmp["fact_check_snippet"] = ""
        tmp["category"] = ""
        tmp["source"] = "AraStance"
        parts.append(tmp[COLS])
    df = pd.concat(parts, ignore_index=True)
    print(f"AraStance      : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
    print("  (Discuss=Unverified; Unrelated dropped; article kept as evidence_text)")
    return df


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw corpora directory not found: {RAW_DIR}. "
            "Phase 1 source-pool restoration requires 'Dataset files/' as a "
            "sibling of 'Code files/'."
        )

    print(f"Reading raw corpora from: {RAW_DIR}")
    afnd = build_afnd()
    non_afnd = pd.concat(
        [build_arcov19_claims(), build_arcov19_tweets(), build_vera_arab(), build_arastance()],
        ignore_index=True,
    )
    non_afnd = non_afnd[non_afnd["text"].str.strip() != ""]

    # Same budget logic as the historical script: keep all non-AFND rows,
    # fill the remaining budget to 120,000 with capped AFND Real/Fake/Unverified.
    TRAIN_TARGET = 120_000
    afnd_df = afnd[afnd["text"].str.strip() != ""]
    afnd_budget = TRAIN_TARGET - len(non_afnd)
    afnd_unverified = afnd_df[afnd_df["label"] == "Unverified"]
    afnd_real_fake_budget = max((afnd_budget - len(afnd_unverified)) // 2, 0)

    afnd_real = afnd_df[afnd_df["label"] == "Real"].sample(
        min(len(afnd_df[afnd_df["label"] == "Real"]), afnd_real_fake_budget), random_state=42
    )
    afnd_fake = afnd_df[afnd_df["label"] == "Fake"].sample(
        min(len(afnd_df[afnd_df["label"] == "Fake"]), afnd_real_fake_budget), random_state=42
    )
    afnd_sampled = pd.concat([afnd_real, afnd_fake, afnd_unverified], ignore_index=True)

    unified = pd.concat([afnd_sampled, non_afnd], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"TOTAL              : {len(unified):>7,} rows  (target={TRAIN_TARGET:,})")
    print(f"  AFND sampled     : {len(afnd_sampled):>7,} | non-AFND kept full: {len(non_afnd):,}")
    print(f"Label split        : {unified['label'].value_counts().to_dict()}")
    print(f"Source breakdown   : {unified['source'].value_counts().to_dict()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
