"""
Builds the supervised training CSV from the four core Arabic misinformation
datasets, as agreed with supervisor.  AraNEWS is excluded so the training
domain better matches the UAE-MVD test set.

Datasets included:
  1. AFND           — Arabic Fake News Dataset
  2. ArCOV19-Rumors — COVID-19 fact-checked claims + labeled tweets
  3. VERA-ARAB      — Verified Arabic tweets
  4. AraStance      — Claim–article stance pairs

Label mapping (3-class, matching UAE-MVD test set):
  AFND      : credible  → Real        |  not credible → Fake
              undecided → Unverified  (capped at AFND_UNVERIFIED_CAP)
  ArCOV19   : True      → Real        |  False        → Fake
  VERA-ARAB : 0         → Real        |  1            → Fake
  AraStance : Agree     → Real        |  Disagree     → Fake
              Discuss   → Unverified  |  Unrelated    → dropped

Output: data/four_dataset_training.csv
"""

import os
import json
import ast
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT   = os.path.join(DATA_DIR, "four_dataset_training.csv")

COLS = ["source", "text", "label",
        "evidence_text", "source_url", "source_name",
        "source_type", "fact_check_snippet", "category"]

parts = []


def _parse_list_field(val):
    if pd.isna(val):
        return ""
    try:
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
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return sep.join(str(x).strip() for x in parsed if str(x).strip())
    except Exception:
        pass
    s = str(val).strip()
    return s if s not in ("[]", "nan", "") else ""


# ── 1. AFND ───────────────────────────────────────────────────────────────────
afnd_sources = os.path.join(DATA_DIR, "AFND", "sources.json")
afnd_dataset = os.path.join(DATA_DIR, "AFND", "Dataset")

with open(afnd_sources, encoding="utf-8") as f:
    source_labels = json.load(f)

AFND_LABEL_MAP = {"credible": "Real", "not credible": "Fake"}
AFND_UNVERIFIED_CAP = 30_000   # cap undecided articles as Unverified (half of 60k to balance classes)

afnd_rows = []
afnd_unverified_buf = []

for source_id, label in source_labels.items():
    mapped = AFND_LABEL_MAP.get(label)
    is_undecided = (label == "undecided")
    if not mapped and not is_undecided:
        continue
    articles_file = os.path.join(afnd_dataset, source_id, "scraped_articles.json")
    if not os.path.exists(articles_file):
        continue
    with open(articles_file, encoding="utf-8") as f:
        data = json.load(f)
    for article in data.get("articles", []):
        title    = article.get("title", "").strip()
        body     = article.get("text", "").strip()
        combined = (title + " " + body).strip()
        if not combined:
            continue
        row = {
            "source":             "AFND",
            "text":               combined,
            "label":              mapped if mapped else "Unverified",
            "evidence_text":      "",
            "source_url":         "",
            "source_name":        source_id,
            "source_type":        "Arabic news website",
            "fact_check_snippet": "",
            "category":           "",
        }
        if is_undecided:
            afnd_unverified_buf.append(row)
        else:
            afnd_rows.append(row)

# Sample undecided articles up to cap to avoid class imbalance
import random
random.seed(42)
if len(afnd_unverified_buf) > AFND_UNVERIFIED_CAP:
    afnd_unverified_buf = random.sample(afnd_unverified_buf, AFND_UNVERIFIED_CAP)
afnd_rows.extend(afnd_unverified_buf)

df = pd.DataFrame(afnd_rows)
parts.append(df[COLS])
print(f"AFND           : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
print(f"  (undecided=Unverified: {len(afnd_unverified_buf):,} articles added, cap={AFND_UNVERIFIED_CAP:,})")


# ── 2. ArCOV19 – tweets ───────────────────────────────────────────────────────
tweets_path = os.path.join(DATA_DIR, "ArCOV19-Rumors",
                           "tweet_verification", "Tweets.txt")
df = pd.read_csv(tweets_path, sep="\t", encoding="utf-8")
df = df.rename(columns={"tweetText": "text"})
label_map = {True: "Real", False: "Fake",
             "True": "Real", "False": "Fake"}
df["label"]              = df["label"].map(label_map)
df                       = df.dropna(subset=["label", "text"])
df["text"]               = df["text"].str.strip()
df["source"]             = "ArCOV19-tweets"
df["evidence_text"]      = ""
df["source_url"]         = "https://twitter.com/i/web/status/" + df["tweetID"].astype(str)
df["source_name"]        = "Twitter"
df["source_type"]        = "social media tweet"
df["fact_check_snippet"] = ""
df["category"]           = ""
parts.append(df[COLS])
print(f"ArCOV19 tweets : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")


# ── 3. ArCOV19 – claims ───────────────────────────────────────────────────────
claims_path = os.path.join(DATA_DIR, "ArCOV19-Rumors",
                           "claim_verification", "claims")
df = pd.read_csv(claims_path, sep="\t", encoding="utf-8")
df = df.rename(columns={"Claim": "text", "ClaimLabel": "orig_label",
                         "ClaimSource": "source_url",
                         "Category": "category"})
label_map = {"True": "Real", "False": "Fake",
             True: "Real", False: "Fake"}
df["label"]              = df["orig_label"].map(label_map)
df                       = df.dropna(subset=["label", "text"])
df["text"]               = df["text"].str.strip()
df["source"]             = "ArCOV19-claims"
df["evidence_text"]      = ""
df["source_name"]        = "ArCOV19 fact-check"
df["source_type"]        = "fact-checked claim"
df["fact_check_snippet"] = df["text"]
parts.append(df[COLS])
print(f"ArCOV19 claims : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")


# ── 4. VERA-ARAB ──────────────────────────────────────────────────────────────
vera_path = os.path.join(DATA_DIR, "VERA-ARAB", "peerj-cs-10-2432-s001.xlsx")
df = pd.read_excel(vera_path, usecols=[
    "text_arabic", "label",
    "urls", "domain", "domain_label",
    "annotations_text", "annotations_type",
    "context_domain", "context_domain_desc",
])
df = df.rename(columns={"text_arabic": "text"})
label_map = {0: "Real", 1: "Fake"}
df["label"]              = df["label"].map(label_map)
df                       = df.dropna(subset=["label", "text"])
df["text"]               = df["text"].str.strip()
df["source"]             = "VERA-ARAB"
df["evidence_text"]      = ""
df["source_url"]         = df["urls"].apply(_parse_list_field)
df["source_name"]        = df["context_domain"].apply(_parse_list_field)
df["source_type"]        = "social media tweet"
df["fact_check_snippet"] = df["annotations_text"].apply(_join_list_field)
df["category"]           = df.apply(
    lambda r: _join_list_field(r["context_domain_desc"]) or str(r["domain"]).strip(),
    axis=1
)
parts.append(df[COLS])
print(f"VERA-ARAB      : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")


# ── 5. AraStance (all splits) ─────────────────────────────────────────────────
arastance_files = [
    os.path.join(DATA_DIR, "AraStance", "arastance_train.csv"),
    os.path.join(DATA_DIR, "AraStance", "arastance_valid.csv"),
    os.path.join(DATA_DIR, "AraStance", "arastance_test.csv"),
]
# Discuss → Unverified (neither agrees nor disagrees — ambiguous/contested claim)
# Unrelated → dropped (no stance toward the claim at all)
label_map = {"Agree": "Real", "Disagree": "Fake", "Discuss": "Unverified"}
df_list = []
for fp in arastance_files:
    tmp = pd.read_csv(fp)
    tmp["label"] = tmp["stance"].map(label_map)
    tmp = tmp.dropna(subset=["label"])
    tmp["text"]               = (tmp["claim"].fillna("") + " " + tmp["article"].fillna("")).str.strip()
    tmp["evidence_text"]      = tmp["article"].fillna("").str.strip()
    tmp["source_url"]         = ""
    tmp["source_name"]        = ""
    tmp["source_type"]        = "news article"
    tmp["fact_check_snippet"] = ""
    tmp["category"]           = ""
    tmp["source"]             = "AraStance"
    df_list.append(tmp[COLS])

df = pd.concat(df_list, ignore_index=True)
parts.append(df)
print(f"AraStance      : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
print("  (Discuss=Unverified; Unrelated dropped; article kept as evidence_text)")


# ── Combine & save ─────────────────────────────────────────────────────────────
# Keep ALL rows from ArCOV19, VERA-ARAB, AraStance (parts[1:])
# Only cap AFND (parts[0]) to fill remaining budget to reach 120,000 total
# (test=30,000 → 20% of 150,000 total)
TRAIN_TARGET = 120_000

non_afnd = pd.concat(parts[1:], ignore_index=True)
non_afnd  = non_afnd[non_afnd["text"].str.strip() != ""]

afnd_df   = parts[0].copy()
afnd_df   = afnd_df[afnd_df["text"].str.strip() != ""]

afnd_budget    = TRAIN_TARGET - len(non_afnd)
afnd_unverified = afnd_df[afnd_df["label"] == "Unverified"]
afnd_real_fake_budget = (afnd_budget - len(afnd_unverified)) // 2

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

print(f"\n{'='*60}")
test_rows  = 30_015
total      = len(unified) + test_rows
test_pct   = 100 * test_rows / total
print(f"TOTAL              : {len(unified):>7,} rows  (target={TRAIN_TARGET:,})")
print(f"  AFND sampled     : {len(afnd_sampled):>7,} | non-AFND kept full: {len(non_afnd):,}")
print(f"Test split check   : {test_rows:,} test / {total:,} total = {test_pct:.1f}%")
print(f"Label split        : {unified['label'].value_counts().to_dict()}")
print(f"Source breakdown   : {unified['source'].value_counts().to_dict()}")
print()
for col in ["evidence_text", "source_url", "source_name",
            "source_type", "fact_check_snippet", "category"]:
    filled = (unified[col].str.strip() != "").sum()
    print(f"  {col:<22}: {filled:>7,} rows filled  ({100*filled/len(unified):.1f}%)")

unified.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT}")
