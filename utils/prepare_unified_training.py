"""
Combines all 5 Arabic misinformation datasets into a single unified training CSV,
enriched with evidence text, source URLs, fact-check snippets, and source metadata.

Label mapping (everything -> credible / not credible):
  AFND           : credible -> credible  |  not credible -> not credible
  AraNEWS        : Orginal  -> credible  |  Machine      -> not credible
  ArCOV19-Rumors : True     -> credible  |  False        -> not credible (claims + tweets)
  VERA-ARAB      : 0        -> credible  |  1            -> not credible
  AraStance      : Agree    -> credible  |  Disagree     -> not credible
                   (Discuss / Unrelated dropped - not a credibility signal)

Enrichment columns added where available:
  evidence_text      - supporting/refuting article or passage
  source_url         - URL of original content or fact-check source
  source_name        - name or ID of the publishing outlet
  source_type        - category of source (news, social media, fact-checking, etc.)
  fact_check_snippet - annotation or snippet from a fact-checking service
  category           - topic domain (health, politics, sports, etc.)

Output: data/unified_training.csv
"""

import os
import sys
import json
import ast
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT   = os.path.join(DATA_DIR, "unified_training.csv")

COLS = ["source", "text", "label",
        "evidence_text", "source_url", "source_name",
        "source_type", "fact_check_snippet", "category"]

parts = []


def _parse_list_field(val):
    """Parse a stringified list; return first element or empty string."""
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
    """Parse a stringified list; join all elements."""
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


# ── 1. AFND (built from raw JSON sources) ────────────────────────────────────
afnd_sources = os.path.join(DATA_DIR, "AFND", "sources.json")
afnd_dataset = os.path.join(DATA_DIR, "AFND", "Dataset")

with open(afnd_sources, encoding="utf-8") as f:
    source_labels = json.load(f)

afnd_rows = []
for source_id, label in source_labels.items():
    if label == "undecided":
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
        if combined:
            afnd_rows.append({
                "source":           "AFND",
                "text":             combined,
                "label":            label,
                "evidence_text":    "",
                "source_url":       "",
                "source_name":      source_id,          # anonymised outlet ID
                "source_type":      "Arabic news website",
                "fact_check_snippet": "",
                "category":         "",
            })

df = pd.DataFrame(afnd_rows)
parts.append(df[COLS])
print(f"AFND           : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")


# ── 2. AraNEWS (train split only) ────────────────────────────────────────────
aranews_path = os.path.join(DATA_DIR, "AraNEWS",
                            "ArNews_train_GW_order_0",
                            "ArNews_train_GW_order_0.tsv")
df = pd.read_csv(aranews_path, encoding="utf-8")
df = df.rename(columns={"Sentence": "text", "label": "orig_label"})
label_map = {"Orginal": "credible", "Machine": "not credible"}
df["label"]            = df["orig_label"].map(label_map)
df                     = df.dropna(subset=["label", "text"])
df["text"]             = df["text"].str.strip()
df["source"]           = "AraNEWS"
df["evidence_text"]    = ""
df["source_url"]       = ""
df["source_name"]      = ""
df["source_type"]      = df["orig_label"].map(
                             {"Orginal": "original news article",
                              "Machine": "machine-generated news"})
df["fact_check_snippet"] = ""
df["category"]         = ""
parts.append(df[COLS])
print(f"AraNEWS        : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")


# ── 3. ArCOV19-Rumors – tweets ───────────────────────────────────────────────
tweets_path = os.path.join(DATA_DIR, "ArCOV19-Rumors",
                           "tweet_verification", "Tweets.txt")
df = pd.read_csv(tweets_path, sep="\t", encoding="utf-8")
df = df.rename(columns={"tweetText": "text"})
label_map = {True: "credible", False: "not credible",
             "True": "credible", "False": "not credible"}
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


# ── 4. ArCOV19-Rumors – claims ───────────────────────────────────────────────
claims_path = os.path.join(DATA_DIR, "ArCOV19-Rumors",
                           "claim_verification", "claims")
df = pd.read_csv(claims_path, sep="\t", encoding="utf-8")
df = df.rename(columns={"Claim": "text", "ClaimLabel": "orig_label",
                         "ClaimSource": "source_url",
                         "Category": "category"})
label_map = {"True": "credible", "False": "not credible",
             True: "credible", False: "not credible"}
df["label"]              = df["orig_label"].map(label_map)
df                       = df.dropna(subset=["label", "text"])
df["text"]               = df["text"].str.strip()
df["source"]             = "ArCOV19-claims"
df["evidence_text"]      = ""
df["source_name"]        = "ArCOV19 fact-check"
df["source_type"]        = "fact-checked claim"
df["fact_check_snippet"] = df["text"]   # the claim itself is the fact-checked statement
parts.append(df[COLS])
print(f"ArCOV19 claims : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")


# ── 5. VERA-ARAB ─────────────────────────────────────────────────────────────
vera_path = os.path.join(DATA_DIR, "VERA-ARAB", "peerj-cs-10-2432-s001.xlsx")
df = pd.read_excel(vera_path, usecols=[
    "text_arabic", "label",
    "urls", "domain", "domain_label",
    "annotations_text", "annotations_type",
    "context_domain", "context_domain_desc",
])
df = df.rename(columns={"text_arabic": "text"})
label_map = {0: "credible", 1: "not credible"}
df["label"]              = df["label"].map(label_map)
df                       = df.dropna(subset=["label", "text"])
df["text"]               = df["text"].str.strip()
df["source"]             = "VERA-ARAB"
df["evidence_text"]      = ""
df["source_url"]         = df["urls"].apply(_parse_list_field)
df["source_name"]        = df["context_domain"].apply(_parse_list_field)
df["source_type"]        = "social media tweet"
df["fact_check_snippet"] = df["annotations_text"].apply(_join_list_field)
# category: prefer context_domain_desc (entity type), fall back to domain
df["category"]           = df.apply(
    lambda r: _join_list_field(r["context_domain_desc"]) or str(r["domain"]).strip(),
    axis=1
)
parts.append(df[COLS])
print(f"VERA-ARAB      : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")


# ── 6. AraStance (all splits) ────────────────────────────────────────────────
arastance_files = [
    os.path.join(DATA_DIR, "AraStance", "arastance_train.csv"),
    os.path.join(DATA_DIR, "AraStance", "arastance_valid.csv"),
    os.path.join(DATA_DIR, "AraStance", "arastance_test.csv"),
]
label_map = {"Agree": "credible", "Disagree": "not credible"}
df_list = []
for f in arastance_files:
    tmp = pd.read_csv(f)
    tmp["label"] = tmp["stance"].map(label_map)
    tmp = tmp.dropna(subset=["label"])          # drops Discuss + Unrelated
    tmp["text"]            = (tmp["claim"].fillna("") + " " + tmp["article"].fillna("")).str.strip()
    tmp["evidence_text"]   = tmp["article"].fillna("").str.strip()
    tmp["source_url"]      = ""
    tmp["source_name"]     = ""
    tmp["source_type"]     = "news article"
    tmp["fact_check_snippet"] = ""
    tmp["category"]        = ""
    tmp["source"]          = "AraStance"
    df_list.append(tmp[COLS])

df = pd.concat(df_list, ignore_index=True)
parts.append(df)
print(f"AraStance      : {len(df):>7,} rows  | {df['label'].value_counts().to_dict()}")
print("  (Discuss / Unrelated dropped; article column kept as evidence_text)")


# ── Combine & save ────────────────────────────────────────────────────────────
unified = pd.concat(parts, ignore_index=True)
unified = unified[unified["text"].str.strip() != ""]

# Coverage summary
print(f"\n{'='*60}")
print(f"TOTAL              : {len(unified):>7,} rows")
print(f"Label split        : {unified['label'].value_counts().to_dict()}")
print(f"Source breakdown   : {unified['source'].value_counts().to_dict()}")
print()
for col in ["evidence_text", "source_url", "source_name",
            "source_type", "fact_check_snippet", "category"]:
    filled = (unified[col].str.strip() != "").sum()
    print(f"  {col:<22}: {filled:>7,} rows filled  ({100*filled/len(unified):.1f}%)")

unified.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT}")
