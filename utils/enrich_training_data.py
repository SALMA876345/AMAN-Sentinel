"""
Enriches unified_training.csv with retrieved evidence, URLs, fact-check snippets,
and source metadata — four phases, all runnable locally without API keys.

Phase 1 – ArCOV19 Claim Evidence from Relevant Tweets
    Reads claim_verification/relevant_tweets/claimXXX.txt for each of the 138 claims.
    Selects up to 2 same-label tweets (supporting) + 1 opposite-label tweet (refuting)
    and stores them as evidence_text.

Phase 2 – TF-IDF In-corpus Evidence Retrieval  (sklearn sparse matrices)
    Builds a TF-IDF matrix from a stratified 20,000-article AFND sample.
    Batch cosine-similarity retrieval finds the top-3 most relevant AFND articles
    for every ArCOV19 claim and AraStance row and appends them to evidence_text.

Phase 3 – Source Domain Classification
    Rule-based from the URL hostname.  Updates source_type for every row with a URL.

Phase 4 – Google Fact Check API  (optional)
    Set env var GOOGLE_FACT_CHECK_API_KEY to enable.
    Queries unique claim texts; merges returned rating + publisher into
    fact_check_snippet and source_name.

Output: data/unified_training_enriched.csv
"""

import os, re, time, glob, warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
INPUT    = os.path.join(DATA_DIR, "unified_training.csv")
OUTPUT   = os.path.join(DATA_DIR, "unified_training_enriched.csv")

GOOGLE_API_KEY = os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "")
AFND_SAMPLE_N  = 10_000   # per label (credible + not credible)


# ── Domain classification ─────────────────────────────────────────────────────
FACT_CHECKERS = {
    "misbar", "kashif", "fatabyyano", "arabfact", "snopes", "factcheck",
    "politifact", "fullfact", "africa-check", "checkyourfact", "verify-sy",
    "mafact", "teyit",
}
NEWS_AGENCIES = {
    "aljazeera", "bbc", "reuters", "apnews", "france24", "alarabiya",
    "skynewsarabia", "cnn", "euronews", "rt", "independent", "guardian",
    "washingtonpost", "nytimes", "huffpost", "albawaba", "middleeasteye",
    "aawsat", "asharq",
}
GOV_HINTS    = {".gov", ".mil", "who.int", "un.org", "unicef.org"}
SOCIAL_MEDIA = {"twitter", "x.com", "facebook", "instagram", "tiktok",
                "youtube", "telegram", "reddit", "linkedin"}


def classify_domain(url: str) -> str:
    if not url or str(url).strip() in ("", "nan"):
        return ""
    host = re.sub(r"https?://", "", url.lower()).split("/")[0]
    for s in SOCIAL_MEDIA:
        if s in host:  return "social media"
    for g in GOV_HINTS:
        if g in host:  return "government"
    for fc in FACT_CHECKERS:
        if fc in host: return "fact-checking"
    for na in NEWS_AGENCIES:
        if na in host: return "news agency"
    return "news website"


# ── Phase 1 ───────────────────────────────────────────────────────────────────
def phase1_arcov_tweet_evidence(df: pd.DataFrame) -> pd.DataFrame:
    rel_dir    = os.path.join(DATA_DIR, "ArCOV19-Rumors",
                              "claim_verification", "relevant_tweets")
    claim_files = sorted(glob.glob(os.path.join(rel_dir, "claim*.txt")))
    file_map    = {int(re.search(r"\d+", os.path.basename(f)).group()): f
                   for f in claim_files}

    mask    = df["source"] == "ArCOV19-claims"
    indices = list(df.index[mask])
    print(f"[Phase 1] Adding tweet evidence for {len(indices)} ArCOV19 claims ...",
          flush=True)

    for pos, idx in enumerate(indices, 1):
        f = file_map.get(pos)
        if f is None:
            continue
        try:
            rel = pd.read_csv(f, sep="\t", encoding="utf-8", on_bad_lines="skip")
        except Exception:
            continue

        claim_lbl    = df.at[idx, "label"]
        align_lbl    = "True"  if claim_lbl == "credible" else "False"
        contrast_lbl = "False" if claim_lbl == "credible" else "True"

        align    = rel[rel["label"] == align_lbl]["tweetText"].dropna()
        contrast = rel[rel["label"] == contrast_lbl]["tweetText"].dropna()

        snippets = ([str(t)[:200] for t in list(align)[:2]] +
                    ["[refuting] " + str(t)[:200] for t in list(contrast)[:1]])
        evidence = " ||| ".join(snippets)

        if evidence:
            existing = str(df.at[idx, "evidence_text"]).strip()
            df.at[idx, "evidence_text"] = (
                existing + " ||| " + evidence
                if existing and existing != "nan" else evidence
            )
        if not str(df.at[idx, "fact_check_snippet"]).strip():
            pool = list(align) + list(contrast)
            if pool:
                df.at[idx, "fact_check_snippet"] = str(pool[0])[:300]

    filled = (df.loc[mask, "evidence_text"].str.strip() != "").sum()
    print(f"  Done. evidence_text filled for {filled}/{len(indices)} claims.",
          flush=True)
    return df


# ── Phase 2 ───────────────────────────────────────────────────────────────────
def phase2_tfidf_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch TF-IDF cosine retrieval — builds one sparse matrix, then does a single
    matrix-multiply for all 1,609 queries.  Typically finishes in < 60 seconds.
    """
    query_mask = df["source"].isin(["ArCOV19-claims", "AraStance"])
    n_query    = query_mask.sum()
    print(f"\n[Phase 2] TF-IDF evidence retrieval for {n_query} claim rows ...",
          flush=True)

    # Stratified AFND sample
    afnd_df = df[df["source"] == "AFND"].reset_index(drop=True)
    s_cred  = afnd_df[afnd_df["label"] == "credible"].sample(
                  n=min(AFND_SAMPLE_N, (afnd_df["label"] == "credible").sum()),
                  random_state=42)
    s_nc    = afnd_df[afnd_df["label"] == "not credible"].sample(
                  n=min(AFND_SAMPLE_N, (afnd_df["label"] == "not credible").sum()),
                  random_state=42)
    sample  = pd.concat([s_cred, s_nc], ignore_index=True)
    del afnd_df

    print(f"  Fitting TF-IDF on {len(sample):,} AFND articles ...", flush=True)
    vec    = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    corpus = vec.fit_transform(sample["text"])          # sparse (20K × 50K)

    print(f"  Transforming {n_query} query texts ...", flush=True)
    query_texts = df.loc[query_mask, "text"].tolist()
    q_vecs      = vec.transform(query_texts)            # sparse (Q × 50K)

    print(f"  Computing cosine similarities (batch) ...", flush=True)
    sims = cosine_similarity(q_vecs, corpus)            # dense  (Q × 20K)

    texts_arr = sample["text"].to_numpy()
    for i, idx in enumerate(df.index[query_mask]):
        row     = sims[i]
        top3    = row.argsort()[-3:][::-1]
        snippets = [texts_arr[k][:200] for k in top3 if row[k] > 0]
        if snippets:
            retrieved = " ||| ".join(snippets)
            existing  = str(df.at[idx, "evidence_text"]).strip()
            df.at[idx, "evidence_text"] = (
                existing + " ||| " + retrieved
                if existing and existing != "nan" else retrieved
            )

    filled = (df.loc[query_mask, "evidence_text"].str.strip() != "").sum()
    print(f"  Done. evidence_text filled for {filled}/{n_query} rows.", flush=True)
    return df


# ── Phase 3 ───────────────────────────────────────────────────────────────────
def phase3_domain_classify(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[Phase 3] Classifying source domains ...", flush=True)
    has_url = df["source_url"].str.strip().fillna("") != ""
    generic = df["source_type"].isin(["", "nan", "Arabic news website",
                                      "news website"])
    classified = df.loc[has_url, "source_url"].apply(classify_domain)
    df.loc[has_url & generic, "source_type"] = classified[has_url & generic]
    print(f"  source_type dist: {df['source_type'].value_counts().to_dict()}",
          flush=True)
    return df


# ── Phase 4 ───────────────────────────────────────────────────────────────────
def phase4_google_factcheck(df: pd.DataFrame) -> pd.DataFrame:
    if not GOOGLE_API_KEY:
        print("\n[Phase 4] Skipped — set GOOGLE_FACT_CHECK_API_KEY to enable.",
              flush=True)
        return df

    import requests
    query_mask = df["source"].isin(["ArCOV19-claims", "AraStance"])
    claims     = df.loc[query_mask, "text"].unique()
    print(f"\n[Phase 4] Google Fact Check API: {len(claims)} queries ...",
          flush=True)
    BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    for i, claim_text in enumerate(claims, 1):
        try:
            r   = requests.get(BASE, params={"query": claim_text[:120],
                                             "languageCode": "ar",
                                             "key": GOOGLE_API_KEY}, timeout=10)
            items = r.json().get("claims", [])
            if not items:
                continue
            rev  = items[0].get("claimReview", [{}])[0]
            snip = (rev.get("textualRating", "") + " — " +
                    rev.get("title", "")).strip(" —")
            pub  = rev.get("publisher", {}).get("name", "")
            url  = rev.get("url", "")
            for idx in df.index[query_mask & (df["text"] == claim_text)]:
                if snip and not df.at[idx, "fact_check_snippet"]:
                    df.at[idx, "fact_check_snippet"] = snip
                if pub  and not df.at[idx, "source_name"]:
                    df.at[idx, "source_name"] = pub
                if url  and not df.at[idx, "source_url"]:
                    df.at[idx, "source_url"]  = url
        except Exception:
            pass
        if i % 50 == 0:
            print(f"  ... {i}/{len(claims)}", flush=True)
        time.sleep(0.5)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {INPUT} ...", flush=True)
    chunks = []
    for chunk in pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False,
                              dtype=str, chunksize=50_000):
        chunks.append(chunk.fillna(""))
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows.", flush=True)

    df = phase1_arcov_tweet_evidence(df)
    df = phase2_tfidf_evidence(df)
    df = phase3_domain_classify(df)
    df = phase4_google_factcheck(df)

    print(f"\n{'='*60}", flush=True)
    print(f"TOTAL rows : {len(df):,}", flush=True)
    for col in ["evidence_text", "source_url", "source_name",
                "source_type", "fact_check_snippet", "category"]:
        filled = (df[col].str.strip() != "").sum()
        print(f"  {col:<22}: {filled:>7,}  ({100*filled/len(df):.1f}%)",
              flush=True)

    print(f"\nWriting {OUTPUT} ...", flush=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
