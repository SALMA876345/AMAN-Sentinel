"""
Build a CLAIM-TRUTH-ALIGNED training set that matches the UAE-MVD test task.

Problem this fixes
──────────────────
The old four_dataset_training.csv was 78% AFND, whose labels mean *source
credibility* (is the outlet trustworthy?), not *claim truthfulness* (is the
statement true?). Training on that and testing on UAE-MVD claim verification
is a task mismatch — baselines scored BELOW the majority-class baseline.

What this builds
────────────────
  Real / Fake  : from claim-truth datasets only — VERA-ARAB, AraStance,
                 ArCOV19 (genuine claim-level truthfulness labels).
  Unverified   : kept from AFND (uncertainty class — less tied to source
                 credibility) + AraStance's Unverified, capped for balance.
  evidence_text: populated for every row, matching the test set which has
                 retrieved evidence:
                   • VERA-ARAB / ArCOV19-claims → fact_check_snippet
                   • AraStance                  → existing evidence_text
                   • rows with no evidence      → BM25-retrieved top-3 passages

Output: data/claim_training.csv  (columns: text, label, evidence_text, source)
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("pip install rank-bm25")

SRC_FILE  = os.path.join(DATA_DIR, "four_dataset_training.csv")
OUT_FILE  = os.path.join(DATA_DIR, "claim_training.csv")

CLAIM_TRUTH_SOURCES = ["VERA-ARAB", "AraStance", "ArCOV19-claims", "ArCOV19-tweets"]
UNVERIFIED_CAP      = 15_000   # cap AFND Unverified so it doesn't dominate
CORPUS_LIMIT        = 10_000   # BM25 evidence pool (10k keeps pure-Python get_scores fast)
TOP_K               = 3


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    print("Loading source ...")
    df = pd.read_csv(SRC_FILE, low_memory=False)

    # ── 1. Real / Fake from claim-truth datasets ──────────────────────────────
    ct = df[df["source"].isin(CLAIM_TRUTH_SOURCES)]
    real_fake = ct[ct["label"].isin(["Real", "Fake"])].copy()

    # ── 2. Unverified: AFND (capped) + any claim-truth Unverified ─────────────
    afnd_unv = df[(df["source"] == "AFND") & (df["label"] == "Unverified")]
    if len(afnd_unv) > UNVERIFIED_CAP:
        afnd_unv = afnd_unv.sample(UNVERIFIED_CAP, random_state=42)
    ct_unv = ct[ct["label"] == "Unverified"].copy()
    unverified = pd.concat([afnd_unv, ct_unv], ignore_index=True)

    out = pd.concat([real_fake, unverified], ignore_index=True)
    print(f"Selected rows: {len(out):,}  |  {out['label'].value_counts().to_dict()}")

    # ── 3. Populate evidence_text ─────────────────────────────────────────────
    # 3a. Use fact_check_snippet where evidence_text is empty
    ev = out["evidence_text"].fillna("").astype(str)
    fc = out["fact_check_snippet"].fillna("").astype(str)
    ev = ev.where(ev.str.strip() != "", fc)
    out["evidence_text"] = ev

    # 3b. BM25-retrieve evidence for rows still empty
    need = out["evidence_text"].str.strip() == ""
    n_need = int(need.sum())
    print(f"Rows needing BM25 evidence: {n_need:,}")

    if n_need > 0:
        print(f"Building BM25 evidence pool (cap={CORPUS_LIMIT:,}) ...")
        pool = df["text"].dropna().astype(str)
        pool = pool[pool.str.len() > 40].head(CORPUS_LIMIT).tolist()
        bm25 = BM25Okapi([normalize_arabic(t).split() for t in pool])
        print(f"  Pool ready — {len(pool):,} passages")

        filled = []
        need_idx = out.index[need].tolist()
        for n, idx in enumerate(need_idx):
            if n % 2000 == 0:
                print(f"  retrieving {n}/{n_need} ...", flush=True)
            q = normalize_arabic(out.at[idx, "text"]).split()
            if not q:
                filled.append("")
                continue
            scores = bm25.get_scores(q)
            top = np.argsort(scores)[::-1][:TOP_K]
            filled.append(" ".join(pool[j] for j in top))
        out.loc[need_idx, "evidence_text"] = filled

    # ── 4. Final tidy + save ──────────────────────────────────────────────────
    out = out[["text", "label", "evidence_text", "source"]].copy()
    out = out.dropna(subset=["text", "label"])
    out = out[out["text"].astype(str).str.strip() != ""].reset_index(drop=True)

    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {OUT_FILE}")
    print(f"Total rows : {len(out):,}")
    print(f"Labels     : {out['label'].value_counts().to_dict()}")
    print(f"Sources    : {out['source'].value_counts().to_dict()}")
    print(f"Evidence populated: {(out['evidence_text'].astype(str).str.strip()!='').mean()*100:.1f}%")


if __name__ == "__main__":
    main()
