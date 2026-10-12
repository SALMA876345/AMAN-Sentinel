"""Leakage-respecting retrieval evaluation for AMAN Sentinel Phase 3.

Evaluates four retrieval methods -- BM25, Dense, Hybrid (existing weighted
min-max fusion), Hybrid-RRF (Reciprocal Rank Fusion, new in this module) --
using `phase3.retrieval` (which wraps the existing `HybridRetriever`).

GROUND TRUTH CAVEAT (read this before trusting the numbers):
--------------------------------------------------------------
The retrieval corpus is built from TRAIN + VALIDATION evidence only (TEST
evidence is never indexed, to avoid leaking the answer). A TEST claim's own
`evidence_text` is its nominal "correct" evidence, but Recall/MRR/HitRate can
only be computed for a query if that exact gold evidence string also happens
to appear in the TRAIN+VALIDATION corpus (otherwise there is no judged
relevant document in the collection to rank, and the metric is undefined for
that query -- it is excluded, not scored as 0).

Empirically, whether that overlap exists is almost entirely a function of
source provenance:
  * AraStance ("native" ground truth: each claim is paired with a unique,
    independently-authored news article) -- essentially NEVER reuses the
    same article across claims, so a TEST claim's gold article almost never
    also appears among TRAIN/VALIDATION evidence. In this dataset it is
    exactly 0 of 169 AraStance TEST rows.
  * ArCOV19-tweets / ArCOV19-claims -- same pattern, 0 of 638 TEST rows.
  * VERA-ARAB ("backfilled" ground truth: evidence was itself assigned by a
    prior TF-IDF retrieval step against a shared passage pool during Phase 1
    restoration, see hackathon/restore_claim_training_fast.py) -- the same
    passage is legitimately shared across many claims, so a large fraction
    of TEST rows (44.7%, 1428/3193) DO have their gold passage present in
    the TRAIN+VALIDATION corpus.

Net effect: under the required "corpus excludes TEST evidence" constraint,
quantitative Recall/MRR/HitRate for the NATIVE (AraStance) group is not
computable at all (N=0) -- this is a structural fact about the data, not a
bug in this script, and it is reported explicitly rather than papered over.
All measurable signal comes from VERA-ARAB, so "backfilled" and "pooled"
numbers in this report are, in effect, VERA-ARAB-only numbers.

Also note: because every query here has exactly one gold relevant document,
Recall@K and HitRate@K are numerically identical (recall's denominator is
always 1). Both are reported for completeness, but they are not independent
signals.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from phase3 import retrieval as R  # noqa: E402

CLEAN_SPLIT_DIR = os.path.join(_REPO_ROOT, "hackathon", "data", "clean_split")
TEST_CSV = os.path.join(CLEAN_SPLIT_DIR, "source_binary_test.csv")
VAL_CSV = os.path.join(CLEAN_SPLIT_DIR, "source_binary_validation.csv")

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
REPORT_JSON = os.path.join(REPORT_DIR, "retrieval_evaluation.json")

SEED = 42
SAMPLE_SIZE = 500  # cap on evaluable TEST queries actually scored, see note in report
K_LIST = [1, 3, 5, 10]
NATIVE_SOURCES = {"AraStance"}
BACKFILLED_SOURCES = {"VERA-ARAB", "ArCOV19-tweets", "ArCOV19-claims"}

METHODS = ["bm25", "dense", "hybrid_weighted", "hybrid_rrf"]


def _gold_doc_lookup(corpus: pd.DataFrame) -> dict[str, int]:
    """evidence_norm -> row position in retriever.df (0-indexed)."""
    return {norm: i for i, norm in enumerate(corpus["evidence_norm"].tolist())}


def _rank_of(scores: np.ndarray, gold_pos: int) -> int:
    """1-indexed rank of gold_pos under descending sort of `scores`."""
    order = np.argsort(-scores, kind="stable")
    rank = int(np.where(order == gold_pos)[0][0]) + 1
    return rank


def _empty_metric_block() -> dict:
    return {
        "n": 0,
        "recall_at_k": {str(k): None for k in K_LIST},
        "hitrate_at_k": {str(k): None for k in K_LIST},
        "mrr": None,
        "note": "no evaluable queries (gold evidence not present in TRAIN+VALIDATION corpus)",
    }


def _aggregate(ranks: list[int]) -> dict:
    n = len(ranks)
    if n == 0:
        return _empty_metric_block()
    ranks_arr = np.array(ranks, dtype=float)
    recall = {str(k): float(np.mean(ranks_arr <= k)) for k in K_LIST}
    return {
        "n": n,
        "recall_at_k": recall,
        "hitrate_at_k": dict(recall),  # identical here: exactly one relevant doc per query
        "mrr": float(np.mean(1.0 / ranks_arr)),
    }


def quick_validation_weight_check(retriever, val_df: pd.DataFrame, gold_lookup: dict, n_probe: int = 80) -> dict:
    """Lightweight sanity check (VALIDATION only, never TEST) on whether the
    existing 50/50 bm25/dense weights or RRF k=60 need adjusting. Validation
    evidence is trivially inside the corpus (it's part of the corpus by
    construction), so this is only a rough directional probe, not a
    held-out metric -- used solely to decide whether default settings are
    worth deviating from given hackathon time constraints."""
    rng = random.Random(SEED)
    rows = val_df[val_df["evidence_norm"].astype(str).str.strip() != ""]
    rows = rows.sample(min(n_probe, len(rows)), random_state=SEED)
    weighted_ranks, rrf_ranks = [], []
    for _, row in rows.iterrows():
        gold_pos = gold_lookup.get(str(row["evidence_norm"]))
        if gold_pos is None:
            continue
        bm25_s, dense_s = R.raw_scores(retriever, row["claim_text"])
        w = R.weighted_fuse(retriever, bm25_s, dense_s)
        rrf = R.rrf_fuse(bm25_s, dense_s, k=R.RRF_K)
        weighted_ranks.append(_rank_of(w, gold_pos))
        rrf_ranks.append(_rank_of(rrf, gold_pos))
    return {
        "n_probe": len(weighted_ranks),
        "hybrid_weighted_mrr": float(np.mean(1.0 / np.array(weighted_ranks))) if weighted_ranks else None,
        "hybrid_rrf_mrr": float(np.mean(1.0 / np.array(rrf_ranks))) if rrf_ranks else None,
        "decision": "kept existing defaults (bm25_weight=0.5, dense_weight=0.5, RRF k=60); "
        "no strong validation-set signal to justify retuning within the hackathon time budget",
    }


def main() -> None:
    t_start = time.time()
    os.makedirs(REPORT_DIR, exist_ok=True)

    corpus = R.build_corpus()
    gold_lookup = _gold_doc_lookup(corpus)
    retriever = R.get_retriever(top_k=10)
    assert len(retriever.df) == len(corpus), "HybridRetriever dropped corpus rows unexpectedly"

    test = pd.read_csv(TEST_CSV)
    test["evidence_norm"] = test["evidence_norm"].fillna("").astype(str)
    test["gold_pos"] = test["evidence_norm"].map(gold_lookup)
    test["evaluable"] = test["gold_pos"].notna()

    coverage_by_source = (
        test.groupby("source")["evaluable"]
        .agg(n_total="count", n_evaluable="sum")
        .reset_index()
    )
    coverage_by_source["n_evaluable"] = coverage_by_source["n_evaluable"].astype(int)

    evaluable = test[test["evaluable"]].copy()
    evaluable_native = evaluable[evaluable["source"].isin(NATIVE_SOURCES)]
    evaluable_backfilled = evaluable[evaluable["source"].isin(BACKFILLED_SOURCES)]

    sampled_native = evaluable_native  # already 0 rows expected; keep logic general
    if len(evaluable_backfilled) > SAMPLE_SIZE:
        sampled_backfilled = evaluable_backfilled.sample(SAMPLE_SIZE, random_state=SEED)
    else:
        sampled_backfilled = evaluable_backfilled

    sample_reduction_note = (
        f"Evaluable TEST queries (gold evidence present in TRAIN+VALIDATION corpus) = "
        f"{len(evaluable)} of {len(test)} total TEST rows. Of these, {len(evaluable_native)} are "
        f"native (AraStance) and {len(evaluable_backfilled)} are backfilled (VERA-ARAB / "
        f"ArCOV19-tweets / ArCOV19-claims). To stay within the ~20-30 minute compute budget for "
        f"this hackathon task, the backfilled-evaluable pool was downsampled with a FIXED SEED "
        f"({SEED}) from {len(evaluable_backfilled)} to {len(sampled_backfilled)} queries "
        f"(cap={SAMPLE_SIZE}). The native-evaluable pool ({len(evaluable_native)}) was used in "
        f"full (no reduction needed)."
    )

    quick_tune_check = quick_validation_weight_check(
        retriever, pd.read_csv(VAL_CSV), gold_lookup
    )

    all_sampled = pd.concat([sampled_native, sampled_backfilled], ignore_index=True)

    ranks: dict[str, list[tuple[str, int]]] = {m: [] for m in METHODS}  # method -> [(source, rank)]
    for _, row in all_sampled.iterrows():
        gold_pos = int(row["gold_pos"])
        bm25_s, dense_s = R.raw_scores(retriever, row["claim_text"])
        scores = {
            "bm25": bm25_s,
            "dense": dense_s,
            "hybrid_weighted": R.weighted_fuse(retriever, bm25_s, dense_s),
            "hybrid_rrf": R.rrf_fuse(bm25_s, dense_s, k=R.RRF_K),
        }
        for m in METHODS:
            r = _rank_of(scores[m], gold_pos)
            ranks[m].append((row["source"], r))

    def ranks_for(method: str, sources: set[str] | None) -> list[int]:
        if sources is None:
            return [r for _, r in ranks[method]]
        return [r for s, r in ranks[method] if s in sources]

    results = {}
    for m in METHODS:
        results[m] = {
            "native": _aggregate(ranks_for(m, NATIVE_SOURCES)),
            "backfilled": _aggregate(ranks_for(m, BACKFILLED_SOURCES)),
            "pooled": _aggregate(ranks_for(m, None)),
        }

    elapsed = time.time() - t_start

    report = {
        "methodology": {
            "corpus": "TRAIN + VALIDATION evidence_text only, deduped by evidence_norm "
            f"({len(corpus)} unique passages). TEST evidence is never indexed.",
            "queries": "TEST split claim_text only (held out, never used to tune anything).",
            "ground_truth": "A query's gold document is its own evidence_norm string, looked up "
            "in the corpus. Queries whose gold evidence is absent from the TRAIN+VALIDATION "
            "corpus are excluded (metric undefined), not scored as misses. See module docstring "
            "for the full native-vs-backfilled explanation.",
            "native_vs_backfilled": "native = AraStance (independently-authored article per claim, "
            "genuine ground truth). backfilled = VERA-ARAB / ArCOV19-tweets / ArCOV19-claims "
            "(evidence_text was itself selected by a prior TF-IDF retrieval step during Phase 1 "
            "restoration -- weaker evidence of true relevance; see "
            "hackathon/restore_claim_training_fast.py).",
            "structural_finding": "0 of 169 AraStance / 0 of 638 ArCOV19(-tweets/-claims) TEST rows "
            "have their gold evidence present in the TRAIN+VALIDATION corpus (each claim's article "
            "is essentially unique, never reused across claims/splits) vs. 1428 of 3193 (44.7%) for "
            "VERA-ARAB (evidence drawn from a shared, reusable passage pool). Consequently NATIVE "
            "metrics below are N=0 (not computable under the leakage-respecting corpus constraint), "
            "and 'backfilled'/'pooled' numbers are, in effect, VERA-ARAB-only numbers. This is "
            "reported explicitly rather than pooling it away.",
            "recall_hitrate_equivalence": "Every query has exactly one gold relevant document, so "
            "Recall@K and HitRate@K are numerically identical here; both are reported for "
            "completeness, not as independent signals.",
            "k_values": K_LIST,
            "rrf_k": R.RRF_K,
            "hybrid_weighted_weights": {"bm25_weight": retriever.bm25_weight, "dense_weight": retriever.dense_weight},
            "dense_model": R.DENSE_MODEL_NAME,
            "sample_reduction": sample_reduction_note,
            "sample_size_used": {"native": len(sampled_native), "backfilled": len(sampled_backfilled)},
            "seed": SEED,
            "validation_only_tuning_check": quick_tune_check,
            "eval_wallclock_seconds": round(elapsed, 1),
        },
        "coverage_by_source": coverage_by_source.to_dict(orient="records"),
        "metrics": results,
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote {REPORT_JSON}")
    print(f"Elapsed: {elapsed:.1f}s, sampled backfilled={len(sampled_backfilled)}, native={len(sampled_native)}")
    for m in METHODS:
        pooled = results[m]["pooled"]
        print(f"{m}: pooled n={pooled['n']} recall@5={pooled['recall_at_k'].get('5')} mrr={pooled['mrr']}")


if __name__ == "__main__":
    main()
