"""Score a single annotator against the silver labels.

With one annotator there is no inter-annotator agreement to report. What the
sample still measures is how often the dataset's silver labels match a human
reading of the evidence under the written codebook. That is a rate, so its
precision depends on the number of rows checked rather than on what fraction
of the corpus they are, and this module reports it with intervals that account
for repeated evidence passages.

Rows 1 to 20 are excluded from the primary figures, as in the original
protocol. Six of them were displayed with their silver labels during data
inspection, so excluding the block keeps the scored rows clean; all 240 are
reported beside them for reference.

Faithfulness is not scored. Annotator A's pass used the v2 wording, answered
High on every row, and so carries no information about the silver field.

Outputs ``results/improvements/annotation/single_annotator_*.csv`` and a JSON
summary used by the report and the workbook.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / "results" / "improvements" / "annotation"
DATA = ROOT / "data" / "final_studies" / "combined_three_class_test.csv"

EXCLUDED_ROWS = 20
BOOTSTRAP = 4000
SEED = 42
VERDICTS = ["Real", "Fake", "Unverified"]
RELEVANCE = ["Relevant", "Partially Relevant", "Irrelevant"]


def arabic(text: str) -> str:
    return re.sub(r"[^؀-ۿ0-9]+", " ", html.unescape(str(text))).strip()


def load(path: Path) -> pd.DataFrame:
    """Read a sheet and attach the key, the evidence groups and the texts.

    A sheet exported from the bench carries only the answers, so the claim and
    evidence text are joined back from the corpus rather than assumed present.
    """
    sheet = pd.read_csv(path, dtype=str).fillna("")
    key = pd.read_csv(ANNOTATION_DIR / "annotation_key.csv", dtype=str).fillna("")
    corpus = pd.read_csv(
        DATA,
        usecols=["sample_id", "evidence_group", "claim_text", "evidence_text"],
        dtype=str,
    ).fillna("")
    sheet = sheet[[c for c in sheet.columns
                   if c not in ("claim_text", "evidence_text")]]
    frame = sheet.merge(key, on="annotation_id").merge(corpus, on="sample_id", how="left")
    frame["id"] = frame["annotation_id"].astype(int)
    frame["corpus"] = frame["stratum"].str.split(" | ", regex=False).str[1]
    words = frame["evidence_text"].map(lambda t: len(html.unescape(str(t)).split()))
    frame["short_evidence"] = words <= 5
    frame["copies_claim"] = [
        len(arabic(c)) >= 20 and arabic(c)[: min(60, len(arabic(c)))] in arabic(e)
        for c, e in zip(frame["claim_text"], frame["evidence_text"])
    ]
    return frame


def cluster_interval(frame: pd.DataFrame, a: str, b: str) -> dict:
    """Agreement and kappa with 95% intervals from a bootstrap over evidence groups."""
    rng = np.random.default_rng(SEED)
    index = frame.groupby("evidence_group").indices
    blocks = list(index.values())
    agree, kappa = [], []
    for _ in range(BOOTSTRAP):
        pick = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        sample = frame.iloc[pick]
        agree.append(float((sample[a] == sample[b]).mean()))
        if sample[a].nunique() > 1 or sample[b].nunique() > 1:
            kappa.append(cohen_kappa_score(sample[a], sample[b]))
    n = len(frame)
    p = float((frame[a] == frame[b]).mean())
    binomial = 1.96 * np.sqrt(p * (1 - p) / n)
    return {
        "n": n,
        "evidence_groups": len(blocks),
        "agreement": p,
        "agreement_ci": [float(np.percentile(agree, 2.5)), float(np.percentile(agree, 97.5))],
        "agreement_half_width_if_independent": float(binomial),
        "kappa": float(cohen_kappa_score(frame[a], frame[b])),
        "kappa_ci": [float(np.percentile(kappa, 2.5)), float(np.percentile(kappa, 97.5))],
    }


PRIMARY = ANNOTATION_DIR / "annotation_sheet_annotator_A.csv"
EARLIER = ANNOTATION_DIR / "completed" / "annotation_sheet_annotator_A_2026-09-12.csv"


def test_retest(primary: pd.DataFrame) -> dict:
    """Agreement between the annotator's two passes over the same rows.

    With one annotator this is the only reliability estimate available. It is
    an upper bound on what two people would achieve, since the same reader
    brings the same habits to both passes.
    """
    if not EARLIER.exists():
        return {}
    earlier = load(EARLIER)[["annotation_id", "verdict", "evidence_relevance"]]
    merged = primary.merge(earlier, on="annotation_id", suffixes=("", "_pass1"))
    merged = merged[merged["id"] > EXCLUDED_ROWS]
    return {
        "verdict": cluster_interval(merged, "verdict", "verdict_pass1"),
        "relevance": cluster_interval(
            merged, "evidence_relevance", "evidence_relevance_pass1"
        ),
        "pass1_vs_silver": cluster_interval(merged, "verdict_pass1", "silver_verdict"),
    }


def main(sheet: str | None = None) -> None:
    frame = load(Path(sheet) if sheet else PRIMARY)
    scored = frame[frame["id"] > EXCLUDED_ROWS]
    summary = {"sheet": str(sheet or PRIMARY.name), "excluded_rows": EXCLUDED_ROWS}

    summary["verdict"] = cluster_interval(scored, "verdict", "silver_verdict")
    summary["verdict_all_240"] = cluster_interval(frame, "verdict", "silver_verdict")
    relevance = scored[scored["silver_evidence_relevance"].str.strip() != ""]
    summary["relevance"] = cluster_interval(
        relevance, "evidence_relevance", "silver_evidence_relevance"
    )
    faith = scored[scored["silver_faithfulness"].str.strip() != ""]
    if len(faith) and faith["faithfulness"].str.strip().ne("").all():
        summary["faithfulness"] = cluster_interval(
            faith, "faithfulness", "silver_faithfulness"
        )
    summary["test_retest"] = test_retest(frame)

    committed_silver = scored[scored["silver_verdict"].isin(["Real", "Fake"])]
    summary["silver_real_fake_called_unverified"] = float(
        (committed_silver["verdict"] == "Unverified").mean()
    )
    uae_relevant = relevance[relevance["silver_evidence_relevance"] == "Relevant"]
    summary["silver_relevant_judged_irrelevant"] = {
        "n": int(len(uae_relevant)),
        "judged_irrelevant": int((uae_relevant["evidence_relevance"] == "Irrelevant").sum()),
    }
    summary["codebook_checks"] = {
        "short_evidence_rows": int(frame["short_evidence"].sum()),
        "short_evidence_marked_unverified": int(
            (frame["short_evidence"] & (frame["verdict"] == "Unverified")).sum()
        ),
        "copied_claim_rows": int(frame["copies_claim"].sum()),
        "copied_claim_marked_unverified": int(
            (frame["copies_claim"] & (frame["verdict"] == "Unverified")).sum()
        ),
    }
    # Sensitivity: the copied-claim rows are the ones that break the codebook.
    # Scoring them as the rule requires shows whether they move the result.
    ruled = scored.copy()
    ruled.loc[ruled["copies_claim"], "verdict"] = "Unverified"
    summary["verdict_with_rule_applied"] = {
        "agreement": float((ruled["verdict"] == ruled["silver_verdict"]).mean()),
        "kappa": float(cohen_kappa_score(ruled["verdict"], ruled["silver_verdict"])),
    }

    per_stratum = (
        scored.assign(agree=scored["verdict"] == scored["silver_verdict"])
        .groupby("stratum")
        .agg(rows=("agree", "size"), agreement=("agree", "mean"),
             human_unverified=("verdict", lambda s: float((s == "Unverified").mean())))
        .reset_index()
    )
    confusion = pd.crosstab(
        scored["verdict"].rename("annotator"),
        scored["silver_verdict"].rename("silver"),
    ).reindex(index=VERDICTS, columns=VERDICTS, fill_value=0)

    # How precise a rate estimate is at other sample sizes, for the question of
    # whether annotating more rows would change anything.
    p = summary["verdict"]["agreement"]
    ratio = (
        (summary["verdict"]["agreement_ci"][1] - summary["verdict"]["agreement_ci"][0])
        / (2 * summary["verdict"]["agreement_half_width_if_independent"])
    )
    summary["cluster_width_ratio"] = float(ratio)
    summary["half_width_by_n"] = {
        str(n): float(1.96 * np.sqrt(p * (1 - p) / n) * ratio)
        for n in [60, 100, 220, 500, 1000, 6369]
    }

    per_stratum.to_csv(ANNOTATION_DIR / "single_annotator_by_stratum.csv",
                       index=False, encoding="utf-8-sig")
    confusion.to_csv(ANNOTATION_DIR / "single_annotator_verdict_confusion.csv",
                     encoding="utf-8-sig")
    (ANNOTATION_DIR / "single_annotator_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    v, r = summary["verdict"], summary["relevance"]
    print(f"VERDICT vs silver, rows 21-240 (n={v['n']}, {v['evidence_groups']} evidence groups)")
    print(f"  agreement {v['agreement']:.1%}  95% CI {v['agreement_ci'][0]:.1%} to {v['agreement_ci'][1]:.1%}")
    print(f"  kappa     {v['kappa']:.3f}  95% CI {v['kappa_ci'][0]:.3f} to {v['kappa_ci'][1]:.3f}")
    print(f"RELEVANCE vs silver (n={r['n']})")
    print(f"  agreement {r['agreement']:.1%}  kappa {r['kappa']:.3f}  95% CI {r['kappa_ci'][0]:.3f} to {r['kappa_ci'][1]:.3f}")
    if "faithfulness" in summary:
        f = summary["faithfulness"]
        print(f"FAITHFULNESS vs silver (n={f['n']})")
        print(f"  agreement {f['agreement']:.1%}  kappa {f['kappa']:.3f}  95% CI {f['kappa_ci'][0]:.3f} to {f['kappa_ci'][1]:.3f}")
    if summary["test_retest"]:
        t = summary["test_retest"]
        print("TEST-RETEST, the same annotator's two passes")
        for field in ["verdict", "relevance"]:
            x = t[field]
            print(f"  {field:10s} agreement {x['agreement']:.1%}  kappa {x['kappa']:.3f}  "
                  f"95% CI {x['kappa_ci'][0]:.3f} to {x['kappa_ci'][1]:.3f}")
        p1 = t["pass1_vs_silver"]
        print(f"  first pass vs silver: agreement {p1['agreement']:.1%}  kappa {p1['kappa']:.3f}")
    print(f"silver Real/Fake the annotator called Unverified: {summary['silver_real_fake_called_unverified']:.1%}")
    s = summary["silver_relevant_judged_irrelevant"]
    print(f"silver 'Relevant' UAE rows judged Irrelevant: {s['judged_irrelevant']} of {s['n']}")
    c = summary["codebook_checks"]
    print(f"codebook: short evidence -> Unverified {c['short_evidence_marked_unverified']}/{c['short_evidence_rows']}, "
          f"copied claim -> Unverified {c['copied_claim_marked_unverified']}/{c['copied_claim_rows']}")
    w = summary["verdict_with_rule_applied"]
    print(f"with the copied-claim rule applied: agreement {w['agreement']:.1%}, kappa {w['kappa']:.3f}")
    print("\nper stratum:")
    print(per_stratum.round(3).to_string(index=False))
    print(f"\nclustering widens the interval {ratio:.2f}x; half-width of the agreement estimate by rows checked:")
    for n, hw in summary["half_width_by_n"].items():
        print(f"  {n:>5} rows -> +/- {hw:.1%}")


if __name__ == "__main__":
    main(*sys.argv[1:])
