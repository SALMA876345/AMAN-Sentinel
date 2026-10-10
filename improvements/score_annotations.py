"""Score completed annotation sheets: Cohen's kappa and silver-label validation.

Reads the two blind sheets produced by ``annotation_sample.py`` and reports,
for each annotated field, agreement between the annotators and agreement of
each annotator with the silver label. The second number is the one the Dataset
section needs, because it is the only direct evidence that the silver labels
mean what the report says they mean.

Refuses to score sheets that are still blank rather than emitting a kappa of
zero that could be mistaken for a measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ANNOTATION_DIR = ROOT / "results" / "improvements" / "annotation"

# Rows 1-20 are the calibration set: both annotators discuss them before the
# main run, so they are no longer independent and are excluded from kappa.
CALIBRATION_ROWS = 20

FIELDS = {
    "verdict": "silver_verdict",
    "evidence_relevance": "silver_evidence_relevance",
    "faithfulness": "silver_faithfulness",
}

# Landis and Koch bands, used verbatim in the reporting guidance.
def interpret(kappa: float) -> str:
    if kappa < 0.0:
        return "poor (worse than chance)"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def load(name: str) -> pd.DataFrame:
    path = ANNOTATION_DIR / f"annotation_sheet_{name}.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run improvements/annotation_sample.py first."
        )
    sheet = pd.read_csv(path, dtype=str).fillna("")
    return apply_v3(apply_recheck(sheet, name), name)


def apply_recheck(sheet: pd.DataFrame, name: str) -> pd.DataFrame:
    """Merge a re-check file (revised verdicts, v3 faithfulness) if one exists.

    Only the columns the re-check was issued for are taken, and only where the
    annotator filled them in; the original sheet is otherwise untouched.
    """
    path = ANNOTATION_DIR / f"{name}_recheck.csv"
    if not path.exists():
        return sheet
    recheck = pd.read_csv(path, dtype=str).fillna("").set_index("annotation_id")
    sheet = sheet.set_index("annotation_id")
    revised = recheck["revised_verdict"].str.strip()
    revised = revised[revised.isin(["Real", "Fake", "Unverified"])]
    sheet.loc[revised.index, "verdict"] = revised
    faith = recheck["faithfulness_v3"].str.strip()
    faith = faith[faith.isin(["High", "Low"])]
    sheet["faithfulness_v3"] = ""
    sheet.loc[faith.index, "faithfulness_v3"] = faith
    print(f"[{name}] re-check merged: {len(revised)} verdicts revised, "
          f"{len(faith)} faithfulness judgments under v3")
    return sheet.reset_index()


def apply_v3(sheet: pd.DataFrame, name: str) -> pd.DataFrame:
    """Faithfulness under codebook v3, applied identically to every annotator.

    Unverified is Low by definition. For Real and Fake the annotator's v3
    judgment is used: the re-check value if one was issued, otherwise the
    sheet's own value, which is v3 for anyone who annotated after 12 September.
    A v2-era value with no v3 re-check is left blank rather than mixed in.
    """
    sheet = sheet.copy()
    verdict = sheet["verdict"].str.strip()
    has_recheck = "faithfulness_v3" in sheet.columns
    v3 = sheet["faithfulness_v3"] if has_recheck else sheet["faithfulness"]
    committed = verdict.isin(["Real", "Fake"])
    sheet["faithfulness"] = ""
    sheet.loc[verdict == "Unverified", "faithfulness"] = "Low"
    sheet.loc[committed, "faithfulness"] = v3[committed].str.strip()
    missing = int((committed & (sheet["faithfulness"] == "")).sum())
    if missing:
        print(f"[{name}] {missing} Real/Fake rows have no v3 faithfulness yet; "
              "they are left out of the faithfulness score")
    return sheet


def refuse_if_derived(merged: pd.DataFrame) -> None:
    """Stop if one sheet was produced by editing the other.

    Two people judging a three-way scale over hundreds of rows never match
    perfectly. A field that agrees on every single row means one sheet was
    copied from the other, and any kappa computed from the pair would look like
    a measurement while being an artefact of the copy.
    """
    for field in ["evidence_relevance", "verdict", "faithfulness"]:
        a, b = f"{field}_a", f"{field}_b"
        if a not in merged or b not in merged:
            continue
        both = merged[(merged[a].str.strip() != "") & (merged[b].str.strip() != "")]
        if len(both) >= 50 and (both[a] == both[b]).all():
            raise SystemExit(
                f"Refusing to score: '{field}' is identical on all {len(both)} "
                "rows both annotators completed. One sheet was derived from the "
                "other, so there is no independent agreement to measure. Use "
                "improvements/score_single_annotator.py, or have the second "
                "annotator start from a blank sheet."
            )


def main() -> None:
    a = load("annotator_A")
    b = load("annotator_B")
    key = pd.read_csv(ANNOTATION_DIR / "annotation_key.csv", dtype=str).fillna("")

    merged = (
        a.merge(b, on="annotation_id", suffixes=("_a", "_b"))
        .merge(key, on="annotation_id")
    )
    refuse_if_derived(merged)
    merged = merged[merged["annotation_id"].astype(int) > CALIBRATION_ROWS]
    print(f"scoring rows {CALIBRATION_ROWS + 1}-240 "
          f"(rows 1-{CALIBRATION_ROWS} are calibration and excluded)")

    # Agreement restricted to rows both annotators marked Sure. Reported beside
    # the full figure so a low overall kappa can be traced to genuinely hard
    # rows rather than to annotators misreading the codebook.
    both_sure = pd.Series(False, index=merged.index)
    if "confidence_a" in merged.columns and "confidence_b" in merged.columns:
        both_sure = (
            (merged["confidence_a"].str.strip() == "Sure")
            & (merged["confidence_b"].str.strip() == "Sure")
        )

    rows = []
    for field, silver_column in FIELDS.items():
        pair = merged[
            (merged[f"{field}_a"].str.strip() != "")
            & (merged[f"{field}_b"].str.strip() != "")
        ]
        if pair.empty:
            print(f"[{field}] no completed rows - skipping")
            continue

        annotator_a = pair[f"{field}_a"].str.strip()
        annotator_b = pair[f"{field}_b"].str.strip()
        silver = pair[silver_column].str.strip()

        inter = cohen_kappa_score(annotator_a, annotator_b)
        against_silver_a = cohen_kappa_score(annotator_a, silver)
        against_silver_b = cohen_kappa_score(annotator_b, silver)
        agreement = float((annotator_a == annotator_b).mean())

        sure = pair[both_sure.reindex(pair.index, fill_value=False)]
        sure_kappa = (
            cohen_kappa_score(
                sure[f"{field}_a"].str.strip(), sure[f"{field}_b"].str.strip()
            )
            if len(sure) >= 10
            else float("nan")
        )

        rows.append(
            {
                "Field": field,
                "N annotated": len(pair),
                "Percent agreement": agreement,
                "Cohen kappa (A vs B)": inter,
                "Interpretation": interpret(inter),
                "N both Sure": len(sure),
                "Kappa on both-Sure rows": sure_kappa,
                "Kappa (A vs silver)": against_silver_a,
                "Kappa (B vs silver)": against_silver_b,
            }
        )

        labels = sorted(set(annotator_a) | set(annotator_b))
        print(f"\n=== {field} (n={len(pair)}) ===")
        print(f"  percent agreement {agreement:.3f}")
        print(f"  Cohen kappa A vs B {inter:.3f} ({interpret(inter)})")
        print(f"  kappa vs silver: A {against_silver_a:.3f}, B {against_silver_b:.3f}")
        print("  A x B confusion matrix over " + ", ".join(labels))
        print(
            pd.DataFrame(
                confusion_matrix(annotator_a, annotator_b, labels=labels),
                index=labels,
                columns=labels,
            ).to_string()
        )

    if not rows:
        raise SystemExit(
            "\nBoth sheets are still blank. Annotate them before scoring;"
            " an unfilled sheet cannot produce a meaningful kappa."
        )

    frame = pd.DataFrame(rows)
    path = ANNOTATION_DIR / "agreement_summary.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
