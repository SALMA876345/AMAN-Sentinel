"""Build the annotation instrument the Dataset section is missing.

The brief requires an Annotation subsection covering labelling method,
guidelines, and inter-annotator agreement. The submitted report has no such
subsection: the UAE relevance, faithfulness, and stance fields are silver
labels derived from metadata, and their quality is asserted in the limitations
rather than measured.

Measuring it does not require re-annotating the corpus. A stratified sample,
annotated blind by two people against a written codebook, yields Cohen's kappa
for annotator agreement and a second kappa against the silver label, which is
the validation the silver labels currently lack.

This script produces the sample and the codebook. It deliberately does not
produce agreement numbers, because there is nothing to agree about until two
people have filled the sheet in - a fabricated kappa would be worse than the
present omission.

Run ``score_annotations`` afterwards on the completed sheet to compute kappa.

Outputs into ``results/improvements/annotation/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "final_studies"
OUT_DIR = ROOT / "results" / "improvements" / "annotation"

SEED = 42
SAMPLE_PER_STRATUM = 40

CODEBOOK = """\
# AFVD annotation codebook (v3)

v3 (12 September 2026) redefines faithfulness after annotator A's pass. The v2
question, "would a reader accept your verdict?", is naturally answered High for
an Unverified verdict on irrelevant evidence, since a reader would accept that
nothing is settled. That reading is reasonable and made the field uninformative:
annotator A marked High on all 240 rows. v3 asks instead whether the evidence
carries a definite verdict, which also matches the meaning of the silver
faithfulness field. A re-judges faithfulness under v3 on the rows where it is
not fixed by the verdict, before any comparison with annotator B is made. v3
also moves the calibration discussion to after both annotators finish, because
A completed the whole sheet first.

v2 closes a gap in v1, found on inspection before annotation began. v1 said
"Real: the evidence states the claim is accurate", which let a post that merely
repeats a rumour count as confirming it. In this corpus the evidence for many
debunked claims is other posts spreading the same story, so v1 would have had
annotators mark known falsehoods as Real. v2 requires evidence to vouch for a
verdict, not just restate the claim.

## Purpose
Validate the silver `three_class_label`, `retrieval_quality` and
`faithfulness_label` fields on a stratified sample, and report Cohen's kappa
between two independent annotators and against the silver label.

## Procedure
Two annotators label every row independently, without seeing each other's
sheet, the silver label, or the `source` column. Work through the sheet in the
order given; it is already shuffled. Do not look the claim up online.

You are not deciding whether the claim is true. You are deciding whether THIS
evidence settles it. Most of the time in this corpus it does not, and when it
does not, Unverified is the correct and confident answer, not a fallback.

## Decide in this order

Stop at the first step that answers the question.

1. **Is there anything to read?** A single word, a place name, or a sentence
   fragment cannot confirm or refute anything. -> Unverified.

2. **Is the evidence independent of the claim?** If it only repeats the claim,
   either the same wording copied back or other posts spreading the same story,
   it is the claim circulating, not evidence about it. -> Unverified.

3. **Is it about this specific event?** Evidence that shares a place, a person
   or a topic but describes a different event does not address the claim.
   -> Unverified.

4. **Does it settle the claim?**
   - Real: a report, statement or fact-check confirms the specific assertion.
   - Fake: a report, statement or fact-check denies, corrects or debunks it.
   - Unverified: it discusses the claim without settling it either way.

Real and Fake both need evidence that vouches. Repetition is never confirmation,
and a rumour cannot debunk itself, so steps 1 to 3 can only ever give
Unverified.

## Fields to fill

### 1. verdict  -  one of {Real, Fake, Unverified}
Apply the four steps above.

### 2. evidence_relevance  -  one of {Relevant, Partially Relevant, Irrelevant}
- **Relevant**            Addresses the specific claim, its entities, and its time frame.
- **Partially Relevant**  Same topic or entity, but not the specific assertion.
- **Irrelevant**          Different topic, or generic boilerplate.

### 3. faithfulness  -  one of {High, Low}
Does this evidence, on its own, carry a definite verdict?
- **High** Your verdict is Real or Fake, and a reader who saw only this
           evidence would reach it without outside knowledge.
- **Low**  Your verdict is Real or Fake but rests partly on outside knowledge
           or on thin evidence, OR your verdict is Unverified.

Unverified is always Low: evidence that settles nothing carries no verdict.

### 4. confidence  -  one of {Sure, Unsure}
Mark Unsure when you could defend a different verdict. This is not a penalty:
it tells adjudication where to look, and it lets agreement be reported on the
confident rows separately from the hard ones.

### 5. notes  -  free text, optional
Use for dialect, sarcasm, temporal mismatch, or conflicting evidence.

## Independence and discussion
The annotators do not discuss any row, or share answers, counts or impressions,
until both have finished all 240. (v2 planned a calibration discussion after
the first 20 rows; annotator A finished the whole sheet first, so discussing
now would only pull B toward A's reading and inflate agreement.) Kappa is
reported on rows 21 to 240, as originally planned, and on all 240 for
reference. Disagreements are discussed and adjudicated afterwards.

## Adjudication
Disagreements are resolved by discussion after both sheets are complete and
kappa has been computed on the pre-adjudication labels. Report the
pre-adjudication kappa; the post-adjudication set becomes the gold subset.

## Reporting
Report, for each of the three fields: Cohen's kappa between annotators, kappa
of each annotator against the silver label, percentage agreement, and the
confusion matrix. Interpret with Landis and Koch: <0.20 slight, 0.21-0.40 fair,
0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect.
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    frame = pd.read_csv(
        DATA_DIR / "combined_three_class_test.csv", low_memory=False
    ).fillna("")

    # Stratify by silver label crossed with corpus, so the sample covers the
    # UAE-only Unverified rows that carry most of the labelling risk rather
    # than mirroring the corpus and burying them.
    frame["_stratum"] = (
        frame["three_class_label"].astype(str)
        + " | "
        + frame["dataset_family"].astype(str)
    )
    picked = []
    for stratum, group in frame.groupby("_stratum"):
        take = min(SAMPLE_PER_STRATUM, len(group))
        picked.append(group.sample(n=take, random_state=SEED))
    sample = pd.concat(picked).sample(frac=1.0, random_state=SEED)

    # The blind sheet withholds every silver field and the publisher name.
    sheet = pd.DataFrame(
        {
            "annotation_id": range(1, len(sample) + 1),
            "claim_text": sample["claim_text"].values,
            "evidence_text": sample["evidence_text"].values,
            "verdict": "",
            "evidence_relevance": "",
            "faithfulness": "",
            "confidence": "",
            "notes": "",
        }
    )

    # The key stays separate so the sheet can be handed out without the answers.
    key = pd.DataFrame(
        {
            "annotation_id": range(1, len(sample) + 1),
            "sample_id": sample["sample_id"].values,
            "stratum": sample["_stratum"].values,
            "silver_verdict": sample["three_class_label"].values,
            "silver_evidence_relevance": sample["retrieval_quality"].values,
            "silver_faithfulness": sample["faithfulness_label"].values,
            "source": sample["source"].values,
        }
    )

    for name in ["annotator_A", "annotator_B"]:
        path = OUT_DIR / f"annotation_sheet_{name}.csv"
        # Never overwrite a sheet someone has started: regenerating would
        # silently replace hours of annotation with blanks.
        if path.exists():
            existing = pd.read_csv(path, dtype=str).fillna("")
            answered = (
                existing.get("verdict", pd.Series(dtype=str)).str.strip() != ""
            ).sum()
            if answered:
                print(f"kept {path.name}: it already holds {answered} answers")
                continue
        sheet.to_csv(path, index=False, encoding="utf-8-sig")
    key.to_csv(OUT_DIR / "annotation_key.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "codebook.md").write_text(CODEBOOK, encoding="utf-8")

    print(f"sampled {len(sheet)} rows across {sample['_stratum'].nunique()} strata")
    print(key["stratum"].value_counts().to_string())
    print(f"\nwrote blind sheets, key and codebook to {OUT_DIR}")
    print("Fill annotation_sheet_annotator_A.csv and _B.csv, then run:")
    print("  python improvements/score_annotations.py")


if __name__ == "__main__":
    main()
