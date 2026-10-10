"""Consolidate every improvement result into one Excel workbook.

The analyses write a dozen CSVs, which is the right format for re-running them
and the wrong one for reading them. This builds a single formatted workbook:
a summary sheet stating each finding in words, then one sheet per result table
with the numbers behind it.

Sheets whose source CSV does not exist yet are skipped, so the workbook can be
built while a slow analysis is still running and rebuilt afterwards.

Outputs ``results/improvements/FAAV_improvement_results.xlsx``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "improvements"
OUT_PATH = RESULTS_DIR / "FAAV_improvement_results.xlsx"

# Palette echoes the figures so the workbook and the PNGs read as one set.
INK = "FF0B0B0B"
MUTED = "FF52514E"
HEADER_FILL = "FF1F3A5F"
BAND_FILL = "FFF4F6F9"
GOOD_FILL = "FFDDF3E7"
BAD_FILL = "FFFBE4E0"
RULE_COLOR = "FFD8DCE3"

THIN = Side(style="thin", color=RULE_COLOR)
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# sheet name -> (csv file, one-line description, columns to render as percent-ish
# decimals). Order is the reading order, not alphabetical.
SHEETS = [
    (
        "Gate vs Confidence",
        "gate_vs_confidence.csv",
        "Does any gate rank errors better than plain softmax confidence? "
        "Positive advantage = gate wins. 22 of 24 say no.",
    ),
    (
        "Selective AURC",
        "selective_aurc.csv",
        "Area under the risk-coverage curve per selector. Lower is better.",
    ),
    (
        "Selective at Coverage",
        "selective_at_matched_coverage.csv",
        "Selective accuracy and macro F1 at 70/80/90% coverage.",
    ),
    (
        "Control Baselines",
        "control_baselines.csv",
        "How much of each label is recoverable from the publisher alone, "
        "with no claim or evidence text.",
    ),
    (
        "Effective Sample Size",
        "evidence_structure.csv",
        "Rows vs independent evidence clusters, and the resulting confidence "
        "interval inflation.",
    ),
    (
        "Ties",
        "best_results_tie_aware.csv",
        "Best-model selections, with three-way ties named instead of resolved "
        "silently.",
    ),
    (
        "Bootstrap CIs",
        "bootstrap_intervals.csv",
        "Macro F1 with cluster-aware 95% intervals, against the naive "
        "row-level interval.",
    ),
    (
        "Significance Tests",
        "significance_tests.csv",
        "Paired McNemar on shared test manifests.",
    ),
    (
        "NLI (not validated)",
        "faithfulness_nli.csv",
        "Entailment faithfulness. Not validated against human judgment, so not "
        "reported as faithfulness. Its disagreement with the silver labels is not "
        "evidence against it: the human audit finds the silver labels unreliable.",
    ),
    (
        "Human Audit",
        "annotation/single_annotator_by_stratum.csv",
        "One annotator against the silver verdicts, rows 21-240, by silver label "
        "and corpus. Second pass: agreement 25.9% (95% CI 18.6-33.0%), kappa "
        "-0.110 (-0.214 to -0.016). First pass: 35.9%, kappa 0.037.",
    ),
    (
        "Dataset Examples",
        "dataset_examples.csv",
        "Real anonymised claim-evidence rows to replace Table 5's constructed "
        "ones.",
    ),
]

SUMMARY = [
    ("FAAV improvement analyses", "", ""),
    (
        "Generated from results/improvements. Regenerate with "
        "improvements/run_improvements.ps1.",
        "",
        "",
    ),
    ("", "", ""),
    ("FINDING", "EVIDENCE", "WHERE"),
    (
        "In 22 of 24 comparisons no faithfulness gate beats plain softmax "
        "confidence: 16 are reliably worse, 6 indistinguishable.",
        "Both shipped gates lose in every in-domain setting: FAAV-Light by "
        "-0.042 to -0.071 AURC, FAAV-Full by -0.027 to -0.031.",
        "Gate vs Confidence",
    ),
    (
        "The single exception is the study the report deleted.",
        "Zero-shot UAE three-class: FAAV-Light beats confidence by +0.0689 "
        "AURC, 95% CI [+0.0512, +0.0937], winning 400 of 400 cluster "
        "resamples. Evidence gating earns its place under domain shift, not "
        "in-domain.",
        "Gate vs Confidence",
    ),
    (
        "The gate collapsed for a fixable reason, not an inherent one.",
        "It fits its 'is this verdict trustworthy?' target on the verifier's "
        "own training rows, where the verifier scores 0.956-0.980 against "
        "0.861-0.896 held out. It sees 2-4% negatives where reality is "
        "10-14%, so it learns to accept everything.",
        "Replication",
    ),
    (
        "Three of six 'best model' selections are three-way ties in which "
        "FAAV reproduces its own verifier row for row.",
        "Zero-shot binary 0.5637, source binary 0.9016, combined three-class "
        "0.8763 - each shared by AraBERT, FAAV-Light and FAAV-Full. "
        "final_best_results.csv names FAAV-Full the winner of the third.",
        "Ties",
    ),
    (
        "A provenance shortcut explains far more of the label than the "
        "majority baseline suggests.",
        "Publisher alone reaches 0.6009 macro F1 on combined three-class "
        "against the best model's 0.8763. Real headroom is 0.275, not the "
        "0.673 the majority floor implies.",
        "Control Baselines",
    ),
    (
        "Test sets are structurally much smaller than their row counts.",
        "The zero-shot UAE set has 6,390 rows but about 26 Kish-effective "
        "evidence clusters; one passage repeats 1,009 times.",
        "Effective Sample Size",
    ),
    (
        "Measured intervals are 1.2-2.8x wider than a naive row bootstrap - "
        "less than effective N implies, but enough to matter.",
        "Kish effective N is a worst-case structural warning, not a "
        "prediction; resampling evidence groups is the number to report. The "
        "combined three-class interval spans 0.042, which is wider than most "
        "between-model gaps in Table 13 - AraBERT and MARBERT are not "
        "separable at this resolution.",
        "Bootstrap CIs",
    ),
    (
        "The entailment metric and the silver labels disagree, and neither can "
        "be trusted as the reference yet.",
        "NLI finds 53-62% neutral relations whatever the silver grade. That "
        "looked like NLI failing, but the silver labels grade unrelated "
        "evidence (a drone claim vs Epstein-leak evidence) as Relevant, so "
        "they cannot adjudicate, and the human audit confirms the silver labels "
        "are unreliable. NLI itself is still unvalidated. Report neither as "
        "faithfulness.",
        "NLI (not validated)",
    ),
    (
        "Three-class results move more between library versions than several "
        "differences the report treats as model effects.",
        "Retraining under transformers 5.5 rather than 4.35 reproduces binary "
        "within 0.005 but shifts source three-class by +0.034.",
        "Replication",
    ),
    (
        "Over half the Real/Fake test rows carry evidence that cannot settle "
        "the claim.",
        "In the combined three-class test, 54.3% of Real/Fake rows have "
        "evidence that copies the claim or runs to 5 words or fewer (61.8% of "
        "source Fake rows). The labels record whether a claim was true, not "
        "whether the attached evidence shows it. Both counts are lower bounds.",
        "annotation/codebook.md",
    ),
    (
        "A human audit finds the silver labels agree with a human reading of "
        "the evidence at chance level.",
        "One annotator, 220 scored rows, second pass: verdict agreement 25.9% "
        "(95% CI 18.6-33.0%), kappa -0.110 (-0.214 to -0.016), below chance; "
        "first pass 35.9%, kappa 0.037. Relevance kappa -0.023. 80.1% of silver "
        "Real/Fake rows judged Unverified. The conclusion holds in both passes.",
        "Human Audit",
    ),
    (
        "Reliability of the audit comes from the annotator labelling the "
        "sample twice, not from a second annotator.",
        "Test-retest over the two passes: verdict kappa 0.578 (0.451-0.697), "
        "relevance 0.560, both moderate. This is an upper bound on what two "
        "people would reach. Human faithfulness vs the silver field: kappa 0.362.",
        "Human Audit",
    ),
    ("", "", ""),
    ("STILL OUTSTANDING", "", ""),
    (
        "Independent second annotator",
        "The B sheet was produced by editing A's, so it carries no independent "
        "agreement and is excluded; the scorer now refuses such a pair. A second "
        "person annotating pairs 21-80 from a blank sheet would give a real "
        "inter-annotator kappa.",
        "annotation/",
    ),
    (
        "Collection time period",
        "Required by the brief under Data Collection; not derivable from the "
        "data.",
        "-",
    ),
    (
        "Word count",
        "Body prose is 5,862 words against a 6,000 floor. The new sections "
        "cover the shortfall.",
        "-",
    ),
]


def style_header(worksheet, row: int, columns: int) -> None:
    for column in range(1, columns + 1):
        cell = worksheet.cell(row=row, column=column)
        cell.font = Font(bold=True, color="FFFFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
        cell.alignment = Alignment(
            horizontal="left", vertical="center", wrap_text=True
        )
        cell.border = BORDER


def autosize(worksheet, frame: pd.DataFrame, start_row: int) -> None:
    for index, name in enumerate(frame.columns, start=1):
        longest = max(
            [len(str(name))]
            + [len(str(v)) for v in frame[name].head(200).tolist()]
        )
        width = min(max(longest + 2, 11), 46)
        worksheet.column_dimensions[get_column_letter(index)].width = width
    worksheet.freeze_panes = worksheet.cell(row=start_row + 1, column=1)


def write_frame(writer, sheet: str, frame: pd.DataFrame, note: str) -> None:
    # Row 1 carries the note, so the table starts at row 3.
    frame.to_excel(writer, sheet_name=sheet, index=False, startrow=2)
    worksheet = writer.sheets[sheet]

    worksheet.cell(row=1, column=1, value=note).font = Font(
        italic=True, color=MUTED, size=9
    )
    worksheet.merge_cells(
        start_row=1, start_column=1, end_row=1, end_column=max(len(frame.columns), 4)
    )
    worksheet.cell(row=1, column=1).alignment = Alignment(
        wrap_text=True, vertical="top"
    )
    worksheet.row_dimensions[1].height = 30

    style_header(worksheet, 3, len(frame.columns))
    autosize(worksheet, frame, start_row=3)

    numeric = frame.select_dtypes("number").columns
    for row in range(4, len(frame) + 4):
        for index, name in enumerate(frame.columns, start=1):
            cell = worksheet.cell(row=row, column=index)
            cell.border = BORDER
            cell.font = Font(size=10, color=INK)
            if name in numeric:
                cell.number_format = "0.0000"
                cell.alignment = Alignment(horizontal="right")
            else:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        if row % 2 == 1:
            for index in range(1, len(frame.columns) + 1):
                worksheet.cell(row=row, column=index).fill = PatternFill(
                    "solid", fgColor=BAND_FILL
                )

    # Green where a gate genuinely beats confidence, red where it loses. The
    # sign of the advantage is the whole point of that sheet, so it is worth
    # making visible without reading the numbers.
    if "AURC advantage" in frame.columns:
        column = get_column_letter(list(frame.columns).index("AURC advantage") + 1)
        span = f"{column}4:{column}{len(frame) + 3}"
        worksheet.conditional_formatting.add(
            span,
            CellIsRule(
                operator="greaterThan",
                formula=["0"],
                fill=PatternFill("solid", fgColor=GOOD_FILL),
            ),
        )
        worksheet.conditional_formatting.add(
            span,
            CellIsRule(
                operator="lessThan",
                formula=["0"],
                fill=PatternFill("solid", fgColor=BAD_FILL),
            ),
        )


def write_summary(writer) -> None:
    frame = pd.DataFrame(SUMMARY, columns=["Finding", "Evidence", "Sheet"])
    frame.to_excel(writer, sheet_name="Summary", index=False, header=False)
    worksheet = writer.sheets["Summary"]
    worksheet.column_dimensions["A"].width = 52
    worksheet.column_dimensions["B"].width = 76
    worksheet.column_dimensions["C"].width = 24

    for row in range(1, len(SUMMARY) + 1):
        first = str(worksheet.cell(row=row, column=1).value or "")
        for column in range(1, 4):
            cell = worksheet.cell(row=row, column=column)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font = Font(size=10, color=INK)
        if row == 1:
            worksheet.cell(row=row, column=1).font = Font(
                bold=True, size=15, color=INK
            )
        elif first in ("FINDING", "STILL OUTSTANDING"):
            for column in range(1, 4):
                cell = worksheet.cell(row=row, column=column)
                cell.font = Font(bold=True, color="FFFFFFFF", size=10)
                cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
        elif row == 2:
            worksheet.cell(row=row, column=1).font = Font(
                italic=True, size=9, color=MUTED
            )
        else:
            worksheet.cell(row=row, column=1).font = Font(bold=True, size=10)
    worksheet.freeze_panes = "A5"


def main() -> None:
    written = []
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        write_summary(writer)

        replication = RESULTS_DIR / "verifier_replication_summary.json"
        if replication.exists():
            raw = pd.read_json(replication)
            frame = pd.DataFrame(
                {
                    "Family": raw["family"],
                    "Task": raw["task"],
                    "Best epoch": raw["best_epoch"],
                    "Validation macro F1": raw["validation_macro_f1"],
                    "Verifier accuracy on train (gate fits here)": raw[
                        "train_accuracy_in_sample"
                    ],
                    "Verifier accuracy held out": raw[
                        "validation_accuracy_held_out"
                    ],
                    "Train rows": raw["train_rows"],
                    "Validation rows": raw["validation_rows"],
                }
            )
            frame["Gate sees this share of errors"] = (
                1 - frame["Verifier accuracy on train (gate fits here)"]
            )
            frame["True error share"] = (
                1 - frame["Verifier accuracy held out"]
            )
            write_frame(
                writer,
                "Replication",
                frame,
                "Retrained verifiers. The last two columns are the root cause "
                "of the all-accept gate: it is fitted where the verifier is "
                "near-perfect, so its target is almost constant.",
            )
            written.append("Replication")

        for sheet, filename, note in SHEETS:
            path = RESULTS_DIR / filename
            if not path.exists():
                print(f"  skipped {sheet} (no {filename} yet)")
                continue
            frame = pd.read_csv(path)
            # Fingerprints are audit data, not reading material.
            frame = frame[
                [c for c in frame.columns if "Fingerprint" not in c]
            ]
            write_frame(writer, sheet, frame, note)
            written.append(sheet)

    print(f"wrote {OUT_PATH}")
    print("sheets:", ", ".join(["Summary"] + written))


if __name__ == "__main__":
    main()
