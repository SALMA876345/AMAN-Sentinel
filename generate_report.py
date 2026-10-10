# -*- coding: utf-8 -*-
"""Build the final FAAV research report.

The report is intentionally conservative: true external testing is treated as
the main evidence, while high in-distribution scores are labelled as an
upper-bound diagnostic rather than final test performance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"
OUT_PATH = ROOT / "Assignment2_Report.docx"


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


external_full = read_csv("all_models_comparison.csv")
external_2class = read_csv("comparison_2class.csv")
internal_2class = read_csv("comparison_2class_indist_3source.csv")
zero_shot = read_csv("zero_shot/zero_shot_model_comparison.csv")
audit = read_json("data_audit.json")


def fmt(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return "-"
    try:
        return f"{float(text):.{digits}f}"
    except ValueError:
        return text


def make_figures() -> None:
    """Create report figures from the current result CSVs."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    GRAPHS_DIR.mkdir(exist_ok=True)

    if not external_full.empty and "Macro F1" in external_full.columns:
        df = external_full.copy()
        df["Macro F1"] = pd.to_numeric(df["Macro F1"], errors="coerce")
        df = df.sort_values("Macro F1", ascending=True)
        colors = ["#6B7280"] * len(df)
        for i, model in enumerate(df["Model"]):
            if "FAAV" in model:
                colors[i] = "#2563EB"
            elif "RAG" in model or "SBERT" in model:
                colors[i] = "#059669"
            elif model in {"MARBERT", "AraBERT"}:
                colors[i] = "#7C3AED"
        plt.figure(figsize=(8, 4.8))
        plt.barh(df["Model"], df["Macro F1"], color=colors)
        plt.xlabel("Macro F1")
        plt.title("External UAE Test: Macro F1 by Model")
        for idx, row in df.iterrows():
            n = row.get("N", "")
            plt.text(row["Macro F1"] + 0.006, list(df.index).index(idx), f"N={n}", va="center", fontsize=8)
        plt.xlim(0, max(0.62, float(df["Macro F1"].max()) + 0.08))
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / "13_external_full_macro_f1.png", dpi=180)
        plt.close()

    if not external_2class.empty and not internal_2class.empty:
        common = sorted(set(external_2class["Model"]) & set(internal_2class["Model"]))
        rows = []
        for model in common:
            ext = external_2class.loc[external_2class["Model"] == model, "Macro F1"].iloc[0]
            ind = internal_2class.loc[internal_2class["Model"] == model, "Macro F1"].iloc[0]
            if str(ext).strip() and str(ind).strip():
                try:
                    ind_v = float(ind)
                    ext_v = float(ext)
                    if pd.notna(ind_v) and pd.notna(ext_v):
                        rows.append((model, ind_v, ext_v))
                except ValueError:
                    pass
        if rows:
            df = pd.DataFrame(rows, columns=["Model", "In-distribution", "External"])
            df = df.sort_values("External", ascending=False).head(10)
            x = range(len(df))
            plt.figure(figsize=(9, 4.5))
            plt.bar([i - 0.18 for i in x], df["In-distribution"], width=0.36, label="In-distribution", color="#2563EB")
            plt.bar([i + 0.18 for i in x], df["External"], width=0.36, label="External UAE-MVD", color="#DC2626")
            plt.xticks(list(x), df["Model"], rotation=35, ha="right")
            plt.ylabel("Macro F1")
            plt.title("Generalisation Gap: Internal vs External 2-class Testing")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(GRAPHS_DIR / "14_internal_external_gap.png", dpi=180)
            plt.close()

    if not external_full.empty:
        rows = []
        for _, row in external_full.iterrows():
            if "FAAV" in str(row.get("Model", "")):
                faith = str(row.get("Faithfulness Acc", "N/A"))
                hall = row.get("Hallucination Rate", "N/A")
                ground = row.get("Grounding Score", "N/A")
                evrel = row.get("Evidence Relevance", "N/A")
                rows.append((row["Model"], faith, hall, ground, evrel))
        if rows:
            metrics = []
            labels = []
            for model, faith, hall, ground, evrel in rows:
                vals = []
                for v in [faith.split()[0], hall, ground, evrel]:
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(float("nan"))
                metrics.append(vals)
                labels.append(model)
            df = pd.DataFrame(metrics, index=labels, columns=["Faithfulness", "Hallucination", "Grounding", "Evidence rel."])
            df = df.dropna(axis=1, how="all")
            if not df.empty:
                plt.figure(figsize=(7.2, 3.8))
                df.plot(kind="bar", ax=plt.gca(), color=["#2563EB", "#DC2626", "#059669", "#7C3AED"])
                plt.ylim(0, 1)
                plt.ylabel("Score")
                plt.title("FAAV Trustworthiness Diagnostics")
                plt.xticks(rotation=15, ha="right")
                plt.legend(frameon=False, fontsize=8)
                plt.tight_layout()
                plt.savefig(GRAPHS_DIR / "15_faav_trustworthiness.png", dpi=180)
                plt.close()


make_figures()


doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.85)
section.bottom_margin = Inches(0.85)
section.left_margin = Inches(0.9)
section.right_margin = Inches(0.9)


def set_font(run, size=None, bold=None, italic=None, color=None):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def configure_styles() -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    normal.element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal.element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    for style_name, size, color, before, after in [
        ("Heading 1", 15, "1F4E79", 14, 7),
        ("Heading 2", 12.5, "1F4E79", 10, 5),
        ("Heading 3", 11.5, "1F4E79", 8, 4),
    ]:
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style.element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")


configure_styles()

fig_no = 0
table_no = 0


def body(text: str, justify: bool = True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if justify else WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    set_font(r, 11)
    return p


def heading(text: str, level: int = 1):
    return doc.add_heading(text, level=level)


def caption(text: str, kind: str) -> None:
    global fig_no, table_no
    if kind == "figure":
        fig_no += 1
        label = f"Figure {fig_no}. {text}"
    else:
        table_no += 1
        label = f"Table {table_no}. {text}"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if kind == "figure" else WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(label)
    set_font(r, 9.5, italic=True)


def add_table(title: str, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    caption(title, "table")
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            for r in p.runs:
                set_font(r, 9.5, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    set_font(r, 9)
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)


def add_bullets(items: list[tuple[str, str]]) -> None:
    for label, text in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(f"{label}. ")
        set_font(r, 11, bold=True)
        r2 = p.add_run(text)
        set_font(r2, 11)


def add_figure(filename: str, title: str, width: float = 5.9) -> None:
    path = GRAPHS_DIR / filename
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width))
        caption(title, "figure")


def page_break() -> None:
    doc.add_page_break()


def section_break() -> None:
    doc.add_section(WD_SECTION.NEW_PAGE)


def paragraph_rule() -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p_pr = p._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "B7C9DD")
    border.append(bottom)
    p_pr.append(border)


def table_from_external_full() -> None:
    if external_full.empty:
        return
    rows = []
    cols = ["Model", "N", "Accuracy", "Macro F1", "MCC"]
    for _, r in external_full.iterrows():
        trust = []
        faith = fmt(r.get("Faithfulness Acc"), 4)
        grounding = fmt(r.get("Grounding Score"), 4)
        evrel = fmt(r.get("Evidence Relevance"), 4)
        if faith != "-":
            trust.append(f"Faith {faith}")
        if grounding != "-":
            trust.append(f"Ground {grounding}")
        if evrel != "-":
            trust.append(f"EvRel {evrel}")
        rows.append(
            [fmt(r.get(c), 4) if c not in {"Model", "N"} else fmt(r.get(c), 0) for c in cols]
            + ["; ".join(trust) if trust else "-"]
        )
    add_table(
        "Primary external UAE test results. N is shown because some computationally expensive models were evaluated on capped subsets.",
        ["Model", "N", "Acc.", "Macro F1", "MCC", "Trust diagnostics"],
        rows,
        widths=[1.85, 0.55, 0.65, 0.75, 0.65, 2.05],
    )


def table_from_zero_shot() -> None:
    if zero_shot.empty:
        return
    rows = []
    for _, r in zero_shot.iterrows():
        rows.append(
            [
                str(r.get("Model", "")),
                fmt(r.get("N"), 0),
                fmt(r.get("Accuracy")),
                fmt(r.get("Macro F1")),
                fmt(r.get("MCC")),
                fmt(r.get("Supported F1")),
                fmt(r.get("Refuted F1")),
                fmt(r.get("NEI F1")),
            ]
        )
    add_table(
        "Strict zero-shot cross-domain results on the frozen UAE verification test. No UAE claims were used for training or threshold selection.",
        ["Model", "N", "Acc.", "Macro F1", "MCC", "Supp. F1", "Ref. F1", "NEI F1"],
        rows,
        widths=[1.72, 0.48, 0.55, 0.68, 0.55, 0.66, 0.62, 0.62],
    )


def table_from_external_2class() -> None:
    if external_2class.empty:
        return
    rows = []
    for _, r in external_2class.iterrows():
        rows.append([r["Model"], fmt(r.get("Accuracy")), fmt(r.get("Macro F1")), fmt(r.get("Weighted F1")), fmt(r.get("MCC")), fmt(r.get("Kappa"))])
    add_table(
        "Independent two-class UAE-MVD stress test. Blank cells indicate that the run was not available for that model.",
        ["Model", "Accuracy", "Macro F1", "Weighted F1", "MCC", "Kappa"],
        rows,
        widths=[2.05, 0.82, 0.82, 0.92, 0.75, 0.75],
    )


def table_from_internal_2class() -> None:
    if internal_2class.empty:
        return
    keep = [
        "MARBERT (after)",
        "FAAV-Full (proposed)",
        "AraBERT (after FT)",
        "BiLSTM",
        "SVM",
        "Logistic Regression",
        "Standard RAG",
        "SBERT-RAG",
        "FAAV-Light",
    ]
    rows = []
    for model in keep:
        rec = internal_2class[internal_2class["Model"] == model]
        if rec.empty:
            continue
        r = rec.iloc[0]
        rows.append([model, fmt(r.get("Accuracy")), fmt(r.get("Macro F1")), fmt(r.get("Weighted F1")), fmt(r.get("MCC")), fmt(r.get("Kappa"))])
    add_table(
        "In-distribution two-class diagnostic split. These values are an upper bound and are not treated as final external-test performance.",
        ["Model", "Accuracy", "Macro F1", "Weighted F1", "MCC", "Kappa"],
        rows,
        widths=[2.05, 0.82, 0.82, 0.92, 0.75, 0.75],
    )


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Faithfulness-Aware Retrieval-Augmented Verification for Arabic Misinformation Under Domain Shift")
set_font(r, 19, bold=True, color="17365D")

for text, size, bold in [
    ("Hamza Alqaruoti", 13, True),
    ("503 Machine Learning - Assignment 2", 11.5, False),
    ("Master of Science in Machine Learning", 11.5, False),
    ("Final research report", 11.5, False),
    ("June 2026", 11.5, False),
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = p.add_run(text)
    set_font(rr, size, bold=bold)

doc.add_paragraph()
paragraph_rule()
lead = doc.add_paragraph()
lead.alignment = WD_ALIGN_PARAGRAPH.CENTER
rr = lead.add_run(
    "Core claim: FAAV improves the auditability of Arabic verification through evidence grounding and faithfulness diagnostics; external-domain accuracy remains an open research challenge."
)
set_font(rr, 11, italic=True, color="404040")
page_break()


# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------
heading("Abstract")
body(
    "Arabic misinformation verification is often evaluated as a claim-only text classification problem, but deployable verification requires evidence retrieval, grounded reasoning, and a mechanism for detecting unsupported explanations. This study proposes Faithfulness-Aware Arabic Verification (FAAV), a retrieval-augmented framework that combines Arabic claim normalisation, evidence retrieval, transformer-based verification, faithfulness scoring, and an interpretable final decision. The system is evaluated against traditional classifiers, deep learning baselines, transformer encoders, and retrieval-augmented baselines."
)
body(
    "The empirical contribution of the study is not a single inflated leaderboard number. Instead, the final protocol introduces a strict zero-shot cross-domain evaluation in which no UAE claims are used for training, relevance supervision, stance supervision, retrieval-corpus construction, or threshold selection. All lightweight models are evaluated on the same frozen set of 6,387 UAE claims with zero exact claim overlap. The best zero-shot model, a fusion logistic-regression system using the provided evidence and metadata, reaches 0.3723 macro F1 and 0.1638 MCC. Evidence improves performance over claim-only classification, but Supported-class F1 remains extremely low, exposing a severe target-domain evidence and label-semantics gap. Earlier in-distribution scores near 0.90 macro F1 are retained only as upper-bound diagnostics."
)
body(
    "These findings support a more publishable conclusion: Arabic misinformation verification under domain shift is limited less by classifier architecture alone than by label semantics, evidence quality, topic drift, dialectal variation, and retrieval mismatch. FAAV is therefore valuable as a faithfulness-aware verification and analysis framework, while future performance gains require target-domain evidence, stronger retrieval, and domain adaptation."
)
kw = doc.add_paragraph()
kr = kw.add_run("Keywords: ")
set_font(kr, 11, bold=True)
rr = kw.add_run("Arabic misinformation; claim verification; retrieval-augmented verification; faithfulness; evidence grounding; MARBERT; domain shift.")
set_font(rr, 11)
page_break()


# ---------------------------------------------------------------------------
# Main paper
# ---------------------------------------------------------------------------
heading("1. Introduction")
body(
    "Misinformation detection systems are frequently optimised for accuracy on fixed benchmark splits. This framing is useful for model comparison, but it is incomplete for real verification. A factual verdict is trustworthy only when it can be connected to relevant evidence, and an explanation is useful only when it remains faithful to that evidence. These requirements are especially important for Arabic social-media misinformation, where claims vary across dialects, regions, sources, and rapidly changing public events."
)
body(
    "This report presents FAAV, a faithfulness-aware retrieval-augmented verification framework for Arabic misinformation. Unlike classification-only baselines, FAAV records evidence relevance, grounding, faithfulness, and hallucination-related diagnostics. The final paper deliberately distinguishes internal validation from true testing: high in-distribution numbers are interpreted as an upper bound, while external UAE testing is treated as the main evidence for publishable conclusions."
)

heading("2. Research Gap and Questions")
body(
    "Existing Arabic misinformation systems are dominated by claim-only classifiers or retrieval heuristics. They often lack an explicit test of whether the predicted verdict is supported by retrieved evidence. They also tend to report results on matched train/test distributions, which can obscure the generalisation problem that appears when models face a new country, topic, source, or label definition."
)
add_bullets(
    [
        ("RQ1", "How do classification-only, retrieval-only, and faithfulness-aware systems compare on Arabic misinformation verification?"),
        ("RQ2", "What does faithfulness-aware evaluation add beyond raw accuracy?"),
        ("RQ3", "How severe is the gap between in-distribution validation and external-domain testing?"),
    ]
)

heading("3. FAAV Framework")
body(
    "FAAV decomposes verification into five stages: Arabic claim processing, evidence retrieval, verdict prediction, faithfulness checking, and explainable final decision. The framework is implemented in two variants. FAAV-Light uses transparent rule-based faithfulness checks, while FAAV-Full uses neural similarity and grounding signals in a final abstention gate. A Real or Fake verifier output is accepted only when evidence length, verifier confidence, claim-evidence similarity, and explanation grounding pass their configured thresholds; otherwise the system returns Unverified."
)
add_table(
    "FAAV stages and outputs.",
    ["Stage", "Component", "Method", "Output"],
    [
        ["1", "Claim processing", "Arabic cleaning and normalisation", "Normalised claim"],
        ["2", "Evidence retrieval", "BM25 or dense sentence embeddings", "Ranked evidence"],
        ["3", "Verification", "Classifier, transformer, or retrieval heuristic", "Verdict"],
        ["4", "Faithfulness check", "Lexical, semantic, and grounding diagnostics", "Faithfulness signal"],
        ["5", "Final decision", "Thresholded fusion of verifier confidence, evidence relevance, and grounding", "Accepted verdict or Unverified abstention"],
    ],
    widths=[0.45, 1.35, 2.55, 1.55],
)

heading("4. Data and Evaluation Design")
body(
    "The project combines Arabic misinformation resources including AraStance, VERA-ARAB or related Arabic claim-verification data, ArCOV19-Rumors, and AFND-derived material. A key methodological finding is that source-credibility labels and claim-truth labels cannot be treated as interchangeable. Source credibility describes a publisher; claim verification describes whether a particular claim is supported, refuted, or not verifiable by evidence."
)
train_rows = audit.get("train", {}).get("rows", "25,263")
test_rows = audit.get("test", {}).get("rows", "5,053")
train_unique = audit.get("train", {}).get("unique_claims", "20,000")
test_unique = audit.get("test", {}).get("unique_claims", "5,050")
body(
    f"The latest audit reports {train_rows} training rows with {train_unique} unique claims and {test_rows} external test rows with {test_unique} unique claims in the audited two-class setup. Exact train-test overlap is reported as zero. The configured full UAE test used by the refreshed evaluator contains 30,015 rows, while some transformer and FAAV runs are intentionally capped for compute; therefore every table reports N and avoids direct overclaiming across unequal test sizes."
)
add_table(
    "Evaluation regimes used in the final paper.",
    ["Regime", "Purpose", "Main file", "How to interpret"],
    [
        ["Strict zero-shot UAE verification", "Primary leakage-free external evaluation", "zero_shot_model_comparison.csv", "Main evidence; identical N=6,387"],
        ["External UAE configured test", "Primary stress test with trust metrics", "all_models_comparison.csv", "Main evidence; N differs by model"],
        ["External two-class UAE-MVD", "Independent Real/Fake generalisation test", "comparison_2class.csv", "Best available two-class external comparison"],
        ["In-distribution two-class split", "Sanity check and upper bound", "comparison_2class_indist_3source.csv", "Not final test performance"],
    ],
    widths=[1.45, 1.8, 1.6, 2.0],
)

heading("5. Experimental Setup")
body(
    "Traditional baselines use TF-IDF features with support vector machines and logistic regression. The deep baseline uses a BiLSTM. Transformer baselines use AraBERT and MARBERT, with MARBERT trained using maximum sequence length 128, batch size 32, learning rate 2e-5, AdamW, class weighting, and three epochs where fine-tuning is performed. Retrieval systems use sparse BM25 and dense sentence-transformer similarity. Metrics include accuracy, weighted F1, macro F1, MCC, faithfulness accuracy, hallucination rate, grounding score, and evidence relevance."
)
body(
    "Macro F1 and MCC are emphasised in the discussion because external datasets are imbalanced and because high accuracy can be produced by majority-class behaviour. FAAV-Full additionally applies a final decision gate with default thresholds of 40 evidence characters, 0.50 verifier confidence, 0.20 claim-evidence semantic similarity, and 0.30 explanation grounding. Faithfulness and grounding metrics remain reportable diagnostics, but they now also control whether a Real or Fake verdict is accepted."
)

heading("6. Results")
body(
    "The strict zero-shot table is the primary result. Legacy tables were generated before the faithfulness-controlled final decision gate was added and remain useful only as pre-gate context."
)
heading("6.1 Strict Zero-Shot Cross-Domain UAE Test", 2)
body(
    "The corrected protocol uses AraStance for claim-evidence supervision, AFND only as a retrieval corpus, grouped non-UAE validation for threshold selection, and a frozen 6,387-row UAE test manifest. The fusion model attains the best macro F1 (0.3723) and MCC (0.1638). Claim-plus-evidence models outperform claim-only classification, confirming that evidence helps. However, the best model achieves only 0.0181 F1 on Supported claims, while NEI F1 exceeds 0.84. This imbalance demonstrates that honest external performance remains limited by target-domain evidence coverage and cross-dataset label semantics."
)
table_from_zero_shot()

heading("6.2 Legacy Configured UAE Test", 2)
body(
    "The primary table is the refreshed configured UAE evaluation. It shows that external Arabic verification is difficult. Full-coverage classical and retrieval models remain low in macro F1. FAAV-Light achieves high apparent accuracy but low macro F1 and near-zero MCC, indicating majority-class behaviour rather than robust verification. FAAV-Full provides the strongest faithfulness diagnostics among FAAV variants but is evaluated on a 4,831-row capped subset. MARBERT and AraBERT transformer results are also capped at 500 rows, so they are informative but not directly comparable with 30,015-row full-coverage systems."
)
table_from_external_full()
add_figure("13_external_full_macro_f1.png", "Macro F1 on the configured external UAE test, with N shown for each model.")

heading("6.3 Independent Two-Class UAE-MVD Stress Test", 2)
body(
    "The independent two-class table is the cleanest external Real/Fake stress test available in the result folder. Its results are far lower than the in-distribution split. MARBERT after fine-tuning reaches 0.5536 macro F1, while most other systems remain between roughly 0.33 and 0.53. This supports the main research finding: train/test similarity, not model architecture alone, drives much of the high internal score."
)
table_from_external_2class()

heading("6.4 In-Distribution Diagnostic Split", 2)
body(
    "The in-distribution split should be reported, but only as a diagnostic upper bound. MARBERT and FAAV-Full reach approximately 0.90 macro F1 when training and test rows are drawn from the same claim-level distribution. This confirms that the modelling pipeline can learn the task, but it does not prove deployment-ready generalisation."
)
table_from_internal_2class()
add_figure("14_internal_external_gap.png", "Generalisation gap between the in-distribution split and external two-class UAE-MVD testing.")

heading("6.5 Faithfulness and Grounding", 2)
body(
    "FAAV's strongest contribution is the additional trustworthiness layer. In the refreshed external evaluation, FAAV-Full reports faithfulness accuracy of 0.6984 on its evaluated subset, grounding score of 0.5659, and evidence relevance of 0.5648. The hallucination-rate proxy is high, which should not be interpreted as factual hallucination in the human sense; it reflects low semantic overlap between template explanations and evidence text. The publishable interpretation is that FAAV makes these weaknesses measurable instead of invisible."
)
add_figure("15_faav_trustworthiness.png", "FAAV trustworthiness diagnostics on the external configured test.")

heading("7. Error Analysis")
body(
    "The failure analysis points to six recurring causes: dialect mismatch, political ambiguity, temporal misinformation, sarcasm or irony, conflicting evidence, and retrieval failures. The largest observed categories are dialectal and political failures, which are consistent with the gap between training corpora dominated by broader Arabic misinformation sources and the UAE-focused external test."
)
add_bullets(
    [
        ("Dialect variation", "Gulf, Levantine, Egyptian, and informal social-media forms reduce lexical overlap and can weaken both sparse retrieval and transformer confidence."),
        ("Political ambiguity", "Political claims often have partially supporting and partially refuting evidence, making single-label decisions brittle."),
        ("Temporal claims", "Static evidence cannot reliably verify claims whose truth value changes with date or event status."),
        ("Retrieval mismatch", "When retrieved evidence is only topically related, a classifier may learn topic association rather than verification."),
        ("Label semantics", "Datasets differ in whether labels mean claim truth, source credibility, stance, or evidence support, and these definitions do not transfer cleanly."),
    ]
)

heading("8. Discussion")
body(
    "RQ1 is answered cautiously. Retrieval-grounded systems do not automatically outperform claim-only systems on external data; retrieval quality and label alignment determine whether retrieval helps. RQ2 is answered more positively: faithfulness-aware evaluation adds measurable diagnostics that ordinary classifiers do not provide. RQ3 is the strongest empirical result: the gap between in-distribution and external-domain testing is large enough that internal performance should be treated as an upper bound, not as final evidence."
)
body(
    "The central conclusion is therefore not that FAAV has solved Arabic misinformation detection. A stronger and more publishable claim is that FAAV exposes the right failure modes: unsupported evidence, weak grounding, retrieval mismatch, label mismatch, and domain shift. This is a credible contribution because Arabic verification research needs systems that can diagnose why they fail, not only systems that report a single accuracy score."
)

heading("9. Limitations and Threats to Validity")
add_bullets(
    [
        ("Unequal N", "Some transformer and FAAV runs are capped for compute, so direct ranking must be interpreted carefully."),
        ("Evidence quality", "Several evidence fields are short snippets rather than full articles; this limits both retrieval and faithfulness scoring."),
        ("Metric proxies", "Hallucination and faithfulness are measured with lexical and semantic proxies, not human expert annotation."),
        ("Dataset semantics", "Combining claim-truth, source-credibility, stance, and evidence-support resources creates unavoidable label harmonisation risk."),
        ("External validity", "UAE-MVD is a valuable stress test but remains one regional domain; broader Gulf and pan-Arab testing is needed."),
    ]
)

heading("10. Conclusion and Future Work")
body(
    "This study finalises FAAV as a faithfulness-aware framework for Arabic misinformation verification under domain shift. The corrected evidence does not support a simplistic claim that FAAV dominates all baselines in accuracy on true external testing. It does support a stronger research conclusion: high internal scores collapse under domain shift, while evidence grounding and faithfulness diagnostics provide the analytical machinery needed to understand and improve Arabic verification systems."
)
body(
    "The next research step is to improve the external setting rather than hide it: collect target-domain labelled claims, retrieve full source articles instead of snippets, tune the new abstention thresholds on a held-out validation set, train a domain-adapted verifier, and validate faithfulness labels with human annotation. These steps would turn FAAV from a strong diagnostic framework into a stronger deployable verifier."
)

heading("References")
refs = [
    "Abdul-Mageed, M., Elmadany, A., & Nagoudi, E. M. B. (2021). ARBERT & MARBERT: Deep bidirectional transformers for Arabic. Proceedings of ACL-IJCNLP.",
    "Alhindi, T., Alabdulkarim, A., Alshehri, A., Abdul-Mageed, M., & Nakov, P. (2021). AraStance: A multi-country and multi-domain dataset of Arabic stance detection for fact checking. Proceedings of NLP4IF.",
    "Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. Proceedings of OSACT.",
    "Haouari, F., Hasanain, M., Suwaileh, R., & Elsayed, T. (2021). ArCOV19-Rumors: Arabic COVID-19 Twitter dataset for misinformation detection. Proceedings of WANLP.",
    "Khalil, A., Jarrah, M., & Aldwairi, M. Arabic Fake News Dataset (AFND). Dataset documentation.",
    "Khouja, J. (2020). Stance prediction and claim verification: An Arabic perspective. arXiv:2005.10410.",
    "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proceedings of EMNLP-IJCNLP.",
    "Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval.",
    "Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: A large-scale dataset for fact extraction and verification. Proceedings of NAACL-HLT.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    r = p.add_run(ref)
    set_font(r, 10.5)


saved_path = OUT_PATH
try:
    doc.save(saved_path)
except PermissionError:
    saved_path = ROOT / "Assignment2_Report_Updated.docx"
    doc.save(saved_path)
print(f"SUCCESS: saved to {saved_path}")
print(f"Tables: {table_no}")
print(f"Figures: {fig_no}")
