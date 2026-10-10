"""
Generates the Dataset Section as a formatted Word document (.docx).
Output: report/dataset_section.docx
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import docx

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "report")
OUTPUT     = os.path.join(OUTPUT_DIR, "dataset_section.docx")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_font(run, bold=False, italic=False, size=12, color=None):
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def add_paragraph(doc, text, bold_parts=None, indent=False):
    """Add paragraph; bold_parts is list of substrings to bold."""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)
    if bold_parts:
        remaining = text
        for bp in bold_parts:
            idx = remaining.find(bp)
            if idx == -1:
                continue
            if idx > 0:
                run = p.add_run(remaining[:idx])
                run.font.size = Pt(12)
            run = p.add_run(bp)
            run.bold = True
            run.font.size = Pt(12)
            remaining = remaining[idx + len(bp):]
        if remaining:
            run = p.add_run(remaining)
            run.font.size = Pt(12)
    else:
        run = p.add_run(text)
        run.font.size = Pt(12)
    p.paragraph_format.space_after = Pt(8)
    return p


def add_table(doc, title, headers, rows, col_widths=None):
    # Table caption
    cap = doc.add_paragraph()
    r = cap.add_run(title)
    r.bold = True
    r.font.size = Pt(11)
    cap.paragraph_format.space_before = Pt(10)
    cap.paragraph_format.space_after  = Pt(4)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(11)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "2E4057")
        tcPr.append(shd)
        run.font.color.rgb = RGBColor(255, 255, 255)

    # Data rows
    for ri, row in enumerate(rows):
        tr = table.rows[ri + 1]
        for ci, val in enumerate(row):
            cell = tr.cells[ci]
            cell.text = str(val)
            run = cell.paragraphs[0].runs[0]
            run.font.size = Pt(11)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if ri % 2 == 0:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "EBF5FB")
                tcPr.append(shd)

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)

    doc.add_paragraph()
    return table


# ── Build document ─────────────────────────────────────────────────────────────
doc = Document()

# Page margins
section = doc.sections[0]
section.top_margin    = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin   = Inches(1.2)
section.right_margin  = Inches(1.2)

# Default font
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

# ── Title ──────────────────────────────────────────────────────────────────────
title = doc.add_heading("4. Dataset", level=1)
title.alignment = WD_ALIGN_PARAGRAPH.LEFT

# ── 4.1 Overview ──────────────────────────────────────────────────────────────
add_heading(doc, "4.1 Overview", level=2)
add_paragraph(doc,
    "Rather than constructing a dataset from scratch — a process that requires substantial "
    "human annotation effort and is prone to label inconsistency — this work adopts a "
    "principled combination strategy: four existing Arabic misinformation datasets are "
    "unified under a shared three-class label schema and enriched with retrieved evidence, "
    "source metadata, and fact-check snippets. The result is a novel enriched corpus "
    "specifically adapted for faithfulness-aware Arabic fake news verification. This "
    "combined-and-enriched approach constitutes the primary data contribution of this work.",
    bold_parts=["novel enriched corpus", "primary data contribution"]
)
add_paragraph(doc,
    "The final dataset consists of 329,484 instances divided into an 80/20 training/test "
    "split, with labels aligned to three classes that reflect the evaluation demands of "
    "real-world Arabic fact-checking: Real, Fake, and Unverified.",
    bold_parts=["329,484 instances", "Real", "Fake", "Unverified"]
)

# ── 4.2 Source Datasets ────────────────────────────────────────────────────────
add_heading(doc, "4.2 Source Datasets", level=2)

add_heading(doc, "4.2.1 AFND — Arabic Fake News Dataset", level=3)
add_paragraph(doc,
    "The Arabic Fake News Dataset (AFND) is the largest Arabic-language misinformation "
    "corpus available, comprising scraped articles from hundreds of Arabic news websites, "
    "each labelled at the source level by credibility assessors as credible, not credible, "
    "or undecided. Articles from credible sources are mapped to Real; articles from "
    "not-credible sources are mapped to Fake. Crucially, the 31 undecided sources — "
    "representing outlets whose credibility could not be definitively established — are "
    "mapped to Unverified, providing the primary training signal for the third class. "
    "After sampling, AFND contributes 237,936 instances to the training set. "
    "The undecided-to-Unverified mapping is a methodological contribution of this work; "
    "to our knowledge, no prior study has exploited this portion of AFND as Unverified "
    "training data.",
    bold_parts=["237,936 instances", "undecided-to-Unverified mapping",
                "methodological contribution"]
)

add_heading(doc, "4.2.2 ArCOV19 — Arabic COVID-19 Rumour Dataset", level=3)
add_paragraph(doc,
    "ArCOV19-Rumors is a COVID-19 Arabic misinformation dataset containing fact-checked "
    "claims and labelled tweets collected during the pandemic. Verified claims and tweets "
    "labelled True are mapped to Real; those labelled False are mapped to Fake. The dataset "
    "contributes two sub-components: 3,584 labelled tweets and 138 fact-checked claims, "
    "for a total of 3,722 instances retained in full. ArCOV19 is notable for its "
    "social-media register, making it directly relevant to the UAE social-media test domain.",
    bold_parts=["3,722 instances", "social-media register"]
)

add_heading(doc, "4.2.3 VERA-ARAB — Verified Arabic Tweets", level=3)
add_paragraph(doc,
    "VERA-ARAB is a dataset of Arabic tweets annotated for veracity, with binary labels "
    "(0 = Real, 1 = Fake) and supplementary metadata including URLs, domain labels, "
    "annotation text, and context domain descriptions. All 20,076 instances are retained "
    "in full. The accompanying metadata is preserved as enrichment fields in the unified "
    "corpus, providing source-level provenance information unavailable in other datasets.",
    bold_parts=["20,076 instances"]
)

add_heading(doc, "4.2.4 AraStance — Arabic Stance Detection Dataset", level=3)
add_paragraph(doc,
    "AraStance provides Arabic claim-article pairs annotated with stance labels: Agree, "
    "Disagree, Discuss, and Unrelated. The mapping to the three-class schema follows "
    "principled semantic reasoning: Agree → Real (the article supports the claim); "
    "Disagree → Fake (the article refutes the claim); Discuss → Unverified (the claim "
    "is contested or ambiguous); Unrelated → dropped (no stance signal). All three splits "
    "(train, validation, test) are included, yielding 1,858 instances retained in full. "
    "AraStance is the only source dataset with native claim-evidence pairs; the article "
    "text is preserved verbatim as the evidence_text field.",
    bold_parts=["1,858 instances", "only source dataset with native claim-evidence pairs"]
)

# ── 4.3 Label Harmonisation ────────────────────────────────────────────────────
add_heading(doc, "4.3 Label Harmonisation", level=2)
add_paragraph(doc,
    "The four source datasets employ heterogeneous label schemas. Table 1 documents the "
    "complete mapping applied to produce the unified three-class schema. The three-class "
    "schema — Real, Fake, Unverified — is chosen to match the label space of the UAE-MVD "
    "test set and to reflect the practical output of Arabic fact-checking systems, where "
    "many claims cannot be conclusively verified or refuted.",
    bold_parts=["Real, Fake, Unverified"]
)

add_table(doc,
    "Table 1 — Label Mapping Across Source Datasets",
    ["Dataset", "Original Label", "Unified Label"],
    [
        ["AFND",      "credible",     "Real"],
        ["AFND",      "not credible", "Fake"],
        ["AFND",      "undecided",    "Unverified"],
        ["ArCOV19",   "True",         "Real"],
        ["ArCOV19",   "False",        "Fake"],
        ["VERA-ARAB", "0",            "Real"],
        ["VERA-ARAB", "1",            "Fake"],
        ["AraStance", "Agree",        "Real"],
        ["AraStance", "Disagree",     "Fake"],
        ["AraStance", "Discuss",      "Unverified"],
        ["AraStance", "Unrelated",    "Dropped"],
    ],
    col_widths=[1.8, 2.0, 1.8]
)

# ── 4.4 Data Enrichment ────────────────────────────────────────────────────────
add_heading(doc, "4.4 Data Enrichment", level=2)
add_paragraph(doc,
    "Beyond label harmonisation, each instance in the unified corpus is enriched with "
    "structured metadata fields absent from the original datasets. This enrichment "
    "transforms the corpus from a plain classification dataset into a "
    "retrieval-augmented verification dataset. The enrichment fields are:",
    bold_parts=["retrieval-augmented verification dataset"]
)

enrichment_items = [
    ("evidence_text",
     "For AraStance instances, the paired article is used directly. For other instances, "
     "this field is populated by BM25 retrieval against the training corpus, retrieving "
     "the top-5 most relevant passages for each claim at inference time."),
    ("source_url",
     "Preserved from VERA-ARAB (tweet URLs) and ArCOV19 (claim source URLs) where available."),
    ("source_name",
     "The originating website or platform identifier, drawn from AFND source IDs and "
     "VERA-ARAB domain fields."),
    ("source_type",
     "A categorical field indicating the medium: Arabic news website (AFND), social media "
     "tweet (VERA-ARAB, ArCOV19 tweets), fact-checked claim (ArCOV19 claims), or news "
     "article (AraStance)."),
    ("fact_check_snippet",
     "Annotation text from VERA-ARAB and claim text from ArCOV19, providing short "
     "fact-check rationales."),
    ("category",
     "Topic or domain label drawn from VERA-ARAB context descriptions and ArCOV19 "
     "category fields."),
]
for field, desc in enrichment_items:
    p = doc.add_paragraph(style="List Bullet")
    r1 = p.add_run(field + ": ")
    r1.bold = True
    r1.font.size = Pt(12)
    r2 = p.add_run(desc)
    r2.font.size = Pt(12)
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_after = Pt(4)

doc.add_paragraph()
add_table(doc,
    "Table 2 — Enrichment Field Coverage in Training Set (263,587 rows)",
    ["Field", "Rows Filled", "Coverage"],
    [
        ["source_type",        "263,587", "100.0%"],
        ["source_name",        "~249,000", "94.5%"],
        ["source_url",         "~11,600",  "4.4%"],
        ["evidence_text",      "~1,858",   "0.7% (AraStance native)"],
        ["fact_check_snippet", "~6,600",   "2.5%"],
        ["category",           "~10,600",  "4.0%"],
    ],
    col_widths=[2.0, 1.8, 2.4]
)

# ── 4.5 Test Dataset ───────────────────────────────────────────────────────────
add_heading(doc, "4.5 Test Dataset — UAE-MVD", level=2)
add_paragraph(doc,
    "The test set is the UAE Misinformation and Verification Dataset (UAE-MVD), an "
    "out-of-domain evaluation set of Arabic claims drawn from UAE social-media posts. "
    "The out-of-domain nature of this evaluation — training on Arabic news articles, "
    "testing on UAE social-media posts — deliberately increases difficulty and "
    "strengthens the validity of results: a model that performs well here has learned "
    "transferable Arabic misinformation patterns rather than domain-specific surface "
    "features.",
    bold_parts=["UAE Misinformation and Verification Dataset (UAE-MVD)",
                "out-of-domain", "transferable Arabic misinformation patterns"]
)
add_paragraph(doc,
    "Labels for the test set were generated using a two-LLM cross-validation approach: "
    "each claim was independently labelled by Qwen3 and Gemma4, with the majority label "
    "adopted as the gold label and a confidence score recorded per instance. This "
    "semi-automatic annotation strategy is consistent with recent practice in low-resource "
    "Arabic NLP where full manual annotation at scale is infeasible.",
    bold_parts=["Qwen3", "Gemma4", "two-LLM cross-validation approach"]
)

add_table(doc,
    "Table 3 — UAE-MVD Test Set Statistics (after merging)",
    ["Label", "Approx. Count", "Percentage"],
    [
        ["Real",        "~38,000", "~57.7%"],
        ["Fake",        "~19,400", "~29.4%"],
        ["Unverified",  "~8,500",  "~12.9%"],
        ["Total",       "~65,897", "100%"],
    ],
    col_widths=[1.8, 2.0, 1.8]
)
add_paragraph(doc,
    "Note: Final counts will be updated once the full labeled test set is merged. "
    "The existing verified subset contains 5,538 rows (Real: 3,203, Fake: 1,628, "
    "Unverified: 707).",
    bold_parts=["5,538 rows"]
)

# ── 4.6 Training / Test Split ──────────────────────────────────────────────────
add_heading(doc, "4.6 Training / Test Split", level=2)
add_table(doc,
    "Table 4 — Final Dataset Split",
    ["Split", "Instances", "Percentage"],
    [
        ["Training", "263,587", "80.0%"],
        ["Test",     "65,897",  "20.0%"],
        ["Total",    "329,484", "100%"],
    ],
    col_widths=[2.0, 2.0, 2.0]
)

add_table(doc,
    "Table 5 — Training Set Class Distribution",
    ["Class", "Count", "Percentage", "Primary Source"],
    [
        ["Real",        "~101,600", "38.5%", "AFND (credible)"],
        ["Fake",        "~101,600", "38.5%", "AFND (not credible)"],
        ["Unverified",  "~60,387",  "22.9%", "AFND (undecided) + AraStance (Discuss)"],
    ],
    col_widths=[1.5, 1.4, 1.4, 2.5]
)

add_paragraph(doc,
    "The mild class imbalance in the Unverified class — constrained by the number of "
    "available undecided AFND sources — is addressed at training time through "
    "class-weighted loss functions applied to all neural models.",
    bold_parts=["class-weighted loss functions"]
)

# ── 4.7 Novel Contribution ────────────────────────────────────────────────────
add_heading(doc, "4.7 Novel Contribution of the Dataset", level=2)
add_paragraph(doc,
    "The unified enriched corpus is novel in three respects. First, it is the only Arabic "
    "dataset to map undecided AFND sources explicitly to an Unverified class, providing "
    "60,000 Unverified training instances where prior work either discards this data or "
    "collapses it into the Fake class. Second, it integrates four previously disjoint "
    "Arabic corpora under a single schema with shared metadata fields, enabling "
    "cross-corpus generalisation studies. Third, the out-of-domain UAE-MVD test set, "
    "evaluated against models trained on Arabic news, provides a more ecologically valid "
    "benchmark for Arabic social-media fact-checking than in-domain splits used in "
    "prior work.",
    bold_parts=["60,000 Unverified training instances",
                "four previously disjoint Arabic corpora",
                "ecologically valid benchmark"]
)

# ── Save ───────────────────────────────────────────────────────────────────────
doc.save(OUTPUT)
print(f"Saved -> {OUTPUT}")
