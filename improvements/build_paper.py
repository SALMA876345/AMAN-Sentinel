"""Typeset a paper in IEEE conference style and in ACL/LREC style.

One content module, two layouts. The styles differ in more than fonts: IEEE
numbers citations and tables with Roman numerals and centres small-caps section
heads, while ACL cites by author and year, numbers sections in arabic, and
orders references alphabetically. Citation markers in the content are resolved
per style, so neither version carries the other's conventions.

    python improvements/build_paper.py a            # Paper A, both styles
    python improvements/build_paper.py b ieee       # Paper B, one style
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT.parent
FIGURES = ROOT / "results" / "improvements" / "figures_report"

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
         "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]

STYLES = {
    "ieee": {
        "margins": (0.75, 1.0, 0.625),
        "column_width": 3.5, "column_gap": 0.25,
        "body_pt": 10, "caption_pt": 8, "ref_pt": 8,
        "title_pt": 24, "author_pt": 11,
        "section_caps": True, "section_centre": True,
        "first_line": 0.2,
    },
    "acl": {
        "margins": (1.0, 1.0, 0.95),
        "column_width": 3.05, "column_gap": 0.25,
        "body_pt": 10, "caption_pt": 9, "ref_pt": 9,
        "title_pt": 15, "author_pt": 12,
        "section_caps": False, "section_centre": False,
        "first_line": 0.18,
    },
}


# ---------------------------------------------------------------- citations
def resolve_citations(style: str, order: list[str]):
    """Return a function turning {c:key} and {t:key} markers into style text."""
    def replace(text: str) -> str:
        def one(match):
            mode, key = match.group(1), match.group(2)
            if key not in REFERENCES:
                return match.group(0)
            if key not in order:
                order.append(key)
            if style == "ieee":
                return f"[{order.index(key) + 1}]"
            inline = REFERENCES[key]["inline"]
            if mode == "t":
                author, year = inline.rsplit(", ", 1)
                return f"{author} ({year})"
            return f"({inline})"
        text = re.sub(r"\{([ct]):([a-z0-9]+)\}", one, text)
        # "[1][2]" reads badly; IEEE wants "[1], [2]".
        text = re.sub(r"\](\[)", r"], \1", text)
        text = re.sub(r"\)\(", r"; ", text)
        return text
    return replace


# ---------------------------------------------------------------- layout
def set_columns(section, count: int, gap: float) -> None:
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(int(gap * 1440)))
    cols.set(qn("w:equalWidth"), "1")


def page_setup(section, cfg) -> None:
    top, bottom, side = cfg["margins"]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(top), Inches(bottom)
    section.left_margin = section.right_margin = Inches(side)


def para(doc, text="", size=10, *, bold=False, italic=False, align=None,
         before=0, after=0, caps=False, first_line=None, keep=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold, run.italic, run.font.small_caps = bold, italic, caps
    fmt = p.paragraph_format
    fmt.space_before, fmt.space_after = Pt(before), Pt(after)
    fmt.line_spacing = 1.0
    fmt.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.keep_with_next = keep
    if first_line:
        fmt.first_line_indent = Inches(first_line)
    return p


def rule_table(doc, rows, width_in, cfg, style):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    tbl = table._tbl
    tbl.remove(tbl.tblPr)
    tblPr = OxmlElement("w:tblPr")
    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(int(width_in * 1440)))
    w.set(qn("w:type"), "dxa")
    tblPr.append(w)
    borders = OxmlElement("w:tblBorders")
    for side in ["top", "bottom"]:
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "8")
        b.set(qn("w:color"), "000000")
        borders.append(b)
    for side in ["left", "right", "insideV", "insideH"]:
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "none")
        borders.append(b)
    tblPr.append(borders)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    tblPr.append(jc)
    tbl.insert(0, tblPr)

    for r_index, row in enumerate(rows):
        for c_index, value in enumerate(row):
            cell = table.rows[r_index].cells[c_index]
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                           if c_index or r_index == 0 else WD_ALIGN_PARAGRAPH.LEFT)
            run = p.add_run(value)
            run.font.size = Pt(cfg["caption_pt"])
            run.bold = r_index == 0
        if r_index == 0:
            for cell in table.rows[0].cells:
                tcPr = cell._tc.get_or_add_tcPr()
                tcb = OxmlElement("w:tcBorders")
                b = OxmlElement("w:bottom")
                b.set(qn("w:val"), "single")
                b.set(qn("w:sz"), "8")
                b.set(qn("w:color"), "000000")
                tcb.append(b)
                tcPr.append(tcb)
    return table


# ---------------------------------------------------------------- build
def build(paper: str, style: str) -> Path:
    content = importlib.import_module(f"improvements.paper_{paper}_content")
    ABSTRACT, AFFILIATIONS, AUTHORS = content.ABSTRACT, content.AFFILIATIONS, content.AUTHORS
    BODY, KEYWORDS, REFERENCES_, TITLE = content.BODY, content.KEYWORDS, content.REFERENCES, content.TITLE
    globals()["REFERENCES"] = REFERENCES_
    cfg = dict(STYLES[style])
    cfg["out"] = ASSIGNMENT / f"{content.OUT_STEM}_{style.upper()}.docx"
    full_width = 2 * cfg["column_width"] + cfg["column_gap"]
    order: list[str] = []
    cite = resolve_citations(style, order)

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(cfg["body_pt"])
    page_setup(doc.sections[0], cfg)
    set_columns(doc.sections[0], 1, cfg["column_gap"])

    # title block, across the page
    para(doc, TITLE, cfg["title_pt"], bold=(style == "acl"),
         align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
    names = ", ".join(f"{n}{'' if style == 'acl' else str(a)}"
                      for n, a in AUTHORS)
    para(doc, names, cfg["author_pt"], align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    for index, affiliation in enumerate(AFFILIATIONS, start=1):
        label = affiliation if style == "acl" else f"{index}{affiliation}"
        para(doc, label, cfg["body_pt"] - 1, italic=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "", 6, after=6)

    if style == "acl":
        # ACL puts the abstract in a narrower block before the two columns.
        para(doc, "Abstract", 12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
             before=4, after=4)
        p = para(doc, cite(ABSTRACT), 10, after=10)
        p.paragraph_format.left_indent = Inches(0.9)
        p.paragraph_format.right_indent = Inches(0.9)
        section = doc.add_section(WD_SECTION.CONTINUOUS)
        page_setup(section, cfg)
        set_columns(section, 2, cfg["column_gap"])
    else:
        section = doc.add_section(WD_SECTION.CONTINUOUS)
        page_setup(section, cfg)
        set_columns(section, 2, cfg["column_gap"])
        p = doc.add_paragraph()
        lead = p.add_run("Abstract—")
        lead.bold = lead.italic = True
        lead.font.size = Pt(9)
        rest = p.add_run(cite(ABSTRACT))
        rest.bold = rest.italic = True
        rest.font.size = Pt(9)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(6)
        p = doc.add_paragraph()
        lead = p.add_run("Index Terms—")
        lead.bold = lead.italic = True
        lead.font.size = Pt(9)
        rest = p.add_run(KEYWORDS)
        rest.bold = rest.italic = True
        rest.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(8)

    columns_now = 2

    def ensure(n: int):
        nonlocal columns_now
        if columns_now != n:
            s = doc.add_section(WD_SECTION.CONTINUOUS)
            page_setup(s, cfg)
            set_columns(s, n, cfg["column_gap"])
            columns_now = n

    section_no = 0
    sub_no = 0
    table_no = 0
    figure_no = 0

    for block in BODY:
        kind = block[0]
        if kind == "h1":
            ensure(2)
            section_no += 1
            sub_no = 0
            label = (f"{ROMAN[section_no - 1]}.  {block[1]}" if style == "ieee"
                     else f"{section_no}  {block[1]}")
            para(doc, label, cfg["body_pt"] + (0 if style == "ieee" else 2),
                 bold=(style == "acl"),
                 align=(WD_ALIGN_PARAGRAPH.CENTER if cfg["section_centre"]
                        else WD_ALIGN_PARAGRAPH.LEFT),
                 caps=cfg["section_caps"], before=10, after=4, keep=True)
        elif kind == "h2":
            ensure(2)
            sub_no += 1
            label = (f"{chr(64 + sub_no)}. {block[1]}" if style == "ieee"
                     else f"{section_no}.{sub_no}  {block[1]}")
            para(doc, label, cfg["body_pt"] + (0 if style == "ieee" else 1),
                 italic=(style == "ieee"), bold=(style == "acl"),
                 align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=2, keep=True)
        elif kind == "p":
            ensure(2)
            para(doc, cite(block[1]), cfg["body_pt"],
                 first_line=cfg["first_line"], after=3)
        elif kind == "table":
            _, caption, rows, wide = block
            ensure(1 if wide else 2)
            table_no += 1
            width = full_width if wide else cfg["column_width"]
            if style == "ieee":
                para(doc, f"TABLE {ROMAN[table_no - 1]}", cfg["caption_pt"],
                     align=WD_ALIGN_PARAGRAPH.CENTER, before=6, keep=True)
                para(doc, cite(caption), cfg["caption_pt"],
                     align=WD_ALIGN_PARAGRAPH.CENTER,
                     caps=len(caption) <= 90, after=2, keep=True)
                rule_table(doc, rows, width, cfg, style)
            else:
                para(doc, f"Table {table_no}: {cite(caption)}",
                     cfg["caption_pt"], align=WD_ALIGN_PARAGRAPH.LEFT,
                     before=6, after=2, keep=True)
                rule_table(doc, rows, width, cfg, style)
            para(doc, "", 4, after=4)
        elif kind == "figure":
            _, caption, filename, wide = block
            path = FIGURES / filename
            if not path.exists():
                continue
            ensure(1 if wide else 2)
            figure_no += 1
            doc.add_picture(str(path),
                            width=Inches(full_width if wide else cfg["column_width"]))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            label = (f"Fig. {figure_no}.  {cite(caption)}" if style == "ieee"
                     else f"Figure {figure_no}: {cite(caption)}")
            para(doc, label, cfg["caption_pt"],
                 align=(WD_ALIGN_PARAGRAPH.CENTER if style == "ieee"
                        else WD_ALIGN_PARAGRAPH.LEFT),
                 after=6)

    # references
    ensure(2)
    heading = "References"
    if style == "ieee":
        para(doc, heading, cfg["body_pt"], align=WD_ALIGN_PARAGRAPH.CENTER,
             caps=True, before=10, after=4, keep=True)
        for index, key in enumerate(order, start=1):
            p = para(doc, f"[{index}] {REFERENCES[key]['ieee']}", cfg["ref_pt"],
                     after=1)
            p.paragraph_format.left_indent = Inches(0.22)
            p.paragraph_format.first_line_indent = Inches(-0.22)
    else:
        para(doc, heading, cfg["body_pt"] + 2, bold=True,
             align=WD_ALIGN_PARAGRAPH.LEFT, before=10, after=4, keep=True)
        for key in sorted(order, key=lambda k: REFERENCES[k]["acl"]):
            p = para(doc, REFERENCES[key]["acl"], cfg["ref_pt"], after=2)
            p.paragraph_format.left_indent = Inches(0.22)
            p.paragraph_format.first_line_indent = Inches(-0.22)

    doc.save(cfg["out"])
    words = len(" ".join(p.text for p in doc.paragraphs).split())
    print(f"{style}: {cfg['out'].name}  {words} words, "
          f"{table_no} tables, {figure_no} figures, {len(order)} references")
    return cfg["out"]


if __name__ == "__main__":
    args = sys.argv[1:]
    paper = args[0] if args and args[0] in ("a", "b") else "a"
    styles = [a for a in args if a in STYLES] or ["ieee", "acl"]
    for target in styles:
        build(paper, target)
