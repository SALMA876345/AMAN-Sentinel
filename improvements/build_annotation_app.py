"""Build the annotation dashboard from the blind sheet.

Produces a single self-contained HTML file with all 240 claim-evidence pairs
embedded. The page is the annotation instrument; `annotation_sample.py` remains
the source of truth for which rows are in it.

Nothing from `annotation_key.csv` is embedded. The silver labels, the stratum
and the publisher name stay out of the file entirely, so the annotator cannot
reach them from the page even by reading its source.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / "results" / "improvements" / "annotation"
OUT_PATH = ANNOTATION_DIR / "annotation_dashboard.html"
STANDALONE_PATH = ANNOTATION_DIR / "AFVD_Annotation_Bench.html"
TEMPLATE_PATH = Path(__file__).resolve().parent / "annotation_app_template.html"

# The corpus carries HTML entities and stray control whitespace from its
# original scrape. Left as-is they render as literal "&nbsp;" in the page.
ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def clean(text: str) -> str:
    text = str(text)
    if ENTITY.search(text):
        text = html.unescape(text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    import pandas as pd

    sheet = pd.read_csv(
        ANNOTATION_DIR / "annotation_sheet_annotator_B.csv", dtype=str
    ).fillna("")

    items = [
        {
            "id": int(row["annotation_id"]),
            "claim": clean(row["claim_text"]),
            "evidence": clean(row["evidence_text"]),
        }
        for _, row in sheet.iterrows()
    ]

    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    # A literal </script> inside the JSON would close the host script element.
    payload = payload.replace("</", "<\\/")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    document = template.replace("/*__ITEMS__*/null", payload)

    # Viewer version: the Artifact host supplies the doctype and head.
    OUT_PATH.write_text(document, encoding="utf-8")
    size = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH} ({size:.0f} KB, {len(items)} items)")

    # Standalone version: a complete document that opens from disk in any
    # browser, for an annotator outside the owner's organization. The page
    # detects that no viewer is present and saves to localStorage instead.
    standalone = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "</head>\n<body>\n"
        + document
        + "\n</body>\n</html>\n"
    )
    STANDALONE_PATH.write_text(standalone, encoding="utf-8")
    size = STANDALONE_PATH.stat().st_size / 1024
    print(f"wrote {STANDALONE_PATH} ({size:.0f} KB, standalone)")


if __name__ == "__main__":
    main()
