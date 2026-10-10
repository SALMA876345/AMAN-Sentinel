"""Real anonymised claim-evidence examples to replace Table 5's invented ones.

Table 5 currently presents "constructed and anonymized" Arabic examples. For a
section carrying 20% of the mark on the strength of a dataset contribution,
invented rows evidence nothing: they show the schema, not the data. Real rows
show what the corpus actually contains, including the messiness that makes the
verification task hard.

Anonymisation here is conservative. URLs are reduced to their domain, @handles
and standalone digit runs that could be identifiers are masked, and only the
publisher name - already a corpus-level field, not a personal one - is kept.
Rows are drawn from the test split so nothing about the training data is
disclosed beyond what the report already describes.

Outputs ``results/improvements/dataset_examples.csv`` and a Markdown table.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final_studies"
OUT_DIR = ROOT / "results" / "improvements"

SEED = 42
MAX_CHARS = 220

HANDLE = re.compile(r"@[\w_]+")
LONG_DIGITS = re.compile(r"\d{5,}")


def anonymise(text: str) -> str:
    text = HANDLE.sub("@USER", str(text))
    text = LONG_DIGITS.sub("#####", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rstrip() + " ..."
    return text


def domain(url: str) -> str:
    url = str(url).strip()
    if not url:
        return ""
    try:
        host = urlparse(url if "//" in url else f"https://{url}").netloc
    except ValueError:
        return ""
    return host.removeprefix("www.")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        DATA_DIR / "combined_three_class_test.csv", low_memory=False
    ).fillna("")

    # Prefer rows that carry the verification metadata, since those are the
    # ones that show what distinguishes this dataset from a claim-only corpus.
    frame = frame[
        (frame["claim_text"].str.len() > 40)
        & (frame["evidence_text"].str.len() > 60)
    ]

    picked = []
    for label in ["Real", "Fake", "Unverified"]:
        group = frame[frame["three_class_label"] == label]
        annotated = group[group["retrieval_quality"].astype(str) != ""]
        pool = annotated if len(annotated) >= 2 else group
        picked.append(pool.sample(n=min(2, len(pool)), random_state=SEED))

    sample = pd.concat(picked)
    table = pd.DataFrame(
        {
            "Claim (anonymised)": sample["claim_text"].map(anonymise),
            "Evidence (anonymised)": sample["evidence_text"].map(anonymise),
            "Label": sample["three_class_label"],
            "Evidence relevance": sample["retrieval_quality"].replace("", "-"),
            "Faithfulness": sample["faithfulness_label"].replace("", "-"),
            "Source": sample["source"],
            "Evidence domain": sample["evidence_url"].map(domain).replace("", "-"),
            "Corpus": sample["dataset_family"],
        }
    )

    path = OUT_DIR / "dataset_examples.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")

    markdown = ["| " + " | ".join(table.columns) + " |",
                "|" + "---|" * len(table.columns)]
    for _, row in table.iterrows():
        cells = [str(v).replace("|", "\\|") for v in row]
        markdown.append("| " + " | ".join(cells) + " |")
    (OUT_DIR / "dataset_examples.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    print(f"wrote {len(table)} real anonymised examples to {path}")
    print(table[["Label", "Evidence relevance", "Faithfulness", "Corpus"]].to_string(index=False))


if __name__ == "__main__":
    main()
