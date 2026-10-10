import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET
import json
import csv

# Change this path if your folder is somewhere else

from config import SOURCES_FILE, AFND_DIR
OUTPUT_FILE = os.path.join(RESULTS_DIR, "afnd_training_sample.csv")

# Load source labels
with open(SOURCES_FILE, "r", encoding="utf-8") as f:
    source_labels = json.load(f)

rows = []

for source_id, label in source_labels.items():
    # Skip undecided sources
    if label == "undecided":
        continue

    source_folder = os.path.join(RESULTS_DIR, source_id)
    articles_file = os.path.join(source_folder, "scraped_articles.json")

    # Skip sources not included in AFND_Selected
    if not os.path.exists(articles_file):
        continue

    with open(articles_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])

    for article in articles:
        title = article.get("title", "").strip()
        text = article.get("text", "").strip()
        published_date = article.get("published date", "").strip()

        if not title and not text:
            continue

        rows.append({
            "source_id": source_id,
            "title": title,
            "text": text,
            "published_date": published_date,
            "label": label
        })

# Save CSV
with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["source_id", "title", "text", "published_date", "label"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Done. Created file: {OUTPUT_FILE}")
print(f"Total articles: {len(rows)}")

# Count labels
label_counts = {}
for row in rows:
    label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

print("Label counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")