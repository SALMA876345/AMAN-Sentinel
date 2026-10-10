"""Heuristic error analysis for the six best final-study systems.

The requested categories are diagnostic indicators rather than human gold
labels. One error may receive multiple categories. Each tag stores a concise
trigger reason so the resulting analysis remains auditable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.faav_common import lexical_overlap, normalize_arabic


DATA_DIR = ROOT / "data" / "final_studies"
RESULTS_DIR = ROOT / "results" / "final_studies"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
GRAPHS_DIR = ROOT / "graphs" / "final_studies"

STUDY_DOMAINS = {
    "Study 1 - Zero-shot UAE": "uae_full",
    "Study 2 - Source-domain": "source",
    "Study 3 - Combined adaptation": "combined",
}
CATEGORIES = [
    "Dialect failure",
    "Sarcasm",
    "Political ambiguity",
    "Conflicting evidence",
    "Retrieval failure",
    "Temporal misinformation",
]

DIALECT_MARKERS = {
    "شو",
    "ليش",
    "وين",
    "شلون",
    "شنو",
    "هسه",
    "هسة",
    "اكو",
    "ماكو",
    "وايد",
    "مب",
    "يبا",
    "الحين",
    "ترى",
    "وش",
    "نبغى",
    "ابغى",
    "مو",
    "مافي",
    "بدي",
    "هلق",
    "هيك",
    "كتير",
    "عايز",
    "عاوز",
    "ازاي",
    "ازاى",
    "كده",
    "لسه",
    "برضه",
}
DIALECT_PHRASES = [
    "شو السالفه",
    "وش السالفه",
    "مش عارف",
    "مو صحيح",
    "مب صحيح",
    "ما في",
]
SARCASM_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"ه{3,}",
        r"خ{3,}",
        r"(?:يا\s+سلام|يا\s+عيني|طبعا\s*[!؟?]|اكيد\s*[!؟?])",
        r"(?:سخريه|ساخر|تهكم|يتريق|تريقه)",
        r"(?:😂|🤣|😏|🙃|🤡|😹|😆)",
    ]
]
POLITICAL_MARKERS = {
    "رئيس",
    "رئيسه",
    "وزير",
    "وزاره",
    "حكومه",
    "مجلس",
    "برلمان",
    "انتخابات",
    "حزب",
    "سياسي",
    "سياسيه",
    "معارضه",
    "نظام",
    "دوله",
    "سفاره",
    "جيش",
    "قوات",
    "احتلال",
    "حرب",
    "هدنه",
    "عقوبات",
    "قرار",
    "مرسوم",
    "ايران",
    "اسرائيل",
    "فلسطين",
    "حماس",
    "الامارات",
    "السعوديه",
    "قطر",
    "مصر",
    "سوريا",
    "اليمن",
    "السودان",
}
AMBIGUITY_MARKERS = {
    "يزعم",
    "زعم",
    "ادعى",
    "ادعاء",
    "انباء",
    "مصادر",
    "بحسب",
    "ربما",
    "قد",
    "يقال",
    "غير مؤكد",
    "نفى",
    "نفت",
    "اتهام",
}
SUPPORT_CUES = {
    "اكد",
    "اكدت",
    "يؤكد",
    "اعلن",
    "اعلنت",
    "صرح",
    "صرحت",
    "اثبت",
    "اثبتت",
    "صحيح",
    "يدعم",
}
REFUTE_CUES = {
    "نفى",
    "نفت",
    "ينفي",
    "تنفي",
    "كاذب",
    "كاذبه",
    "مضلل",
    "مضلله",
    "زائف",
    "مزيف",
    "مفبرك",
    "غير صحيح",
    "لا صحه",
    "لم يحدث",
    "لا يوجد دليل",
}
CURRENT_TIME_MARKERS = {
    "اليوم",
    "الان",
    "حاليا",
    "امس",
    "غدا",
    "هذه الساعه",
    "هذا الاسبوع",
    "هذا الشهر",
    "هذا العام",
    "حديثا",
    "عاجل",
}
OLD_TIME_MARKERS = {
    "قديم",
    "قديمه",
    "سابق",
    "سابقا",
    "ارشيف",
    "ارشيفي",
    "اعيد نشر",
    "منذ سنوات",
}
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def slug(study: str, task: str, model: str) -> str:
    return (
        f"{study}_{task}_{model.lower().replace(' ', '_')}"
        .replace("+", "plus")
    )


def tokens(text: object) -> set[str]:
    return set(normalize_arabic(text).split())


def contains_phrase(text: str, phrases: list[str] | set[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def extract_years(text: object) -> set[str]:
    normalized = str(text).translate(ARABIC_DIGITS)
    return set(YEAR_RE.findall(normalized))


def short_text(text: object, limit: int = 150) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def classify_error(
    row: pd.Series,
    conflicting_evidence_groups: set[str],
) -> tuple[list[str], dict[str, str]]:
    claim = normalize_arabic(row.get("claim_text", ""))
    evidence = normalize_arabic(row.get("evidence_text", ""))
    claim_tokens = set(claim.split())
    evidence_tokens = set(evidence.split())
    combined = f"{claim} {evidence}".strip()
    categories = []
    reasons: dict[str, str] = {}

    dialect_hits = sorted(
        (claim_tokens & DIALECT_MARKERS)
        | set(contains_phrase(claim, DIALECT_PHRASES))
    )
    if dialect_hits:
        categories.append("Dialect failure")
        reasons["Dialect failure"] = "markers=" + ",".join(dialect_hits[:5])

    sarcasm_hits = [
        pattern.pattern
        for pattern in SARCASM_PATTERNS
        if pattern.search(str(row.get("claim_text", "")))
    ]
    if sarcasm_hits:
        categories.append("Sarcasm")
        reasons["Sarcasm"] = "sarcasm_marker"

    political_hits = sorted(claim_tokens & POLITICAL_MARKERS)
    ambiguity_hits = sorted(claim_tokens & AMBIGUITY_MARKERS)
    if political_hits and (
        ambiguity_hits
        or "؟" in str(row.get("claim_text", ""))
        or "?" in str(row.get("claim_text", ""))
        or len(political_hits) >= 2
    ):
        categories.append("Political ambiguity")
        reason = "political=" + ",".join(political_hits[:4])
        if ambiguity_hits:
            reason += "; ambiguity=" + ",".join(ambiguity_hits[:3])
        reasons["Political ambiguity"] = reason

    support_hits = sorted(evidence_tokens & SUPPORT_CUES)
    refute_hits = sorted(
        set(contains_phrase(evidence, REFUTE_CUES))
        | (evidence_tokens & REFUTE_CUES)
    )
    overlap = float(
        row.get("lexical_overlap", lexical_overlap(claim, evidence)) or 0.0
    )
    evidence_chars = len(evidence)
    conflict_reasons = []
    if support_hits and refute_hits:
        conflict_reasons.append("support_and_refute_cues")
    evidence_group = str(row.get("evidence_group", ""))
    if (
        evidence_group
        and evidence_group in conflicting_evidence_groups
        and evidence_chars >= 80
        and overlap >= 0.10
    ):
        conflict_reasons.append("evidence_reused_across_labels")
    claim_truth = str(row.get("claim_truth_label", "")).strip()
    stance = str(row.get("verification_label", "")).strip()
    if (
        (claim_truth == "Real" and stance == "Refuted")
        or (claim_truth == "Fake" and stance == "Supported")
    ):
        conflict_reasons.append("claim_truth_evidence_stance_mismatch")
    if conflict_reasons:
        categories.append("Conflicting evidence")
        reasons["Conflicting evidence"] = "|".join(conflict_reasons)

    retrieval_quality = str(row.get("retrieval_quality", "")).strip()
    retrieval_reasons = []
    if evidence_chars < 40:
        retrieval_reasons.append("short_or_empty_evidence")
    if overlap < 0.05:
        retrieval_reasons.append(f"low_overlap={overlap:.3f}")
    if retrieval_quality == "Irrelevant":
        retrieval_reasons.append("gold_irrelevant")
    elif retrieval_quality == "Partially Relevant" and overlap < 0.10:
        retrieval_reasons.append("partial_and_low_overlap")
    if retrieval_reasons:
        categories.append("Retrieval failure")
        reasons["Retrieval failure"] = "|".join(retrieval_reasons)

    claim_years = extract_years(row.get("claim_text", ""))
    evidence_years = extract_years(row.get("evidence_text", ""))
    temporal_reasons = []
    if claim_years and evidence_years and claim_years.isdisjoint(evidence_years):
        temporal_reasons.append(
            "year_mismatch="
            + ",".join(sorted(claim_years))
            + "!="
            + ",".join(sorted(evidence_years))
        )
    current_hits = sorted(claim_tokens & CURRENT_TIME_MARKERS)
    old_hits = sorted(evidence_tokens & OLD_TIME_MARKERS)
    if current_hits and old_hits:
        temporal_reasons.append(
            "current_claim_vs_old_evidence="
            + ",".join(current_hits[:2])
            + "/"
            + ",".join(old_hits[:2])
        )
    if temporal_reasons:
        categories.append("Temporal misinformation")
        reasons["Temporal misinformation"] = "|".join(temporal_reasons)

    return categories, reasons


def load_best_errors() -> pd.DataFrame:
    best = pd.read_csv(RESULTS_DIR / "final_best_results.csv")
    frames = []
    for _, setting in best.iterrows():
        study = str(setting["Study"])
        task = str(setting["Task"])
        model = str(setting["Model"])
        domain = STUDY_DOMAINS[study]
        manifest_name = (
            f"uae_{task}_full"
            if domain == "uae_full"
            else f"{domain}_{task}_test"
        )
        manifest = pd.read_csv(
            DATA_DIR / f"{manifest_name}.csv",
            low_memory=False,
        )
        label_column = "binary_label" if task == "binary" else "three_class_label"
        conflicting_groups = set(
            manifest.groupby("evidence_group")[label_column]
            .nunique()
            .loc[lambda values: values > 1]
            .index.astype(str)
        )
        prediction_path = (
            PREDICTIONS_DIR
            / f"{slug(study, task, model)}_predictions.csv"
        )
        predictions = pd.read_csv(prediction_path, low_memory=False)
        metadata_columns = [
            "sample_id",
            "claim_norm",
            "evidence_norm",
            "claim_group",
            "evidence_group",
            "claim_truth_label",
            "verification_label",
            "retrieval_quality",
            "faithfulness_label",
            "evidence_url",
        ]
        metadata_columns = [
            column for column in metadata_columns if column in manifest
        ]
        merged = predictions.merge(
            manifest[metadata_columns],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        errors = merged[
            merged["label"].astype(str) != merged["prediction"].astype(str)
        ].copy()
        errors["Study"] = study
        errors["Task"] = task
        errors["Model"] = model
        errors["Test N"] = len(merged)
        errors["Error N"] = len(errors)
        errors["error_confidence"] = pd.to_numeric(
            errors.get(
                "confidence",
                errors.get("verifier_confidence", np.nan),
            ),
            errors="coerce",
        )
        classified = errors.apply(
            lambda row: classify_error(row, conflicting_groups),
            axis=1,
        )
        errors["error_categories"] = [
            "|".join(result[0]) if result[0] else "Other / unclassified"
            for result in classified
        ]
        errors["error_reasons"] = [
            json.dumps(result[1], ensure_ascii=False, sort_keys=True)
            for result in classified
        ]
        for category in CATEGORIES:
            errors[category] = [
                category in result[0] for result in classified
            ]
        errors["Other / unclassified"] = ~errors[CATEGORIES].any(axis=1)
        frames.append(errors)
    return pd.concat(frames, ignore_index=True)


def setting_summary(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, task, model), group in errors.groupby(
        ["Study", "Task", "Model"],
        sort=False,
    ):
        error_n = int(len(group))
        row = {
            "Study": study,
            "Task": task,
            "Model": model,
            "Test N": int(group["Test N"].iloc[0]),
            "Error N": error_n,
            "Error Rate": error_n / int(group["Test N"].iloc[0]),
        }
        for category in CATEGORIES + ["Other / unclassified"]:
            count = int(group[category].sum())
            row[f"{category} N"] = count
            row[f"{category} Rate"] = count / error_n if error_n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def category_summary(errors: pd.DataFrame) -> pd.DataFrame:
    total = len(errors)
    rows = []
    for category in CATEGORIES + ["Other / unclassified"]:
        subset = errors[errors[category]]
        rows.append(
            {
                "Category": category,
                "Tagged Error Instances": int(len(subset)),
                "Share of Error Instances": len(subset) / total if total else 0.0,
                "Mean Error Confidence": float(
                    subset["error_confidence"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Tagged Error Instances",
        ascending=False,
    )


def representative_examples(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        candidates = errors[errors[category]].copy()
        if candidates.empty:
            continue
        candidates = candidates.drop_duplicates(
            ["sample_id", "claim_text"],
            keep="first",
        )
        study_3 = candidates[
            candidates["Study"] == "Study 3 - Combined adaptation"
        ]
        if not study_3.empty:
            candidates = pd.concat(
                [
                    study_3,
                    candidates.loc[
                        candidates.index.difference(study_3.index)
                    ],
                ]
            )
        candidates = candidates.sort_values(
            "error_confidence",
            ascending=False,
            na_position="last",
        )
        candidates = candidates.drop_duplicates("claim_text", keep="first")
        for _, row in candidates.head(2).iterrows():
            reason_map = json.loads(row["error_reasons"])
            rows.append(
                {
                    "Category": category,
                    "Study": row["Study"],
                    "Task": row["Task"],
                    "Model": row["Model"],
                    "Sample ID": row["sample_id"],
                    "Gold": row["label"],
                    "Prediction": row["prediction"],
                    "Confidence": row["error_confidence"],
                    "Trigger": reason_map.get(category, ""),
                    "Claim Snippet": short_text(row["claim_text"]),
                    "Evidence Snippet": short_text(row["evidence_text"]),
                }
            )
    return pd.DataFrame(rows)


def create_figure(summary: pd.DataFrame) -> None:
    plot = summary[
        summary["Category"].isin(CATEGORIES)
    ].sort_values("Share of Error Instances")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        plot["Category"],
        plot["Share of Error Instances"],
        color="#3568A8",
    )
    ax.set_xlabel("Share of model-setting error instances")
    ax.set_xlim(0, max(0.05, float(plot["Share of Error Instances"].max()) * 1.2))
    ax.set_title("Heuristic error indicators across the six best systems")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, plot["Share of Error Instances"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHS_DIR / "04_error_category_rates.png", dpi=200)
    plt.close(fig)


def main() -> None:
    errors = load_best_errors()
    setting = setting_summary(errors)
    category = category_summary(errors)
    examples = representative_examples(errors)

    row_columns = [
        "Study",
        "Task",
        "Model",
        "sample_id",
        "dataset_family",
        "source",
        "label",
        "prediction",
        "error_confidence",
        "error_categories",
        "error_reasons",
        "claim_text",
        "evidence_text",
    ]
    row_columns += [
        column
        for column in [
            "retrieval_quality",
            "faithfulness_label",
            "verification_label",
            "evidence_url",
        ]
        if column in errors
    ]
    errors[row_columns].to_csv(
        RESULTS_DIR / "final_error_analysis_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    setting.to_csv(
        RESULTS_DIR / "final_error_analysis_by_setting.csv",
        index=False,
        encoding="utf-8-sig",
    )
    category.to_csv(
        RESULTS_DIR / "final_error_category_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    examples.to_csv(
        RESULTS_DIR / "final_error_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    definitions = {
        "scope": (
            "Misclassified rows from the best model in each of the six final "
            "study/task settings. Counts are model-setting error instances, "
            "not unique claims."
        ),
        "method": (
            "Multi-label heuristic indicators. Percentages may sum above 100%. "
            "The tags support diagnosis but are not human gold causal labels."
        ),
        "categories": {
            "Dialect failure": (
                "Conservative Arabic dialect markers detected in the claim."
            ),
            "Sarcasm": (
                "Explicit laughter, sarcasm terms, or sarcasm-associated emoji."
            ),
            "Political ambiguity": (
                "Political entities or institutions combined with attribution, "
                "uncertainty, questioning, or multiple political actors."
            ),
            "Conflicting evidence": (
                "Mixed support/refutation cues, or substantial evidence reused "
                "across opposing labels."
            ),
            "Retrieval failure": (
                "Short/empty evidence, very low claim-evidence overlap, or "
                "irrelevant/weak retrieval annotation."
            ),
            "Temporal misinformation": (
                "Explicit year mismatch or current-event wording paired with "
                "old/archive evidence."
            ),
        },
    }
    (RESULTS_DIR / "final_error_analysis_definitions.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    create_figure(category)
    print("\nSetting summary:")
    print(
        setting[
            ["Study", "Task", "Model", "Test N", "Error N", "Error Rate"]
        ].to_string(index=False)
    )
    print("\nCategory summary:")
    print(category.to_string(index=False))


if __name__ == "__main__":
    main()
