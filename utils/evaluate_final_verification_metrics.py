"""Compute final FAAV verification metrics on annotated UAE test rows.

Source-domain manifests do not contain gold retrieval-quality, faithfulness,
verification-stance, or citation annotations. Verification metrics are
therefore evaluated only on UAE rows, including the UAE subset of Study 3.
The annotation fields are used for evaluation only and are never model inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "final_studies"
RESULTS_DIR = ROOT / "results" / "final_studies"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS = ["FAAV-Light", "FAAV-Full"]
STUDIES = {
    "Study 1 - Zero-shot UAE": "uae_full",
    "Study 2 - Source-domain": "source",
    "Study 3 - Combined adaptation": "combined",
}
TASKS = ["binary", "three_class"]
RELEVANCE_WEIGHTS = {
    "Relevant": 1.0,
    "Partially Relevant": 0.5,
    "Irrelevant": 0.0,
}
VERDICT_TO_STANCE = {
    "Real": "Supported",
    "Fake": "Refuted",
    "Unverified": "Not Enough Evidence",
}


def slug(study: str, task: str, model: str) -> str:
    return (
        f"{study}_{task}_{model.lower().replace(' ', '_')}"
        .replace("+", "plus")
    )


def truthy(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def annotated_mask(frame: pd.DataFrame) -> pd.Series:
    required = [
        "retrieval_quality",
        "faithfulness_label",
        "verification_label",
    ]
    mask = pd.Series(True, index=frame.index)
    for column in required:
        if column not in frame:
            return pd.Series(False, index=frame.index)
        mask &= frame[column].fillna("").astype(str).str.strip().ne("")
    return mask


def effective_output(
    frame: pd.DataFrame,
    task: str,
) -> tuple[pd.Series, pd.Series]:
    if task == "binary":
        committed = (
            truthy(frame["accepted"])
            if "accepted" in frame
            else frame["faithfulness_status"].eq("Verified")
        )
        final_output = frame["prediction"].where(committed, "Unverified")
    else:
        final_output = frame["prediction"].astype(str)
        committed = final_output.ne("Unverified")
    return final_output, committed


def evaluate_setting(
    *,
    study: str,
    domain: str,
    task: str,
    model: str,
) -> dict:
    manifest_name = (
        f"uae_{task}_full"
        if domain == "uae_full"
        else f"{domain}_{task}_test"
    )
    manifest = pd.read_csv(DATA_DIR / f"{manifest_name}.csv", low_memory=False)
    prediction_path = (
        PREDICTIONS_DIR
        / f"{slug(study, task, model)}_predictions.csv"
    )
    predictions = pd.read_csv(prediction_path, low_memory=False)
    annotation_columns = [
        "sample_id",
        "retrieval_quality",
        "faithfulness_label",
        "verification_label",
        "evidence_url",
    ]
    annotation_columns = [
        column for column in annotation_columns if column in manifest
    ]
    merged = predictions.merge(
        manifest[annotation_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    annotated = merged.loc[annotated_mask(merged)].copy()
    base = {
        "Study": study,
        "Task": task,
        "Model": model,
        "Test N": int(len(merged)),
        "Verification N": int(len(annotated)),
    }
    if annotated.empty:
        return {
            **base,
            "Citation N": 0,
            "Evidence Relevance": np.nan,
            "Verified Evidence Relevance": np.nan,
            "Faithfulness Score": np.nan,
            "Hallucination Rate": np.nan,
            "Citation Support Accuracy": np.nan,
            "Commitment Rate": np.nan,
            "Citation Coverage": np.nan,
        }

    final_output, committed = effective_output(annotated, task)
    relevance = annotated["retrieval_quality"].map(RELEVANCE_WEIGHTS)
    predicted_faithfulness = pd.Series(
        np.where(committed, "High", "Low"),
        index=annotated.index,
    )
    faithfulness_score = float(
        (
            predicted_faithfulness.values
            == annotated["faithfulness_label"].astype(str).values
        ).mean()
    )
    hallucination_rate = (
        float(
            annotated.loc[committed, "faithfulness_label"]
            .astype(str)
            .eq("Low")
            .mean()
        )
        if committed.any()
        else np.nan
    )

    citation_mask = (
        annotated["evidence_url"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    expected_stance = final_output.map(VERDICT_TO_STANCE)
    citation_support = (
        float(
            (
                expected_stance.loc[citation_mask].values
                == annotated.loc[
                    citation_mask,
                    "verification_label",
                ].astype(str).values
            ).mean()
        )
        if citation_mask.any()
        else np.nan
    )
    return {
        **base,
        "Citation N": int(citation_mask.sum()),
        "Evidence Relevance": float(relevance.mean()),
        "Verified Evidence Relevance": (
            float(relevance.loc[committed].mean())
            if committed.any()
            else np.nan
        ),
        "Faithfulness Score": faithfulness_score,
        "Hallucination Rate": hallucination_rate,
        "Citation Support Accuracy": citation_support,
        "Commitment Rate": float(committed.mean()),
        "Citation Coverage": float(citation_mask.mean()),
    }


def main() -> None:
    rows = [
        evaluate_setting(
            study=study,
            domain=domain,
            task=task,
            model=model,
        )
        for study, domain in STUDIES.items()
        for task in TASKS
        for model in MODELS
    ]
    verification = pd.DataFrame(rows)
    verification.to_csv(
        RESULTS_DIR / "faav_verification_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison = pd.read_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        low_memory=False,
    )
    final = comparison.merge(
        verification,
        on=["Study", "Task", "Model", "Test N"],
        how="left",
    )
    final.to_csv(
        RESULTS_DIR / "final_evaluation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    definitions = {
        "scope": (
            "Study 1 verification metrics use the complete annotated UAE task "
            "dataset. Study 3 values use its frozen UAE test subset; Study 2 "
            "has no verification gold."
        ),
        "annotation_status": (
            "retrieval_quality, faithfulness_label, and verification_label are "
            "silver annotations and are used for evaluation only"
        ),
        "metrics": {
            "Evidence Relevance": (
                "Mean retrieval-quality weight: Relevant=1, Partially "
                "Relevant=0.5, Irrelevant=0."
            ),
            "Faithfulness Score": (
                "Accuracy of FAAV High/Low faithfulness status against the "
                "silver faithfulness label."
            ),
            "Hallucination Rate": (
                "Fraction of committed Real/Fake outputs whose evidence has a "
                "silver Low faithfulness label."
            ),
            "Citation Support Accuracy": (
                "Accuracy of final Real/Fake/Unverified output mapped to "
                "Supported/Refuted/Not Enough Evidence on rows with a URL."
            ),
        },
    }
    (RESULTS_DIR / "verification_metric_definitions.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        verification[
            [
                "Study",
                "Task",
                "Model",
                "Verification N",
                "Evidence Relevance",
                "Faithfulness Score",
                "Hallucination Rate",
                "Citation Support Accuracy",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
