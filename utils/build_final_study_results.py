"""Consolidate final-study metrics and generate paper-ready tables/figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "final_studies"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
GRAPHS_DIR = ROOT / "graphs" / "final_studies"
MODEL_ORDER = [
    "Majority",
    "Logistic Regression",
    "Linear SVM",
    "BiLSTM",
    "AraBERT",
    "MARBERT",
    "FAAV-Light",
    "FAAV-Full",
]


def best_model_rank(task: str, model: str) -> int:
    if task == "three_class":
        preferred = [
            "FAAV-Full",
            "FAAV-Light",
            "AraBERT",
            "MARBERT",
            "BiLSTM",
            "Linear SVM",
            "Logistic Regression",
        ]
    else:
        preferred = [
            "AraBERT",
            "MARBERT",
            "BiLSTM",
            "Linear SVM",
            "Logistic Regression",
            "FAAV-Full",
            "FAAV-Light",
        ]
    for index, prefix in enumerate(preferred):
        if str(model).startswith(prefix):
            return index
    return len(preferred)


def prediction_abstention(metrics_path: Path) -> float:
    prediction_path = (
        PREDICTIONS_DIR
        / metrics_path.name.replace("_metrics.json", "_predictions.csv")
    )
    if not prediction_path.exists():
        return float("nan")
    predictions = pd.read_csv(
        prediction_path,
        usecols=["prediction"],
        low_memory=False,
    )["prediction"]
    return float((predictions == "Unverified").mean())


def row_from_payload(path: Path, payload: dict) -> dict:
    metrics = payload["test_metrics"]
    report = metrics["classification_report"]
    validation_macro = payload.get("validation_macro_f1")
    if validation_macro is None:
        validation_macro = payload.get("validation_metrics", {}).get("macro_f1")
    if validation_macro is None:
        sweep = payload.get("validation_sweep", [])
        validation_macro = max(
            (float(row.get("macro_f1", np.nan)) for row in sweep),
            default=np.nan,
        )
    return {
        "Study": payload["study"],
        "Task": payload["task"],
        "Model": payload["model"],
        "Train N": int(payload["train_rows"]),
        "Validation N": int(payload["validation_rows"]),
        "Test N": int(payload["test_rows"]),
        "Accuracy": float(metrics["accuracy"]),
        "Macro Precision": float(metrics["macro_precision"]),
        "Macro Recall": float(metrics["macro_recall"]),
        "Macro F1": float(metrics["macro_f1"]),
        "Weighted F1": float(metrics["weighted_f1"]),
        "MCC": float(metrics["mcc"]),
        "Real Precision": report.get("Real", {}).get("precision", np.nan),
        "Real Recall": report.get("Real", {}).get("recall", np.nan),
        "Real F1": report.get("Real", {}).get("f1-score", np.nan),
        "Fake Precision": report.get("Fake", {}).get("precision", np.nan),
        "Fake Recall": report.get("Fake", {}).get("recall", np.nan),
        "Fake F1": report.get("Fake", {}).get("f1-score", np.nan),
        "Unverified Precision": report.get("Unverified", {}).get(
            "precision", np.nan
        ),
        "Unverified Recall": report.get("Unverified", {}).get(
            "recall", np.nan
        ),
        "Unverified F1": report.get("Unverified", {}).get(
            "f1-score", np.nan
        ),
        "Abstention Rate": payload.get(
            "abstention_rate",
            prediction_abstention(path),
        ),
        "Faithfulness Coverage": payload.get("coverage", np.nan),
        "Selective Accuracy": payload.get("selective_accuracy", np.nan),
        "Selective Macro F1": payload.get("selective_macro_f1", np.nan),
        "Selective MCC": payload.get("selective_mcc", np.nan),
        "Validation Macro F1": validation_macro,
        "Test Fingerprint": payload["test_fingerprint_sha256"],
    }


def load_model_rows() -> list[dict]:
    rows = []
    for path in RESULTS_DIR.glob("*_metrics.json"):
        if path.name == "manifest_audit.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "test_metrics" not in payload or "model" not in payload:
            continue
        rows.append(row_from_payload(path, payload))
    return rows


def load_majority_rows() -> list[dict]:
    path = RESULTS_DIR / "classical_model_comparison.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[frame["Model"].astype(str).str.startswith("Majority")].copy()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "Study": row["Study"],
                "Task": row["Task"],
                "Model": row["Model"],
                "Train N": int(row["Train N"]),
                "Validation N": int(row["Validation N"]),
                "Test N": int(row["Test N"]),
                "Accuracy": row["Accuracy"],
                "Macro Precision": np.nan,
                "Macro Recall": np.nan,
                "Macro F1": row["Macro F1"],
                "Weighted F1": row["Weighted F1"],
                "MCC": row["MCC"],
                "Real Precision": np.nan,
                "Real Recall": np.nan,
                "Real F1": row["Real F1"],
                "Fake Precision": np.nan,
                "Fake Recall": np.nan,
                "Fake F1": row["Fake F1"],
                "Unverified Precision": np.nan,
                "Unverified Recall": np.nan,
                "Unverified F1": row["Unverified F1"],
                "Abstention Rate": np.nan,
                "Faithfulness Coverage": np.nan,
                "Selective Accuracy": np.nan,
                "Selective Macro F1": np.nan,
                "Selective MCC": np.nan,
                "Validation Macro F1": np.nan,
                "Test Fingerprint": row["Test Fingerprint"],
            }
        )
    return rows


def model_rank(name: str) -> int:
    for index, prefix in enumerate(MODEL_ORDER):
        if str(name).startswith(prefix):
            return index
    return len(MODEL_ORDER)


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "Study",
        "Task",
        "Model",
        "Test N",
        "Accuracy",
        "Macro F1",
        "MCC",
        "Real F1",
        "Fake F1",
        "Unverified F1",
    ]
    display = frame[columns].copy()
    for column in [
        "Accuracy",
        "Macro F1",
        "MCC",
        "Real F1",
        "Fake F1",
        "Unverified F1",
    ]:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    headers = display.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_figures(frame: pd.DataFrame) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    plot = frame[~frame["Model"].str.startswith("Majority")].copy()

    best = (
        plot.sort_values("Macro F1", ascending=False)
        .groupby(["Study", "Task"], as_index=False)
        .first()
    )
    study_short = {
        "Study 1 - Zero-shot UAE": "S1 Zero-shot",
        "Study 2 - Source-domain": "S2 Source",
        "Study 3 - Combined adaptation": "S3 Combined",
    }
    task_short = {"binary": "Binary", "three_class": "3-class"}
    labels = [
        f"{study_short[row.Study]}\n{task_short[row.Task]}"
        for row in best.itertuples()
    ]
    colors = [
        "#13795B" if task == "three_class" else "#3568A8"
        for task in best["Task"]
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, best["Macro F1"], color=colors)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("Best validated model in each final study setting")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, best.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.3f}\n{row.Model}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "01_best_macro_f1_by_study.png", dpi=200)
    plt.close(fig)

    combined = plot[
        plot["Study"] == "Study 3 - Combined adaptation"
    ].copy()
    combined = combined.sort_values(["Task", "Macro F1"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{row.Model}\n{row.Task}" for row in combined.itertuples()]
    colors = [
        "#3568A8" if task == "binary" else "#C4553D"
        for task in combined["Task"]
    ]
    bars = ax.bar(labels, combined["Macro F1"], color=colors)
    ax.set_ylabel("Macro F1")
    ax.set_title("Combined-domain results on the full 20% test")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=25)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "02_combined_domain_macro_f1.png", dpi=200)
    plt.close(fig)

    three = plot[plot["Task"] == "three_class"].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    study_colors = {
        "Study 1 - Zero-shot UAE": "#7A5195",
        "Study 2 - Source-domain": "#2F6B4F",
        "Study 3 - Combined adaptation": "#D17A22",
    }
    for study, group in three.groupby("Study"):
        ax.scatter(
            group["Accuracy"],
            group["Macro F1"],
            label=study,
            color=study_colors[study],
            s=60,
            alpha=0.85,
        )
        row = group.sort_values("Macro F1", ascending=False).iloc[0]
        ax.annotate(
            f"Best: {row['Model']}",
            (row["Accuracy"], row["Macro F1"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro F1")
    ax.set_title("Three-class accuracy versus balanced performance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "03_three_class_accuracy_vs_macro_f1.png", dpi=200)
    plt.close(fig)


def main() -> None:
    rows = load_majority_rows() + load_model_rows()
    frame = pd.DataFrame(rows)
    frame["Model Rank"] = frame["Model"].map(model_rank)
    frame = (
        frame.sort_values(
            ["Task", "Study", "Macro F1", "Model Rank"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["Study", "Task", "Model"], keep="last")
        .drop(columns=["Model Rank"])
        .reset_index(drop=True)
    )
    output = RESULTS_DIR / "final_model_comparison.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    write_markdown(frame, RESULTS_DIR / "final_model_comparison.md")

    non_majority = frame[~frame["Model"].str.startswith("Majority")].copy()
    non_majority["Best Rank"] = [
        best_model_rank(task, model)
        for task, model in zip(
            non_majority["Task"],
            non_majority["Model"],
        )
    ]
    best = (
        non_majority.sort_values(
            ["Macro F1", "Best Rank"],
            ascending=[False, True],
        )
        .drop_duplicates(["Study", "Task"], keep="first")
        .drop(columns=["Best Rank"])
        .sort_values(["Study", "Task"])
        .reset_index(drop=True)
    )
    non_majority = non_majority.drop(columns=["Best Rank"])
    best.to_csv(
        RESULTS_DIR / "final_best_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fingerprint_counts = (
        non_majority.groupby(["Study", "Task"])["Test Fingerprint"]
        .nunique()
        .to_dict()
    )
    manifest = json.loads(
        (RESULTS_DIR / "manifest_audit.json").read_text(encoding="utf-8")
    )
    study_3_union_checks = {}
    study_1_full_uae_checks = {}
    all_tests_twenty_percent = True
    for task in ["binary", "three_class"]:
        source_test = pd.read_csv(
            ROOT / "data" / "final_studies" / f"source_{task}_test.csv",
            usecols=["sample_id"],
        )
        uae_test = pd.read_csv(
            ROOT / "data" / "final_studies" / f"uae_{task}_test.csv",
            usecols=["sample_id"],
        )
        combined_test = pd.read_csv(
            ROOT / "data" / "final_studies" / f"combined_{task}_test.csv",
            usecols=["sample_id"],
        )
        expected = set(source_test["sample_id"]) | set(uae_test["sample_id"])
        actual = set(combined_test["sample_id"])
        study_3_union_checks[task] = actual == expected
        uae_full = pd.read_csv(
            ROOT / "data" / "final_studies" / f"uae_{task}_full.csv",
            usecols=["sample_id"],
        )
        study_1_rows = non_majority[
            (non_majority["Study"] == "Study 1 - Zero-shot UAE")
            & (non_majority["Task"] == task)
        ]
        study_1_full_uae_checks[task] = (
            not study_1_rows.empty
            and int(study_1_rows["Test N"].iloc[0]) == len(uae_full)
            and study_1_rows["Test Fingerprint"].nunique() == 1
        )
        for domain in ["source", "uae", "combined"]:
            fraction = manifest["splits"][f"{domain}_{task}_test"][
                "fraction_of_task_dataset"
            ]
            all_tests_twenty_percent = (
                all_tests_twenty_percent and abs(fraction - 0.20) <= 0.001
            )
    audit = {
        "protocol": "final_three_study_full_uae_zero_shot_v3",
        "models_in_final_table": sorted(non_majority["Model"].unique().tolist()),
        "settings_evaluated": int(
            non_majority[["Study", "Task"]].drop_duplicates().shape[0]
        ),
        "same_test_fingerprint_within_each_setting": all(
            count == 1 for count in fingerprint_counts.values()
        ),
        "source_uae_split_and_combined_tests_are_approximately_20_percent": (
            all_tests_twenty_percent
        ),
        "study_1_uses_complete_uae_task_dataset": study_1_full_uae_checks,
        "study_3_test_is_union_of_source_and_uae_tests": study_3_union_checks,
        "manifest_overlap_checks": manifest["overlap_checks"],
        "best_results": best[
            ["Study", "Task", "Model", "Test N", "Accuracy", "Macro F1", "MCC"]
        ].to_dict(orient="records"),
        "interpretation_warning": (
            "Study 1 evaluates the complete UAE dataset, including rows used "
            "for Study 3 training and validation. It is a full external-domain "
            "zero-shot diagnostic and is not row-for-row comparable with "
            "Study 3's frozen 20% combined test."
        ),
    }
    (RESULTS_DIR / "final_results_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    create_figures(frame)
    print(
        best[
            ["Study", "Task", "Model", "Test N", "Accuracy", "Macro F1", "MCC"]
        ].to_string(index=False)
    )
    print(f"\nSaved {output}")
    print(f"Saved audit and figures under {RESULTS_DIR} and {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
