"""Rebuild the best-model table with ties made explicit.

``results/final_studies/final_best_results.csv`` selects one winner per study
and task. In Study 3 three-class, AraBERT, FAAV-Light and FAAV-Full produce
byte-identical metrics, because the validation sweep selected an all-accept
gate and the two FAAV variants reduce to the verifier they wrap. The table
resolves that three-way tie in favour of FAAV-Full, and the report's headline
table inherits the choice, so the proposed system appears to win a setting in
which it did not act.

This module re-derives the same table from the shipped metric JSONs, marks
every tie at a stated tolerance, and names the tied models, so the selection is
visible rather than implicit. It writes a new file and leaves the original
alone.

Outputs ``results/improvements/best_results_tie_aware.csv``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "final_studies"
OUT_DIR = ROOT / "results" / "improvements"

# Four decimals is the precision the report tables are quoted at, so models
# that are indistinguishable there are indistinguishable to a reader.
TOLERANCE = 1e-4


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in FINAL_DIR.glob("*_metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("test_metrics")
        if not metrics:
            continue
        records.append(
            {
                "Study": payload["study"],
                "Task": payload["task"],
                "Model": payload["model"],
                "Test N": payload.get("test_rows"),
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "MCC": metrics["mcc"],
                "Abstention rate": payload.get("abstention_rate"),
            }
        )

    frame = pd.DataFrame(records)
    rows = []
    for (study, task), group in frame.groupby(["Study", "Task"]):
        best = group["Macro F1"].max()
        tied = group[group["Macro F1"] >= best - TOLERANCE]
        winner = tied.sort_values("Model").iloc[0]
        rows.append(
            {
                "Study": study,
                "Task": task,
                "Best macro F1": best,
                "Tied models": len(tied),
                "Tie": len(tied) > 1,
                "Models at best score": ", ".join(sorted(tied["Model"])),
                "Reported winner": winner["Model"],
                "Selection note": (
                    "unique best"
                    if len(tied) == 1
                    else f"{len(tied)}-way tie within {TOLERANCE:g} macro F1;"
                    " naming any single winner is arbitrary"
                ),
                "Accuracy": winner["Accuracy"],
                "MCC": winner["MCC"],
                "Test N": winner["Test N"],
            }
        )

    result = pd.DataFrame(rows).sort_values(["Task", "Study"])
    path = OUT_DIR / "best_results_tie_aware.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(
        result[
            [
                "Study",
                "Task",
                "Best macro F1",
                "Tied models",
                "Models at best score",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
