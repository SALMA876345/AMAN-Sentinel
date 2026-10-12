"""Phase 2 Step 5: model disagreement analysis on the TEST split.

Works with whichever model prediction files are present among
{svm, bilstm, transformer} under phase2/reports/ — so it can be run with
2 models now and re-run automatically picking up a 3rd once available.
No model is retrained here; this only reads existing prediction CSVs.

For every pair of available models, reports:
  - disagreement rate
  - P(error | both models agree)
  - P(error | models disagree)
  where "error" means the FIRST model's prediction in the pair is wrong
  (reported symmetrically for both orderings so either reading is covered).

Also reports a 3-way (if 3 models available) breakdown: accuracy of a
simple majority vote conditioned on how many of the 3 models agreed
(3/3 unanimous vs 2/1 split).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Code files/
REPORTS_DIR = ROOT / "phase2" / "reports"

MODEL_FILES = {
    "svm": REPORTS_DIR / "svm_predictions_test.csv",
    "bilstm": REPORTS_DIR / "bilstm_predictions_test.csv",
    "transformer": REPORTS_DIR / "transformer_predictions_test.csv",
}


def load_available_models() -> dict[str, pd.DataFrame]:
    available = {}
    for name, path in MODEL_FILES.items():
        if path.exists():
            df = pd.read_csv(path, low_memory=False).set_index("sample_id")
            available[name] = df
    return available


def pairwise_analysis(models: dict[str, pd.DataFrame]) -> dict:
    names = list(models.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common_idx = models[a].index.intersection(models[b].index)
            pa = models[a].loc[common_idx, "prediction"]
            pb = models[b].loc[common_idx, "prediction"]
            true = models[a].loc[common_idx, "true_label"]

            agree = pa.to_numpy() == pb.to_numpy()
            a_correct = pa.to_numpy() == true.to_numpy()
            b_correct = pb.to_numpy() == true.to_numpy()

            disagreement_rate = float((~agree).mean())
            p_error_a_given_agree = float((~a_correct[agree]).mean()) if agree.sum() else float("nan")
            p_error_a_given_disagree = float((~a_correct[~agree]).mean()) if (~agree).sum() else float("nan")
            p_error_b_given_agree = float((~b_correct[agree]).mean()) if agree.sum() else float("nan")
            p_error_b_given_disagree = float((~b_correct[~agree]).mean()) if (~agree).sum() else float("nan")
            # at least one of the pair wrong, conditioned on agree/disagree
            either_wrong = ~a_correct | ~b_correct
            p_either_error_given_agree = float(either_wrong[agree].mean()) if agree.sum() else float("nan")
            p_either_error_given_disagree = float(either_wrong[~agree].mean()) if (~agree).sum() else float("nan")

            results[f"{a}_vs_{b}"] = {
                "n": int(len(common_idx)),
                "disagreement_rate": disagreement_rate,
                "p_error_given_agree": {
                    a: p_error_a_given_agree, b: p_error_b_given_agree,
                    "either": p_either_error_given_agree,
                },
                "p_error_given_disagree": {
                    a: p_error_a_given_disagree, b: p_error_b_given_disagree,
                    "either": p_either_error_given_disagree,
                },
            }
    return results


def three_way_analysis(models: dict[str, pd.DataFrame]) -> dict | None:
    names = list(models.keys())
    if len(names) < 3:
        return None
    common_idx = models[names[0]].index
    for n in names[1:]:
        common_idx = common_idx.intersection(models[n].index)

    preds = {n: models[n].loc[common_idx, "prediction"] for n in names}
    true = models[names[0]].loc[common_idx, "true_label"]

    frame = pd.DataFrame(preds)
    frame["true_label"] = true.to_numpy()
    frame["n_agree_on_majority"] = frame[names].apply(lambda row: row.value_counts().iloc[0], axis=1)
    frame["majority_vote"] = frame[names].apply(lambda row: row.value_counts().idxmax(), axis=1)
    frame["majority_correct"] = frame["majority_vote"] == frame["true_label"]

    breakdown = {}
    for n_agree in sorted(frame["n_agree_on_majority"].unique()):
        subset = frame[frame["n_agree_on_majority"] == n_agree]
        breakdown[f"{n_agree}_of_{len(names)}_agree"] = {
            "n": int(len(subset)),
            "majority_vote_accuracy": float(subset["majority_correct"].mean()),
        }
    return {"n": int(len(frame)), "breakdown": breakdown}


def main() -> None:
    models = load_available_models()
    print(f"Available models for disagreement analysis: {list(models.keys())}")
    if len(models) < 2:
        print("Need at least 2 models with predictions; nothing to do yet.")
        return

    pairwise = pairwise_analysis(models)
    for pair, result in pairwise.items():
        print(f"\n[{pair}]  n={result['n']}  disagreement_rate={result['disagreement_rate']:.4f}")
        print(f"  P(error | agree)    : {result['p_error_given_agree']}")
        print(f"  P(error | disagree) : {result['p_error_given_disagree']}")

    three_way = three_way_analysis(models)
    if three_way:
        print(f"\n[3-way majority vote breakdown]  n={three_way['n']}")
        for key, val in three_way["breakdown"].items():
            print(f"  {key}: n={val['n']}  majority_vote_accuracy={val['majority_vote_accuracy']:.4f}")

    out = {"models_available": list(models.keys()), "pairwise": pairwise, "three_way": three_way}
    out_path = REPORTS_DIR / "disagreement_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
