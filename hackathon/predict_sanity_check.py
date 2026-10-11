"""Phase 1 Step 8: minimal prediction sanity test.

NOT the AMAN UI. This only proves the persisted vectorizer + model can be
loaded from disk and used to score real Arabic text in the format the
model expects (claim_norm + ' [SEP] ' + evidence_norm), end to end.

Examples are drawn from the clean TEST split (held-out, never used for
training or hyperparameter selection) by fixed row position — not hand-
picked, not invented. Gold labels are reported alongside the prediction
for comparison only; they are not used to select which rows to show.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "artifacts" / "svm"
CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
TASK = "binary"
N_EXAMPLES = 8
SEED = 42


def load_model():
    vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.joblib")
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    config = json.loads((ARTIFACT_DIR / "config.json").read_text(encoding="utf-8"))
    return vectorizer, model, config


def predict(vectorizer, model, claim_norm: str, evidence_norm: str) -> dict:
    model_text = f"{claim_norm} [SEP] {evidence_norm}"
    matrix = vectorizer.transform([model_text])
    prediction = model.predict(matrix)[0]
    score = model.decision_function(matrix)
    decision_score = float(score[0]) if np.ndim(score) == 1 else float(score[0].max())
    return {
        "prediction": str(prediction),
        "decision_score": decision_score,
        "model_used": "LinearSVC (clean, hackathon/train_svm_clean.py)",
    }


def main() -> None:
    if not (ARTIFACT_DIR / "model.joblib").exists():
        raise FileNotFoundError("Run hackathon/train_svm_clean.py first.")

    vectorizer, model, config = load_model()
    test = pd.read_csv(CLEAN_DIR / f"source_{TASK}_test.csv", low_memory=False).fillna("")

    # Fixed, deterministic sample of real held-out rows (not cherry-picked)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(test), size=min(N_EXAMPLES, len(test)), replace=False)
    sample = test.iloc[idx].reset_index(drop=True)

    print(f"Model config: C={config['selected_C']}  class_weight={config['selected_class_weight']}")
    print(f"Sanity-checking {len(sample)} real held-out examples from "
          f"source_{TASK}_test.csv (sample_ids not used for training or tuning)\n")

    results = []
    n_correct = 0
    for i, row in sample.iterrows():
        out = predict(vectorizer, model, row["claim_norm"], row["evidence_norm"])
        out["sample_id"] = row["sample_id"]
        out["gold_label"] = row["task_label"]
        out["matches_gold"] = out["prediction"] == out["gold_label"]
        n_correct += int(out["matches_gold"])
        results.append(out)
        claim_preview = row["claim_text"][:80].replace("\n", " ")
        print(f"[{i + 1}] sample_id={row['sample_id']}")
        print(f"    claim (preview): {claim_preview}...")
        print(f"    gold={out['gold_label']:<6} prediction={out['prediction']:<6} "
              f"decision_score={out['decision_score']:+.4f}  "
              f"{'MATCH' if out['matches_gold'] else 'MISMATCH'}")
        print()

    print(f"Sanity check: {n_correct}/{len(sample)} predictions matched gold label on this sample "
          f"(this is a loadability/inference smoke test, not a statistically meaningful accuracy "
          f"estimate — see hackathon/evaluate_svm_clean.py for the full test-set evaluation).")

    out_path = ARTIFACT_DIR / "prediction_sanity_check.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
