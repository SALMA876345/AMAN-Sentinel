"""Entailment-based faithfulness, replacing the gate-dependent proxy.

The shipped RAG table reports "Faithfulness Score" and "Hallucination Rate"
columns that are computed two different ways depending on the row. Ungated
systems get a retrieval heuristic over overlap, evidence length and confidence;
gated systems get ``mean(accepted)``, which makes faithfulness numerically
identical to coverage and hallucination identical to the abstention rate. The
two halves of the column therefore measure different quantities and cannot be
compared, and a gate that abstains less mechanically looks more faithful.

This module computes one quantity for every system, independent of any gate.
A multilingual NLI model reads the retrieved evidence as premise and the claim
as hypothesis, giving the evidence-claim relation on its own terms:

    entailment    -> the evidence supports the claim   -> Real
    contradiction -> the evidence refutes the claim    -> Fake
    neutral       -> the evidence settles nothing      -> Unverified

A committed verdict is faithful when it matches that relation. Hallucination is
its complement over committed rows only, so abstention neither helps nor hurts
the score, and coverage is reported separately as the distinct quantity it is.

Outputs ``results/improvements/faithfulness_nli.csv`` plus per-row scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "improvements"
PREDICTIONS_DIR = OUT_DIR / "predictions"

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_LEN = 256

# NLI relation -> the verdict it licenses.
RELATION_TO_VERDICT = {
    "entailment": "Real",
    "contradiction": "Fake",
    "neutral": "Unverified",
}

STUDY_TITLES = {
    "study_1_zero_shot_uae": "Study 1 - Zero-shot UAE",
    "study_2_source_domain": "Study 2 - Source-domain",
    "study_3_combined_adaptation": "Study 3 - Combined adaptation",
}


def load_nli():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE)
    model.eval()
    # Label order differs between NLI checkpoints, so read it off the config
    # rather than assuming the usual entailment/neutral/contradiction order.
    id_to_relation = {
        int(i): str(name).lower() for i, name in model.config.id2label.items()
    }
    return tokenizer, model, id_to_relation


def nli_relations(
    tokenizer,
    model,
    id_to_relation: dict[int, str],
    premises: list[str],
    hypotheses: list[str],
) -> tuple[list[str], np.ndarray]:
    relations = []
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(premises), BATCH_SIZE):
            batch_premise = premises[start : start + BATCH_SIZE]
            batch_hypothesis = hypotheses[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch_premise,
                batch_hypothesis,
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(**encoded).logits
            probability = torch.softmax(logits.float(), dim=1).cpu().numpy()
            probabilities.append(probability)
            relations.extend(
                id_to_relation[int(i)] for i in probability.argmax(axis=1)
            )
    return relations, np.vstack(probabilities)


def main() -> None:
    if not PREDICTIONS_DIR.exists():
        raise SystemExit(
            "No predictions found. Run improvements/train_verifiers.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, model, id_to_relation = load_nli()
    print(f"NLI label map: {id_to_relation}", flush=True)

    rows = []
    for study_slug, study_title in STUDY_TITLES.items():
        for task in ["binary", "three_class"]:
            path = PREDICTIONS_DIR / f"{study_slug}_{task}_arabert.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False).fillna("")
            print(f"scoring {study_title} / {task} ({len(frame)} rows)", flush=True)

            relations, probability = nli_relations(
                tokenizer,
                model,
                id_to_relation,
                frame["evidence_norm"].astype(str).tolist(),
                frame["claim_norm"].astype(str).tolist(),
            )
            frame["nli_relation"] = relations
            frame["nli_verdict"] = [
                RELATION_TO_VERDICT.get(r, "Unverified") for r in relations
            ]
            for index, name in id_to_relation.items():
                frame[f"nli_{name}"] = probability[:, index]

            # Binary systems cannot emit Unverified, so a neutral premise-claim
            # relation can never be matched by any binary verdict. Faithfulness
            # is therefore scored only where the relation licenses a verdict the
            # system is able to produce; the neutral share is reported instead.
            committed = frame["prediction"] != "Unverified"
            if task == "binary":
                scorable = committed & frame["nli_verdict"].isin(["Real", "Fake"])
            else:
                scorable = committed

            faithful = (frame["prediction"] == frame["nli_verdict"]) & scorable
            n_scorable = int(scorable.sum())
            faithfulness = float(faithful.sum() / n_scorable) if n_scorable else float("nan")

            frame.to_csv(
                OUT_DIR / f"nli_{study_slug}_{task}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            rows.append(
                {
                    "Study": study_title,
                    "Task": task,
                    "Test N": len(frame),
                    "Commitment rate": float(committed.mean()),
                    "Scorable N": n_scorable,
                    "NLI faithfulness": faithfulness,
                    "NLI hallucination": 1.0 - faithfulness,
                    "Neutral evidence share": float(
                        (frame["nli_relation"] == "neutral").mean()
                    ),
                    "Entailment share": float(
                        (frame["nli_relation"] == "entailment").mean()
                    ),
                    "Contradiction share": float(
                        (frame["nli_relation"] == "contradiction").mean()
                    ),
                    # How often the verifier and an independent entailment model
                    # reach the same verdict, as a retrieval-quality read.
                    "Verifier/NLI agreement": float(
                        (frame["prediction"] == frame["nli_verdict"]).mean()
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    path = OUT_DIR / "faithfulness_nli.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print()
    print(frame.round(4).to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
