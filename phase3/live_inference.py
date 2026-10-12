"""Phase 3 integration: run SVM, BiLSTM, frozen-MARBERT-probe, and the
novelty detector on a SINGLE new (claim, evidence) pair, live.

Phase 1/2 only ever ran these models in batch over pre-built CSVs. For
`verify_content(text)` to work on arbitrary new input, each model needs a
single-example inference path. This module builds exactly that, reusing
every persisted artifact unchanged -- nothing here retrains anything.

Reuses:
  - utils/faav_common.normalize_arabic
  - artifacts/svm/{vectorizer,model}.joblib            (Phase 1)
  - artifacts/bilstm/{model.pt,vocab.json,config.json} (Phase 1)
  - artifacts/transformers/marbert_clean_binary/ + linear_probe_head.pt (Phase 2)
  - phase2/novelty_detector.load_fitted_detector        (Phase 2)
  - phase2/calibration fit parameters (recomputed identically: Platt a/b
    for SVM is NOT applied -- Phase 2 found Platt scaling makes SVM worse
    on held-out data, so production uses SVM's RAW confidence, exactly as
    AMAN_PHASE2_INTELLIGENCE_REPORT.md recommends)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parents[1]  # Code files/
sys.path.insert(0, str(ROOT))

from utils.faav_common import normalize_arabic  # noqa: E402

SVM_DIR = ROOT / "artifacts" / "svm"
BILSTM_DIR = ROOT / "artifacts" / "bilstm"
TRANSFORMER_DIR = ROOT / "artifacts" / "transformers"
LABELS = ["Real", "Fake"]
MAX_LEN = 128

# BiLSTM temperature from Phase 2 calibration (phase2/reports/calibration_results.json)
BILSTM_TEMPERATURE = 1.0624358735699397
TRANSFORMER_TEMPERATURE = None  # loaded lazily from calibration_results.json


@dataclass
class ModelVote:
    model: str
    prediction: str
    confidence: float          # calibrated where Phase 2 found calibration helps; raw for SVM
    raw_confidence: float


def _model_text(claim_norm: str, evidence_norm: str) -> str:
    return f"{claim_norm} [SEP] {evidence_norm}"


class LiveSVM:
    def __init__(self) -> None:
        self.vectorizer = joblib.load(SVM_DIR / "vectorizer.joblib")
        self.model = joblib.load(SVM_DIR / "model.joblib")

    def predict(self, claim_norm: str, evidence_norm: str) -> ModelVote:
        matrix = self.vectorizer.transform([_model_text(claim_norm, evidence_norm)])
        prediction = str(self.model.predict(matrix)[0])
        scores = self.model.decision_function(matrix)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        confidence = float(softmax(scores, axis=1).max(axis=1)[0])
        # Phase 2 finding: Platt-scaled SVM confidence is LESS calibrated on
        # held-out data than the raw value. Production uses raw.
        return ModelVote(model="svm", prediction=prediction, confidence=confidence, raw_confidence=confidence)


class _BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.dropout = nn.Dropout(0.30)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, ids, lengths):
        embedded = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(representation))


class LiveBiLSTM:
    def __init__(self) -> None:
        self.vocab = json.loads((BILSTM_DIR / "vocab.json").read_text(encoding="utf-8"))
        config = json.loads((BILSTM_DIR / "config.json").read_text(encoding="utf-8"))
        self.model = _BiLSTMClassifier(config["vocab_size"], len(LABELS))
        self.model.load_state_dict(torch.load(BILSTM_DIR / "model.pt", map_location="cpu"))
        self.model.eval()

    def predict(self, claim_norm: str, evidence_norm: str) -> ModelVote:
        text = _model_text(claim_norm, evidence_norm)
        ids = [self.vocab.get(tok, 1) for tok in text.split()[:MAX_LEN]]
        length = max(len(ids), 1)
        ids = ids or [1]
        ids += [0] * (MAX_LEN - len(ids))
        ids_t = torch.tensor([ids], dtype=torch.long)
        len_t = torch.tensor([length], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(ids_t, len_t)
        label_to_id = {label: i for i, label in enumerate(LABELS)}
        z = float(logits[0, label_to_id["Real"]] - logits[0, label_to_id["Fake"]])
        p_real_raw = float(torch.softmax(logits, dim=1)[0, label_to_id["Real"]])
        p_real_calibrated = 1.0 / (1.0 + np.exp(-z / BILSTM_TEMPERATURE))
        prediction = "Real" if p_real_raw >= 0.5 else "Fake"
        raw_conf = p_real_raw if prediction == "Real" else 1 - p_real_raw
        cal_conf = p_real_calibrated if prediction == "Real" else 1 - p_real_calibrated
        return ModelVote(model="bilstm", prediction=prediction, confidence=float(cal_conf), raw_confidence=float(raw_conf))


class LiveTransformer:
    """Frozen MARBERT backbone + linear probe head (Phase 2 strategy --
    full fine-tuning was CPU-infeasible; see AMAN_PHASE2_INTELLIGENCE_REPORT.md)."""

    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        checkpoint = TRANSFORMER_DIR / "marbert_clean_binary"
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        self.backbone = AutoModel.from_pretrained(str(checkpoint), local_files_only=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        head_config = json.loads((checkpoint / "linear_probe_head_config.json").read_text(encoding="utf-8"))
        hidden_dim = head_config["hidden_size"]
        # Saved state dict uses key "classifier.weight"/"classifier.bias"
        # (the training script wrapped the Linear in a `.classifier` attr,
        # same naming convention as BiLSTMClassifier elsewhere in this repo).
        self.head = nn.Module()
        self.head.classifier = nn.Linear(hidden_dim, len(head_config["labels"]))
        state = torch.load(checkpoint / "linear_probe_head.pt", map_location="cpu")
        self.head.load_state_dict(state)
        self.head.eval()

        global TRANSFORMER_TEMPERATURE
        if TRANSFORMER_TEMPERATURE is None:
            calib_path = ROOT / "phase2" / "reports" / "calibration_results.json"
            if calib_path.exists():
                results = json.loads(calib_path.read_text(encoding="utf-8"))
                for r in results:
                    if r.get("model") == "transformer":
                        TRANSFORMER_TEMPERATURE = r["temperature"]
            if TRANSFORMER_TEMPERATURE is None:
                TRANSFORMER_TEMPERATURE = 1.0

    def _mean_pool(self, claim_norm: str, evidence_norm: str) -> torch.Tensor:
        claim = claim_norm[:400]
        evidence = evidence_norm[:1200]
        encoded = self.tokenizer(claim, evidence, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
        with torch.inference_mode():
            outputs = self.backbone(**encoded)
        hidden = outputs.last_hidden_state  # (1, seq, dim)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def predict(self, claim_norm: str, evidence_norm: str) -> ModelVote:
        pooled = self._mean_pool(claim_norm, evidence_norm)
        with torch.inference_mode():
            logits = self.head.classifier(pooled)
        label_to_id = {label: i for i, label in enumerate(LABELS)}
        z = float(logits[0, label_to_id["Real"]] - logits[0, label_to_id["Fake"]])
        probs = torch.softmax(logits, dim=1)[0]
        prediction = LABELS[int(torch.argmax(probs))]
        raw_conf = float(probs.max())
        p_real_calibrated = 1.0 / (1.0 + np.exp(-z / TRANSFORMER_TEMPERATURE))
        cal_conf = p_real_calibrated if prediction == "Real" else 1 - p_real_calibrated
        return ModelVote(model="transformer", prediction=prediction, confidence=float(cal_conf), raw_confidence=raw_conf)


class LiveNoveltyDetector:
    def __init__(self) -> None:
        from phase2.novelty_detector import load_fitted_detector
        self.detector = load_fitted_detector()

    def score(self, claim_text: str) -> tuple[float, str]:
        claim_norm = normalize_arabic(claim_text)
        score = float(self.detector.score_texts([claim_norm])[0])
        category = self.detector.categorize(np.array([score]))[0]
        return score, str(category)


class LiveModelEnsemble:
    """Loads all four live components once; call `.score(claim, evidence)`
    repeatedly for cheap per-request inference."""

    def __init__(self, include_transformer: bool = True) -> None:
        self.svm = LiveSVM()
        self.bilstm = LiveBiLSTM()
        self.transformer = LiveTransformer() if include_transformer else None
        self.novelty = LiveNoveltyDetector()

    def score(self, claim_text: str, evidence_text: str = "") -> dict:
        claim_norm = normalize_arabic(claim_text)
        evidence_norm = normalize_arabic(evidence_text)

        votes = [self.svm.predict(claim_norm, evidence_norm), self.bilstm.predict(claim_norm, evidence_norm)]
        if self.transformer is not None:
            votes.append(self.transformer.predict(claim_norm, evidence_norm))

        predictions = [v.prediction for v in votes]
        all_agree = len(set(predictions)) == 1

        novelty_score, novelty_category = self.novelty.score(claim_text)

        svm_vote = next(v for v in votes if v.model == "svm")
        primary_conf = svm_vote.raw_confidence  # raw, per Phase 2's calibration finding

        novelty_penalty = {"FAMILIAR": 0.0, "MODERATELY NOVEL": 0.10, "HIGHLY NOVEL": 0.30}.get(novelty_category, 0.0)
        disagreement_penalty = 0.0 if all_agree else 0.50
        trust_score = primary_conf - disagreement_penalty - novelty_penalty

        return {
            "votes": [{"model": v.model, "prediction": v.prediction, "confidence": v.confidence,
                       "raw_confidence": v.raw_confidence} for v in votes],
            "all_agree": all_agree,
            "svm_prediction": svm_vote.prediction,
            "svm_confidence": primary_conf,
            "novelty_score": novelty_score,
            "novelty_category": novelty_category,
            "trust_score": trust_score,
        }


if __name__ == "__main__":
    ensemble = LiveModelEnsemble()
    result = ensemble.score(
        "وصول ميسي والمنتخب الأرجنتيني إلى قطر",
        "صحيفة الخليج الخليج الرياضي",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
