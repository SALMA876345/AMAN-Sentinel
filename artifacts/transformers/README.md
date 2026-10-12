# Transformers — attempted in Phase 2 (MARBERT, frozen linear probe)

Phase 1 deferred this with an explicit decision point: confirm HF cache /
network access, then decide whether to fine-tune. Phase 2 did exactly
that.

## What was used and why

**Model:** `UBC-NLP/MARBERTv2` (MARBERT) — the brief's preferred choice,
and it was successfully obtained (tokenizer + weights resolved from the
local HuggingFace cache or a live download).

**Strategy: frozen-backbone linear probe, not full fine-tuning.** A timed
probe on the real model and real clean-split data measured **12.22
seconds per training batch** (batch size 16). Extrapolated to a full
epoch (875 batches) that is **10,694 seconds**, and **32,083 seconds for
3 epochs** — far beyond the realistic CPU-only time budget for this
experiment. Full fine-tuning was therefore judged intractable on this
hardware (`torch 2.12.0+cpu`, no CUDA) and the run fell back to: freeze
every MARBERT parameter, compute mean-pooled (attention-mask-weighted)
last-hidden-state embeddings once per split, and train only a linear
classification head on those cached features. This is a materially
weaker setup than full fine-tuning — it cannot adapt MARBERT's internal
representations to this task at all — and the result below should be
read with that firmly in mind, not compared as if it were a fully
fine-tuned MARBERT.

## Result — reported honestly, not improved over the existing baselines

| Model | Macro F1 | Accuracy | MCC |
|---|---:|---:|---:|
| SVM (Phase 1, clean split) | **0.8640** | 0.8645 | 0.7281 |
| BiLSTM (Phase 1, clean split) | 0.7989 | 0.8003 | 0.5993 |
| MARBERT frozen linear probe (Phase 2) | 0.7396 | 0.7410 | 0.4799 |

Confusion matrix (test, n=4,000, labels `[Real, Fake]`):

```
          pred_Real  pred_Fake
true_Real      1336        561
true_Fake       475       1628
```

**The frozen-probe MARBERT does not improve on either existing baseline —
it is the weakest of the three.** This is the expected consequence of
freezing the backbone rather than fine-tuning it: a linear head over
generic pooled embeddings has far less task-specific signal to work with
than either a tuned TF-IDF+SVM decision boundary or a (admittedly small)
trained BiLSTM. The honest conclusion is that MARBERT was successfully
obtained and evaluated exactly as the brief required, and under the
compute constraints actually available, it did not provide a meaningful
improvement — full fine-tuning remains untested and would need either
more CPU time budget, a GPU, or a reduced-epoch/frozen-middle-layers
compromise not attempted here.

## Confidence values

`confidence` / `prob_real` / `prob_fake` in the prediction CSVs below are
**raw softmax output from the trained linear head** — not temperature-
scaled. Phase 2's calibration step (`phase2/calibration.py`) can be
extended to this model's logits the same way it was applied to SVM and
BiLSTM.

## Files in this directory

- `marbert_clean_binary/` — the frozen MARBERT backbone + tokenizer
  (`model.safetensors`, 621 MB) plus the trained linear head
  (`linear_probe_head.pt`, 8 KB) and its config.
- `config.json` — model choice reasoning, the timed probe that justified
  the linear-probe fallback, all hyperparameters, full per-epoch training
  history (21 epochs, best at epoch 16 by validation macro F1), device,
  wall-clock (1,805 seconds total).
- `clean_evaluation_metrics.json` — the test-set metrics above in full
  (precision/recall/F1 per class, confusion matrix, MCC).
- `clean_validation_predictions.csv`, `clean_test_predictions.csv` —
  per-row `sample_id, true_label, prediction, confidence, prob_real,
  prob_fake`.

## How to reload for inference

```python
import torch
from transformers import AutoModel, AutoTokenizer

checkpoint = "artifacts/transformers/marbert_clean_binary"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
backbone = AutoModel.from_pretrained(checkpoint)
head_state = torch.load(f"{checkpoint}/linear_probe_head.pt", map_location="cpu")
# see artifacts/transformers/marbert_clean_binary/linear_probe_head_config.json
# for the head's input/output dimensions and the mean-pooling convention
# used to go from backbone output -> head input.
```

## Train / validation / test discipline

Fit (the linear head) on TRAIN only. Model selection (best epoch) and
early stopping on VALIDATION macro F1 only. TEST touched exactly once,
after the head was already selected, for the metrics reported above.
