# Transformers — not attempted in Phase 1

Per the Phase 1 priority order (SVM first, BiLSTM second if resources are
available, transformers only after the classical baseline works), AraBERT
and MARBERT fine-tuning was **not attempted** in this phase.

Reasons, stated plainly rather than silently skipped:

1. **Priority ordering.** The brief explicitly gates transformer retraining
   on the classical baseline succeeding first, and Phase 1's scope is
   "restore + validate + create a clean reproducible baseline" — the SVM
   baseline is that baseline.
2. **No local checkpoint for the current label space.** The only
   transformer checkpoint on disk is `Results files/arabert_finetuned/`
   (516 MB, legacy 15-claim-era), and the audit found its `config.json`
   carries no `id2label` — its label order cannot be recovered from the
   file alone. It cannot be reused as a drop-in verifier without
   independently re-establishing what each output index means.
3. **No cached base-model weights confirmed.** `final_studies_neural.py`
   loads `aubmindlab/bert-base-arabertv02` and `UBC-NLP/MARBERTv2` with
   `local_files_only=True`, which assumes a populated local HuggingFace
   cache. Whether one exists on this machine was not verified in Phase 1,
   and fine-tuning either model from a cold download was judged out of
   scope for a "clean baseline" restoration pass on CPU-only hardware
   (`torch 2.12.0+cpu` per the original audit).

This directory exists so the artifact layout matches the brief
(`artifacts/{svm,bilstm,transformers}/`) and is **empty on purpose**. No
checkpoint, config, or metric file has been fabricated here.

**Phase 2 decision point:** confirm HF cache / network access, decide
whether to fine-tune AraBERT on the clean split (recommended — it is the
project's strongest historical verifier) or defer further.
