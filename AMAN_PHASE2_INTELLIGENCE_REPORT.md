# AMAN Phase 2 Intelligence Report

**Branch:** `aman-hackathon` (continues directly from the Phase 1 commit; Phase 1 artifacts, clean splits, and reports are untouched)
**Scope:** uncertainty-aware multi-model verification layer — transformer baseline, calibration, disagreement analysis, novelty/OOD detection, decision router, ensemble experiments. SOURCE-DOMAIN ONLY throughout; UAE-MVD remains unavailable and was not fabricated or substituted.
**Method note:** heavy, isolated workstreams (transformer fine-tuning, novelty/OOD construction) were delegated to independent subagents per the session's token-efficiency instructions; their outputs were verified against disk artifacts before being incorporated here, and one design bug surfaced during integration was found and fixed (see "Model Disagreement Analysis" caveat is not affected; see Router section).

---

# Phase 2 Executive Summary

Three findings decide the production architecture:

1. **SVM remains the strongest individual model.** Neither BiLSTM nor a transformer beat it on this clean split.
2. **The transformer did not improve on the baselines — and the reason is load-bearing, not a dead end.** CPU-only hardware made full MARBERT fine-tuning measurably intractable (a timed probe projected 32,083 seconds for 3 epochs against a ~3,000-second remaining budget), so the experiment ran as a frozen-backbone linear probe instead. That variant scored below both existing baselines (Macro F1 0.7396). This is reported as what it is — a result of a compute-constrained substitute, not evidence that MARBERT itself can't work here.
3. **Model disagreement is a strong, real signal even though the weaker models lose on their own.** When all three models agree (67% of test rows), majority-vote accuracy is 92.6%; when they split 2-1, it falls to 64.9%. This 27.7-point swing is larger than any single calibration or ensemble gain found elsewhere in this phase, and it is the basis for the router's design.

Recommended Phase 3 starting point: **SVM as the primary predictor, kept honest by a novelty detector and a confidence-and-disagreement router; BiLSTM and the frozen-probe transformer retained only as auxiliary disagreement signals, not as production predictors in their own right.** Ensembling by vote was tested and explicitly rejected — it underperforms SVM alone.

---

# SVM Baseline

Unchanged from Phase 1 (clean, leakage-free source-domain split; see `AMAN_PHASE1_RESTORATION_REPORT.md`):

| Metric | Value |
|---|---:|
| Accuracy | 0.8645 |
| Macro F1 | **0.8640** |
| MCC | 0.7281 |
| Macro Precision | 0.8645 |
| Macro Recall | 0.8636 |

Confusion matrix (test, n=4,000): `[[1605, 292], [250, 1853]]` (rows=true, cols=pred, order `[Real, Fake]`).

# BiLSTM Baseline

Unchanged from Phase 1:

| Metric | Value |
|---|---:|
| Accuracy | 0.8003 |
| Macro F1 | **0.7989** |
| MCC | 0.5993 |

# Transformer Results

**Model:** `UBC-NLP/MARBERTv2` (MARBERT) — the brief's preferred choice, and it was successfully obtained (local HF cache or live download succeeded; no fallback to AraBERT was needed).

**Strategy actually used: frozen-backbone linear probe, not full fine-tuning.** A timed probe on the real model and real training data measured 12.22 seconds per training batch (batch size 16). Extrapolated: one full epoch (875 batches) ≈ 10,694 seconds; 3 epochs ≈ 32,083 seconds — far beyond the practical CPU-only budget for this experiment (`torch 2.12.0+cpu`, no CUDA anywhere on this machine). Full fine-tuning was therefore judged intractable, and the backbone was frozen entirely: mean-pooled (attention-mask-weighted) last-hidden-state embeddings were cached once per split, and only a linear classification head was trained on those fixed features. Train/validation/test discipline was preserved exactly (head fit on TRAIN only; best of 21 epochs — epoch 16 — selected by VALIDATION macro F1; TEST touched exactly once for the numbers below).

| Metric | Value |
|---|---:|
| Accuracy | 0.7410 |
| Macro Precision | 0.7407 |
| Macro Recall | 0.7392 |
| Macro F1 | **0.7396** |
| MCC | 0.4799 |

Per-class: Real — precision 0.7377, recall 0.7043, F1 0.7206 (n=1,897). Fake — precision 0.7437, recall 0.7741, F1 0.7586 (n=2,103).

Confusion matrix: `[[1336, 561], [475, 1628]]`.

**This does not meet the bar of "meaningful improvement" — reported honestly, as instructed.** It is also not a fair test of MARBERT's ceiling: a frozen linear probe cannot adapt the model's internal representations to this task at all, unlike the historical FAAV project's own fully fine-tuned AraBERT/MARBERT results (0.90/0.88 macro F1 on comparable but leaky splits). Full fine-tuning of MARBERT on this clean split remains untested and is the clearest "redo with more compute" item for a future phase.

Artifacts: `artifacts/transformers/` (checkpoint, config with full probe/strategy justification and training history, metrics, validation/test prediction CSVs). Full detail in `artifacts/transformers/README.md`.

# Model Comparison

| Model | Accuracy | Macro F1 | MCC | Strategy |
|---|---:|---:|---:|---|
| **SVM** | **0.8645** | **0.8640** | **0.7281** | TF-IDF word+char → LinearSVC, validation-tuned |
| BiLSTM | 0.8003 | 0.7989 | 0.5993 | 1-layer BiLSTM, 3 epochs, early-stopped |
| MARBERT (frozen probe) | 0.7410 | 0.7396 | 0.4799 | Frozen backbone + linear head (full fine-tune infeasible on CPU) |

SVM wins outright. The ranking SVM > BiLSTM > transformer-linear-probe is consistent with how much of each model's capacity was actually allowed to adapt to this task: SVM's features are tuned directly on the data; BiLSTM trains its full (small) architecture; the transformer's backbone was frozen and only a linear head adapted.

# Confidence Calibration

Fit on VALIDATION only in every case; ECE/Brier reported on TEST only (never on validation).

| Model | Method | ECE before → after | Brier before → after | Verdict |
|---|---|---|---|---|
| SVM | Platt/sigmoid scaling | 0.0360 → **0.0959** | 0.1027 → **0.1151** | **Calibration made it worse — use raw confidence in production** |
| BiLSTM | Temperature scaling (T=1.062) | 0.0308 → 0.0244 | 0.1416 → 0.1412 | Small genuine improvement |
| Transformer | Temperature scaling (recovered from log-prob difference) | 0.0680 → **0.0168** | 0.1819 → 0.1770 | Large genuine improvement |

**SVM's negative result is reported as found, not hidden.** `softmax(decision_function)` was already reasonably calibrated on this split (ECE 3.6%); Platt scaling, fit on only 2,000 validation rows, measurably overfit and produced worse calibration on held-out test (ECE 9.6%, Brier 0.115). The router (below) therefore uses SVM's **raw** confidence, not its Platt-scaled version, while using the **calibrated** confidence for BiLSTM and the transformer, where calibration genuinely helped. For the transformer, whose raw softmax has no `predict_proba`-style guarantee either, temperature scaling was derived exactly (not approximated) via `z = log(p_real) − log(p_fake)`, the only quantity a 2-class softmax is sensitive to.

Full sweep: `phase2/reports/calibration_results.json`. Calibrated per-row predictions: `phase2/reports/{model}_predictions_{validation,test}_calibrated.csv`.

# Model Disagreement Analysis

Computed on the TEST split (n=4,000), all pairwise combinations plus a 3-way majority breakdown:

| Pair | Disagreement rate | P(error \| agree) | P(error \| disagree), per model |
|---|---:|---:|---|
| SVM vs BiLSTM | 17.9% | 9.5% | SVM 32.1% / BiLSTM 67.9% |
| SVM vs Transformer | 25.5% | 9.4% | SVM 25.8% / Transformer 74.2% |
| BiLSTM vs Transformer | 22.6% | 15.0% | BiLSTM 36.9% / Transformer 63.1% |

**3-way majority vote, conditioned on agreement level:**

| Agreement | n | Majority-vote accuracy |
|---|---:|---:|
| 3 of 3 agree | 2,680 (67.0%) | **92.6%** |
| 2 of 3 agree | 1,320 (33.0%) | **64.9%** |

This is the single most useful signal found in Phase 2. Unanimous agreement marks a row as substantially more trustworthy than any one model's confidence alone would suggest; a split vote marks it as close to a coin flip. Note the "P(error | disagree) = 100% for 'either'" figure in the raw JSON output is a tautology for any 2-model binary pairwise comparison (if two models disagree on a binary task, exactly one must be wrong by definition) — the per-model breakdown is the informative part, and it consistently shows SVM is the more trustworthy member of every pair it's in.

Full data: `phase2/reports/disagreement_results.json`.

# Novelty / OOD Method

Built and validated by an isolated subagent workstream; independently verified against disk artifacts before use here.

- **Method:** kNN density — mean cosine distance to the k=10 nearest neighbors in a TRAIN-only reference embedding set. Deliberately the simplest defensible method; Mahalanobis/Isolation Forest were not added, per the brief's explicit instruction not to add methods for complexity's sake.
- **Embedding model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — reused from this repo's own existing usage in `pipeline/hybrid_retrieval.py`, not a new dependency.
- **Text representation:** `claim_norm` only (not evidence), because evidence in this restored dataset is retrieval-backfilled for many rows and would measure retrieval-corpus coverage rather than genuine claim novelty.
- **Reference fit:** TRAIN split (13,999 rows) only.
- **Category thresholds:** percentiles of the VALIDATION split's *unsupervised* novelty-score distribution — P80 = 0.2756 (FAMILIAR cutoff), P95 = 0.3713 (HIGHLY NOVEL cutoff). Labels were never consulted for this.
- **Output:** `novelty_score` (continuous) and `novelty_category` ∈ {FAMILIAR, MODERATELY NOVEL, HIGHLY NOVEL} for every validation/test row.

TEST split category distribution (in-distribution sanity check): FAMILIAR 80.77%, MODERATELY NOVEL 14.77%, HIGHLY NOVEL 4.45% — close to the 80/15/5 split implied by the validation-derived percentile cutoffs, confirming the thresholds generalize sensibly to unseen in-distribution rows.

**Explicit framing, preserved from the detector's own design:** novelty/OOD is an unfamiliarity signal, never a misinformation signal. `HIGHLY NOVEL` does not mean `Fake`; `FAMILIAR` does not mean `Real`.

Artifacts: `artifacts/novelty/` (reference embeddings, config with full threshold derivation, validation/test scores). Code: `phase2/novelty_detector.py`.

# OOD Evaluation Limitations

No genuine labeled OOD misinformation benchmark exists anywhere in this repository — confirmed by a direct search (documented in full in `phase2/reports/ood_corpus_search.md`), not assumed. `Dataset files/AraNEWS/` was confirmed, by reading the actual restoration scripts line-by-line, to be unused anywhere in building the clean split; `Dataset files/VERA-ARAB/` supplementary files `s002`–`s008` were inspected directly and are codebooks, setup notes, or analysis scripts — not usable text corpora.

AraNEWS was used for one honest, bounded sanity check: a 500-row sample (seed 42) of its `Sentence` text — a **different task** (original vs. machine-generated text, no misinformation labels at all) — was scored with the exact same fitted novelty detector.

| Category | Clean TEST (n=4,000) | AraNEWS sample (n=500) |
|---|---:|---:|
| FAMILIAR | 80.77% | 17.0% |
| MODERATELY NOVEL | 14.77% | 27.4% |
| HIGHLY NOVEL | 4.45% | 55.6% |

The detector's scores shift sharply and in the expected direction under this real text-distribution shift (mean novelty_score roughly doubles, from 0.17 to 0.38). **This validates only that the detector responds sensibly to distribution shift. It is not, and is not presented as, a measurement of OOD misinformation-detection accuracy.**

**External cross-domain OOD classifier performance (e.g., on UAE-style data) is explicitly NOT YET ESTABLISHED.** No claim about it is made anywhere in this report.

# Ensemble Experiments

Tested exactly the two methods specified: majority voting and calibrated-confidence-weighted voting, over all three models (SVM, BiLSTM, transformer) on the TEST split.

| Approach | Accuracy | Macro F1 | MCC |
|---|---:|---:|---:|
| **SVM alone (best individual)** | **0.8645** | **0.8640** | **0.7281** |
| Majority vote | 0.8347 | 0.8336 | 0.6688 |
| Confidence-weighted vote | 0.8347 | 0.8336 | 0.6688 |

**Neither ensemble beats SVM alone.** Both are dragged down by the two weaker models' errors, which outnumber the cases where they correct an SVM mistake. **Conclusion: do not use a voting ensemble in production.** SVM's standalone accuracy is the one to beat, and nothing tested here beats it. (Both ensemble strategies land on identical numbers in this run — with 3 models, the weighted vote only diverges from a plain majority when weights are closely split in a way that didn't occur often enough on this test set to change any outcome.)

Full data: `phase2/reports/ensemble_results.json`.

# Selective Prediction / Routing Results

**Design note — a bug was found and fixed during this phase.** The router's first implementation used model-agreement and non-high-novelty as *hard* pass/fail filters before ranking by confidence. That construction is mathematically broken: the hard filters alone reject ~36% of rows unconditionally, which makes 70/80/90% coverage targets unreachable no matter where the confidence threshold is set (the first run measured actual coverage of 40–48% against those targets — confirmed, not assumed, before the fix). The corrected design ranks every row on one continuous **trust score** — calibrated primary-model confidence, minus a fixed penalty for disagreement (0.50) and a graduated penalty for novelty (0.10 moderate / 0.30 high) — so any coverage target from 0–100% is reachable by moving one threshold, matching the standard selective-prediction construction the original FAAV project's own `improvements/selective_evaluation.py` already used. Confidence remains the dominant term, consistent with that project's own finding that hand-built multi-signal gates lose to plain confidence in 22 of 24 historical comparisons — disagreement and novelty demote risky rows rather than compete with confidence on equal footing.

Threshold for each coverage target is the `(1 − target)` quantile of the VALIDATION trust-score distribution only; no validation or test labels are used anywhere in threshold selection.

| Target coverage | Actual coverage | Selective accuracy | Selective risk | n accepted (of 4,000) |
|---:|---:|---:|---:|---:|
| 100% | 100.0% | 86.45% | 13.55% | 4,000 |
| 90% | 87.35% | 90.21% | 9.79% | 3,494 |
| 80% | 76.90% | 92.65% | 7.35% | 3,076 |
| 70% | 72.70% | 92.71% | 7.29% | 2,908 |

The curve is monotonic and sane: accuracy rises and risk falls smoothly as coverage tightens, with no inversions. Actual coverage tracks the targets reasonably closely (not exactly, because the trust-score distribution has clusters around the two penalty levels rather than being smooth) — a known, reported limitation of this first-pass design rather than a hidden one. At the 90% operating point, roughly half the test set that would otherwise be misclassified gets correctly escalated rather than auto-accepted, at the cost of reviewing more rows manually.

Per-row routing decisions (at the 90% operating point): `phase2/reports/router_decisions_test.csv`. Full sweep: `phase2/reports/router_results.json`.

# Recommended Production Models

- **SVM** — primary predictor. Strongest individual model on every metric; cheap; its raw (not Platt-scaled) confidence is the router's dominant signal.
- **Novelty detector (kNN/SBERT)** — production-ready as an auxiliary signal. Internally validated; cheap to run.
- **Confidence-and-disagreement router** — production-ready as an *experimental* first-pass gate. Clear monotonic risk/coverage behavior; known coverage-targeting imprecision stated above.
- **BiLSTM and the frozen-probe transformer — retained, but only as disagreement-signal inputs to the router, not as independent predictors.** Their own standalone accuracy is below SVM's and a vote across all three underperforms SVM alone — but their *disagreement* with SVM is real, measured signal (the 92.6%-vs-64.9% split). Keeping them running purely to feed that signal into the trust score is a defensible, narrow use; using either of their own predictions as the delivered verdict is not.

# Models Rejected From Production and Why

- **Voting ensemble (majority or confidence-weighted)** — measurably worse than SVM alone (0.8336 vs 0.8640 macro F1). Rejected outright, not kept "just in case."
- **BiLSTM as a standalone predictor** — correct, cheap, and real, but strictly weaker than SVM on every metric. No role as the delivered verdict.
- **MARBERT (frozen linear probe) as a standalone predictor** — weakest of the three models tested. Rejected *in this form*. This is explicitly not a verdict on MARBERT itself — full fine-tuning was never tested due to CPU constraints, and that remains open for a future phase with more compute.
- **Platt-scaled SVM confidence** — calibration made SVM's confidence measurably worse on held-out data (ECE 3.6%→9.6%). Use raw SVM confidence instead.
- **The original hard-filter router design** — mathematically incapable of reaching its own stated coverage targets. Replaced with the composite trust-score design before being reported as a result.

# Proposed Phase 3 Architecture

```
Arabic claim (+ evidence, source-domain only)
        │
        ├──────────────► SVM (primary classifier)
        │                        │
        ├──────────────► BiLSTM ─┤  (disagreement signal only,
        │                        │   not delivered as a verdict)
        ├──────────────► MARBERT ┤  (disagreement signal only;
        │   (frozen probe)       │   revisit with full fine-tune
        │                        │   given more compute)
        └──────────────► Novelty detector (kNN/SBERT, claim_norm)
                                 │
                                 ▼
                  Confidence-and-disagreement router
                  trust_score = SVM raw confidence
                                − disagreement penalty
                                − novelty penalty
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
          VERIFIED_PREDICTION      ESCALATE_FOR_DEEP_VERIFICATION
          (SVM verdict delivered)   (routed to a future evidence-
                                      retrieval / LLM-reasoning stage
                                      — not built in Phase 2)
```

Before this becomes a real Phase 3 build: (1) get GPU or a larger CPU time budget to test full MARBERT/AraBERT fine-tuning properly rather than only the frozen-probe fallback; (2) tighten the router's coverage-targeting precision (the trust-score clustering issue above); (3) decide, with the user, whether UAE-domain data will ever become available, since every result in both phases remains source-domain-only.

# Remaining Risks

1. **The transformer result is a compute-constrained substitute, not a ceiling.** Reporting "transformer underperforms" is honest for what was tested, but a reader could wrongly generalize it to "transformers don't help here" — they weren't properly tested.
2. **The router's coverage-targeting is imprecise** (actual vs. target coverage differs by up to ~3 points) due to clustering in the trust-score distribution at the two fixed penalty levels. Functionally monotonic and safe, but not a precision instrument yet.
3. **Disagreement and novelty penalties (0.50 / 0.10 / 0.30) are fixed constants chosen by the main agent for this integration, not learned from validation data the way the confidence threshold itself is.** This is a reasonable first pass but is the most arbitrary part of the router and the best candidate for a validation-driven refinement in Phase 3.
4. **Phase 2 remains entirely source-domain.** Every number in this report — including the novelty detector's internal validation — says nothing about true UAE-domain or other genuine cross-domain behavior. This is Phase 1's limitation, carried forward unchanged, not newly introduced.
5. **SVM's Platt-scaling negative result was fit on only 2,000 validation rows.** It is a real, reproducible finding on this split, but a larger validation set might calibrate more successfully; this was not re-tested with more data because no more leakage-free validation data exists without rebuilding the split.
6. **The frozen MARBERT checkpoint (621 MB) and its calibration are both legitimate but expensive.** If the production router ships with the transformer in the loop purely for its disagreement signal, that is a meaningfully heavier deployment than SVM+BiLSTM+novelty alone for a signal that could potentially be replaced by something cheaper (e.g., a second classical model) — worth evaluating in Phase 3 rather than assumed necessary.

---

**New or modified files this phase:** `phase2/` (12 scripts + tests + 20 report files, all new), `artifacts/transformers/` (checkpoint + metrics + predictions, new; `README.md` updated from Phase 1's placeholder), `artifacts/novelty/` (new, entirely). No file under `hackathon/`, `data/final_studies/`, `results/`, `Results files/`, `artifacts/svm/`, or `artifacts/bilstm/` was modified. Full test suite: **29/29 passing** (7 historical `utils.test_final_decision` + 8 Phase 1 `hackathon.tests.test_phase1` + 14 Phase 2 `phase2.tests.test_phase2`).
