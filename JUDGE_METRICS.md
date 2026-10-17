# JUDGE_METRICS

A 30-second scan of AMAN Sentinel's validated numbers. Every value below is copied verbatim from `phase5/VALIDATED_METRICS.md` — that file is the single source of truth; nothing here has been rounded or retyped differently.

---

## Classification (SVM, clean leakage-checked split)

| Metric | Value |
|---|---|
| Accuracy | **86.45%** |
| Macro F1 | **86.40%** |
| MCC | **0.7281** |

---

## Model Agreement

| Condition | Share of test rows | Accuracy |
|---|---|---|
| All 3 models agree (SVM + BiLSTM + MARBERT-probe) | 67% | **92.6%** |
| 2-of-3 agreement (split vote) | 33% | **64.9%** |

---

## Router (Selective Prediction)

At the selected **80% target coverage** operating point:

| Metric | Value |
|---|---|
| Actual coverage | **76.90%** |
| Selective accuracy (trusted/accepted subset only) | **92.65%** |
| Selective risk | 7.35% |

> Correct reading: *at the selected 80% routing target, the trusted subset achieved 92.65% selective accuracy at 76.90% coverage.* This is not an overall system accuracy figure — it describes only the ~77% of inputs the router accepted without escalation.

---

## Retrieval

| Method | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.076 | 0.140 | 0.0618 |
| Dense (multilingual SBERT) | 0.092 | 0.144 | 0.0596 |
| **Hybrid-RRF (k=60) — selected** | **0.108** | **0.202** | **0.0787** |

**Disclosure:** these numbers are measured only on the subset of the test corpus where reusable in-corpus gold evidence actually exists (the VERA-ARAB-sourced rows). The other source corpora (AraStance, ArCOV19) have **zero** reusable gold evidence in the retrieval corpus by construction, so retrieval quality for those is **not measured and not claimed**.

---

## Tests

**63/63 tests passing** (44 from Phase 1-3 unittest suites + 19 from Phase 4 pytest suites), plus 3 gated slow end-to-end tests independently run by hand and confirmed passing.

---

## AMD Status

**AMD production deployment prepared; final MI300X validation pending cloud-credit activation.**

---

## NVIDIA GPU End-to-End Validation

**Completed.** AMAN Sentinel has been validated end-to-end with Qwen2.5-7B-Instruct on a university NVIDIA GPU environment (NVIDIA GB10, CUDA 13.0, via Ollama — `ollama ps` confirmed 100% GPU utilization). Two real claims were run through the full pipeline (FastAPI → classifiers → novelty detector → trust router → retrieval → deep verification → LLM → safeguards); both correctly returned INSUFFICIENT_EVIDENCE rather than fabricating a verdict. This is execution validation, not a labeled accuracy benchmark — no accuracy percentage is claimed for it. The AMD deployment path is prepared for AMD Instinct MI300X using ROCm and vLLM and remains pending final AMD execution.

Full detail: `phase5/reports/UNIVERSITY_GPU_VALIDATION.md`.

---

*Footnote: a separate, limited development-model deep-verification experiment using `qwen2.5:1.5b` measured 60% verdict accuracy on a 25-example held-out sample where the model committed to a verdict. This is a small-sample development-model result, not a system-level accuracy claim, and is intentionally excluded from the table above.*
