# AMAN Sentinel — Results (Verified Metrics Only)

Every number below traces to a raw artifact (see `phase5/VALIDATED_METRICS.md` for exact sources). Nothing here is estimated or rounded up.

## Classification (SVM, clean leakage-checked split, n=4,000 test rows)

| Metric | Value |
|---|---:|
| Accuracy | 86.45% |
| Macro F1 | 86.40% |
| MCC | 0.7281 |

## Model agreement

| Condition | Share of test rows | Accuracy |
|---|---:|---:|
| All 3 classifiers agree | 67% | 92.6% |
| 2-of-3 agreement (split vote) | 33% | 64.9% |

## Selective routing — full coverage/accuracy curve

| Target | Actual coverage | Selective accuracy (accepted subset) |
|---|---:|---:|
| 100% | 100.00% | 86.45% |
| 90% | 87.35% | 90.21% |
| **80% (selected)** | **76.90%** | **92.65%** |
| 70% | 72.70% | 92.71% |

> **Correct statement:** "At the selected operating point, the router accepts 76.90% of claims with 92.65% selective accuracy on the accepted subset." **Never:** "AMAN Sentinel's accuracy is 92.65%." That figure describes only the accepted subset, not the whole system and not the escalated remainder.

## Retrieval (n=500 queries, VERA-ARAB-evaluable subset)

| Method | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| BM25 | 0.076 | 0.140 | 0.0618 |
| Dense (multilingual SBERT) | 0.092 | 0.144 | 0.0596 |
| Hybrid, weighted fusion | 0.092 | 0.176 | 0.0723 |
| **Hybrid-RRF, k=60 (selected)** | **0.108** | **0.202** | **0.0787** |

Caveat: reusable gold evidence exists only for VERA-ARAB-sourced queries; AraStance and ArCOV19 retrieval quality is not measured by these figures.

## Deep verification — exploratory comparison (n=25, identical sample for both models)

| Model | Attempted | Committed | Abstained | Correct | Committed accuracy (95% Wilson CI) |
|---|---:|---:|---:|---:|---|
| qwen2.5:1.5b | 25 | 15 | 10 (40%) | 9 | 60.0% (35.7–80.2%) |
| qwen2.5:7b-instruct | 25 | 8 | 17 (68%) | 3 | 37.5% (13.7–69.4%) |

Fisher's exact test: p = 0.40, odds ratio = 2.5. **Neither model is shown to outperform the other** — the sample is too small to distinguish them statistically. The 7B model's higher abstention rate is reported as an observed behavior, not interpreted as evidence of being "more cautious" in any validated sense. This experiment is exploratory and underpowered by design disclosure, not by omission.

## Testing

**63/63 tests passing** (44 unittest + 19 pytest), plus 3 gated slow end-to-end tests independently confirmed passing by hand. Re-confirmed as part of final submission cleanup.

## Infrastructure validation

- **NVIDIA GPU (development): validated end-to-end.** Full pipeline run on a university NVIDIA GB10 GPU server with Qwen2.5-7B-Instruct via Ollama. See `phase5/reports/UNIVERSITY_GPU_VALIDATION.md`.
- **AMD MI300X (production target): pending.** AMD Developer Cloud was configured and access requested; execution has not yet occurred. No AMD/ROCm performance numbers exist and none are claimed.

## What is deliberately not claimed here

- No overall "AMAN Sentinel accuracy" figure — the system's output quality depends on which path a claim takes, and that is what the coverage/selective-accuracy table above actually reports.
- No claim that the 7B model verifies more accurately than the 1.5B model, or vice versa.
- No AMD execution, AMD performance, or ROCm benchmark of any kind.
- No OOD-detection accuracy for the novelty detector — only a distribution-shift sanity check exists (AraNEWS: 17%/27%/56% Familiar/Moderate/High-novelty vs. 80.8%/14.8%/4.5% in-distribution), not a labeled benchmark.
