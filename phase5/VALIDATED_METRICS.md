# AMAN Sentinel — Validated Metrics (single source of truth)

**Every number below is measured and already reported in a phase report.** Any document that cites a metric about this project — README, ARCHITECTURE doc, DEMO_SCRIPT, SUBMISSION_COPY, PITCH_DECK_CONTENT, or anything else — must pull from this file verbatim, not retype from memory or round differently. If a number isn't here, it isn't validated — don't cite it.

## Classification (Phase 1/2 — SVM, clean leakage-checked split)

| Metric | Value |
|---|---|
| Accuracy | **86.45%** (0.8645) |
| Macro F1 | **86.40%** (0.8640) |
| MCC | **0.7281** |

Source: `AMAN_PHASE1_RESTORATION_REPORT.md`, `AMAN_PHASE2_INTELLIGENCE_REPORT.md`.

## Model agreement (Phase 2)

| Condition | Share of test rows | Accuracy |
|---|---|---|
| All 3 models agree (SVM + BiLSTM + MARBERT-probe) | 67% | **92.6%** |
| 2-of-3 agreement (split vote) | 33% | **64.9%** |

This 27.7-point swing is *why* disagreement is a routing signal. Source: `AMAN_PHASE2_INTELLIGENCE_REPORT.md`.

## Router / selective prediction (Phase 2)

At the **80% target coverage** operating point:

| Metric | Value |
|---|---|
| Actual coverage | **76.90%** |
| Selective accuracy (on the accepted/trusted subset) | **92.65%** |
| Selective risk | 7.35% (= 1 − 0.9265) |

**Correct framing, always:** *"At the selected 80% routing target, the trusted subset achieved 92.65% selective accuracy at 76.90% coverage."* **Never say** "AMAN is 92.65% accurate" or "92.65% overall accuracy" — that number describes only the ~77% of inputs the router accepted without escalation, not the whole system, and not the escalated ~23%.

Source: `AMAN_PHASE2_INTELLIGENCE_REPORT.md`.

## Retrieval (Phase 3)

| Method | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.076 | 0.140 | 0.0618 |
| Dense (multilingual SBERT) | 0.092 | 0.144 | 0.0596 |
| **Hybrid-RRF (k=60) — selected** | **0.108** | **0.202** | **0.0787** |

**Mandatory disclosure whenever these numbers appear:** these are measured only on the subset of the test corpus where reusable in-corpus gold evidence actually exists — the VERA-ARAB-sourced rows. The other source corpora (AraStance, ArCOV19) have **zero** reusable gold evidence in the retrieval corpus by construction (each claim's evidence is a unique, non-reused article), so retrieval quality for those is **not measured, not claimed**. Source: `AMAN_PHASE3_DEEP_VERIFICATION_REPORT.md`.

## Deep verification (Phase 3) — development-model experiment, NOT "AMAN accuracy"

On a 25-example held-out sample, using the **development LLM** (`qwen2.5:1.5b` via local Ollama — *not* the production target):

- 100% retrieval success, 100% structured-output validity, 100% evidence-citation validity (zero hallucinated citations).
- 60% verdict accuracy on the subset where the model committed to SUPPORTED/CONTRADICTED (9/15; INSUFFICIENT_EVIDENCE rows are abstentions, not scored as wrong).

**Mandatory framing whenever this 60% figure is mentioned:** label it explicitly as *"a limited development-model deep-verification experiment using qwen2.5:1.5b"* — never as "AMAN's accuracy," "AMAN's verification accuracy," or any unqualified system-level claim. It has not been re-measured against the production target (Qwen2.5-7B-Instruct). Source: `AMAN_PHASE3_DEEP_VERIFICATION_REPORT.md`.

## Testing (Phase 4)

**63/63 tests passing** before final AMD deployment validation (44 from Phases 1–3's unittest suites + 19 from Phase 4's pytest suites), plus 3 gated slow end-to-end tests independently run by hand and confirmed passing. Source: `AMAN_PHASE4_PRODUCTIZATION_REPORT.md`.

## AMD status — say this exactly, every time it comes up

**AMD execution has not occurred.** This sandbox has no AMD hardware, ROCm stack, or cloud credentials — confirmed directly (`phase3/reports/amd_investigation.md`). The production path (Qwen2.5-7B-Instruct via vLLM on AMD Instinct MI300X / ROCm) is fully designed, documented, and the code has a ready-to-flip provider switch (`LLM_PROVIDER=vllm`), but it has never connected to real AMD hardware.

**Required exact wording wherever AMD status is stated:**
> "AMD production deployment prepared; final MI300X validation pending cloud-credit activation."

**Never write or imply:** that an AMD run occurred, that latency/throughput numbers exist for AMD hardware, that ROCm was tested, or that ROCm was active it any point. All performance/latency numbers in this project were measured on CPU-only hardware (Intel) or via local Ollama — label them as such if cited at all.
