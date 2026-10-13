# PITCH_DECK_CONTENT

Slide-by-slide content for the AMAN Sentinel pitch deck. Every metric is pulled verbatim from `phase5/VALIDATED_METRICS.md`. Keep bullets as bullets when building slides — this is deliberately terse, not prose.

---

## Slide 1 — AMAN Sentinel

- **Title:** AMAN Sentinel
- **Tagline:** Arabic-first AI verification that knows when to classify, when to retrieve evidence, and when to abstain.
- AMD-sponsored hackathon submission

**Speaker note:** Open on the tagline — the whole pitch is "it knows when it doesn't know."

---

## Slide 2 — The Problem

- Arabic-language misinformation is widespread and fast-moving
- Verification tooling is built mostly for English — Arabic is underserved
- The "obvious" fix — ask an LLM — is slow, expensive, and often ungrounded
- LLMs answer fluently from training-time associations, not checkable evidence
- They rarely admit when they don't actually know

**Speaker note:** Set up why this is a real, specific gap — not just "misinformation is bad."

---

## Slide 3 — Why Existing Approaches Fail

- Naive "always ask an LLM" pipelines pay full LLM cost on every single input
- No distinction between "easy, obvious" claims and genuinely hard ones
- No evidence grounding — answers are plausible, not verifiable
- No abstention — a confident-sounding wrong answer looks the same as a right one
- Hard to audit: unclear why the model said what it said

**Speaker note:** Contrast briefly — this is the setup for why AMAN routes instead of brute-forcing.

---

## Slide 4 — Our Solution

- **One sentence:** AMAN Sentinel routes each claim through a cheap trust-scoring step first, and only escalates to evidence retrieval and grounded LLM reasoning when the input is genuinely uncertain — abstaining rather than guessing when evidence doesn't settle it.

**Speaker note:** This is the single slide to remember if nothing else lands.

---

## Slide 5 — How AMAN Works

- **Classify:** validated SVM makes the primary call
- **Measure uncertainty:** BiLSTM + frozen MARBERT probe (disagreement signals) + SBERT/kNN novelty detector
- **Trust score → route:**
  - High trust → fast result, no LLM call
  - Low trust → escalate
- **Escalation path:** BM25 + dense + Hybrid-RRF retrieval → LLM reasons strictly from retrieved evidence → cites real passage IDs or returns INSUFFICIENT_EVIDENCE

**Speaker note:** Walk this left-to-right as the pipeline diagram — classify, score, route, then one of two outcomes.

---

## Slide 6 — Technical Validation

**Classification (SVM)**

| Metric | Value |
|---|---|
| Accuracy | 86.45% |
| Macro F1 | 86.40% |
| MCC | 0.7281 |

**Model agreement**

| Condition | Accuracy |
|---|---|
| All 3 models agree (67% of rows) | 92.6% |
| 2-of-3 split vote (33% of rows) | 64.9% |

**Router** — at the selected 80% target, trusted subset: **92.65% selective accuracy at 76.90% coverage**

**Retrieval (Hybrid-RRF, selected)** — Recall@5 0.108, Recall@10 0.202, MRR 0.0787 (measured on VERA-ARAB subset only — see appendix/notes)

**Testing** — 63/63 tests passing

**Speaker note:** Read the router line exactly as written — it describes the trusted ~77% subset, not overall system accuracy.

---

## Slide 7 — AMD Architecture

- Production target: Qwen2.5-7B-Instruct served via vLLM on AMD Instinct MI300X, through ROCm
- Provider abstraction already built with a ready-to-flip switch to the AMD-hosted model
- Development today runs on a small local model (`qwen2.5:1.5b` via Ollama, CPU-only) so the full pipeline could be built and demoed without AMD hardware access
- **AMD production deployment prepared; final MI300X validation pending cloud-credit activation.**

**Speaker note:** Say this status line exactly as written — do not imply a run happened.

---

## Slide 8 — Demo / Impact

- Three real pipeline paths, demonstrated live (not hard-coded):
  - **High-confidence, no escalation** — all 3 models agree, fast result, no LLM call
  - **Escalation with evidence** — models disagree, retrieval + LLM produce an evidence-grounded verdict with citations
  - **Insufficient evidence** — escalates, retrieval runs, but the citation safeguard rejects an unsupported verdict and the system abstains instead of guessing
- Each path is driven by real held-out test-split examples, run live through the actual pipeline

**Speaker note:** The third path is the one worth dwelling on — it's the safeguard actively working, not a missing feature.

---

## Slide 9 — Future Vision

- Run real AMD MI300X / ROCm / vLLM validation once cloud credits activate
- Re-evaluate deep verification against the production LLM target (Qwen2.5-7B-Instruct), not the development stand-in
- Expand retrieval ground truth beyond the current VERA-ARAB subset to cover all source corpora
- Extend beyond text to additional input modalities, per the architecture's documented extensibility
- Broaden language coverage beyond Arabic/English/mixed-language text

**Speaker note:** Keep this technical and product-focused — no market size, revenue, or user-count claims; none have been measured.
