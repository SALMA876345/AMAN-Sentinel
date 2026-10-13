# AMAN Sentinel — Architecture

This document traces the real, running system end to end: every box below names the actual file/function/artifact behind it, so a technical judge can open the cited file and verify the claim directly. Nothing here was redesigned to write this document — it describes `phase3/aman_pipeline.py` and what it calls, exactly as it exists on `aman-hackathon` today.

For headline numbers, see the table in `README.md` or the single source of truth, `phase5/VALIDATED_METRICS.md`. This document does not repeat metrics beyond what's needed to cite the right threshold/operating point for each box.

## Two categories you will see throughout this document

Every component below is tagged with one of:

- **[VALIDATED LOCALLY]** — exercised, measured, and tested on this machine (CPU-only, Intel), with numbers in `phase5/VALIDATED_METRICS.md`.
- **[AMD PRODUCTION TARGET — PENDING FINAL RUN]** — designed, coded, and ready to run, but never executed against real AMD hardware. No ROCm stack or AMD cloud credentials exist in this development environment (`phase3/reports/amd_investigation.md`). **AMD production deployment prepared; final MI300X validation pending cloud-credit activation.**

Do not read the two categories as interchangeable. A box in the second category has code but zero execution evidence on AMD silicon.

## Pipeline overview

```
 INPUT (Arabic / English / mixed-language text, + optional evidence_text)
        │
        ▼
 PREPROCESSING                                           [VALIDATED LOCALLY]
   utils/faav_common.normalize_arabic()
   - strips URLs, tatweel (ـ), diacritics
   - unifies alef variants -> ا, alef maqsura (ى) -> ي, ta marbuta (ة) -> ه
   - collapses non-word chars / whitespace
   Applied to claim_text and evidence_text independently
   (phase3/live_inference.py: LiveModelEnsemble.score())
        │
        ▼
 INTELLIGENCE LAYER (all four components run on every request)  [VALIDATED LOCALLY]
   phase3/live_inference.py: LiveModelEnsemble
   ┌───────────────────────────────────────────────────────────┐
   │ SVM (primary classifier)        -- LiveSVM                │
   │   artifacts/svm/{vectorizer,model}.joblib (Phase 1)        │
   │   raw softmax confidence used as-is (Phase 2 found Platt   │
   │   scaling makes it LESS calibrated on held-out data)       │
   │                                                             │
   │ BiLSTM (disagreement signal)    -- LiveBiLSTM              │
   │   artifacts/bilstm/{model.pt,vocab.json,config.json}       │
   │   confidence calibrated via temperature 1.0624             │
   │   (phase2/reports/calibration_results.json)                │
   │                                                             │
   │ MARBERT-probe (disagreement signal) -- LiveTransformer     │
   │   artifacts/transformers/marbert_clean_binary/ + frozen    │
   │   backbone + linear_probe_head.pt (Phase 2: full            │
   │   fine-tuning was CPU-infeasible, so only a linear head     │
   │   is trained on top of frozen MARBERT embeddings)           │
   │                                                             │
   │ Novelty / OOD detector (SBERT + kNN) -- LiveNoveltyDetector│
   │   phase2/novelty_detector.py, loaded via                   │
   │   load_fitted_detector()                                    │
   │   - embeds claim_norm ONLY (not evidence) with              │
   │     sentence-transformers/paraphrase-multilingual-          │
   │     MiniLM-L12-v2                                            │
   │   - score = mean cosine distance to k=10 nearest neighbors  │
   │     in the TRAIN claim-embedding reference set               │
   │   - category thresholds (FAMILIAR / MODERATELY NOVEL /       │
   │     HIGHLY NOVEL) = P80 / P95 of VALIDATION's novelty-score  │
   │     distribution, persisted in artifacts/novelty/config.json│
   │   NOTE: novelty is NOT a misinformation signal — a novel     │
   │   claim could be true or false; it only flags "the           │
   │   classifier has little training-distribution basis to       │
   │   judge this input," which is why it feeds the router below  │
   │   as a trust penalty, never as a verdict.                     │
   └───────────────────────────────────────────────────────────┘
        │
        ▼
 DECISION ROUTER                                          [VALIDATED LOCALLY]
   phase3/router_decision.py: route(trust_score, threshold)
   trust_score = svm_raw_confidence
                 − disagreement_penalty(0.50 if the 3 models don't
                   unanimously agree, else 0.0)
                 − novelty_penalty(0.10 MODERATELY NOVEL /
                   0.30 HIGHLY NOVEL / 0.0 FAMILIAR)
   (composite formula + penalty values: phase2/reports/router_results.json,
   "composite_score")

   threshold = phase3.router_decision.load_default_threshold(), which reads
   phase2/reports/router_results.json → coverage_sweep.target_80pct
   .trust_score_threshold. This is the threshold actually wired into the
   live pipeline — fit on VALIDATION trust-score quantiles only, no labels
   used — and it matches the operating point `phase5/VALIDATED_METRICS.md`
   headlines (92.65% selective accuracy / 76.90% coverage), so the number
   every judge-facing document quotes is the number the demo actually runs
   at. (Through Phase 3/4 the default was the 90%-target row instead —
   also a genuine measurement from the same file, just a different
   operating point. Caught during Phase 5's documentation cross-check and
   switched to 80%-target on 2026-10-13 so code and docs agree; see
   `phase3/router_decision.py`'s docstring and
   `AMAN_PHASE5_SUBMISSION_READINESS_REPORT.md` for the change record.)
        │
   ┌────┴─────────────────────────────────────────┐
   ▼ trust_score ≥ threshold                        ▼ trust_score < threshold
 PATH A: VERIFIED_PREDICTION                   PATH B: ESCALATE_FOR_DEEP_VERIFICATION
 (phase3/router_decision.py                    (same module, same function —
  VERIFIED_PREDICTION constant)                 ESCALATE_FOR_DEEP_VERIFICATION
        │                                        constant)
        ▼                                              │
 Cautious classifier-only result                       ▼
 status = VERIFIED_AUTOMATICALLY                 Evidence retrieval
 classification = SVM's own prediction           phase3/retrieval.py:
 (Real/Fake) — UI never calls this a              retrieve_evidence(claim_text,
 factual "verdict"; the frontend's                top_k=5, method="hybrid_rrf")
 StatusBadge (frontend/src/components/                  │
 StatusBadge.jsx) renders it as                         ▼
 "LIKELY RELIABLE" / "LIKELY MISLEADING"          BM25 (rank_bm25.BM25Okapi)
 under a "Quick classifier check (no             + Dense (sentence-transformers/
 evidence)" sub-label — a visibly                paraphrase-multilingual-MiniLM-
 different UI tier from Path B's results          L12-v2, same model as novelty)
 (phase3/aman_pipeline.py,                        + Hybrid-RRF fusion (k=60,
 verify_content() VERIFIED_PREDICTION             Cormack et al. 2009 default,
 branch)                                          not tuned here)
                                                   -> top-5 "Evidence Pack"
                                                   (phase3/retrieval.py:
                                                   rrf_fuse(), RRF_K=60)
                                                         │
                                                         ▼
                                                   Evidence-grounded LLM
                                                   verification
                                                   phase3/verification_llm.py:
                                                   verify_claim_against_evidence()
                                                   phase3/llm_provider.py:
                                                   get_default_provider()
                                                   - dev/running today:
                                                     qwen2.5:1.5b via local
                                                     Ollama (OllamaProvider)
                                                     [VALIDATED LOCALLY]
                                                   - production target:
                                                     Qwen2.5-7B-Instruct via
                                                     vLLM on AMD Instinct
                                                     MI300X / ROCm
                                                     (VLLMProvider,
                                                     LLM_PROVIDER=vllm)
                                                     [AMD PRODUCTION TARGET —
                                                     PENDING FINAL RUN]
                                                         │
                                                         ▼
                                                   Safeguards (all in
                                                   phase3/verification_llm.py
                                                   _validate_and_coerce())
                                                   1. Schema validation —
                                                      malformed JSON retried
                                                      once, then forced to
                                                      INSUFFICIENT_EVIDENCE
                                                   2. Evidence-ID validation —
                                                      any cited id not
                                                      literally present in the
                                                      retrieved Evidence Pack
                                                      forces
                                                      INSUFFICIENT_EVIDENCE
                                                      (the model can never
                                                      cite evidence it wasn't
                                                      given)
                                                   3. Citation validation /
                                                      unsupported-verdict
                                                      downgrade — SUPPORTED or
                                                      CONTRADICTED with an
                                                      empty evidence_ids list
                                                      is force-downgraded to
                                                      INSUFFICIENT_EVIDENCE
                                                         │
                                                         ▼
                                                   Final evidence_verdict:
                                                   SUPPORTED | CONTRADICTED |
                                                   INSUFFICIENT_EVIDENCE
                                                   -> status
                                                   VERIFIED_WITH_EVIDENCE
                                                   (SUPPORTED/CONTRADICTED) or
                                                   UNCERTAIN
                                                   (INSUFFICIENT_EVIDENCE)
        │                                                     │
        └───────────────────────┬─────────────────────────────┘
                                 ▼
 PRODUCT                                                  [VALIDATED LOCALLY]
   backend/main.py — FastAPI, POST /api/verify, GET /health
     - wraps phase3.aman_pipeline.verify_content() via
       backend/pipeline_wrapper.py (raw-dict -> VerifyResponse schema)
     - startup lifespan warms the SVM/BiLSTM/MARBERT/novelty singletons
       once (~30-90s) so no request pays that cost
     - each request runs verify_content() in a 4-worker thread pool,
       bounded by a 180s asyncio.wait_for timeout (504 on timeout, 503 on
       internal pipeline error, 422 on invalid input)
   frontend/ — React 19 + Vite + Tailwind
     - frontend/src/api.js: calls POST /api/verify when VITE_API_URL is
       set; otherwise falls back to a clearly-labeled mock fixture
       (frontend/src/mockData.js) so the UI reviews without a live backend
     - frontend/src/components/StatusBadge.jsx: classifyResult() enforces
       the two-tier UI language rule described above
     - frontend/src/components/TransparencyModal.jsx: surfaces the exact
       validated numbers (accuracy, agreement, selective accuracy,
       retrieval Recall@5/10/MRR) inline in the product, bilingually
        │
        ▼
 AMD PRODUCTION TARGET — PENDING FINAL RUN
   Qwen2.5-7B-Instruct served via vLLM on AMD Instinct MI300X / ROCm 6.x.
   Full runbook: phase4/reports/amd_deployment_instructions.md. The
   provider-switch seam (LLM_PROVIDER=vllm -> VLLMProvider in
   phase3/llm_provider.py) exists and is unit-tested to fail honestly
   (raises, does not fabricate a response) when no real endpoint is
   configured — see backend/tests/test_api.py::TestLLMProviderEnvSwitch.
   No AMD hardware, ROCm install, or vLLM process has ever been reached
   from this sandbox (phase3/reports/amd_investigation.md).
```

## Why the router exists at all (not just a nice-to-have)

`phase5/VALIDATED_METRICS.md`'s model-agreement numbers are the empirical justification for the whole Path A / Path B split: when all three models agree, accuracy on that subset is 92.6%; when they split 2-of-3, it drops to 64.9% — a 27.7-point swing. That gap is why disagreement (and, separately, novelty) is treated as a *routing* signal rather than ignored: an input the models actually agree on is cheap and trustworthy to answer immediately; an input they don't agree on is exactly the case worth paying for retrieval + LLM reasoning on.

## Why escalation is expensive and bounded, not free

The escalation path's own numbers (`AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §10, live HTTP measurements) are 20 seconds to ~2 minutes per request, versus roughly 1-2 seconds on the trusted path (after the one-time model warm-up). This is why `backend/main.py` runs every request in a worker thread with a 180-second ceiling (`asyncio.wait_for`) rather than leaving it unbounded, and why the router's whole purpose is to send only the genuinely uncertain slice of traffic down that path.

## Known gap between validated classifier accuracy and the product's default UX

Phase 1/2's 86.45% accuracy figure was measured on `(claim, evidence)` pairs — evidence was always present. The deployed single-text-input frontend sends `evidence_text=""` by default (`frontend/src/api.js`, `backend/schemas.py: VerifyRequest.evidence_text` defaults to `""`). `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §11 confirms directly that the same claim text can score meaningfully differently with real evidence present versus none supplied. This is disclosed, not hidden — see the Limitations section of `README.md`.

## Retrieval corpus construction (why Recall@5/10/MRR are scoped the way they are)

`phase3/retrieval.py: build_corpus()` indexes only TRAIN + VALIDATION evidence passages (TEST evidence is deliberately excluded so held-out retrieval evaluation can't leak the answer). `phase5/VALIDATED_METRICS.md`'s mandatory disclosure applies here: reusable gold evidence for retrieval evaluation only exists for the VERA-ARAB-sourced rows of the test set; AraStance and ArCOV19 rows have no reusable in-corpus gold evidence by construction, so retrieval quality for those sources is not measured.

## Deep-verification evaluation scope

`phase3/evaluate_deep_verification.py` ran the 25-example held-out evaluation behind the 60% figure, using `OllamaProvider` (`qwen2.5:1.5b`) — the development model, not Qwen2.5-7B-Instruct. See the mandatory framing for this number in `phase5/VALIDATED_METRICS.md`; it is not re-stated here beyond pointing at the source.
