# AMAN Sentinel

**An Arabic-first AI misinformation verification system that routes uncertain claims through evidence retrieval and evidence-grounded verification rather than forcing unsupported answers.**

---

## 1. Problem

Arabic-language misinformation spreads fast and is hard to verify reliably — especially when the only check available is "ask an LLM and trust whatever it says." A raw LLM answer has no guarantee it's grounded in anything real: it can sound confident while citing nothing, or citing something that was never actually retrieved. Arabic-specific tooling is also underserved relative to English, where most fact-checking demos are tuned and evaluated.

## 2. Solution

AMAN Sentinel answers one narrow, concrete question for a piece of Arabic, English, or mixed-language text: **"Can I trust what I just received?"** It does this by triaging every input cheaply first (three classifiers, a disagreement check, a semantic-novelty check), and only paying for the expensive step — evidence retrieval plus a citation-constrained LLM call — when that cheap triage is genuinely uncertain. When even that can't resolve the claim, the system explicitly abstains rather than guessing.

## 3. Why AMAN Sentinel is different

Most "AI fact-checker" demos take a shortcut: paste text in, ask an LLM, print whatever comes back. AMAN deliberately does not do this:

- **It escalates only when uncertain**, instead of sending every claim through an expensive LLM call.
- **It cannot hallucinate a citation.** Every evidence ID the LLM cites is checked against what was actually retrieved; an invented or non-existent ID forces a downgrade to `INSUFFICIENT_EVIDENCE`.
- **It distinguishes two different kinds of answer.** A fast classifier-only read is shown as a cautious `LIKELY RELIABLE` / `LIKELY MISLEADING` judgment — never worded as a sourced fact. An evidence-grounded answer is shown as `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT_EVIDENCE`, a structurally different and stronger claim. The two are never conflated.
- **It reports uncertainty instead of hiding it** — selective accuracy is reported on the accepted subset at a stated coverage, not as one unconditional accuracy number for the whole system.

## 4. Architecture

```
Arabic / English / mixed-language text
            │
            ▼
   SVM + BiLSTM + MARBERT-probe + novelty detector   (cheap, ~1-2s)
            │
            ▼
      Confidence + disagreement + novelty → trust score
            │
   ┌────────┴──────────────────────────┐
   ▼ trust_score ≥ threshold            ▼ trust_score < threshold
PATH A: VERIFIED_AUTOMATICALLY     PATH B: ESCALATE_FOR_DEEP_VERIFICATION
(classifier verdict, no LLM             │
 call, fast)                            ▼
                              Hybrid-RRF evidence retrieval (BM25 + dense)
                                         │
                                         ▼
                              Evidence-grounded LLM verification
                              (cites only retrieved evidence, or abstains)
                                         │
                                         ▼
                   SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
            │                            │
            └─────────────┬──────────────┘
                           ▼
            FastAPI (backend/) ──► React frontend (frontend/)
```

Full technical detail, with every box traced to a real file/function, is in **[`ARCHITECTURE.md`](ARCHITECTURE.md)**. One Python entry point ties it all together: `phase3.aman_pipeline.verify_content(text, evidence_text="") -> dict`.

| Layer | What it does | Real code |
|---|---|---|
| Preprocessing | Arabic text normalization (tatweel, diacritics, alef/ta-marbuta variants, URLs) | `utils/faav_common.normalize_arabic` |
| Intelligence layer | SVM (primary) + BiLSTM + MARBERT-probe (disagreement signals) + SBERT/kNN novelty detector | `phase3/live_inference.py` |
| Decision router | confidence + agreement + novelty → trust score → route | `phase3/router_decision.py`, threshold from `phase2/reports/router_results.json` |
| Path A | High trust → cautious classifier-only result, never worded as a factual verdict | `phase3/aman_pipeline.py` |
| Path B | Escalation → BM25 + Dense → Hybrid-RRF (k=60) → Evidence Pack → evidence-grounded LLM → safeguards → verdict | `phase3/retrieval.py`, `phase3/verification_llm.py` |
| Product | FastAPI backend, React frontend | `backend/`, `frontend/` |

## 5. Key Features

- Three-model classifier ensemble trained on a leakage-audited, connected-component-grouped data split.
- A selective-prediction router that escalates low-confidence, disagreeing, or semantically novel claims instead of forcing a verdict.
- Hybrid BM25 + dense retrieval (Reciprocal Rank Fusion) feeding an evidence-grounded LLM verifier.
- Programmatic safeguards that force an explicit abstention (`INSUFFICIENT_EVIDENCE`) whenever a citation can't be verified — not a prompt instruction alone.
- A provider abstraction (`LLM_PROVIDER=ollama|vllm|mock`) so the same verification logic runs against a local model today and a production vLLM endpoint later with no code change.
- Bilingual (Arabic/English) React frontend with a transparency view showing confidence, novelty, model agreement, and whether deep verification was used.

## 6. Demo

See **[`phase5/DEMO_SCRIPT.md`](phase5/DEMO_SCRIPT.md)** for the full ~2–3 minute narration script and **[`phase5/DEMO_SHOT_LIST.md`](phase5/DEMO_SHOT_LIST.md)** for what should be on screen at each point. Both use the three real examples in `phase4/demo_examples.json` — nothing hard-coded, nothing invented. See also **[`submission/DEMO_GUIDE.md`](submission/DEMO_GUIDE.md)** for exact run-it-yourself steps.

## 7. Installation

```bash
git clone <this-repo>
cd "Code files"
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**LLM provider** (only used on the escalation path): development uses [Ollama](https://ollama.com) running locally with `qwen2.5:1.5b` or `qwen2.5:7b-instruct` pulled (`ollama pull <model>`). If Ollama isn't running, the escalation path still completes — it safely downgrades to `INSUFFICIENT_EVIDENCE` rather than failing. Copy `backend/.env.example` to `backend/.env` to configure `LLM_PROVIDER` / `OLLAMA_HOST` / `VLLM_*`.

## 8. Running the Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

First request after startup pays a one-time model warm-up (SVM/BiLSTM/MARBERT/novelty detector load into memory before any request is accepted). API docs: `http://localhost:8000/docs`. Health check: `GET /health` → `{"status": "ok", "pipeline_loaded": true}`.

## 9. Running the Frontend

```bash
cd frontend
npm install
cp .env.example .env.local      # set VITE_API_URL=http://localhost:8000
npm run dev
```

Opens at `http://localhost:5173`. **Without `VITE_API_URL` set, the UI falls back to a clearly console-labeled local mock fixture** (`frontend/src/mockData.js`, every string prefixed `[DEV FIXTURE]`) so the interface can still be reviewed with no backend running — make sure `VITE_API_URL` is set before a judge-facing demo so real backend output is shown, not the fixture.

## 10. Example Verification Request

```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "الاسطوره الايفواريه ديديه دروجبا يعتنق الإسلام", "evidence_text": ""}'
```

Response shape (`VerifyResponse` in `backend/schemas.py`):

```json
{
  "status": "VERIFIED_AUTOMATICALLY | VERIFIED_WITH_EVIDENCE | UNCERTAIN",
  "classification": "Real | Fake | null",
  "confidence": 0.0,
  "novelty": { "level": "FAMILIAR | MODERATELY NOVEL | HIGHLY NOVEL", "score": 0.0 },
  "model_agreement": "ALL_AGREE | PARTIAL_AGREEMENT",
  "deep_verification_used": false,
  "evidence_verdict": "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE | null",
  "explanation": "...",
  "evidence": [ { "id": "...", "text": "...", "source": "...", "retrieval_method": "...", "retrieval_score": 0.0 } ],
  "metadata": { "trust_score": 0.0, "router_threshold": 0.0, "model_votes": [ { "model": "svm", "prediction": "Real", "confidence": 0.0 } ], "processing_time_seconds": 0.0 }
}
```

## 11. Evaluation Results

Every number below is a measured result, sourced from `phase5/VALIDATED_METRICS.md`. See **[`submission/RESULTS.md`](submission/RESULTS.md)** for full detail and caveats.

| Metric | Value |
|---|---|
| SVM classifier accuracy / macro F1 / MCC (clean leakage-checked split) | **86.45% / 86.40% / 0.7281** |
| Accuracy when all 3 models agree (67% of test rows) | **92.6%** |
| Accuracy on 2-of-3 split-vote rows (33% of test rows) | 64.9% |
| Router: coverage / selective accuracy at the selected 80%-target operating point | **76.90% coverage / 92.65% selective accuracy on the accepted subset** |
| Retrieval — Hybrid-RRF (k=60): Recall@5 / Recall@10 / MRR | **0.108 / 0.202 / 0.0787** |
| Deep verification — qwen2.5:1.5b: committed accuracy / abstention (n=25) | 60.0% (9/15 committed), 40% abstained, Wilson 95% CI 35.7–80.2% |
| Deep verification — qwen2.5:7b-instruct: committed accuracy / abstention (n=25) | 37.5% (3/8 committed), 68% abstained, Wilson 95% CI 13.7–69.4% |
| Tests passing | **63/63** (+ 3 gated slow end-to-end tests, confirmed passing by hand) |

> **Never say** "AMAN Sentinel accuracy = 92.65%." The correct statement is: *"At the selected operating point, the router accepts 76.90% of claims with 92.65% selective accuracy on the accepted subset."* That number describes only the accepted subset, not the whole system and not the escalated remainder.

> **The 1.5B-vs-7B comparison above is exploratory, not a benchmark result.** Fisher's exact test on the two models' committed-verdict outcomes gives p = 0.40 (odds ratio 2.5) — **neither model is shown to outperform the other** at this sample size. The 7B model's higher abstention rate is an observed behavior on this sample, not evidence that it is "more cautious" in any validated sense.

## 12. AMD / Infrastructure Status

- **Development validation: COMPLETED.** A real end-to-end run of the full pipeline (FastAPI → classifiers → novelty → router → retrieval → deep verification → LLM → safeguards) was executed on a university GPU server (NVIDIA GB10, CUDA 13.0) running Qwen2.5-7B-Instruct via **Ollama**. Full detail: **[`phase5/reports/UNIVERSITY_GPU_VALIDATION.md`](phase5/reports/UNIVERSITY_GPU_VALIDATION.md)**.
- **AMD production validation: PENDING.** AMD Developer Cloud was configured for MI300X/ROCm and GPU access was requested; execution is still pending GPU-limit/access approval. **No AMD/ROCm/MI300X execution has occurred.** The NVIDIA validation above does not substitute for it.
- **Production target:** Qwen2.5-7B-Instruct via vLLM on AMD Instinct MI300X/ROCm — fully designed and coded with a ready-to-flip provider switch (`LLM_PROVIDER=vllm`, `phase3/llm_provider.py`), unexecuted. Runbook: **[`phase4/reports/amd_deployment_instructions.md`](phase4/reports/amd_deployment_instructions.md)**.

## 13. Repository Structure

```
backend/           FastAPI service wrapping verify_content()
frontend/          React + Vite web application
phase2/            Classifier ensemble, calibration, disagreement, novelty detector, router
phase3/            Retrieval, evidence-grounded LLM verification, live inference, aman_pipeline.py
phase4/            Demo examples, integration tests, deployment reports
phase5/            Validated metrics, judge-facing demo script/shot list, AMD evidence
hackathon/         Phase 1 restoration scripts and the leakage-safe clean data split
utils/, pipeline/  Shared helpers (`faav_common`, `build_final_study_manifests`, `hybrid_retrieval` — all live runtime dependencies)
artifacts/         Trained model artifacts (svm/, bilstm/, transformers/, novelty/)
submission/        Judge-facing project description, demo guide, architecture summary, results, and final checklist
```

**Historical research code** (`baselines/`, `eval/`, `experiments/`, `graphs/`, `improvements/`, `report/`, and the root-level `generate_*.py`/`run_*.py` scripts) is the original FAAV research project this work builds on — preserved in place because several AMAN modules import from adjacent shared utilities at a fixed directory depth, and because it documents where the models and methodology originated. It is not part of the AMAN Sentinel product and judges do not need to open it. The university assignment submission and the separate conference-paper manuscript that this project's leakage-audit findings also fed into have been moved out of this repository entirely (to a sibling `ACADEMIC_ARCHIVE_NOT_FOR_HACKATHON/` folder) and are not part of this submission.

## 14. Team

**Team:** AMAN AI
**Project:** AMAN Sentinel

- Hamza Alqaruoti
- Salma Amarah

---

## Testing

```bash
python -m pytest utils/test_final_decision.py hackathon/tests/test_phase1.py phase2/tests/test_phase2.py phase3/tests/test_phase3.py backend/tests/test_api.py phase4/tests/test_integration.py -q
```

**63/63 tests passing**, re-confirmed as part of this submission's final cleanup pass (2 gated real end-to-end tests skipped by default; see `phase5/VALIDATED_METRICS.md`).

## Responsible AI / Transparency

- **AMAN abstains rather than guesses.** When evidence doesn't resolve a claim, the system returns `INSUFFICIENT_EVIDENCE` instead of forcing a verdict it can't back up.
- **The LLM can never cite evidence that wasn't actually retrieved** — an invented or non-existent citation ID forces a downgrade to `INSUFFICIENT_EVIDENCE`.
- **A committed verdict with no citation is rejected** by the same safeguard.
- **The UI distinguishes classifier-only results from evidence-grounded ones** so a user never mistakes a fast heuristic read for a sourced fact-check.

## Limitations

- **The single-text-input UX sends `evidence_text=""` by default.** The classifiers were validated on `(claim, evidence)` pairs with evidence always present; a bare text submission presents a different input distribution. Disclosed, not hidden — see `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §11.
- **Retrieval metrics are measured on a subset of the test corpus.** Reusable gold evidence exists only for VERA-ARAB-sourced rows.
- **The deep-verification figures are small-sample, exploratory results**, not AMAN's overall verification accuracy — see §11 above.
- **No AMD/ROCm/MI300X execution has occurred anywhere in this project** — see §12 above.

## Screenshots

*(No screenshots captured yet — run `npm run dev` with a live backend connected and capture before presenting. See `submission/DEMO_GUIDE.md`.)*
