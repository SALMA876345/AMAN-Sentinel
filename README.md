# AMAN Sentinel

**Arabic-first AI information verification agent.**

Built across four phases on top of **FAAV** (*Faithfulness-Aware Arabic Verification*), the original research project this repository started from — AI503 Machine Learning, Assignment 2, by Hamza Alqaruoti. FAAV's research artifacts (papers, the original three-study evaluation protocol, the full audit trail) are preserved; AMAN Sentinel is the hackathon productization built on top of it, branch `aman-hackathon`.

## The problem

Arabic-language misinformation spreads fast and is hard to verify quickly or reliably — especially when the only check available is "ask an LLM and trust whatever it says." A raw LLM answer has no guarantee it's grounded in anything real: it can sound confident while citing nothing, or citing something that was never actually retrieved.

AMAN Sentinel exists to answer one narrow, concrete question for a piece of Arabic, English, or mixed-language content:

> **"Can I trust what I just received?"**

## The key innovation: AMAN does not blindly send every claim to an LLM

Most "AI fact-checker" demos take a shortcut: paste text in, ask an LLM, print whatever comes back. AMAN deliberately does not do this. Every input is cheaply triaged first — by a classifier ensemble, a disagreement check, and a novelty/out-of-distribution check — and the expensive step (evidence retrieval + a grounded LLM call) is only paid for when that cheap triage is genuinely uncertain.

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

Full technical detail, with every box traced to a real file/function, is in **[`ARCHITECTURE.md`](ARCHITECTURE.md)**.

One Python entry point ties it all together: `phase3.aman_pipeline.verify_content(text, evidence_text="") -> dict`.

## Why Arabic-first

Arabic misinformation-verification tooling is underserved relative to English: most off-the-shelf fact-checking demos are tuned and evaluated on English text and treat Arabic as an afterthought. AMAN's classifiers and normalization are built around real Arabic text behavior that generic tooling tends to mishandle — tatweel elongation, diacritics, alef/ta-marbuta/alef-maqsura spelling variants, and mixed Arabic/English (code-switched) input (`utils/faav_common.normalize_arabic`, and the LLM verification layer's explicit instruction to handle MSA, dialectal variation, and mixed-language text carefully, `phase3/verification_llm.py`). This is a deliberately scoped claim: AMAN handles Arabic/English/mixed text as its three language categories; it does not claim dialect-specific NLP models or capabilities beyond what's listed here.

## Architecture (condensed — see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full trace)

| Layer | What it does | Real code |
|---|---|---|
| Preprocessing | Arabic text normalization (tatweel, diacritics, alef/ta-marbuta variants, URLs) | `utils/faav_common.normalize_arabic` |
| Intelligence layer | SVM (primary) + BiLSTM + MARBERT-probe (disagreement signals) + SBERT/kNN novelty detector | `phase3/live_inference.py` |
| Decision router | confidence − disagreement penalty − novelty penalty → trust score → route | `phase3/router_decision.py`, threshold from `phase2/reports/router_results.json` |
| Path A | High trust → cautious classifier-only result, never worded as a factual verdict | `phase3/aman_pipeline.py` (`VERIFIED_PREDICTION` branch) |
| Path B | Escalation → BM25 + Dense → Hybrid-RRF (k=60) → Evidence Pack → evidence-grounded LLM → safeguards → verdict | `phase3/retrieval.py`, `phase3/verification_llm.py` |
| Product | FastAPI backend, React frontend | `backend/`, `frontend/` |
| AMD production target | Qwen2.5-7B-Instruct via vLLM on AMD Instinct MI300X/ROCm — **prepared, not executed** | `phase3/llm_provider.py` (`VLLMProvider`), `phase4/reports/amd_deployment_instructions.md` |

## Key validated metrics

Every number below is pulled verbatim from `phase5/VALIDATED_METRICS.md` — that file is the single source of truth; see it for full framing rules.

| Metric | Value | Source |
|---|---|---|
| SVM classifier accuracy (clean leakage-checked split) | **86.45%** | Phase 1/2 |
| SVM classifier Macro F1 | **86.40%** | Phase 1/2 |
| SVM classifier MCC | **0.7281** | Phase 1/2 |
| Accuracy when all 3 models agree (67% of test rows) | **92.6%** | Phase 2 |
| Accuracy on 2-of-3 split-vote rows (33% of test rows) | **64.9%** | Phase 2 |
| Selective accuracy at 80% target coverage (actual coverage 76.90%) | **92.65%** | Phase 2 |
| Selective risk at that operating point | 7.35% | Phase 2 |
| Retrieval — Hybrid-RRF (k=60), Recall@5 / Recall@10 / MRR | **0.108 / 0.202 / 0.0787** | Phase 3 |
| Retrieval — BM25, Recall@5 / Recall@10 / MRR | 0.076 / 0.140 / 0.0618 | Phase 3 |
| Retrieval — Dense (multilingual SBERT), Recall@5 / Recall@10 / MRR | 0.092 / 0.144 / 0.0596 | Phase 3 |
| Deep-verification dev-model experiment (qwen2.5:1.5b, 25 held-out examples) | 100% retrieval success, 100% structured-output validity, 100% citation validity, **60%** verdict accuracy on committed cases (9/15) | Phase 3 — **development-model experiment, not AMAN's accuracy; see Limitations** |
| Tests passing | **63/63** (+ 3 gated slow end-to-end tests run by hand, confirmed passing) | Phase 4 |

**Never say** "92.65% overall accuracy" — that number describes only the ~77% of inputs the router accepted without escalation at the 80%-target operating point, not the whole system and not the escalated ~23%. See `phase5/VALIDATED_METRICS.md` for the exact required framing of every number in this table.

## Screenshots

![Landing screen](docs/screenshots/landing.png)
*Landing screen — single text input, language toggle, demo-example shortcuts.*

![Result dashboard — trusted path](docs/screenshots/result-trusted-path.png)
*Result dashboard for a `VERIFIED_AUTOMATICALLY` case — confidence meter, model-agreement badge, "no deep verification needed" explanation.*

![Result dashboard — escalated path](docs/screenshots/result-escalated-path.png)
*Result dashboard for a `VERIFIED_WITH_EVIDENCE` case — evidence cards with source/retrieval method, evidence-grounded verdict badge.*

![Transparency modal](docs/screenshots/transparency-modal.png)
*Transparency modal — the validated metrics table surfaced directly inside the product, bilingually.*

*(Placeholders — no screenshots exist yet. Capture these from a running `npm run dev` session before submission.)*

## Demo

See **[`phase5/DEMO_SCRIPT.md`](phase5/DEMO_SCRIPT.md)** for the full ~2-3 minute narration script (with timestamps) and **[`phase5/DEMO_SHOT_LIST.md`](phase5/DEMO_SHOT_LIST.md)** for exactly what should be on screen at each point. Both use the three real examples in `phase4/demo_examples.json` — nothing hard-coded, nothing invented.

## Local installation

### Backend (FastAPI)

From the repo root (this directory):

```bash
pip install -r requirements.txt fastapi uvicorn pydantic
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

First request after startup pays a one-time ~30-90s model warm-up (SVM/BiLSTM/MARBERT/novelty detector load into memory at startup, before any request is accepted — see `backend/main.py`'s `lifespan`). API docs at `http://localhost:8000/docs`. Health check: `GET /health`.

**LLM provider** (only used on the escalation path): development uses [Ollama](https://ollama.com) running locally with `qwen2.5:1.5b` pulled (`ollama pull qwen2.5:1.5b`). If Ollama isn't running, the escalation path still completes — it downgrades to `INSUFFICIENT_EVIDENCE` safely rather than failing. Copy `backend/.env.example` to `backend/.env` to configure `LLM_PROVIDER` / `OLLAMA_HOST` / `VLLM_*`.

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. Copy `frontend/.env.example` to `frontend/.env.local` and set `VITE_API_URL=http://localhost:8000` to point it at a real backend; without it, the UI falls back to a clearly-labeled local mock fixture (`frontend/src/mockData.js`) so the interface can be reviewed without a running backend.

## API usage

`POST /api/verify` — request/response fields match `backend/schemas.py` exactly.

```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "الاسطوره الايفواريه ديديه دروجبا يعتنق الإسلام",
    "evidence_text": ""
  }'
```

```bash
# httpie
http POST localhost:8000/api/verify \
  text="شاهد: ضبط مصنع يحول زيت السيارات إلى عسل" \
  evidence_text="القبض على مجموعة يديرون مصنع يحول زيت السيارات إلى عسل..."
```

`text` is required (1-5000 chars, not whitespace-only); `evidence_text` is optional and defaults to `""` (see Limitations below for what that default implies). A successful response shape (`VerifyResponse` in `backend/schemas.py`):

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
  "processing_stages": ["analyzing_language", "running_classifiers", "..."],
  "metadata": { "trust_score": 0.0, "router_threshold": 0.0, "model_votes": [ { "model": "svm", "prediction": "Real", "confidence": 0.0 } ], "processing_time_seconds": 0.0 }
}
```

`GET /health` returns `{"status": "ok", "pipeline_loaded": true|false}`.

## Frontend usage

`npm run dev` serves the Vite dev server at `http://localhost:5173`. `VITE_API_URL` (set in `frontend/.env.local`, copied from `frontend/.env.example`) controls whether the UI talks to a real backend or falls back to the mock fixture — see `frontend/src/api.js`. The language toggle and the two-tier result presentation (classifier-only vs. evidence-grounded) are handled in `frontend/src/i18n.jsx` and `frontend/src/components/StatusBadge.jsx` respectively.

## Testing

```bash
python -m unittest hackathon.tests.test_phase1 phase2.tests.test_phase2 phase3.tests.test_phase3 utils.test_final_decision -v
python -m pytest backend/tests/test_api.py phase4/tests/test_integration.py -v
```

**63/63 tests passing** (44 from Phases 1-3's unittest suites + 19 from Phase 4's pytest suites), plus 3 gated slow end-to-end tests independently run by hand and confirmed passing. Source: `phase5/VALIDATED_METRICS.md`, `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §9.

## AMD deployment

> **AMD production deployment prepared; final MI300X validation pending cloud-credit activation.**

The production path (Qwen2.5-7B-Instruct via vLLM on AMD Instinct MI300X / ROCm) is fully designed and coded with a ready-to-flip provider switch (`LLM_PROVIDER=vllm` in `backend/.env`, backed by `VLLMProvider` in `phase3/llm_provider.py`), but it has never connected to real AMD hardware — this development sandbox has no AMD hardware, ROCm stack, or cloud credentials (confirmed directly in `phase3/reports/amd_investigation.md`). The full step-by-step runbook a human operator would follow once credits are approved is in **[`phase4/reports/amd_deployment_instructions.md`](phase4/reports/amd_deployment_instructions.md)**.

## Limitations

- **The single-text-input UX sends `evidence_text=""` by default.** Phase 1/2 validated 86.45% accuracy on `(claim, evidence)` pairs — evidence was always present. A brand-new user submission through the frontend's single text field presents the classifiers with a different input distribution than what was validated. This was confirmed directly: the same claim scores meaningfully differently with real evidence present versus none supplied. It is disclosed here, not hidden — see `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §11 for the full detail, and it was not "fixed" because doing so would require a pipeline redesign out of scope for this phase.
- **Retrieval metrics are measured on a subset of the test corpus only.** Reusable in-corpus gold evidence exists only for the VERA-ARAB-sourced rows; AraStance and ArCOV19 rows have zero reusable gold evidence by construction, so retrieval quality for those sources is not measured and not claimed. See `phase5/VALIDATED_METRICS.md`.
- **The 60% deep-verification figure is a limited development-model experiment, not AMAN's accuracy.** It used `qwen2.5:1.5b` via local Ollama — not the production target (Qwen2.5-7B-Instruct) — on a 25-example held-out sample, and has not been re-measured against the production model. Never cite it as "AMAN's verification accuracy." See `phase5/VALIDATED_METRICS.md`.
- **No AMD/ROCm execution has occurred anywhere in this project.** See the AMD deployment section above.

## Responsible AI / transparency

- **AMAN abstains rather than guesses.** When evidence doesn't resolve a claim, the evidence-grounded LLM layer is required to return `INSUFFICIENT_EVIDENCE` instead of forcing a SUPPORTED/CONTRADICTED call it can't back up (`phase3/verification_llm.py`).
- **The LLM can never cite evidence that wasn't actually retrieved.** Every cited evidence id is checked against the real Evidence Pack returned by retrieval; an invented or non-existent id forces a downgrade to `INSUFFICIENT_EVIDENCE` (`phase3/verification_llm.py: _validate_and_coerce`).
- **A SUPPORTED/CONTRADICTED verdict with no citations is rejected.** The safeguard layer force-downgrades any committed verdict that cites nothing to `INSUFFICIENT_EVIDENCE` rather than letting an uncited claim through.
- **The UI distinguishes classifier-only results from evidence-grounded ones.** A `VERIFIED_AUTOMATICALLY` result is shown as a cautious "LIKELY RELIABLE"/"LIKELY MISLEADING" classifier read, visually and linguistically distinct from an evidence-grounded `SUPPORTED`/`CONTRADICTED`/`INSUFFICIENT_EVIDENCE` verdict (`frontend/src/components/StatusBadge.jsx`), so a user can never mistake a fast heuristic read for a sourced fact-check.

## Project phases (each phase's report has full methodology and numbers)

| Phase | What it did | Report |
|---|---|---|
| 1 | Restored the broken FAAV pipeline, fixed an undisclosed evidence-leakage bug in the historical data split, trained a clean reproducible SVM baseline | `AMAN_PHASE1_RESTORATION_REPORT.md` |
| 2 | Added BiLSTM/MARBERT as disagreement signals, a novelty/OOD detector, confidence calibration, and the confidence-and-disagreement router | `AMAN_PHASE2_INTELLIGENCE_REPORT.md` |
| 3 | Built the escalation path: evidence retrieval (BM25/dense/Hybrid-RRF) and an evidence-grounded LLM verification layer with hallucination safeguards | `AMAN_PHASE3_DEEP_VERIFICATION_REPORT.md` |
| 4 | Productized everything behind a FastAPI backend and a React frontend; demo mode; AMD deployment preparation | `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` |
| 5 | Judge-facing documentation and validated-metrics consolidation (this phase) | `phase5/VALIDATED_METRICS.md`, this `README.md`, `ARCHITECTURE.md`, `phase5/DEMO_SCRIPT.md`, `phase5/DEMO_SHOT_LIST.md` |

## Repository layout (AMAN-specific additions)

```
backend/           FastAPI service wrapping verify_content()
frontend/          React + Vite web application
phase1/ → hackathon/    Phase 1 restoration scripts and clean split (data/ is gitignored, regenerate via scripts)
phase2/            Transformer baseline, calibration, disagreement, novelty detector, router
phase3/            Retrieval, evidence-grounded LLM verification, live inference, aman_pipeline.py
phase4/            Demo examples, integration tests, deployment reports
phase5/            Validated metrics, judge-facing demo script/shot list
artifacts/         Trained model artifacts (svm/, bilstm/, transformers/, novelty/)
```

Everything under `baselines/`, `pipeline/`, `experiments/`, `eval/`, `improvements/`, `results/`, and the root-level `.docx`/`.pdf` files is the original FAAV research project — preserved as-is, not part of the AMAN productization, but the source of the models and methodology AMAN builds on.

## Team

_Team name / members — fill in before submission._

## Hackathon

_[Hackathon name/track — fill in]_ — an AMD-sponsored hackathon track; no further event-specific detail was found in the repository's existing documentation at the time this file was written.
