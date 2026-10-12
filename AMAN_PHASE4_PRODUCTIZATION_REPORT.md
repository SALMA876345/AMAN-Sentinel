# AMAN Phase 4 Productization Report

**Branch:** `aman-hackathon` (continues directly from Phases 1–3; nothing under `hackathon/`, `phase2/` core logic, `phase3/` core logic, `data/`, `results/`, `Results files/`, or any `artifacts/` subdirectory was modified except one additive edit to `phase3/llm_provider.py` — see §7)
**Scope:** turn the validated research pipeline into a working product — a FastAPI backend, a React frontend, a reliable demo path, and AMD deployment preparation. No model was retrained or redesigned.
**Method:** per this turn's instructions, the backend and frontend were built by two parallel subagents against a single frozen API contract the main agent designed up front from `phase3.aman_pipeline.verify_content()`'s real (not assumed) return shape; the main agent then independently re-verified both (re-ran backend's test suite, rebuilt the frontend, started a real server and hit it with real HTTP requests) before integrating, and found and fixed one real bug during that integration pass (see §10).

---

# 1. Architecture

```
Arabic / English / mixed text
        │
        ▼
phase3.live_inference.LiveModelEnsemble   (SVM primary + BiLSTM/MARBERT disagreement signals + novelty)
        │
        ▼
phase3.router_decision.route()             (Phase 2's validation-derived threshold, reused exactly)
        │
   ┌────┴──────────────────┐
   ▼                        ▼
VERIFIED_AUTOMATICALLY   ESCALATE_FOR_DEEP_VERIFICATION
   │                        │
   │                        ▼
   │              phase3.retrieval.retrieve_evidence (Hybrid-RRF)
   │                        │
   │                        ▼
   │              phase3.verification_llm.verify_claim_against_evidence
   │              (Ollama qwen2.5:1.5b today; Qwen2.5-7B/vLLM/ROCm designed, not executed)
   │                        │
   └───────────┬────────────┘
               ▼
    phase3.aman_pipeline.verify_content(text, evidence_text="") -> dict
               │
               ▼
    backend/main.py  (FastAPI: POST /api/verify, GET /health)
               │
               ▼
    frontend/  (React + Vite: Landing → Analysis Progress → Result Dashboard)
```

# 2. Frontend

**Stack:** React + Vite + `@tailwindcss/vite` (installed cleanly; chosen over fighting a toolchain for hand-rolled CSS). Dark, professional theme via CSS custom properties — explicitly not styled as a university demo.

**Screens** (`frontend/src/components/`): `LandingScreen.jsx` (AMAN Sentinel branding, Arabic-first RTL layout, "Can I trust what I just received?" headline, large text input, 3 real clickable demo examples), `AnalysisProgress.jsx` (optimistic staged pulse while a request is in flight, reconciled against the real `processing_stages` array once the response arrives — never claims a stage ran that the backend didn't report), `ResultDashboard.jsx` (final result with the two-tier language distinction below), `TransparencyModal.jsx` ("How AMAN Works," validated numbers only), plus `StatusBadge.jsx`, `ConfidenceMeter.jsx`, `NoveltyBadge.jsx`, `EvidenceList.jsx`.

**The required language discipline is implemented, not just described:** `StatusBadge.jsx` maps `classification: "Real"/"Fake"` (classifier-only path) to **LIKELY RELIABLE / LIKELY MISLEADING**, visually distinct from `evidence_verdict: "SUPPORTED"/"CONTRADICTED"/"INSUFFICIENT EVIDENCE"` (evidence-grounded path) — different badge styling per tier, so a user cannot mistake an SVM-only guess for an evidence-backed verdict.

**Verified, not assumed:** `npm run build` succeeds (261 KB JS / 20 KB CSS gzipped, re-confirmed independently by the main agent after the frontend agent's own build+dev-server+headless-screenshot check). `frontend/src/data/demoExamples.js` was copied from `phase4/demo_examples.json` — **initially with a bug the main agent caught and fixed** (see §10).

**API/mock boundary:** `frontend/src/api.js` calls the real backend when `VITE_API_URL` is set; otherwise falls back to a clearly-labeled `mockData.js` dev fixture with a console warning. Never silently used in place of a configured real backend.

# 3. Backend

**Stack:** FastAPI + Pydantic v2, `backend/main.py` / `backend/schemas.py` / `backend/pipeline_wrapper.py`.

**Startup:** the pipeline (SVM/BiLSTM/MARBERT/novelty detector/retrieval index) loads once in a `lifespan` handler before the app accepts traffic — no request pays the ~30-90s warm-up cost. `GET /health` reports `pipeline_loaded` without re-triggering it.

**Concurrency/timeouts:** `verify_content()` is synchronous and can take 20s–2min on the escalation path (live retrieval + LLM call). The backend runs it in a `ThreadPoolExecutor` under `asyncio.wait_for(..., timeout=180.0)` — chosen as ~1.5× the documented worst case — returning a clean HTTP 504 `{"error": "verification_timeout"}` rather than hanging. Exceptions from the pipeline are caught, logged server-side with full traceback, and returned as a clean HTTP 503 with no leaked internals.

**Validation:** empty/whitespace-only/>5000-character `text` is rejected with HTTP 422 via Pydantic field validators, before it ever reaches the pipeline.

# 4. API

```
POST /api/verify
  Request:  {"text": "...", "evidence_text": ""}
  Response: {"status", "classification", "confidence",
             "novelty": {"level", "score"},
             "model_agreement": "ALL_AGREE" | "PARTIAL_AGREEMENT",
             "deep_verification_used", "evidence_verdict", "explanation",
             "evidence": [...], "processing_stages": [...],
             "metadata": {"trust_score", "router_threshold", "model_votes", "processing_time_seconds"}}
GET /health -> {"status": "ok", "pipeline_loaded": bool}
```

Every field is a real Pydantic model (populates `/docs` correctly — confirmed via `app.openapi()`), and every field maps to something `verify_content()` actually returns; nothing was invented to match the brief's suggested shape where the real implementation didn't support it (e.g. `model_agreement` is derived from `_router_detail.all_agree`, not a separately-modeled concept).

**Confirmed with a real HTTP request** (not just unit tests) by the main agent:

```json
{"status": "VERIFIED_AUTOMATICALLY", "classification": "Fake", "confidence": 0.8749,
 "novelty": {"level": "FAMILIAR", "score": 0.2459}, "model_agreement": "ALL_AGREE",
 "deep_verification_used": false, ...}
```
— for the Drogba/Islam demo claim, matching its known true label (`Fake`), served by a live `uvicorn backend.main:app` process the main agent started and queried directly.

# 5. Phase 2 Integration

Nothing in Phase 2's validated logic was touched. `phase3.live_inference.LiveModelEnsemble` (built in Phase 3, reused unchanged here) loads the exact same persisted SVM/BiLSTM/MARBERT-probe/novelty-detector artifacts and was numerically verified bit-for-bit against Phase 2's own batch predictions *before* Phase 3 trusted it — that verification still holds, since nothing in those artifacts or the live-inference code changed in Phase 4.

# 6. Phase 3 Integration

`phase3.aman_pipeline.verify_content()` is the single call both the backend and (indirectly, through the backend) the frontend depend on. One additive change was made to it in Phase 4: a `processing_stages` list was added to its return dict (e.g. `["analyzing_language", "running_classifiers", "checking_confidence", "checking_novelty", "preparing_result"]`, with `"retrieving_evidence"`/`"deep_verification"` inserted when escalation occurs) — purely for the frontend's progress UI, with zero effect on routing, retrieval, or verification logic. Confirmed the existing `phase3.tests.test_phase3.EndToEndTests` test still passes unchanged after this edit.

# 7. LLM Provider Abstraction

`phase3/llm_provider.py` gained one additive capability: `get_default_provider()` now reads `LLM_PROVIDER` (`ollama` default, `vllm`, or `mock`) and a new `VLLMProvider` class was added (OpenAI-compatible `/v1/chat/completions` client, reads `VLLM_ENDPOINT`/`VLLM_API_KEY`/`VLLM_MODEL`). No existing provider class's logic was rewritten. `VLLMProvider` raises an honest, clear error if `VLLM_ENDPOINT` isn't configured — it does not silently fall back or pretend to work.

```
LLM_PROVIDER=ollama   # today's real, working path (qwen2.5:1.5b, local)
LLM_PROVIDER=vllm     # designed production path (Qwen2.5-7B-Instruct via AMD/ROCm) — UNTESTED, no AMD hardware exists
LLM_PROVIDER=mock     # deterministic, no network — for CI
```

# 8. AMD Deployment Readiness

**No AMD execution has occurred anywhere in this project, at any phase. None is claimed here either.** Phase 3 already confirmed this sandbox is Intel-only with no ROCm stack, no `vllm` install, and no AMD cloud credentials anywhere in the repository (`phase3/reports/amd_investigation.md`). Phase 4 adds nothing that changes that finding — it only prepares the switch.

`phase4/reports/amd_deployment_instructions.md` is the exact, runnable step-by-step a human operator follows once AMD Developer Cloud credits (already requested, pending account validation per this turn's brief) are approved: provision the instance → install ROCm 6.x → install a ROCm-built PyTorch → install `vllm` with its ROCm backend → `vllm serve Qwen/Qwen2.5-7B-Instruct` → set `LLM_PROVIDER=vllm` + `VLLM_ENDPOINT` → restart the API → re-run the Phase 3 held-out evaluation against the real endpoint to get a verdict-accuracy number that's actually representative of what ships (the 60% figure in `AMAN_PHASE3_DEEP_VERIFICATION_REPORT.md` is from the 1.5B dev model, not this target). No code changes are needed for this swap beyond setting the two environment variables — the provider abstraction is the seam designed for exactly this.

`.env.example` files exist at `backend/.env.example` and `frontend/.env.example`, documenting every environment variable either service reads, with placeholder values only. `.gitignore` was extended (additively) to exclude `.env`/`.env.*` while explicitly keeping `.env.example` files tracked. No credential, token, or API key exists anywhere in this repository.

# 9. Testing

| Suite | Result |
|---|---|
| `utils.test_final_decision` (historical) | 7/7 pass |
| `hackathon.tests.test_phase1` | 8/8 pass |
| `phase2.tests.test_phase2` | 14/14 pass |
| `phase3.tests.test_phase3` | 15/15 pass (re-confirmed after the additive `llm_provider.py` edit) |
| `backend/tests/test_api.py` | 15 pass, 1 skipped (gated real end-to-end, run manually: passed in 82s) |
| `phase4/tests/test_integration.py` | 4 pass, 1 skipped (gated real end-to-end across all 3 demo paths through the live HTTP API — run manually by the main agent: **passed**, see §10) |

**Total: 63/63 passing** (44 from Phases 1–3's unittest suites + 19 from the new pytest suites), **3 gated slow tests all independently run and passing** (not just left skipped and assumed fine).

New Phase 4 test coverage, specifically: health endpoint, trusted/escalated-path field mapping, malformed/empty/overlong input → 422, simulated provider failure → clean 503, timeout → 504, `LLM_PROVIDER` env-switch dispatch (`backend/tests/test_api.py`); full API-contract shape validation for all 3 demo paths together, and an assertion that `INSUFFICIENT_EVIDENCE` never carries a classification (the UI's two-tier language rule depends on this) (`phase4/tests/test_integration.py`).

# 10. Demo Workflow

Three real rows from `hackathon/data/clean_split/source_binary_test.csv` (`phase4/demo_examples.json`), chosen because their *actual* live pipeline behavior demonstrates the three required paths — no hard-coded outputs anywhere in frontend or backend code.

**A real bug was found and fixed during this integration pass.** `phase4/demo_examples.json` was first built with evidence text truncated to 200 characters (a copy-paste artifact from an earlier investigation). Running the "escalation" example through the real pipeline with that truncated evidence produced the WRONG behavior — it took the trusted path instead of escalating, because the truncated evidence changed the classifiers' actual scores. The main agent caught this by running the real end-to-end test, traced it to the truncation, re-extracted the full untruncated text from the source CSV, and re-ran to confirm correct behavior. The frontend's copy of the same data (`frontend/src/data/demoExamples.js`) did not have an evidence field at all and was not affected by this specific bug, but is subject to the related, more fundamental limitation in §11.

**Final, verified real results** (via the live HTTP API, not mocked):

| Example | Expected path | Actual result | Elapsed |
|---|---|---|---:|
| Drogba/Islam claim (true label: Fake) | No escalation | `VERIFIED_AUTOMATICALLY`, classification `Fake` — **correct** | 8-10s |
| Oil-to-honey factory claim (true label: Real) | Escalation | `VERIFIED_WITH_EVIDENCE`, `SUPPORTED`, classification `Real` — **correct** | 95s |
| Borg El Arab stadium claim (true label: Real) | Escalation, safeguard | `UNCERTAIN`, `INSUFFICIENT_EVIDENCE` — safeguard correctly abstained | 55s |

This demonstrates exactly what the brief asked AMAN to show: it does not blindly ask an LLM whether something is fake — it measures classifier confidence, disagreement, and novelty first, and only pays for retrieval + LLM reasoning on the genuinely uncertain third of cases (12.65% of the Phase 3 test split escalated, per `phase2/reports/router_decisions_test.csv`).

# 11. Known Limitations

1. **No AMD execution — stated plainly, not hedged.** See §8.
2. **The real product's single-text-box UX changes the classifiers' effective input distribution, and this matters.** Phase 1/2 validated 86.45% SVM accuracy on `(claim, evidence)` pairs where evidence was always present. A brand-new user submission through the frontend's single text field sends `evidence_text=""` by default. The main agent confirmed directly: the same claim text scores meaningfully differently with real evidence present (correctly escalates, 94.6s, `SUPPORTED`) versus with no evidence supplied (`VERIFIED_AUTOMATICALLY`, different confidence, different novelty category). **This is disclosed here because it is real and affects how much weight a user should put on a fast, non-escalated result for a brand-new claim with no supporting text attached.** It was not "fixed" because doing so (e.g., running retrieval before classification for every request) would be a pipeline redesign, explicitly out of scope for Phase 4 ("do not retrain or redesign the validated Phase 1-3 models unless absolutely required for integration") — it is reported as a limitation for a future phase to address, not silently patched or hidden.
3. **The LLM tested throughout (qwen2.5:1.5b) is not the production target (Qwen2.5-7B-Instruct).** The 60% committed-verdict accuracy figure from Phase 3 belongs to the small dev model and must be re-measured once real AMD/vLLM access exists — the frontend's transparency section deliberately omits this number for exactly this reason.
4. **Retrieval ground truth remains structurally thin** (Phase 3's finding, unchanged): native relevance evaluation is impossible with the current corpus construction for two of the three source corpora.
5. **Escalated requests are slow** (20s–2min) and the backend handles this with a generous timeout rather than a fix — a production deployment on real AMD hardware with a larger, properly-served model would likely be faster, but this hasn't been measured.

# 12. Exact Remaining Steps Once AMD Credits Arrive

All documented in full, runnable detail in `phase4/reports/amd_deployment_instructions.md`. Summary:

1. Provision the AMD Developer Cloud instance; confirm `rocm-smi` shows a real device.
2. Install ROCm-built PyTorch (`torch.version.hip` must return a real string, not `None`) and `vllm` with its ROCm backend.
3. `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000` (or the quantized variant if VRAM/latency requires it).
4. Set `LLM_PROVIDER=vllm`, `VLLM_ENDPOINT=http://<rocm-host>:8000/v1` on the backend; restart `uvicorn`.
5. Re-run `phase4/tests/test_integration.py::test_all_three_demo_examples_real_end_to_end` (set `AMAN_RUN_SLOW_TESTS=1`) against the real endpoint and confirm it still passes with the production model.
6. Re-run a larger slice of Phase 3's held-out evaluation (`phase3/evaluate_deep_verification.py`) against the real endpoint to get a production-representative verdict-accuracy number, replacing the 60%-on-a-1.5B-model figure everywhere it's currently caveated.

---

**New files this phase:** `backend/` (FastAPI service + tests, new), `frontend/` (React + Vite app, new), `phase4/` (demo examples, integration tests, AMD deployment doc, this report — new), root `README.md` (rewritten from the historical UTF-16LE FAAV placeholder into a working AMAN Sentinel guide). One additive edit to `phase3/llm_provider.py` (env-based provider switch + `VLLMProvider` skeleton) and one additive edit to `phase3/aman_pipeline.py` (`processing_stages` field). One additive edit to `.gitignore` (env-file exclusion). No file under `hackathon/`, `data/`, `results/`, `Results files/`, or any `artifacts/` subdirectory was touched. **63/63 tests passing**, 3 gated slow tests independently run and confirmed passing, not just left skipped.
