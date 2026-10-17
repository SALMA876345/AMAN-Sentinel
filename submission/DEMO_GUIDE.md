# AMAN Sentinel — Demo Guide

Exact steps to run and show the live demo. For the narrated walkthrough script, see `phase5/DEMO_SCRIPT.md` and `phase5/DEMO_SHOT_LIST.md`.

## 1. Start the backend

```bash
cd "Code files"
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete` in the log (model warm-up takes roughly 30–90s the first time). Confirm it's healthy:

```bash
curl http://localhost:8000/health
# {"status":"ok","pipeline_loaded":true}
```

**Optional — for the escalation path to use a real LLM instead of abstaining:**
```bash
ollama pull qwen2.5:7b-instruct   # or qwen2.5:1.5b for a faster, smaller model
```
Copy `backend/.env.example` to `backend/.env` and set `LLM_PROVIDER=ollama`. If Ollama isn't running, escalated claims still complete — they safely return `INSUFFICIENT_EVIDENCE` rather than failing.

## 2. Start the frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# edit .env.local: VITE_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`.

**Before demoing to judges: confirm `VITE_API_URL` is actually set.** Without it, the UI silently serves a local mock fixture (clearly labeled `[DEV FIXTURE]` in the console and in the response text) instead of real backend output. Check the browser console for the `[AMAN] VITE_API_URL is not set` warning — if you see it, the demo is not connected to the real backend.

## 3. Run the three validated demo examples

Source: `phase4/demo_examples.json` (no invented claims).

| Example | Expected path | Expected result |
|---|---|---|
| High-confidence claim (Drogba/Islam rumor) | Fast path | `VERIFIED_AUTOMATICALLY`, classification `Fake` |
| Escalation claim (oil-to-honey factory) | Escalated | `VERIFIED_WITH_EVIDENCE`, `SUPPORTED` |
| Insufficient-evidence claim (Borg El Arab stadium) | Escalated | `UNCERTAIN`, `INSUFFICIENT_EVIDENCE` — correct abstention |

Paste each claim's text into the frontend's input box, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "<claim text from phase4/demo_examples.json>", "evidence_text": ""}'
```

## 4. What to point out to judges

- The **routing decision itself** — show a high-confidence claim resolve instantly with no LLM call, then an uncertain claim escalate and visibly take longer.
- The **two different verdict vocabularies** on screen: `LIKELY RELIABLE`/`LIKELY MISLEADING` for the fast path vs. `SUPPORTED`/`CONTRADICTED`/`INSUFFICIENT_EVIDENCE` for the evidence-grounded path — never the same badge for both.
- The **evidence cards** on an escalated result — each one shows its retrieval method and score, so the verdict is auditable, not a black box.
- The **abstention case** — this is the strongest demo moment: show the system explicitly declining to confirm or deny a claim because the retrieved evidence didn't support either answer, rather than guessing.
- If asked about AMD: state the status exactly as written in `README.md` §12 — NVIDIA development validation completed, AMD MI300X execution pending access approval. Do not say "it runs on AMD."

## 5. Known constraints to mention proactively

- The frontend's single text box defaults to no supplied evidence (`evidence_text=""`); this is a different input distribution than the classifiers were validated on and is disclosed in `README.md`'s Limitations.
- The deep-verification accuracy figures (60.0% for the 1.5B model, 37.5% for the 7B model on the same 25-case sample) are exploratory and not statistically distinguishable from each other (Fisher's p=0.40) — don't present either as "the" verification accuracy.
