# Final AMD Run Checklist

**Purpose:** everything that happens the moment AMD Developer Cloud credits activate, so that run is fast, mechanical, and produces the evidence a judge needs — not a scramble. Nothing here has been executed. No step requires guessing; every command references a real file already in this repo.

**Before you start:** read `phase4/reports/amd_deployment_instructions.md` for the full reasoning behind each step (model choice, why ROCm, etc.). This checklist is the condensed, ordered, check-the-box version of that document plus the application-level steps (backend/frontend/tests) it doesn't cover.

**Note — NVIDIA execution-path validation already completed:** the end-to-end request path this checklist exercises (FastAPI → classifiers → novelty detector → trust router → retrieval → deep verification → LLM → safeguards) has already been run successfully, start to finish, on NVIDIA GPU hardware (university GPU server, via Ollama) — see `phase5/reports/UNIVERSITY_GPU_VALIDATION.md`. That reduces execution risk for the AMD run below but is NOT AMD/ROCm/MI300X validation and does not complete any step of this checklist. Every item below remains unchecked and pending until actually run on real AMD hardware.

---

- [ ] **1. Confirm the $100 credit appears** in the AMD Developer Cloud account dashboard. Do not proceed until the balance is visible and active.

- [ ] **2. Provision an AMD Instinct MI300X instance** (or the closest available Instinct-series GPU if MI300X specifically isn't offered at provisioning time — note which one you actually got in `phase5/AMD_EVIDENCE_TEMPLATE.md`).

- [ ] **3. Verify GPU/ROCm on the instance:**
  ```bash
  rocm-smi                          # must show at least one AMD device
  python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"
  # torch.version.hip must be a real version string (not None)
  # torch.cuda.is_available() must be True (ROCm exposes itself via the CUDA-compatible HIP API)
  ```
  Record the raw output in the evidence template — this is the proof real AMD hardware was used, not a CPU fallback.

- [ ] **4. Install the required environment** on the instance:
  ```bash
  # ROCm-built PyTorch (exact index URL depends on the ROCm version installed — check
  # https://pytorch.org/get-started/locally/ for the current ROCm wheel index)
  pip install torch --index-url <rocm-pytorch-wheel-index-for-the-installed-ROCm-version>
  pip install vllm   # with ROCm backend — follow vLLM's ROCm install docs for the exact
                      # flags/Docker image current at run time (e.g. VLLM_TARGET_DEVICE=rocm,
                      # or the official ROCm vLLM container)
  ```
  Then clone/copy this repository onto the instance and:
  ```bash
  pip install -r requirements.txt fastapi uvicorn pydantic
  ```

- [ ] **5. Start vLLM**, serving the production model with an OpenAI-compatible endpoint:
  ```bash
  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
  # add --quantization awq (or gptq) here only if VRAM/latency on the actual
  # instance requires it — try unquantized first and note the result either way
  ```

- [ ] **6. Load Qwen2.5-7B-Instruct** — confirmed by step 5 actually completing and the server reporting ready (vLLM logs "Uvicorn running on..." or equivalent). Record model load time.

- [ ] **7. Configure AMAN's backend** to point at the real vLLM endpoint instead of local Ollama:
  ```bash
  export LLM_PROVIDER=vllm
  export VLLM_ENDPOINT=http://localhost:8000/v1      # adjust host:port if vLLM runs on a different interface
  export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
  # VLLM_API_KEY only if the vLLM server was started with an API key requirement
  ```
  (See `backend/.env.example` for the full documented list of variables.)

- [ ] **8. Start the backend:**
  ```bash
  uvicorn backend.main:app --host 0.0.0.0 --port 8000
  ```
  Wait for `GET /health` to report `{"status": "ok", "pipeline_loaded": true}` before proceeding — this pays the one-time SVM/BiLSTM/MARBERT/novelty-detector/retrieval-index warm-up.

  **Port conflict note:** vLLM and the backend both default to port 8000 in the commands above — run them on different ports on the actual instance (e.g. vLLM on 8000, backend on 8080) and adjust `VLLM_ENDPOINT` accordingly.

- [ ] **9. Start the frontend:**
  ```bash
  cd frontend
  npm install
  echo "VITE_API_URL=http://<backend-host>:<backend-port>" > .env.local
  npm run dev
  ```

- [ ] **10. Execute all three demo paths** against the real stack (`phase4/demo_examples.json` has the exact claim text for each). Confirm, for each:
  - High-confidence example → `VERIFIED_AUTOMATICALLY`, no LLM call, classification matches the known true label (`Fake`).
  - Escalation example → `VERIFIED_WITH_EVIDENCE`, `SUPPORTED`, classification matches the known true label (`Real`).
  - Insufficient-evidence example → `UNCERTAIN`, `INSUFFICIENT_EVIDENCE`.
  Record the actual JSON response for each in the evidence template.

- [ ] **11. Execute the gated end-to-end tests** against the real endpoint:
  ```bash
  AMAN_RUN_SLOW_TESTS=1 python -m pytest backend/tests/test_api.py::test_real_pipeline_end_to_end_through_live_api phase4/tests/test_integration.py::test_all_three_demo_examples_real_end_to_end -v -s
  ```
  Both must pass. Record pass/fail and the printed per-example timing.

- [ ] **12. Record latency** for each of the three demo paths (wall-clock, from request sent to response received) — compare against the CPU/Ollama baseline already documented in `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §10 (8-10s / 95s / 55s) to show the real speedup (or lack thereof) from AMD hardware.

- [ ] **13. Record successful AMD hardware detection** — the raw `rocm-smi` and `torch.version.hip` output from step 3, plus vLLM's own startup log confirming it loaded onto a ROCm device (not a CPU fallback).

- [ ] **14. Save logs and screenshots**: vLLM startup log, backend startup log, the three demo-path JSON responses, a frontend screenshot of the Result Dashboard for each of the three paths, and the pytest output from step 11. Use `phase5/AMD_EVIDENCE_TEMPLATE.md` to organize all of this into one place.

- [ ] **15. Update `README.md`'s AMD status section** — replace "AMD production deployment prepared; final MI300X validation pending cloud-credit activation" with the real outcome (success, partial success, or whatever actually happened — do not overwrite with a success claim if the run didn't fully succeed).

- [ ] **16. Update `SUBMISSION_COPY.md`'s AMD TECHNOLOGY section** and `phase5/VALIDATED_METRICS.md`'s "AMD status" section to match — these must stay consistent with each other and with the README.

- [ ] **17. Destroy the GPU droplet/instance immediately after testing** to avoid unnecessary credit usage. Confirm in the AMD Developer Cloud dashboard that no instance is still running before closing out.

---

**Reminder:** no step in this checklist should be marked done, and no document should be updated to claim success, until the corresponding command has actually been run and its real output recorded. Do not fabricate any part of this.
