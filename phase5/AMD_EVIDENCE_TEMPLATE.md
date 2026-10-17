# AMD Execution Evidence

**Fill this in during and immediately after running `phase5/FINAL_AMD_RUN_CHECKLIST.md`. Every field below is a placeholder until a real AMD run happens — do not fill in any field with an invented value. An unfilled field is more honest than a guessed one.**

**Note — separate NVIDIA validation, not AMD evidence:** a real end-to-end execution-path validation has separately been completed on **NVIDIA** GPU hardware (university GPU server, NVIDIA GB10, via Ollama) — see `phase5/reports/UNIVERSITY_GPU_VALIDATION.md`. That run is NOT AMD/ROCm/MI300X and is documented in its own report, not here. Every field in this template remains specifically about the still-pending AMD run and stays unfilled until that run actually happens.

---

## Hardware / environment

| Field | Value |
|---|---|
| GPU | _[e.g. AMD Instinct MI300X — fill in the exact model actually provisioned]_ |
| `rocm-smi` output confirming device | _[paste raw output]_ |
| ROCm version | _[e.g. ROCm 6.x — fill in exact version from the instance]_ |
| `torch.version.hip` | _[must be a real version string, not None]_ |
| `torch.cuda.is_available()` | _[must be True]_ |
| AMD instance type / provider | _[e.g. AMD Developer Cloud, instance SKU]_ |
| Test timestamp (UTC) | _[exact date/time of the run]_ |

## Model serving

| Field | Value |
|---|---|
| Model served | _[Qwen/Qwen2.5-7B-Instruct, or the exact variant actually used — note if quantized]_ |
| vLLM version | _[`pip show vllm` output]_ |
| vLLM startup log excerpt confirming ROCm device | _[paste]_ |
| Model load time | _[seconds]_ |

## AMAN configuration used

```
LLM_PROVIDER=vllm
VLLM_ENDPOINT=[fill in — do not record the real value if it contains a private IP/credential beyond what's safe to share; redact if needed]
VLLM_MODEL=[fill in]
```

## End-to-end test result

| Field | Value |
|---|---|
| `backend/tests/test_api.py::test_real_pipeline_end_to_end_through_live_api` | _[PASS / FAIL — paste relevant output]_ |
| `phase4/tests/test_integration.py::test_all_three_demo_examples_real_end_to_end` | _[PASS / FAIL — paste relevant output]_ |

## Demo path results (real responses, not summarized)

### 1. High-confidence example (Drogba/Islam claim)
```json
[paste the real JSON response here]
```
Latency: _[seconds]_ — compare to CPU/Ollama baseline: 8-10s

### 2. Escalation example (oil-to-honey factory claim)
```json
[paste the real JSON response here]
```
Latency: _[seconds]_ — compare to CPU/Ollama baseline: ~95s

### 3. Insufficient-evidence example (Borg El Arab stadium claim)
```json
[paste the real JSON response here]
```
Latency: _[seconds]_ — compare to CPU/Ollama baseline: ~55s

## Screenshots

- [ ] Landing screen
- [ ] Result Dashboard — example 1 (classifier-only path)
- [ ] Result Dashboard — example 2 (evidence-grounded, SUPPORTED)
- [ ] Result Dashboard — example 3 (INSUFFICIENT_EVIDENCE)
- [ ] vLLM server startup log (showing the real AMD device)
- [ ] `rocm-smi` output

_[Link or embed screenshot files here once captured — suggested path: `phase5/evidence/*.png`]_

## Logs

_[Attach or link: vLLM startup log, backend startup log, full pytest output from the gated E2E run]_

## Final notes

_[Anything that didn't go as planned, any deviation from the checklist, any caveat a judge should know before trusting this evidence. If the run partially failed, say exactly what worked and what didn't — do not round up to "success."]_

---

**Sign-off:** this evidence was collected by _[name]_ on _[date]_, running the exact steps in `phase5/FINAL_AMD_RUN_CHECKLIST.md` against real AMD hardware. Nothing in this document was filled in before the corresponding step actually ran.
