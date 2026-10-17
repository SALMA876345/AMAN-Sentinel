# University GPU Validation — NVIDIA GB10 (Ollama) — COMPLETED

**Status at a glance:**
- **NVIDIA GPU end-to-end validation: COMPLETED.** AMAN Sentinel was successfully validated end-to-end with Qwen2.5-7B-Instruct running on an NVIDIA GB10 GPU through Ollama, on a university GPU server.
- **AMD production validation (Qwen2.5-7B-Instruct via vLLM on AMD Instinct MI300X / ROCm): STILL PENDING.** This NVIDIA run is unrelated infrastructure and does not change that status in any way — no AMD hardware, ROCm stack, or vLLM-on-AMD path was touched or exercised.

This document records a real, hands-on end-to-end run of the full AMAN Sentinel pipeline on university-provided GPU hardware: backend startup, a dependency issue hit and resolved on that server, and two real `/api/verify` requests exercising classification, novelty detection, trust routing, retrieval, and LLM-based deep verification. It supplements — and does not replace — the metrics already on file in `phase5/VALIDATED_METRICS.md`.

---

## 1. Environment

| Item | Value |
|---|---|
| Host | University GPU server |
| GPU vendor | **NVIDIA** |
| GPU | NVIDIA GB10 |
| NVIDIA driver | 580.173.02 |
| CUDA | 13.0 |
| LLM runtime | Ollama |
| Model | `qwen2.5:7b-instruct` |
| Ollama-reported model size | 5.7 GB |
| Context length | 32768 |
| `ollama ps` processor check | `PROCESSOR: 100% GPU` |

No AMD hardware, ROCm driver stack, or vLLM-on-AMD serving path was present or used anywhere in this environment.

---

## 2. Dependency/environment issue encountered and resolved

During the first attempt to exercise deep verification, the FastAPI backend raised:

```
ImportError: rank-bm25 is required for BM25 retrieval
```

**This was checked and confirmed to be a server-side installation gap, not a repo bug.** The repository's root `requirements.txt` already declares the dependency correctly, at line 14, under the "Retrieval + semantic similarity" section:

```
rank-bm25>=0.2
```

Because the declaration in the repo was already present and correct, no file needed to be changed, and none was. The failure happened on the university server specifically — most likely because the Python environment running the backend had not completed (or had only partially completed) `pip install -r requirements.txt`, e.g. a virtual environment created and partially populated before the full dependency set was installed, or a stale environment carried over from an earlier setup step. This is an **environment/installation issue local to that server**, not a failure of the model, not a logic bug in the pipeline, and not a gap in the repo's dependency manifest.

**Resolution:** `rank-bm25` was installed into the server's active Python environment, and the backend was restarted. After that, the pipeline ran successfully end to end, as documented below.

---

## 3. Backend startup validation

After the dependency fix and restart, FastAPI startup completed successfully:

- Pipeline warm-up completed in approximately **2.0 seconds**.
- Log confirmed: `Application startup complete.`
- Uvicorn served the app at `127.0.0.1:8000`.
- `GET /health` was available.
- `POST /api/verify` was available.
- OpenAPI metadata identified the service as **"AMAN Sentinel Verification API"**, version **1.0.0**.

---

## 4. Test #1 — real end-to-end request

**Input claim:**
> "The World Health Organization announced that drinking water cures all diseases permanently."

**Output:**

| Field | Value |
|---|---|
| status | UNCERTAIN |
| classification | null |
| confidence | 0.5 |
| novelty level | HIGHLY NOVEL (score 0.5516) |
| model_agreement | PARTIAL_AGREEMENT |
| deep_verification_used | true |
| evidence_verdict | INSUFFICIENT_EVIDENCE |
| evidence | [] (empty) |
| trust_score | 0.0192 |
| router_threshold | 0.4972 |

**Model votes:**

| Model | Vote | Confidence |
|---|---|---|
| SVM | Real | 0.8192151085815039 |
| BiLSTM | Fake | 0.9020961143188332 |
| Transformer | Fake | 0.6964469805306206 |

**Timing:** processing_time reported as 23.9849 seconds; wall-clock `curl` time measured 24.000 seconds.

**Returned explanation (Arabic):** stated that the available evidence did not directly address the claim and therefore could not establish it.

**Interpretation (explicit):** The system correctly escalated a highly novel/disputed claim to deep verification (triggered by a near-threshold trust score and disagreement among the three classifiers), retrieved no sufficiently relevant evidence, and correctly returned `INSUFFICIENT_EVIDENCE` rather than inventing or guessing at supporting evidence. This result does **not** prove the claim is true or false — `INSUFFICIENT_EVIDENCE` is an abstention, not a verdict in either direction, and must not be read as one.

---

## 5. Test #2 — real end-to-end request

**Output:**

| Field | Value |
|---|---|
| status | UNCERTAIN |
| classification | null |
| confidence | 0.5 |
| novelty level | HIGHLY NOVEL (score 0.5278) |
| model_agreement | PARTIAL_AGREEMENT |
| deep_verification_used | true |
| evidence_verdict | INSUFFICIENT_EVIDENCE |
| evidence | [] (empty) |
| trust_score | -0.1672 |
| router_threshold | 0.4972 |

**Model votes:**

| Model | Vote | Confidence |
|---|---|---|
| SVM | Fake | 0.6327849041326779 |
| BiLSTM | Real | 0.7856193373649447 |
| Transformer | Fake | 0.529358140653572 |

**Timing:** processing_time reported as 3.7213 seconds; wall-clock `curl` time measured 3.735 seconds.

**Interpretation (explicit):** The router again detected a low (here, negative) trust score together with model disagreement and correctly invoked deep verification, again correctly abstaining with `INSUFFICIENT_EVIDENCE` when no supporting evidence was found. The much lower latency compared to Test #1 (3.72s vs 23.98s) is explained by the model already being loaded into GPU memory from the prior request — this is a **warm-run observation only**. It must **not** be generalized into a benchmark, an average, or a guaranteed production latency figure. Two data points on one server are not a performance benchmark.

---

## 6. What this validates

The two runs above exercised the complete pipeline, confirming each stage actually executes against real GPU-backed infrastructure, not mocks or stubs:

```
User claim
  → FastAPI API (/api/verify)
  → AMAN preprocessing
  → SVM / BiLSTM / Transformer signals (3-way vote, captured above)
  → Novelty detection (HIGHLY NOVEL scores observed)
  → Trust router (trust_score vs router_threshold ≈ 0.4972, correctly escalating both requests)
  → Retrieval (ran; returned no usable evidence for either claim)
  → Deep verification (invoked; produced INSUFFICIENT_EVIDENCE rather than fabricating a verdict)
  → Qwen2.5-7B-Instruct via Ollama (the actual generation backend for deep verification)
  → Safeguard / final response assembly (Arabic explanation, structured JSON fields)
```

The presence of the Qwen2.5-7B-Instruct process on the GPU was independently confirmed outside the application via:

```
ollama ps
```

reporting `qwen2.5:7b-instruct`, size `5.7 GB`, `PROCESSOR: 100% GPU`, `CONTEXT: 32768` — i.e., the model was actually resident and running on GPU during the validation window, not falling back to CPU.

**Claim this run supports:** AMAN Sentinel was successfully validated end-to-end with Qwen2.5-7B-Instruct running on an NVIDIA GB10 GPU through Ollama.

---

## 7. What this does NOT validate (AMD — read this section even if you skip everything else)

**AMD execution has not occurred, on this server or anywhere else in this project.** This run used NVIDIA hardware and the Ollama runtime exclusively. Do **not** say, imply, or allow anyone to infer any of the following from this report:

- "Validated on AMD" — **false.** No AMD GPU was used.
- "Validated on MI300X" — **false.** No AMD Instinct MI300X hardware was used.
- "Validated with ROCm" — **false.** No ROCm driver stack was installed, loaded, or exercised.
- "Validated with vLLM on AMD" — **false.** vLLM was not used in this run at all; Ollama was the runtime, and it ran on NVIDIA GB10.

The project's AMD production target — Qwen2.5-7B-Instruct served via vLLM on AMD Instinct MI300X / ROCm — remains **completely unexecuted**. Nothing in this NVIDIA run changes, substitutes for, or partially satisfies that pending validation. Per `phase5/VALIDATED_METRICS.md`, the required exact wording for AMD status wherever it is stated is:

> "AMD production deployment prepared; final MI300X validation pending cloud-credit activation."

---

## 8. Relationship to existing validated metrics

This report is **new execution evidence**, not a replacement for, override of, or correction to any figure already recorded in `phase5/VALIDATED_METRICS.md`. In particular, it does not alter:

- Phase 1 SVM clean-split metrics (86.45% accuracy / 86.40% macro F1 / MCC 0.7281).
- The router operating point (76.90% coverage / 92.65% selective accuracy on the accepted subset — never cited as overall system accuracy).
- Phase 3 retrieval results (Hybrid-RRF selected: R@5 = 0.108, R@10 = 0.202, MRR = 0.0787), measured only on the VERA-ARAB-sourced subset with reusable gold evidence.
- The prior development-model deep-verification experiment (60.0% verdict accuracy on 9/15 committed cases out of a 25-example sample, using `qwen2.5:1.5b`) — that experiment remains a limited development-model result and is not superseded by the two runs documented here. These two new runs are execution/plumbing validation on the production-target model size (7B), not a new labeled benchmark, and must not be merged with or presented as an update to the 60% figure.
- The Phase 4/5 test suite status (63/63 tests passing) — this validation run did not re-run that suite, so no change to that figure is claimed here.

In short: this document proves the pipeline runs correctly end-to-end on real NVIDIA GPU hardware with the production-sized 7B model. It is additive evidence about *execution correctness*, separate from the *accuracy/quality* metrics already on file, and it leaves the AMD validation gap exactly where it was — open and pending.
