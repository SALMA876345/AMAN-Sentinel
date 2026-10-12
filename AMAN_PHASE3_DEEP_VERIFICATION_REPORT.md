# AMAN Phase 3 Deep Verification Report

**Branch:** `aman-hackathon` (continues directly from Phase 1 + Phase 2; nothing in `hackathon/`, `phase2/`, `data/`, `results/`, `Results files/`, `artifacts/{svm,bilstm,transformers,novelty}/` was modified)
**Scope:** the escalation path — evidence retrieval, evidence-grounded LLM reasoning, hallucination safeguards, and the live `verify_content(text)` integration point for Phase 4. SOURCE-DOMAIN ONLY throughout; UAE-MVD remains unavailable and untouched.
**Method:** per this turn's explicit instruction, the two heavy independent workstreams (retrieval evaluation/implementation, LLM/AMD design/implementation) were delegated to parallel subagents with disjoint file ownership; the main agent built and verified the live single-input inference path, the routing integration, the end-to-end evaluation, and this report.

---

# Executive Summary

Three findings decide where AMAN stands going into Phase 4:

1. **No AMD hardware exists in this environment, and none was faked.** This machine has an Intel-only GPU/CPU stack. The AMD/ROCm path is fully *designed* (model choice, deployment architecture, provider abstraction already built to accept it) but not executed. The exact steps a human needs to take are documented below.
2. **A real, working escalation pipeline runs end-to-end today**, using a genuinely available local alternative (Ollama, `qwen2.5:1.5b`) standing in for the AMD-hosted production model. Retrieval → evidence-grounded LLM → programmatically-validated structured output → final AMAN verdict all work, verified on 25 real held-out TEST-split examples, not just unit tests.
3. **Retrieval is the weak link, and that's consistent with the rest of this project's history**, not a surprise. Recall@10 tops out around 20% even with the best method (Hybrid-RRF), and — more importantly — genuine ("native") relevance ground truth doesn't exist in the available corpus at all; every retrieval number reported here is measured on a weaker, partially-circular ground truth subset. This is reported honestly rather than presented as a clean result.

---

# Existing Retrieval Reused

`pipeline/hybrid_retrieval.py`'s `HybridRetriever` class was reused directly — not reimplemented. It already provides BM25 (`rank_bm25.BM25Okapi`) and dense retrieval (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, the same embedding model already used by Phase 2's novelty detector) with a weighted min-max score fusion for `"hybrid"` mode. Phase 3 adds one new fusion method on top — Reciprocal Rank Fusion — and a thin wrapper, `phase3/retrieval.py`, exposing a clean `retrieve_evidence(claim_text, top_k, method) -> Evidence Pack` function and a cached, persisted index (`phase3/data/retrieval_index/`) so the corpus isn't rebuilt per call.

**Corpus:** TRAIN + VALIDATION evidence passages only (8,250 deduplicated passages) — TEST evidence was deliberately excluded from the index to avoid leaking the answer into retrieval.

**The ground-truth caveat, stated plainly:** AraStance and ArCOV19 claims are each paired with an essentially unique, non-reused article — by construction, **0 of these claims' gold evidence passages exist anywhere in the retrieval corpus** (0/169 AraStance TEST rows, 0/638 ArCOV19 TEST rows). Only VERA-ARAB rows have reusable passages, because many VERA-ARAB rows' evidence was itself populated by a prior TF-IDF retrieval step during Phase 1 restoration (`hackathon/restore_claim_training_fast.py`) from a shared passage pool — so 44.7% of VERA-ARAB TEST rows (1,428/3,193) have their "gold" passage present in-corpus. **Every retrieval metric below is therefore measured only on this VERA-ARAB subset, evaluating whether one retrieval method can recover a passage a different retrieval method already chose — not against independent human relevance judgments.** This is reported here instead of being papered over with a single clean-looking number.

# BM25 Results

Measured on a fixed-seed sample of 500 TEST queries (downsampled from the 1,428 evaluable VERA-ARAB rows to stay within the hackathon compute budget; seed 42, documented in `phase3/reports/retrieval_evaluation.json`):

| Recall@1 | Recall@5 | Recall@10 | MRR |
|---:|---:|---:|---:|
| — | 0.076 | 0.140 | 0.0618 |

# Dense Results

| Recall@1 | Recall@5 | Recall@10 | MRR |
|---:|---:|---:|---:|
| — | 0.092 | 0.144 | 0.0596 |

# Hybrid Results

Two hybrid methods were measured, not assumed:

| Method | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Hybrid — existing weighted min-max fusion | 0.092 | 0.176 | 0.0723 |
| **Hybrid — Reciprocal Rank Fusion (k=60), new** | **0.108** | **0.202** | **0.0787** |

Hybrid-RRF wins on Recall@5, Recall@10, and MRR (weighted hybrid edges it slightly on Recall@1 only). **Hybrid did beat BM25 and dense alone here — but only by a modest margin, and the absolute numbers are low across every method.** This was measured, not assumed, and a different corpus/query mix could plausibly have gone the other way; that is exactly why the brief asked for measurement rather than assumption.

# Selected Retrieval Architecture

**Hybrid-RRF**, `k=60`, `top_k=5` for production calls. Chosen because escalated claims benefit more from reliable top-5/top-10 recall (giving the LLM several chances to find something relevant) than from a marginally-better top-1, and RRF won on every cutoff that matters for that use case. Full reasoning: `phase3/reports/retrieval_method_recommendation.md`.

**Context for the low absolute numbers:** 60.7% of errors in the *original* FAAV project's final studies were tagged "retrieval failure" (per the Phase 0 audit). Phase 3's low recall numbers are consistent with that established project history, not a new or surprising weakness — evidence retrieval over this corpus has been the hard part of this whole project from the start.

# Evidence Pack

Exact schema in production use, produced by `phase3.retrieval.retrieve_evidence`:

```json
{
  "claim": "...",
  "retrieved_evidence": [
    {"id": "...", "text": "...", "source": "...",
     "retrieval_method": "hybrid_rrf", "retrieval_score": 0.0}
  ]
}
```

`source` is preserved from the original corpus metadata (`VERA-ARAB`, `AraStance`, `ArCOV19-tweets`, `ArCOV19-claims`) — no source URL or provenance was invented for any passage.

# LLM Model Selection

**Production target: Qwen2.5-7B-Instruct** (Apache 2.0), chosen over Jais-13B-chat (more Arabic-native but ~2× the deployment cost/latency, thinner vLLM/ROCm ecosystem support) and Llama-3.1-8B-Instruct (comparable size, weaker Arabic). Qwen2.5-7B-Instruct has explicit Arabic training coverage, a 128K context window (comfortably fits a claim plus 5-10 evidence passages), and reliable structured/JSON output — important because every output here is machine-parsed and validated, not just read by a human. Full reasoning: `phase3/reports/llm_model_selection.md`.

**What actually ran in this environment:** `qwen2.5:1.5b` via a locally-reachable Ollama server (`http://localhost:11434`) — a genuinely smaller, different model, used honestly as a development/smoke-test stand-in, never confused with the production recommendation above.

# AMD Integration

**Finding: no AMD hardware, driver stack, or cloud access exists in this sandbox.** Verified directly, not assumed:

- GPU: Intel Iris Xe Graphics. CPU: Intel Core i5-1235U. No AMD device anywhere.
- `rocm-smi`: not installed. No `HIP_*`/`ROCM_*`/`AMD_*` environment variables set.
- `torch.version.hip == None`, `torch.cuda.is_available() == False` — confirms (again) this torch build has no CUDA *and* no ROCm backend compiled in.
- `vllm`: not installed.
- No AMD Developer Cloud credentials, endpoint URLs, or config files found anywhere in the repository.

**No AMD execution occurred, and none is claimed.** Full findings: `phase3/reports/amd_investigation.md`.

**What a human needs to do for real AMD execution** (all four required, none present today):
1. Provision an AMD Developer Cloud instance (or on-prem access to AMD Instinct/Radeon hardware) and obtain its endpoint/credentials.
2. Supply those credentials as environment variables (e.g., `AMD_CLOUD_ENDPOINT`, `AMD_CLOUD_API_KEY`, or SSH access).
3. Install a ROCm 6.x driver stack on that target, a ROCm-built PyTorch, and `vllm` with its ROCm backend.
4. Re-run `phase3/llm_provider.py` / `phase3/verification_llm.py` pointed at Qwen2.5-7B-Instruct served via vLLM on that instance — the provider abstraction already supports this swap with no changes to `verification_llm.py` (an `OpenAICompatProvider` hitting vLLM's OpenAI-compatible endpoint is the only new class needed).

Continuing all other work did not wait on this — per the parallel-execution rule, the rest of Phase 3 proceeded on the available local Ollama path while this finding was established.

# Verification Prompt Design

`phase3/verification_llm.py` builds a prompt that: provides the claim and every retrieved evidence passage (with its `id`); instructs the model to reason only from the provided passages; explicitly distinguishes "evidence says nothing about this" (→ `INSUFFICIENT_EVIDENCE`) from "evidence says the opposite" (→ `CONTRADICTED`); instructs the explanation to default to the claim's own language; and names Modern Standard Arabic, dialect variation, mixed Arabic/English, paraphrasing, negation, dates, named entities, numbers, and conflicting evidence among passages as things to handle explicitly. Language detection (`detect_language`) classifies `ar`/`en`/`mixed` from the raw claim text before prompting.

# Hallucination Safeguards

Implemented in code, not trusted to the prompt alone, in `_validate_and_coerce` and `verify_claim_against_evidence`:

1. **JSON repair.** A malformed response triggers one corrective retry; if still malformed, the result is `INSUFFICIENT_EVIDENCE` rather than a crash or a guess.
2. **Citation-existence check.** Every id in `evidence_ids` is checked against the Evidence Pack's real ids. Any id that doesn't exist forces the verdict to `INSUFFICIENT_EVIDENCE`.
3. **Empty-citation downgrade.** A `SUPPORTED`/`CONTRADICTED` verdict with zero cited evidence is force-downgraded — a verdict without a citation is never accepted.

**Measured, not assumed, that these work:** across the 13-example design smoke test (Agent B) and the 25-example held-out evaluation (this report's own held-out pass), **citation validity was 100%/100% — zero hallucinated evidence ids ever reached a returned result.** All 9 `INSUFFICIENT_EVIDENCE` verdicts in the held-out evaluation were safeguard-triggered downgrades (see Error Analysis) — meaning the safeguards were actively doing load-bearing work, not sitting idle.

# Structured Output

Schema (validated programmatically against this exact shape, every field required):

```json
{"verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
 "confidence": 0.0,
 "explanation": "...",
 "evidence_ids": ["..."],
 "claim_language": "ar" | "en" | "mixed"}
```

# Deep Verification Evaluation

Design-time prompt iteration used 13 **VALIDATION**-split examples only (`phase3/reports/llm_smoketest_results.json`). The numbers below are the one and only pass over 25 real, fixed-seed (42) **TEST**-split rows that Phase 2's own router already escalated (`phase2/reports/router_decisions_test.csv`) — never touched during prompt design.

| Metric | Value |
|---|---:|
| Escalated fraction of TEST | 12.65% (506/4,000) |
| End-to-end success rate | 96.0% (24/25) |
| Retrieval success rate (of completed) | 100.0% |
| Structured-output validity rate | 100.0% |
| Evidence citation validity rate | 100.0% |
| Safeguard-triggered rate | 37.5% (9/24) |
| Insufficient-evidence rate | 37.5% (9/24) — **100% of these were safeguard-forced, not natively produced** |
| Verdict distribution | SUPPORTED 15, CONTRADICTED 0, INSUFFICIENT_EVIDENCE 9 |
| Verdict accuracy when committed (SUPPORTED/CONTRADICTED only) | **60.0% (9/15)** |

**Honest reads on each of these:**
- The one end-to-end failure (`CLEAN-005699-b3040d8c2e`) was a genuine request timeout, not a code bug — caught cleanly by the try/except wrapper.
- 100%/100%/100% on retrieval success / schema validity / citation validity is a real, clean result for the plumbing.
- **The small dev model never once produced `INSUFFICIENT_EVIDENCE` on its own** — every abstention observed was the safeguard correcting an invalid output (an uncited or malformed response). This is reported as a limitation of the 1.5B dev-tier model, not of the safeguard design — the safeguard is precisely what prevented those cases from becoming confident, ungrounded verdicts.
- 60% verdict accuracy on committed verdicts, from a 1.5B parameter stand-in model with no citation relevance filtering, is modest and is reported as such — not claimed as representative of the production Qwen2.5-7B-Instruct target, which was never run here.
- **Latency note:** typical end-to-end latency (retrieval + LLM call) was median 43.1s, range 16.1–121.1s, across 22 of the 25 calls. Two calls recorded anomalous multi-hour elapsed times (7,336s and 25,915s) and are excluded from that median — almost certainly an artifact of this long multi-hour session's host machine sleeping/resuming mid-run (the session's own system clock advanced a full day during this conversation), not representative per-call latency. Reported here rather than silently dropped.

Full data: `phase3/reports/deep_verification_eval_summary.json`, `phase3/reports/deep_verification_eval_examples.csv`.

# Error Analysis

**BM25 vs. dense disagreement** (8 real VERA-ARAB TEST queries, the only subset with any in-corpus gold evidence): 1 BM25-wins, 0 dense-wins, 1 both-succeed, **6 both-fail**. Consistent with the low pooled recall numbers above — most of the time, neither sparse nor dense retrieval recovers the paired passage from this corpus.

**Retrieval succeeds but the LLM still abstains** — e.g. `CLEAN-006985-1aa60ddff7` ("استاد برج العرب غرق بمياه الامطار" / a stadium flooding claim): evidence was retrieved, but the model's answer didn't survive the citation check and was safeguard-downgraded to `INSUFFICIENT_EVIDENCE`.

**Evidence genuinely insufficient, correctly abstained** — the same pattern repeats across `CLEAN-003067-c716462a6f` and `CLEAN-012342-fddc5b449a`: retrieved passages existed but didn't resolve the claim, and the system said so rather than guessing.

**A committed verdict that was correct** — `CLEAN-019553-27c98aedd4` ("شاهد: ضبط مصنع يحول زيت السيارات إلى عسل") and `CLEAN-002844-7b00d384b8` (an Eid-prayer claim): both `SUPPORTED`, both matched the true label — good demo candidates.

**A committed verdict that was wrong** — `CLEAN-010813-d617d50326` (a COVID-19/Italy claim): LLM said `SUPPORTED`, true label was `Fake`. A representative failure mode for the small dev model worth showing honestly in a demo, not hiding.

**Language drift** — noted during the design-time smoke test (Agent B): one explanation for a World Cup / country-list claim drifted into Chinese despite the explicit language instruction, even though the final structured fields remained schema-valid. A real, reportable small-model quirk; full detail in `phase3/reports/llm_smoketest_results.json`.

These examples (claim text, verdict, and correctness) are saved verbatim in `phase3/reports/deep_verification_eval_examples.csv` for pitch/demo use.

# Final Phase 3 Architecture

```
Arabic claim (+ evidence, source-domain only)
        │
        ▼
phase3.live_inference.LiveModelEnsemble
   SVM (primary) + BiLSTM + MARBERT-probe (disagreement signals only)
   + Novelty detector (kNN/SBERT)
        │
        ▼
phase3.router_decision.route()  (Phase 2's validation-derived threshold, reused exactly)
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
VERIFIED_AUTOMATICALLY     ESCALATE_FOR_DEEP_VERIFICATION
(SVM verdict delivered,            │
 no LLM call)                      ▼
                        phase3.retrieval.retrieve_evidence (Hybrid-RRF, top_k=5)
                                   │
                                   ▼
                        phase3.verification_llm.verify_claim_against_evidence
                        (Ollama qwen2.5:1.5b today; Qwen2.5-7B-Instruct via
                         vLLM/ROCm in the designed production path)
                                   │
                        programmatic safeguards (JSON repair, citation
                        existence check, empty-citation downgrade)
                                   │
                                   ▼
                  SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
                                   │
                                   ▼
                   VERIFIED_WITH_EVIDENCE | UNCERTAIN
```

Single Python entry point for Phase 4: `phase3.aman_pipeline.verify_content(claim_text, evidence_text="") -> dict`, returning:

```json
{"status": "VERIFIED_AUTOMATICALLY" | "VERIFIED_WITH_EVIDENCE" | "UNCERTAIN",
 "classification": "Real" | "Fake" | null,
 "confidence": 0.0,
 "novelty_score": 0.0,
 "novelty_category": "FAMILIAR" | "MODERATELY NOVEL" | "HIGHLY NOVEL",
 "deep_verification_used": true,
 "evidence_verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE" | null,
 "explanation": "...",
 "evidence": [ ... ]}
```

Verified numerically against Phase 2's own batch predictions before being trusted (see "live inference" verification below) and exercised end-to-end on real TEST-split rows.

# Remaining Blockers

1. **AMD execution is entirely undone, by necessity.** Four concrete prerequisites are listed above under "AMD Integration" — none exist in this environment, and this is a blocker only a human with cloud/hardware access can resolve.
2. **Retrieval ground truth is structurally thin.** Native (AraStance/ArCOV19) relevance evaluation is impossible with the current corpus construction (N=0). Fixing this would require either a different corpus-building strategy or new relevance annotations — out of scope for Phase 3.
3. **The LLM smoke-tested here (qwen2.5:1.5b) is not the production target (Qwen2.5-7B-Instruct).** 60% committed-verdict accuracy and the "never natively abstains" finding are properties of the small dev model and may not transfer to the larger production model — this needs re-validation once real AMD/vLLM access exists.
4. **Two anomalous multi-hour latency outliers** in the held-out evaluation are attributed to probable host sleep/resume during this session, not confirmed with certainty — worth a clean re-run in a shorter, uninterrupted session before quoting latency numbers externally.
5. **The router's 0.50/0.10/0.30 disagreement/novelty penalty constants** (carried over from Phase 2, unchanged here) remain fixed design choices, not validation-tuned — flagged already in the Phase 2 report, still open.

# Phase 4 Recommendations

1. Build the API and frontend around `phase3.aman_pipeline.verify_content(text, evidence_text="")` — the interface is stable and tested.
2. Pursue real AMD Developer Cloud access as the top priority if live AMD execution is a judging criterion; the moment credentials exist, only one new provider class (`OpenAICompatProvider` → vLLM) is needed, zero changes to the verification logic.
3. Re-run the 25-example (or a larger) held-out evaluation once a bigger/production model is reachable, to get a verdict-accuracy number that's actually representative of what ships.
4. Consider a second, independent relevance-annotation pass (even a small one) to get a genuine native retrieval benchmark — the current one honestly cannot measure AraStance/ArCOV19 retrieval quality at all.
5. Surface `_router_detail` (trust score, model votes, novelty category) in the Phase 4 UI for transparency/demo purposes — it's already computed and returned, just needs a place to be shown.

---

**New files this phase:** `phase3/` (11 Python modules + tests + report files, all new — `retrieval.py`/`evaluate_retrieval.py` by the retrieval workstream, `llm_provider.py`/`verification_llm.py` by the LLM/AMD workstream, `live_inference.py`/`router_decision.py`/`aman_pipeline.py`/`evaluate_deep_verification.py`/tests by the main agent as integrator). No file outside `phase3/` was created or modified. **44/44 tests passing** (7 historical + 8 Phase 1 + 14 Phase 2 + 15 Phase 3).
