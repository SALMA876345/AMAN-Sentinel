# AMAN Sentinel — Architecture (Judge Summary)

For the full file-by-file trace, see the repository root `ARCHITECTURE.md`. This is the condensed version.

## Pipeline

```
Input text (Arabic / English / mixed)
        │
        ▼
Preprocessing — Arabic normalization (tatweel, diacritics, alef/ta-marbuta variants)
        │
        ▼
Classification ensemble — SVM (primary) + BiLSTM + MARBERT-based probe
        │
        ▼
Uncertainty signals — confidence, 3-way agreement, semantic-novelty score
        │
        ▼
Selective-prediction router — trust score vs. a validation-selected threshold
        │
   ┌────┴─────────────────────────────┐
   ▼ high trust                       ▼ low trust / disagreement / novel
VERIFIED_AUTOMATICALLY          ESCALATE_FOR_DEEP_VERIFICATION
(classifier verdict only)              │
                                        ▼
                         Hybrid retrieval — BM25 + dense, fused by
                         Reciprocal Rank Fusion (k=60)
                                        │
                                        ▼
                         Evidence-grounded LLM verification
                         (cites only retrieved passages)
                                        │
                                        ▼
                         Programmatic safeguards
                         (force abstention on any unverifiable citation)
                                        │
                                        ▼
                    SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE
        │                                      │
        └──────────────────┬───────────────────┘
                            ▼
              FastAPI backend  →  React frontend
```

## Why two different verdict vocabularies

- **Classifier-only path:** `LIKELY RELIABLE` / `LIKELY MISLEADING` — a confidence judgment from pattern-matching, not a claim of having checked anything.
- **Evidence-grounded path:** `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT_EVIDENCE` — a claim about what retrieved evidence actually shows (or doesn't).

Keeping these visually and linguistically distinct (see `frontend/src/components/StatusBadge.jsx`) means a user can never mistake a fast heuristic read for a sourced fact-check.

## Why the router exists

Running every claim through retrieval and an LLM call is slow and expensive. The router exists to spend that cost only where it's actually needed — when the fast classifiers disagree, lack confidence, or see something semantically unlike anything in training. The three signals (confidence, agreement, novelty) are incorporated as complementary inputs to one trust score; no formal ablation isolates each signal's individual contribution, which is disclosed as a limitation rather than implied otherwise.

## Why the verification stage can't hallucinate a citation

The LLM is instructed to cite only retrieved evidence IDs — but the system does not trust the instruction alone. Every cited ID is checked in code against the actual Evidence Pack returned by retrieval. Any citation of a non-existent ID, any malformed output, and any committed verdict with zero citations are all programmatically forced to `INSUFFICIENT_EVIDENCE`. This is a mechanical guarantee, not a hope that the prompt is followed.

## Production deployment path

The LLM call is behind a provider abstraction (`phase3/llm_provider.py`) selected by one environment variable (`LLM_PROVIDER=mock|ollama|vllm`). The same verification logic runs unchanged against a local Ollama model today and a production vLLM endpoint on AMD Instinct MI300X hardware once that infrastructure is available — see `README.md` §12 for the exact, current status of that transition.

## Key files

| Component | File |
|---|---|
| End-to-end entry point | `phase3/aman_pipeline.py` (`verify_content`) |
| Classifier ensemble | `phase3/live_inference.py` |
| Router / trust score | `phase3/router_decision.py`, `phase2/router.py` |
| Novelty detector | `phase2/novelty_detector.py` |
| Retrieval | `phase3/retrieval.py`, `pipeline/hybrid_retrieval.py` |
| Evidence-grounded verification + safeguards | `phase3/verification_llm.py` |
| LLM provider abstraction | `phase3/llm_provider.py` |
| FastAPI backend | `backend/main.py`, `backend/pipeline_wrapper.py` |
| React frontend | `frontend/src/` |
