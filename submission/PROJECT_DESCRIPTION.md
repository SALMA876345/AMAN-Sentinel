# AMAN Sentinel — Project Description

**Team:** AMAN AI — Hamza Alqaruoti, Salma Amarah

## What it is

AMAN Sentinel is an Arabic-first AI misinformation verification system. It answers one question for a piece of Arabic, English, or mixed-language text: *can this be trusted?* Instead of sending every input straight to an LLM and printing whatever comes back, it routes each claim based on how confident and well-corroborated the cheap signals are — accepting the fast classifier's answer when trust is high, and escalating to evidence retrieval plus a citation-constrained LLM verifier when it isn't. When even that can't resolve the claim, it says so explicitly rather than guessing.

## The problem

Arabic misinformation spreads fast, and Arabic-specific verification tooling lags English tooling badly. Most demo fact-checkers compound this by skipping any real evidence step — an LLM can sound completely confident while citing nothing, or citing something that was never retrieved.

## The approach

1. **Classify cheaply first.** Three independently-trained classifiers (SVM, BiLSTM, a MARBERT-based probe) vote on the claim in under two seconds.
2. **Measure uncertainty, don't hide it.** Classifier confidence, three-way agreement, and a semantic-novelty score (how unlike anything seen in training) combine into a trust score.
3. **Route, don't force.** High-trust claims get a fast, clearly-labeled classifier verdict. Low-trust claims escalate.
4. **Retrieve real evidence.** A hybrid BM25 + dense retriever (fused by Reciprocal Rank Fusion) finds supporting passages.
5. **Verify against evidence, with teeth.** An LLM is instructed to reason only from retrieved passages and cite only what it was given — and a programmatic safeguard, not just a prompt, forces an explicit abstention if a citation can't be verified against what was actually retrieved.

## Why this matters

A verification system that is wrong *and* confident is more dangerous than one that is uncertain and says so. AMAN Sentinel is built around that principle at every layer: the router reports a coverage/accuracy trade-off instead of one accuracy number; the verifier abstains instead of hallucinating a citation; and every validated metric in this submission is reported with its actual scope, not rounded up into a bigger claim.

## What's validated vs. what's prepared

- The full pipeline, dataset, classifiers, router, retrieval, and evidence-grounded verification logic are built, tested (63/63), and validated end-to-end — including a real run on university NVIDIA GPU infrastructure with the production-target model (Qwen2.5-7B-Instruct).
- Production deployment on AMD Instinct MI300X via ROCm/vLLM is fully designed and coded (one environment-variable switch away) but has not yet executed — AMD Developer Cloud access was requested and is still pending approval. See `README.md` §12 for the exact, non-overclaiming status.

See `submission/RESULTS.md` for every validated number and its caveats, and `submission/ARCHITECTURE.md` for how the pieces fit together.
