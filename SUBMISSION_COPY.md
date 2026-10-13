# SUBMISSION_COPY

Ready-to-paste text for the AMD hackathon submission portal. Every metric below is pulled verbatim from `phase5/VALIDATED_METRICS.md` — the single source of truth for this project's numbers. Do not edit numbers here without updating that file first.

---

## PROJECT TITLE

AMAN Sentinel

---

## ONE-LINE TAGLINE

Arabic-first AI verification that knows when to classify, when to retrieve evidence, and when to abstain.

---

## SHORT DESCRIPTION

AMAN Sentinel is an Arabic-first misinformation-verification system that doesn't blindly ask an LLM about every claim. It first classifies and measures its own uncertainty using lightweight signals, and only escalates to evidence retrieval and grounded LLM reasoning when the input is genuinely hard to call. The result is a system that is cheaper, faster, and more auditable than "always call an LLM" — and one that would rather say "insufficient evidence" than guess.

---

## LONG DESCRIPTION

Misinformation verification tools built for English don't transfer cleanly to Arabic, and most "AI fact-checking" demos quietly skip the hardest question: what happens when the model doesn't actually know? AMAN Sentinel is built around that question. It takes Arabic, English, or mixed-language text and decides whether it can be trusted — not by handing every input to a large language model and hoping for a grounded answer, but by first asking cheaper, faster signals whether the input even needs that.

A validated SVM classifier (the project's strongest individual model) makes the primary call. Two auxiliary signals — a BiLSTM and a frozen MARBERT probe — are run alongside it purely to measure disagreement, because disagreement between models turned out to be a much stronger signal of risk than any single model's own confidence score. A novelty/out-of-distribution detector, built on multilingual SBERT embeddings and k-nearest-neighbor distance, separately flags inputs that look unlike anything the classifier was trained on. These signals combine into a single trust score that decides whether a claim can be resolved immediately or needs to be escalated.

Escalated claims enter the second path: evidence retrieval (BM25 lexical search, dense SBERT retrieval, and a Hybrid Reciprocal-Rank-Fusion combination of the two) pulls the most relevant passages from the evidence corpus, and an LLM is asked to reason strictly from those retrieved passages — never from its own training-time "knowledge" of the topic. The LLM must cite real retrieved passage IDs for any verdict it gives, and if the evidence doesn't clearly support or contradict the claim, it is required to abstain with an INSUFFICIENT_EVIDENCE verdict rather than guess.

This selective-computation design — route cheap, escalate only when needed, abstain over guessing — is what distinguishes AMAN Sentinel from a thin LLM wrapper. Development today runs on a small local model (`qwen2.5:1.5b` via Ollama, CPU-only) so the full pipeline can be demonstrated end-to-end without any special hardware. The production target is a larger model (Qwen2.5-7B-Instruct) served via vLLM on AMD Instinct MI300X hardware through ROCm — fully designed, built behind a ready-to-flip provider switch, and prepared for deployment as soon as cloud-credit access activates.

---

## PROBLEM

Arabic-language misinformation is widespread, fast-moving, and poorly served by verification tooling built primarily for English. Naively routing every claim through a large language model is slow, expensive, and — critically — ungrounded: an LLM asked "is this true?" will often answer fluently from its own training-time associations rather than from any checkable evidence, and it rarely admits when it doesn't actually know. That combination — expensive, unauditable, and overconfident — is a poor foundation for a verification tool that people are meant to trust.

---

## SOLUTION

AMAN Sentinel separates "can this be resolved cheaply and reliably" from "does this need deeper investigation," and only pays the cost of evidence retrieval and LLM reasoning for the subset of claims that actually need it. A validated classifier plus two disagreement signals and a novelty detector produce a trust score; low-uncertainty claims are resolved immediately, while higher-uncertainty claims are escalated to a retrieval-grounded LLM that must cite real evidence or abstain. The system is explicit, at every step, about which of those two paths produced a given result.

---

## WHAT MAKES IT UNIQUE

- **Selective computation, not blanket LLM calls.** Most claims never need an LLM at all; the trust score decides.
- **Disagreement as a signal, not noise.** The auxiliary BiLSTM and MARBERT-probe models aren't there to outperform the classifier — they're there to disagree with it when a claim is genuinely ambiguous, and that disagreement is one of the strongest routing signals found in this project.
- **Evidence-grounded, never knowledge-grounded.** The escalation-path LLM is constrained to reason only from retrieved passages, must cite real passage IDs, and is required to abstain (INSUFFICIENT_EVIDENCE) rather than answer from its own training knowledge.
- **Honest about its own limits.** The system reports two distinct kinds of results (resolved-without-escalation vs. evidence-grounded-with-citations vs. abstained) rather than presenting one undifferentiated "verdict," and the project's own validation reporting follows the same honesty standard — it says what was measured, on what subset, and what wasn't.

---

## HOW IT WORKS

1. **Input.** A claim comes in — Arabic, English, or mixed-language text.
2. **Classify.** A validated SVM model (the strongest individual classifier found in this project) makes a primary Real/Fake call.
3. **Measure uncertainty.** Two auxiliary signals run alongside the classifier purely to check for disagreement: a BiLSTM and a frozen MARBERT linear probe. Separately, a novelty/out-of-distribution detector (multilingual SBERT embeddings + k-nearest-neighbor distance) checks whether the claim looks unlike anything the classifier was trained on.
4. **Combine into a trust score.** These signals are combined into a single score representing how much the system trusts its own primary call.
5. **Route.** If the trust score is high, the system returns its result immediately — no LLM call, fast and cheap. If the trust score indicates genuine uncertainty, the claim is escalated.
6. **Retrieve evidence (escalation path only).** The system searches the evidence corpus using BM25 (lexical), dense SBERT retrieval, and a Hybrid Reciprocal-Rank-Fusion combination of the two, selected because it recovers more relevant passages across the board than either method alone.
7. **Reason from evidence, or abstain.** An LLM is given the claim plus the retrieved passages and instructed to answer strictly from that evidence — never from its own training knowledge — citing only real retrieved passage IDs. If the evidence doesn't clearly settle the claim, the system returns INSUFFICIENT_EVIDENCE instead of a guess.
8. **Result.** The user sees which path produced the answer: a fast classifier-only result, or an evidence-grounded result with citations, or an explicit abstention.

---

## AMD TECHNOLOGY

AMAN Sentinel's production LLM path is designed to run Qwen2.5-7B-Instruct via vLLM on AMD Instinct MI300X hardware through ROCm. The code already includes a ready-to-flip provider switch so the system can move from its local development LLM to the AMD-hosted production model without any pipeline redesign.

**AMD production deployment prepared; final MI300X validation pending cloud-credit activation.**

Development today runs entirely on CPU (Intel) using a small local model (`qwen2.5:1.5b` via Ollama) specifically so the full pipeline could be built, tested, and demonstrated end-to-end before any AMD hardware access was available. No AMD run, ROCm execution, or AMD performance number is claimed anywhere in this project.

---

## RESPONSIBLE AI

- **Abstention over guessing.** When retrieved evidence doesn't clearly support or contradict a claim, the system returns INSUFFICIENT_EVIDENCE rather than forcing a verdict.
- **Citation safeguards.** The escalation-path LLM must cite real retrieved passage IDs for any verdict; verdicts that don't survive citation validation are downgraded rather than surfaced as confident answers.
- **No unsupported LLM world-knowledge.** The LLM is instructed to reason strictly from retrieved evidence passages, not from its own training-time associations about a topic — reducing the risk of fluent-but-ungrounded answers.
- **Transparent two-tier result language.** Every result tells the user which path produced it — resolved without escalation, evidence-grounded with citations, or an explicit abstention — rather than presenting one undifferentiated "verdict."

---

## TECH STACK

- **Frontend:** React 19 + Vite, Tailwind CSS
- **Backend:** FastAPI (Python), Pydantic for request/response validation
- **Classification:** scikit-learn (SVM / LinearSVC, TF-IDF features)
- **Disagreement signals:** PyTorch (BiLSTM), HuggingFace Transformers (frozen MARBERT linear probe)
- **Novelty / OOD detection:** sentence-transformers (multilingual SBERT) + k-nearest-neighbor distance
- **Evidence retrieval:** rank-bm25 (BM25 lexical), sentence-transformers (dense retrieval), custom Reciprocal Rank Fusion
- **LLM serving:** Ollama (`qwen2.5:1.5b`, development/local CPU) today; vLLM on AMD ROCm (Qwen2.5-7B-Instruct) as the designed production target

---

## KEY METRICS

(Verbatim from `phase5/VALIDATED_METRICS.md` — see that file for sourcing.)

**Classification (SVM, clean leakage-checked split)**

| Metric | Value |
|---|---|
| Accuracy | 86.45% |
| Macro F1 | 86.40% |
| MCC | 0.7281 |

**Model agreement**

| Condition | Share of test rows | Accuracy |
|---|---|---|
| All 3 models agree | 67% | 92.6% |
| 2-of-3 agreement (split vote) | 33% | 64.9% |

**Router / selective prediction** — at the selected 80% routing target, the trusted subset achieved **92.65% selective accuracy at 76.90% coverage**. (This describes only the ~77% of inputs the router accepted without escalation — not an overall system accuracy figure.)

**Retrieval** (measured only on the VERA-ARAB-sourced subset of the test corpus where reusable in-corpus gold evidence exists — see Limitations)

| Method | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.076 | 0.140 | 0.0618 |
| Dense (multilingual SBERT) | 0.092 | 0.144 | 0.0596 |
| Hybrid-RRF (k=60) — selected | 0.108 | 0.202 | 0.0787 |

**Testing** — 63/63 tests passing (44 Phase 1-3 unittest + 19 Phase 4 pytest), plus 3 gated slow end-to-end tests independently run and confirmed passing.

**AMD status** — AMD production deployment prepared; final MI300X validation pending cloud-credit activation.

---

## LIMITATIONS

- AMD execution has not occurred yet — the production deployment path is designed and ready, but final MI300X validation is pending cloud-credit activation.
- The retrieval metrics above are measured only on the VERA-ARAB subset of the test corpus where reusable gold evidence exists in-corpus; the other source corpora have no reusable gold evidence by construction, so retrieval quality is not measured or claimed for them.
- End-to-end LLM verification has only been exercised against the development model (`qwen2.5:1.5b`), not the production target (Qwen2.5-7B-Instruct); results have not been re-measured on the larger model.
- Some escalated claims legitimately return no usable retrieved evidence given the corpus's current coverage, which is reflected honestly in the system's abstention rate rather than hidden.

---

## FUTURE WORK

- Run the real AMD MI300X / ROCm / vLLM validation once cloud credits activate, and publish genuine AMD latency/throughput numbers.
- Re-run deep-verification evaluation against the production LLM target (Qwen2.5-7B-Instruct) instead of the development stand-in.
- Expand retrieval ground truth beyond the current VERA-ARAB subset so retrieval quality can be measured across all source corpora.
- Extend input modalities beyond text, per the architecture's documented extensibility.
