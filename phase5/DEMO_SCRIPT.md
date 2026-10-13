# AMAN Sentinel — Demo Video Narration Script

Target length: ~2:30-2:45. Uses the three real demo rows from `phase4/demo_examples.json` (sourced from `hackathon/data/clean_split/source_binary_test.csv`) — nothing in this script is invented or hard-coded; every result named below is what the live pipeline actually produced, per `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §10. Numbers cited come from `phase5/VALIDATED_METRICS.md`; see that file for the exact mandatory framing if quoting any metric aloud.

Companion file: `phase5/DEMO_SHOT_LIST.md` — exactly what to have on screen at each timestamp.

---

**0:00-0:15 — The problem**

"Arabic misinformation spreads fast, and it's hard to verify quickly or reliably — especially when the only check available is 'ask an AI and trust whatever it says.' A raw LLM answer can sound confident while citing nothing real, or citing something that was never actually retrieved."

**0:15-0:35 — Introduce AMAN**

"This is AMAN Sentinel — an Arabic-first AI verification agent. It answers one question for a piece of text: can I trust what I just received? And it answers that honestly — it doesn't blindly send every claim to an LLM. It checks cheaply first, and only escalates to real evidence retrieval and grounded LLM reasoning when it's genuinely uncertain."

**0:35-0:55 — Architecture in brief**

"Every input goes through a classifier ensemble — an SVM, a BiLSTM, and a MARBERT probe — plus a novelty check for how unfamiliar the claim is. Their confidence, agreement, and novelty combine into a trust score. High trust: AMAN answers immediately, cheaply, no LLM call. Low trust: AMAN escalates — retrieves real evidence with a hybrid BM25-plus-dense search, and only then asks an LLM to reason, strictly grounded in what was retrieved."

**0:55-1:20 — Demo #1: high confidence, no escalation**

"Here's a real claim from our test set: a rumor that footballer Didier Drogba converted to Islam. Watch what happens." *[submit claim]* "All three models agree — this is Fake — with high confidence. AMAN returns `VERIFIED_AUTOMATICALLY` in seconds. No LLM call happened here at all — and that's deliberate. When the cheap signals already agree confidently, spending an expensive LLM reasoning step would be wasted cost with no benefit. AMAN avoided it on purpose."

**1:20-1:50 — Demo #2: uncertain claim, escalation**

"Now a harder one: a claim about a factory that supposedly converts car oil into honey." *[submit claim]* "This time the models disagree, or aren't confident enough — so AMAN escalates. It runs hybrid retrieval — BM25 plus dense embeddings, fused with Reciprocal Rank Fusion — to pull real evidence passages, then hands those passages to an LLM that is only allowed to reason from what was actually retrieved. In our verified run, this came back `SUPPORTED`, grounded in retrieved evidence and citing it — not guessed from the model's own prior knowledge."

**1:50-2:10 — Demo #3: insufficient evidence, responsible abstention**

"One more: a claim that a stadium flooded during a rainstorm." *[submit claim]* "Retrieval succeeds here too — but the LLM's answer doesn't survive AMAN's citation safeguard, so instead of forcing a guess, AMAN returns `INSUFFICIENT_EVIDENCE`. That's not a failure — that's the system working as designed. AMAN would rather say 'I don't know' than manufacture false confidence."

**2:10-2:30 — AMD architecture: prepared, not yet run**

"Today's LLM step runs a small development model locally. The production design targets Qwen2.5-7B-Instruct served through vLLM on AMD Instinct MI300X GPUs over ROCm — and the code is ready for that today: a single environment variable switches the provider, with no pipeline changes needed. To be precise: that AMD run has not happened yet. This development environment has no AMD hardware or cloud credentials. AMD production deployment is prepared; final MI300X validation is pending cloud-credit activation."

**2:30-end — Impact and closing**

"AMAN Sentinel's whole point is trustworthy triage: don't pay for expensive reasoning you don't need, do pay for it — and ground it strictly in real evidence — when it matters, and abstain rather than fabricate when the evidence genuinely isn't there. That's AMAN Sentinel: an Arabic-first verification agent that knows the difference between a quick read and a sourced fact-check — and tells you which one you're getting."

---

**Total runtime target:** ~2:30-2:45, allowing natural pacing around the three live demo calls (demo #1 is fast, ~8-10s; demos #2 and #3 are genuinely slow — 55-95s — because of the real LLM calls; consider a timelapse/speed-up cut during the escalation waits rather than narrating in real time, see `phase5/DEMO_SHOT_LIST.md`).
