# AMAN Sentinel — Demo Shot List

Pairs with `phase5/DEMO_SCRIPT.md`. Specifies exactly what must be visible on screen at each timestamp so recording doesn't require guessing. All UI element names reference real components: `frontend/src/components/LandingScreen.jsx`, `ResultDashboard.jsx`, `StatusBadge.jsx`, `ConfidenceMeter.jsx`, `NoveltyBadge.jsx`, `EvidenceList.jsx`, `AnalysisProgress.jsx`, `TransparencyModal.jsx`. Run the frontend against the real backend (`VITE_API_URL=http://localhost:8000`, not the mock fixture) for every shot — the mock fixture must never appear in the final recording.

---

**0:00-0:15 — Problem**
- Screen: a simple title card or the AMAN landing screen with no input yet. Optionally overlay 1-2 lines of example misinformation-style Arabic text (static, not yet submitted) to set the scene visually.
- No UI interaction yet.

**0:15-0:35 — Introduce AMAN**
- Screen: `LandingScreen.jsx` — AMAN Sentinel logo/header (`Header.jsx`), the single text-input box, the language toggle (ar/en), and the three demo-example quick-select buttons/tags (`frontend/src/data/demoExamples.js`: "مسار سريع / Fast path", "تصعيد + أدلة / Escalation + evidence", "أدلة غير كافية / Insufficient evidence").
- Slow pan/zoom across the input box and demo-example chips — no claim submitted yet.

**0:35-0:55 — Architecture in brief**
- Screen: cut to the `ARCHITECTURE.md` diagram (or a slide built from it) showing: Preprocessing → Intelligence layer (SVM + BiLSTM + MARBERT + novelty) → Decision Router → Path A / Path B split → Evidence retrieval → Grounded LLM → Safeguards → Verdict.
- Highlight the fork point (trust score vs. threshold) as the narration says "high trust / low trust."

**0:55-1:20 — Demo #1: high-confidence, no escalation**
- Click the "Fast path" demo chip, which fills the input with the Drogba/Islam claim (`الاسطوره الايفواريه ديديه دروجبا يعتنق الإسلام...`), or type/paste it.
- Submit. Show `AnalysisProgress.jsx` briefly cycling through its stages (`analyzing_language`, `running_classifiers`, `checking_confidence`, `checking_novelty`, `preparing_result`) — this should resolve in roughly 8-10 seconds (per `AMAN_PHASE4_PRODUCTIZATION_REPORT.md` §10), so no cut/speed-up needed here.
- Result Dashboard (`ResultDashboard.jsx`) showing:
  - `StatusBadge` — amber "LIKELY MISLEADING" badge with the ⚡ icon, sub-label "Quick classifier check (no evidence)" ("فحص تصنيف سريع (دون أدلة)" in Arabic).
  - `ConfidenceMeter` — the SVM's confidence value for this call.
  - Model-agreement indicator — "ALL_AGREE" (all 3 classifiers voted the same way).
  - `NoveltyBadge` — whatever novelty category this claim actually scores (familiar/moderately novel/highly novel — read the real value off screen, don't pre-script it).
  - Explanation text: "...no deep verification needed."
  - `EvidenceList` area empty/absent — no evidence was retrieved, no LLM was called.

**1:20-1:50 — Demo #2: escalation with evidence**
- Click the "Escalation + evidence" demo chip (oil-to-honey factory claim: `شاهد: ضبط مصنع يحول زيت السيارات إلى عسل`).
- Submit. Show `AnalysisProgress.jsx` advancing through the additional stages this path adds: `retrieving_evidence`, `deep_verification`. This call is genuinely slow (~95s per the verified Phase 4 run) — cut to a sped-up/timelapse version of the progress indicator rather than showing real-time dead air; narration continues live over the timelapse.
- Result Dashboard showing:
  - `StatusBadge` — green "SUPPORTED" badge with the 🛡 icon, sub-label "Evidence-grounded verification" ("تحقق مستند إلى أدلة حقيقية").
  - `EvidenceList` populated with the retrieved passages: each card showing source, retrieval method tag (`hybrid_rrf`), and retrieval score.
  - Explanation text citing the evidence.
  - Classification shown as `Real`, matching the dataset's true label for this row.

**1:50-2:10 — Demo #3: insufficient evidence, abstention**
- Click the "Insufficient evidence" demo chip (stadium-flooding claim: `استاد برج العرب غرق بمياه الامطار`).
- Submit. Same escalation-path progress indicator; this run took ~55s in the verified Phase 4 run — again cut to a sped-up timelapse.
- Result Dashboard showing:
  - `StatusBadge` — purple "INSUFFICIENT EVIDENCE" badge with the 🛡 icon, same "Evidence-grounded verification" sub-label (still evidence-grounded tier, just an abstention verdict).
  - `EvidenceList` still populated (retrieval succeeded) — visually contrast this with demo #2 to make clear retrieval worked but the verdict was still withheld.
  - Explanation text referencing the safeguard (model's answer didn't survive the citation check) — read the real explanation text on screen rather than narrating a paraphrase that doesn't match.
  - No `classification` shown (null) — point this out on screen if the UI renders it as blank/dash.

**2:10-2:30 — AMD architecture: prepared, not run**
- Screen: cut to a slide/diagram (from `ARCHITECTURE.md`'s "AMD PRODUCTION TARGET — PENDING FINAL RUN" block) showing: today's box (qwen2.5:1.5b via Ollama, CPU, labeled "VALIDATED LOCALLY") next to the production-target box (Qwen2.5-7B-Instruct via vLLM on AMD Instinct MI300X/ROCm, labeled "AMD PRODUCTION TARGET — PENDING FINAL RUN").
- Optionally show a snippet of `backend/.env.example`'s `LLM_PROVIDER` block to visually back up "one environment variable switches the provider."
- Do not show any live AMD dashboard, GPU utilization graph, or benchmark number here — none exists. Keep this shot purely diagrammatic/code-based.

**2:30-end — Impact and closing**
- Screen: return to the AMAN Sentinel landing screen or logo, optionally with the `TransparencyModal.jsx` open in the background showing the validated-metrics table (accuracy, model agreement, selective accuracy, retrieval Recall@5/10/MRR) as a visual trust signal while the closing line plays.
- End card: project name, team placeholder, hackathon placeholder (both to be filled in from `README.md`'s Team/Hackathon sections before final export).

---

## General recording notes

- Always run the frontend against the live backend (`VITE_API_URL` set) — never record against the mock fixture (`frontend/src/mockData.js`); a judge who reads `frontend/src/api.js` will know the difference, and the whole pitch is that these are real pipeline calls.
- Do not pre-fill or hard-code any result shown — let the live API call produce it, exactly as `phase4/demo_examples.json`'s own comment insists ("the actual verdict for each is produced live by the backend at request time").
- If a demo's timing or exact output drifts slightly from the numbers quoted in this file or in `DEMO_SCRIPT.md` on a given recording run (e.g. a slightly different confidence value), narrate what's actually on screen — these documents describe the previously-verified real run, not a scripted fake one.
