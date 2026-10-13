# AMAN Sentinel — Security & Submission-Readiness Audit

Branch: `aman-hackathon`
Scope: all 270 files returned by `git ls-files` (tracked files only; gitignored/untracked bulk
data directories were deliberately not scanned — see item 5).
Method: read-only. No files were modified, deleted, or created except this report.

Severity scale: CRITICAL / HIGH / MEDIUM / LOW / INFO.

---

## 1. Secrets / credentials / API keys / tokens / passwords

**Verdict: no issues found.**

Commands run (all against `git grep`, i.e. tracked files only):

```
git grep -nIE "(api[_-]?key|secret|token|password|passwd|bearer|aws_access_key|aws_secret|private[_-]?key)" -- .
git grep -nIE "AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-|AIza[0-9A-Za-z_-]{30,}|-----BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY-----" -- .
git grep -nIE "(api_key|apikey|secret_key|access_token|auth_token|password)\s*[:=]\s*['\"][A-Za-z0-9+/_\-]{12,}['\"]" -- .
```

- The generic keyword search only matched `tokenizer`/`token_type_ids`/ML-vocabulary usages
  (e.g. `baselines/baseline_marbert.py`, `experiments/final_studies_neural.py`) and one literal
  wordpiece vocabulary entry `"secret"` inside
  `artifacts/transformers/marbert_clean_binary/vocab.txt` / `tokenizer.json` (a BERT subword —
  not a credential).
- The high-signal pattern search (AWS key prefixes, OpenAI/GitHub/Slack/Google key prefixes, PEM
  private-key headers, `key/secret/token/password = "<long string>"` assignments) returned **zero
  matches** anywhere in the tracked tree.
- `backend/.env.example` (`backend/.env.example:1-46`) contains only documentation and
  explicit placeholders: `VLLM_ENDPOINT=http://REPLACE-WITH-REAL-ROCM-HOST:8000` and
  `VLLM_API_KEY=replace-with-a-real-api-key-in-your-local-.env-only`, with an explicit comment
  "NEVER put a real key in this checked-in .env.example file." Confirmed clean.
- `frontend/.env.example` (4 lines) contains only `VITE_API_URL=http://localhost:8000` and
  comments. Confirmed clean.
- No real `.env` file is tracked (`git ls-files | grep -iE "\.env($|\.[^e])"` → empty); only the
  two `.env.example` files are tracked.
- No private key files (`.pem`, `.key`, `id_rsa`, etc.) are tracked.

## 2. Personal information

**Verdict: no issues found beyond expected/legitimate occurrences.**

- `README.md:5` and `improvements/REPORT_CORRECTIONS.md:3` legitimately name the project author,
  "Hamza Alqaruoti," in a byline/report-correction context (course name: AI503 Machine Learning,
  Assignment 2). This matches the task's explicit carve-out for author bylines.
- `git grep -nIoE "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}" -- .` found exactly two distinct
  real-looking email addresses, both inside a dataset artifact, not developer-authored code:
  `artifacts/svm/clean_test_predictions.csv:25560` (`alarab@alarab.com`) and
  `artifacts/svm/clean_test_predictions.csv:25594,25596` (`contact@jtc.tn`). These are generic
  news-outlet contact addresses embedded in the original Arabic misinformation-dataset claim
  text being classified, not personal/developer emails or secrets. No action needed.
- No emails at all were found inside any `.md` file (`git grep` over `*.md` returned empty).
- No phone-number-like patterns were found (a regex sweep over `.md/.py/.js/.jsx/.json` only
  matched floating-point metric values such as `0.8011435446237605`, not phone numbers).

## 3. Absolute local paths hardcoded in source

**Verdict: HIGH / MEDIUM — several hardcoded absolute Windows paths exist in tracked `.py` files,
beyond the single known historical one the task flagged.** None are in `backend/` or `frontend/`
(the core AMAN hackathon demo path audits clean); all are in historical FAAV data-prep / report-
generation tooling.

| Severity | File : line | Content | Note |
|---|---|---|---|
| LOW (known, out of scope) | `utils/prepare_uae_test_dataset.py:34` | `SOURCE_CSV = r"c:\Users\Hamza\social_dataset_agent\datasets\UAE-MVD-merged.csv"` | The historical example named explicitly in the task brief. Part of the original FAAV research project, not AMAN. Left untouched per instructions. |
| MEDIUM (new finding) | `utils/prepare_test_dataset.py:23` (also referenced in its docstring, line 4) | `SRC = r"c:\Users\Hamza\social_dataset_agent\datasets\UAE-MVD-test-final.csv"` | Same category/pattern as the known issue above — a sibling script in `utils/` with the identical class of non-portable, developer-machine-specific path. Would break for anyone else running this script. Also historical FAAV tooling, not AMAN backend/frontend. |
| MEDIUM (new finding) | `improvements/build_report_ieee.py:34-38` | `SCRATCH = Path(r"C:\Users\Hamza\AppData\Local\Temp\claude\c--Users-Hamza-Downloads-Master-Study-503-Machine-Learning-Assignment-2\cf201820-c096-41e6-81b7-35175c70970f\scratchpad\ieee_media")` | A hardcoded absolute path into a *previous Claude Code session's temp scratchpad* (session UUID + a different assignment's folder name baked in), actually used at `build_report_ieee.py:257` (`SCRATCH.mkdir(...)`) and `:261` (writing extracted media there) — not just a comment. This is a real, used path, not portable to another machine, and leaks an unrelated prior-assignment directory name + session UUID. Low security impact (no secret), but a genuine portability bug and the most "accidentally leftover debugging path" of the three. |
| INFO | `hackathon/restore_four_dataset_training.py:21` | `(c:\\Users\\Hamza\\social_dataset_agent\\datasets\\UAE-MVD-merged.csv) does\nnot exist anywhere in this project tree` | This is inside a docstring, explicitly documenting that the path does **not** exist and the pipeline is "legitimately blocked, not reconstructed." Informational only — not executable, not a bug. |

No hardcoded absolute paths were found in any `backend/*.py`, `frontend/src/**`, `phase2/*.py`,
`phase3/*.py`, or `phase4/*` file (confirmed via `git grep -nIE "[Cc]:\\Users\\"` and
`git grep -nIE "/home/|/Users/"` across `.py/.js/.jsx/.json/.ps1`, restricted to those paths —
no hits).

## 4. Large unnecessary artifacts / temp files / caches / generated junk

**Verdict: LOW — one stray debug-dump file; otherwise clean.**

- No `__pycache__/`, `.pyc`, `.vscode/`, `.idea/`, or editor-junk files are tracked
  (`git ls-files | grep -iE "__pycache__|\.pyc$|\.vscode|\.idea"` → empty). `.gitignore` already
  excludes these.
- **LOW**: `phase3/reports/out.txt` is a small (20-line) raw stdout-style dump of three
  verification examples (`=== CLEAN-011130-... ===`, `VERDICT:`, `EXPLANATION:`, `CITED:`),
  distinct in naming/shape from the other, clearly-labeled files in the same directory
  (`deep_verification_eval_examples.csv`, `llm_smoketest_results.json`,
  `retrieval_method_recommendation.md`). The generic name `out.txt` suggests it was an
  unredirected/forgotten terminal capture rather than an intentionally named report artifact. It
  is small (~1 KB) and contains no secrets or PII, so impact is cosmetic — a judge skimming
  `phase3/reports/` may wonder what it is. Recommend renaming/deleting before final submission,
  but not blocking.
- No other stray `.log`, `.tmp`, or debug-named files were found among tracked files.
- The repo's largest tracked file is `artifacts/novelty/train_reference_embeddings.npy` at
  ~21.5 MB — well under GitHub's 100 MB/file limit, and confirmed **not** junk: it is read by
  `phase2/novelty_detector.py` and referenced in `phase2/reports/ood_corpus_search.md` and
  `phase2/tests/test_phase2.py` as the runtime reference-embedding set the novelty detector
  needs at inference time. Keep as is.
- `phase2/reports/*_test_calibrated.csv` and similar large CSVs (up to ~600 KB) are intentional,
  already-noted-as-fine report/result files per the task brief, not junk.

## 5. Accidental datasets that should not be public

**Verdict: no issues found.**

- `git ls-files | grep -iE "^(data/|results/|Results files/)"` returned **zero** matches — none
  of `data/`, `results/`, or `Results files/` are tracked.
- `.gitignore` lines 2-3 (`data/`, `results/`) correctly exclude these at the repo root.
- No `.zip`, `.safetensors`, `.bin`, or `.pt` model/dataset binaries are tracked (excluded by
  `.gitignore` lines 9-12); the only large tracked binary artifacts are the small, intentional
  JSON/CSV/`.npy` model-metadata/report files under `artifacts/` discussed in items 4 and 6.
- No `node_modules/` or Python virtualenv directories are tracked.

## 6. Broken imports / dead demo files

**Verdict: no issues found.**

- `backend/main.py` was actually imported in this environment (`python -c "import backend.main"`)
  and loaded cleanly with no errors.
- A full static-AST scan of every tracked `.py` file's `import`/`from ... import` statements,
  checked against the complete set of tracked `.py` module paths for all local package roots
  (`backend`, `phase2`, `phase3`, `phase4`, `hackathon`, `utils`, `pipeline`, `improvements`,
  `baselines`, `eval`, `experiments`, `config`), found **zero** unresolved local imports across
  all 270 tracked files.
- `frontend/src/App.jsx`'s import graph was traced by hand and via
  `git grep -nIE "^import " -- frontend/src`: every relative import (`./i18n.jsx`, `./api.js`,
  `./components/Header.jsx`, `./components/LandingScreen.jsx`,
  `./components/AnalysisProgress.jsx`, `./components/ResultDashboard.jsx`,
  `./components/TransparencyModal.jsx`, `./data/demoExamples`, `./mockData`, and each
  component's own imports of `ConfidenceMeter.jsx`/`EvidenceList.jsx`/`NoveltyBadge.jsx`/
  `StatusBadge.jsx`) resolves to a file present in `git ls-files`. No missing imports.
- No orphaned/dead `.py`/`.jsx`/`.js` files were identified under `backend/`, `frontend/`,
  `phase2/`, `phase3/`, or `phase4/`. Root-level standalone orchestration scripts outside those
  directories (`ablation_server.py`, `run_repair.py`, `kaggle_session2.py`) are self-contained,
  intentionally run directly (`python script.py`), not meant to be imported, and are documented
  in their own docstrings as deliberate one-off/repair/server scripts from the historical FAAV
  pipeline — not accidental leftovers.
- `phase4/tests/test_integration.py` correctly imports `backend.pipeline_wrapper` and
  `backend.main`, and reads `phase4/demo_examples.json` (present, tracked) — no broken references.

## 7. `.gitignore` problems

**Verdict: no issues found; the known historical fix is confirmed present and working.**

- Read in full (`.gitignore`, 35 lines). The bare `data/` / `results/` exclusion (lines 2-3) is
  immediately followed by an explicit, commented override for the frontend source directory
  (lines 4-8):
  ```
  !frontend/src/data/
  !frontend/src/data/**
  ```
- Confirmed the override actually works:
  `git check-ignore -v frontend/src/data/demoExamples.js` → **exit code 1, no output**, meaning
  the file is correctly **not** ignored. (It is also already present in `git ls-files`, consistent
  with this.)
- The `.env` exclusion (lines 31-34) is similarly paired with explicit un-ignore rules for the
  example files (`!.env.example`, `!*/.env.example`), and both `backend/.env.example` and
  `frontend/.env.example` are confirmed tracked while no real `.env` file is tracked (see item 1).
- No pattern was found that is suspiciously *too narrow* (e.g. missing `node_modules`,
  `__pycache__`, or venv coverage) — `frontend/.gitignore` separately covers `node_modules`,
  `dist`, logs, and editor files for the frontend subtree, and the root `.gitignore` covers
  `__pycache__/`, `*.pyc`, `.vscode/`, `.idea`-style patterns are not explicitly repeated there but
  no such files are tracked regardless (item 4).
- No other directory-swallowing pattern as broad as the historical `data/` one was found.

## 8. Missing dependencies

**Verdict: HIGH for the backend demo — `fastapi`, `pydantic`, and `uvicorn` are required to run
`backend/main.py` but are absent from `requirements.txt`. A judge who clones the repo and runs
only `pip install -r requirements.txt` cannot start the API server.**

- `requirements.txt` (repo root, 18 lines) lists: `pandas`, `numpy`, `scikit-learn`, `openpyxl`,
  `torch`, `transformers`, `accelerate`, `sentence-transformers`, `rank-bm25`, `matplotlib`,
  `seaborn`. It does **not** list `fastapi`, `pydantic`, `uvicorn`, `joblib`, `scipy`, or `pytest`.
- Confirmed actual imports requiring these, by file:
  - `fastapi` — `backend/main.py:44-46` (`from fastapi import FastAPI, Request`, etc.),
    `backend/tests/test_api.py`, `phase4/tests/test_integration.py:27`
    (`from fastapi.testclient import TestClient`).
  - `pydantic` — `backend/schemas.py:12` (`from pydantic import BaseModel, Field,
    field_validator`), which `backend/main.py` and `backend/pipeline_wrapper.py` both import.
  - `uvicorn` — not imported in Python code (by design — `backend/main.py`'s own docstring says
    to launch it via `uvicorn backend.main:app --host 0.0.0.0 --port 8000`), but it is the
    documented, required way to actually run the server and is not installable via
    `requirements.txt` as it stands.
  - `joblib` — used by `hackathon/evaluate_svm_clean.py`, `hackathon/predict_sanity_check.py`,
    `hackathon/train_svm_clean.py`, `hackathon/tests/test_phase1.py`,
    `phase2/generate_base_predictions.py`, and `phase3/live_inference.py` (the module that
    actually loads the BiLSTM model at inference time, imported transitively by
    `phase3/aman_pipeline.py` which `backend/pipeline_wrapper.py` calls into).
  - `scipy` — used by `phase3/live_inference.py` (`from scipy.special import softmax`,
    on the live inference path the backend depends on), plus
    `experiments/final_studies_classical.py`, `experiments/final_studies_rag.py`,
    `improvements/significance.py`, `phase2/calibration.py`.
  - `pytest` — required to run every test suite under `backend/tests/`, `phase2/tests/`,
    `phase3/tests/`, `hackathon/tests/`, `phase4/tests/`; not declared anywhere.
  - Verified in this sandbox that `fastapi`, `pydantic`, `uvicorn`, `joblib`, and `scipy` are in
    fact installed and that `backend.main` imports successfully — i.e. they were installed ad hoc
    into this dev environment, not via the committed `requirements.txt`, which is exactly the gap
    a fresh clone would hit.
  - `phase3/llm_provider.py` deliberately uses only the stdlib `urllib.request` (not `requests`
    or `httpx`) for its Ollama/vLLM HTTP calls — this is intentional (its own docstring at
    line ~231 says "using only `urllib` to avoid adding [a dependency]") and is **not** a missing
    dependency.
  - `improvements/requirements-pinned.txt` (a separate, pinned-version file for the
    report-generation subproject) also does not list `fastapi`/`pydantic`/`uvicorn`/`joblib`, so
    it does not fill this gap either.
- **Recommendation**: add `fastapi`, `pydantic>=2`, `uvicorn`, `joblib`, `scipy`, and `pytest` to
  `requirements.txt` (or add a `backend/requirements.txt`) before final submission, so a judge's
  clean `pip install -r requirements.txt` + `uvicorn backend.main:app` actually works.
- `frontend/package.json` was checked against every import statement in `frontend/src/**`
  (`git grep -nIE "^import " -- frontend/src`): the only non-relative imports are `react` and
  `react-dom/client`, both declared in `dependencies`. **No missing frontend dependencies.**

---

## Overall summary

| # | Category | Verdict |
|---|---|---|
| 1 | Secrets/credentials | Clean — no issues found |
| 2 | Personal information | Clean — only expected author byline + two dataset-embedded generic org emails |
| 3 | Hardcoded absolute paths | 2 new MEDIUM findings beyond the known one (historical FAAV tooling, not AMAN core) |
| 4 | Junk/artifacts | 1 LOW finding (`phase3/reports/out.txt`) |
| 5 | Accidental datasets | Clean — `.gitignore` confirmed working |
| 6 | Broken imports/dead code | Clean — no issues found |
| 7 | `.gitignore` | Clean — historical fix confirmed present and working |
| 8 | Missing dependencies | **HIGH** — `fastapi`/`pydantic`/`uvicorn`/`joblib`/`scipy` missing from `requirements.txt`, breaking a clean-clone run of the backend demo |

**Submission readiness: needs one fix before release.** No secrets, PII, or data leaks were found,
and the AMAN hackathon core (`backend/`, `frontend/`) is clean on paths/imports/gitignore. The one
blocking item is `requirements.txt` not listing the packages `backend/main.py` actually needs
(item 8) — this should be fixed so judges can run the demo from a clean clone. Everything else is
LOW/MEDIUM cosmetic/historical-tooling findings that do not need to block release.
