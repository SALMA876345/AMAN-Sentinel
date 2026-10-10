"""
SESSION 3 — UNIVERSITY SERVER  (best GPU, no time limit — leave overnight)
Estimated time: ~7–14 hours depending on GPU type.

Steps:
  1. Ablation study         (~5-6 hrs on A100 / ~10-11 hrs on V100)
  2. Error analysis         (~5-10 min — needs ALL prior results present)
  3. All 11 graphs          (~5 min   — needs ALL prior results present)

FAAV-Full moved to Kaggle session — server is ablation only.

Ablation study — 7 MARBERT configurations, each trained from scratch:
  1. Full MARBERT            reference — complete system
  2. No Class Weights        does balancing Real/Fake/Unverified matter?
  3. No Unverified Train     does the 3rd class during training help?
  4. Claim Only              does BM25-retrieved evidence add value?
  5. Binary Only             is 3-class harder than Real vs Fake only?
  6. Half Training Data      how much does data volume matter?
  7. No Arabic Normalization does Arabic text cleaning (hamza/taa) matter?

BEFORE RUNNING
──────────────
Copy these result files from the laptop and Kaggle sessions first:

  From laptop results/ :
    svm_claim_results.csv         svm_metrics.txt
    lr_claim_results.csv          lr_claim_metrics.txt
    cross_validation_results.csv  cross_validation_metrics.txt
    arabert_before_finetune_results.csv  arabert_before_finetune_metrics.txt
    bilstm_claim_results.csv      bilstm_metrics.txt
    standard_rag_results.csv      standard_rag_metrics.txt

  From Kaggle results/ :
    marbert_claim_results.csv     marbert_metrics.txt
    marbert_before_finetune_results.csv  marbert_before_finetune_metrics.txt
    marbert_training_loss.csv
    faav_full_results.csv         faav_full_metrics.txt

Then run:
    python run_server.py
"""
import os, sys, subprocess, time
from datetime import datetime

ROOT     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(ROOT, "results", "server_run_log.txt")

STEPS = [
    ("pipeline/ablation_study.py",    "Ablation — 7 MARBERT configs (GPU, ~5-11 hrs)"),
    ("eval/error_analysis.py",        "Error analysis — 6 failure modes (~10 min)"),
    ("graphs/generate_all_graphs.py", "All 11 graphs (~5 min)"),
]

EXPECTED_FROM_LAPTOP = [
    "svm_metrics.txt",
    "lr_claim_metrics.txt",
    "cross_validation_metrics.txt",
    "arabert_before_finetune_metrics.txt",
    "bilstm_metrics.txt",
    "standard_rag_metrics.txt",
    "faithfulness_aware_rag_metrics.txt",   # FAAV-Light  (run_extra_laptop.py)
    "sbert_rag_metrics.txt",                # SBERT-RAG   (run_extra_laptop.py)
]
EXPECTED_FROM_KAGGLE = [
    "marbert_metrics.txt",
    "marbert_before_finetune_metrics.txt",
    "marbert_training_loss.csv",
    "faav_full_metrics.txt",
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_prior_results():
    missing_laptop = [f for f in EXPECTED_FROM_LAPTOP
                      if not os.path.exists(os.path.join(ROOT, "results", f))]
    missing_kaggle = [f for f in EXPECTED_FROM_KAGGLE
                      if not os.path.exists(os.path.join(ROOT, "results", f))]
    return missing_laptop, missing_kaggle


def main():
    log("=" * 60)
    log("SERVER SESSION STARTED — FAAV + ABLATION (overnight run)")
    log("=" * 60)

    missing_l, missing_k = check_prior_results()
    if missing_l:
        log("WARNING: missing laptop results — copy from laptop first:")
        for f in missing_l:
            log(f"   MISSING -> results/{f}")
    if missing_k:
        log("WARNING: missing Kaggle results — copy from Kaggle first:")
        for f in missing_k:
            log(f"   MISSING -> results/{f}")
    if not missing_l and not missing_k:
        log("Prior results check passed — all laptop + Kaggle files present.")
    log("")

    results = []
    for script, desc in STEPS:
        path = os.path.join(ROOT, script)
        if not os.path.exists(path):
            log(f"SKIP  {desc} — file not found: {script}")
            results.append((desc, "SKIPPED", 0))
            continue

        log(f"START  {desc}")
        t0   = time.time()
        proc = subprocess.run(
            [sys.executable, path],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        mins = (time.time() - t0) / 60

        if proc.returncode == 0:
            log(f"OK     {desc}  ({mins:.1f} min)")
            results.append((desc, "OK", mins))
        else:
            log(f"FAIL   {desc}  ({mins:.1f} min) — exit {proc.returncode}")
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            log(f"       Last output:\n{tail}")
            results.append((desc, "FAILED", mins))

    log("=" * 60)
    log("SERVER SESSION FINISHED — ALL THREE SESSIONS COMPLETE")
    log("=" * 60)
    total = 0.0
    for desc, status, mins in results:
        total += mins
        log(f"  {status:<8} {desc}  ({mins:.1f} min)")
    log(f"Total server wall time: {total/60:.1f} hours")

    failed = [d for d, s, _ in results if s == "FAILED"]
    if failed:
        log(f"ATTENTION: {len(failed)} step(s) failed: {failed}")
    else:
        log("All server steps completed successfully.")

    log("")
    log("FINAL OUTPUTS:")
    log("  results/faav_full_metrics.txt      — FAAV-Full all metrics")
    log("  results/ablation_results.csv       — 7-config ablation table")
    log("  results/ablation_metrics.txt       — ablation metrics summary")
    log("  results/error_analysis_report.txt  — 6 failure modes")
    log("  graphs/*.png                       — all 11 graphs (300 DPI)")


if __name__ == "__main__":
    main()
