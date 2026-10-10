"""
Lab runner — executes the full experiment pipeline in order.
Logs every step to results/lab_run_log.txt and continues past failures,
so one broken script never kills the whole overnight run.

Usage (after merging test data and updating config.py):
    python run_all_lab.py
"""
import os, sys, subprocess, time
from datetime import datetime

ROOT     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(ROOT, "results", "lab_run_log.txt")

# (script, description) — in execution order
STEPS = [
    ("baselines/baseline_svm.py",            "SVM baseline"),
    ("baselines/baseline_lr_claim_eval.py",  "Logistic Regression baseline"),
    ("baselines/cross_validation.py",        "5-fold CV (SVM + LR)"),
    ("baselines/baseline_bilstm.py",         "BiLSTM baseline"),
    ("pipeline/standard_rag_experiment.py",  "Standard RAG (BM25)"),
    ("baselines/baseline_arabert.py",        "AraBERT before fine-tuning only (~5 min, no training)"),
    ("baselines/baseline_marbert.py",        "MARBERT before/after fine-tune (GPU, ~2-3 hrs)"),
    ("pipeline/faav_pipeline_full.py",       "FAAV-Full pipeline (GPU, ~2-3 hrs)"),
    # ablation_study.py skipped — 7x MARBERT training adds 14-20 hrs
    ("eval/error_analysis.py",               "Error analysis (6 failure modes)"),
    ("graphs/generate_all_graphs.py",        "Generate all 11 graphs"),
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log("=" * 60)
    log("LAB RUN STARTED")
    log("=" * 60)

    results = []
    for script, desc in STEPS:
        path = os.path.join(ROOT, script)
        if not os.path.exists(path):
            log(f"SKIP  {desc} — file not found: {script}")
            results.append((desc, "SKIPPED", 0))
            continue

        log(f"START {desc}  ({script})")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, path],
            cwd=ROOT, capture_output=True, text=True,
        )
        mins = (time.time() - t0) / 60

        if proc.returncode == 0:
            log(f"OK    {desc}  ({mins:.1f} min)")
            results.append((desc, "OK", mins))
        else:
            log(f"FAIL  {desc}  ({mins:.1f} min) — exit {proc.returncode}")
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            log(f"      Last output:\n{tail}")
            results.append((desc, "FAILED", mins))

    log("=" * 60)
    log("LAB RUN FINISHED — SUMMARY")
    log("=" * 60)
    total = 0.0
    for desc, status, mins in results:
        total += mins
        log(f"  {status:<8} {desc}  ({mins:.1f} min)")
    log(f"Total wall time: {total/60:.1f} hours")
    failed = [d for d, s, _ in results if s == "FAILED"]
    if failed:
        log(f"ATTENTION: {len(failed)} step(s) failed — re-run them individually: {failed}")
    else:
        log("All steps completed successfully.")


if __name__ == "__main__":
    main()
