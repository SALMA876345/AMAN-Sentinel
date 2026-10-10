"""
REPAIR RUN — re-run only the 3 steps that failed or were not reached in Session 1.

  1. AraBERT before fine-tuning  (was fixed: Dataset import bug removed)
  2. Standard RAG BM25           (was not completed)
  3. All graphs                  (was not reached)

SVM, LR, 5-fold CV, BiLSTM results are already valid — not repeated.

Usage:
    python run_repair.py
"""
import os, sys, subprocess, time
from datetime import datetime

ROOT     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(ROOT, "results", "laptop_run_log.txt")

STEPS = [
    ("baselines/baseline_arabert.py",      "AraBERT before fine-tuning — inference only (~5 min)"),
    ("pipeline/standard_rag_experiment.py","Standard RAG BM25 (~10-20 min)"),
    ("graphs/generate_all_graphs.py",      "All graphs (~5 min)"),
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("=" * 60)
    log("REPAIR RUN — AraBERT + Standard RAG + Graphs")
    log("=" * 60)

    results = []
    for script, desc in STEPS:
        path = os.path.join(ROOT, script)
        if not os.path.exists(path):
            log(f"SKIP  {desc} — file not found")
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
            tail = (proc.stderr or proc.stdout or "")[-3000:]
            log(f"       Last output:\n{tail}")
            results.append((desc, "FAILED", mins))

        log("")

    log("=" * 60)
    log("REPAIR RUN FINISHED")
    log("=" * 60)
    total = 0.0
    for desc, status, mins in results:
        total += mins
        log(f"  {status:<8} {desc}  ({mins:.1f} min)")
    log(f"Total: {total:.1f} min")

    failed = [d for d, s, _ in results if s == "FAILED"]
    if failed:
        log(f"ATTENTION: {len(failed)} step(s) still failed: {failed}")
    else:
        log("All repair steps passed — Session 1 is now complete.")
        log("Next: run run_extra_laptop.py for SBERT-RAG + FAAV-Light (~15 min)")


if __name__ == "__main__":
    main()
