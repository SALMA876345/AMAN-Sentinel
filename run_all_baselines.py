"""
Run ALL laptop models in BOTH 2-class and 3-class modes.

For each (mode, script) the script is launched in its own process with the
CLASS_MODE environment variable set, so each writes class-tagged output files
(e.g. svm_metrics_2class.txt, svm_metrics_3class.txt).

Models (claim-only baselines + evidence-based RAG):
    SVM, Logistic Regression, BiLSTM, AraBERT-before,
    Standard RAG, SBERT-RAG, FAAV-Light

MARBERT and FAAV-Full run on Kaggle (kaggle_session2.ipynb).

Usage:
    python run_all_baselines.py            # both modes, all scripts
    python run_all_baselines.py 2          # 2-class only
    python run_all_baselines.py 3          # 3-class only
"""
import os, sys, subprocess, time
from datetime import datetime

ROOT     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(ROOT, "results", "all_baselines_log.txt")

SCRIPTS = [
    ("baselines/baseline_svm.py",                     "SVM"),
    ("baselines/baseline_lr_claim_eval.py",           "Logistic Regression"),
    # BiLSTM moved to Kaggle (laptop CUDA/triton crashes) — see kaggle_session2.ipynb
    ("baselines/baseline_arabert.py",                 "AraBERT (before fine-tune)"),
    ("pipeline/standard_rag_experiment.py",           "Standard RAG"),
    ("pipeline/sbert_rag_experiment.py",              "SBERT-RAG"),
    ("pipeline/faithfulness_aware_rag_experiment.py", "FAAV-Light"),
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    modes = sys.argv[1:] or ["2", "3"]
    log("=" * 60)
    log(f"RUN ALL BASELINES — modes={modes}")
    log("=" * 60)

    results = []
    for mode in modes:
        env = dict(os.environ, CLASS_MODE=mode, PYTHONIOENCODING="utf-8")
        log(f"\n########## CLASS_MODE = {mode} ##########")
        for script, name in SCRIPTS:
            path = os.path.join(ROOT, script)
            if not os.path.exists(path):
                log(f"SKIP  [{mode}c] {name} — not found")
                continue
            log(f"START [{mode}c] {name}")
            t0 = time.time()
            proc = subprocess.run([sys.executable, path], cwd=ROOT, env=env,
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            mins = (time.time() - t0) / 60
            if proc.returncode == 0:
                # pull the Accuracy / Macro F1 line from stdout if present
                acc_line = ""
                for ln in (proc.stdout or "").splitlines():
                    if "Accuracy=" in ln or ln.strip().startswith("Accuracy"):
                        acc_line = ln.strip(); break
                log(f"OK    [{mode}c] {name}  ({mins:.1f} min)  {acc_line}")
                results.append((mode, name, "OK", mins))
            else:
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                log(f"FAIL  [{mode}c] {name}  ({mins:.1f} min)\n{tail}")
                results.append((mode, name, "FAILED", mins))

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for mode, name, status, mins in results:
        log(f"  [{mode}c] {status:<7} {name}  ({mins:.1f} min)")
    failed = [(m, n) for m, n, s, _ in results if s == "FAILED"]
    if failed:
        log(f"\nATTENTION — failed: {failed}")
    else:
        log("\nAll baselines completed in all requested modes.")
    log("Next: python make_comparison.py  to build the side-by-side table.")


if __name__ == "__main__":
    main()
