"""
SESSION 1 — LAPTOP  (RTX 5070 Ti 12 GB, 31 GB RAM, CUDA 12.8)
Estimated time: ~50–70 minutes.

Steps:
  1. SVM baseline              (~5 min, CPU TF-IDF)
  2. Logistic Regression       (~3 min, CPU TF-IDF)
  3. 5-fold Cross-Validation   (~15-20 min, CPU, capped at 15k rows/class)
  4. AraBERT before fine-tune  (~5 min, GPU inference only — no training)
  5. BiLSTM                    (~10-15 min, GPU)
  6. Standard RAG BM25         (~10 min, CPU)
  7. Dataset + split graphs    (~2 min)

Crash protection:
  - Each script runs in its own subprocess → memory fully released after each step
  - RAM + VRAM reported before every step
  - Pre-flight check runs first — stops if anything is missing

Usage:
    python run_laptop.py
"""
import os, sys, subprocess, time, ctypes
from datetime import datetime

ROOT     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(ROOT, "results", "laptop_run_log.txt")

STEPS = [
    ("baselines/baseline_svm.py",           "SVM baseline"),
    ("baselines/baseline_lr_claim_eval.py", "Logistic Regression"),
    ("baselines/cross_validation.py",       "5-fold CV — SVM + LR"),
    ("baselines/baseline_arabert.py",       "AraBERT before fine-tuning — inference only"),
    ("baselines/baseline_bilstm.py",        "BiLSTM — GPU"),
    ("pipeline/standard_rag_experiment.py", "Standard RAG BM25"),
    ("graphs/generate_all_graphs.py",       "Dataset + split graphs"),
]

REQUIRED_PACKAGES = [
    "pandas", "numpy", "sklearn", "torch",
    "transformers", "rank_bm25", "matplotlib", "seaborn",
    "openpyxl", "sentence_transformers",
]
REQUIRED_FILES = [
    os.path.join(ROOT, "data", "four_dataset_training.csv"),
    os.path.join(ROOT, "data", "uae_test_dataset_final.xlsx"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ram_free_gb():
    class _MEM(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = _MEM(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024**3, m.ullTotalPhys / 1024**3


def vram_free_gb():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / 1024**3, total / 1024**3
    except Exception:
        pass
    return None, None


def report_memory(label=""):
    avail_ram, total_ram = ram_free_gb()
    free_vram, total_vram = vram_free_gb()
    ram_str  = f"RAM {avail_ram:.1f}/{total_ram:.1f} GB free"
    vram_str = f"VRAM {free_vram:.1f}/{total_vram:.1f} GB free" if free_vram is not None else "VRAM N/A"
    log(f"  [{label}] {ram_str}  |  {vram_str}")


def preflight_check():
    log("PRE-FLIGHT CHECK")
    ok = True

    # Packages
    missing_pkg = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkg.append(pkg)
    if missing_pkg:
        log(f"  MISSING packages: {missing_pkg}  --> pip install {' '.join(missing_pkg)}")
        ok = False
    else:
        log(f"  Packages : all {len(REQUIRED_PACKAGES)} present")

    # Data files
    for fpath in REQUIRED_FILES:
        if os.path.exists(fpath):
            mb = os.path.getsize(fpath) / 1024**2
            log(f"  File OK  : {os.path.basename(fpath)}  ({mb:.0f} MB)")
        else:
            log(f"  MISSING  : {fpath}")
            ok = False

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            log(f"  GPU      : {torch.cuda.get_device_name(0)}  "
                f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB VRAM)  CUDA OK")
        else:
            log("  GPU      : no CUDA — will use CPU (slower but safe)")
    except Exception as e:
        log(f"  GPU      : check failed — {e}")

    # Scripts
    missing_scripts = [s for s, _ in STEPS if not os.path.exists(os.path.join(ROOT, s))]
    if missing_scripts:
        log(f"  MISSING scripts: {missing_scripts}")
        ok = False
    else:
        log(f"  Scripts  : all {len(STEPS)} present")

    report_memory("start")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    log("=" * 60)
    log("LAPTOP SESSION STARTED  (RTX 5070 Ti, 31 GB RAM)")
    log("=" * 60)

    if not preflight_check():
        log("ABORT — fix missing items above before running.")
        sys.exit(1)

    log("=" * 60)
    log("Pre-flight passed — starting steps")
    log("=" * 60)

    results = []
    for script, desc in STEPS:
        path = os.path.join(ROOT, script)
        if not os.path.exists(path):
            log(f"SKIP  {desc} — file not found")
            results.append((desc, "SKIPPED", 0))
            continue

        report_memory(f"before: {desc[:30]}")
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
            # Print last 3000 chars of output to help diagnose
            tail = (proc.stderr or proc.stdout or "")[-3000:]
            log(f"       Last output:\n{tail}")
            results.append((desc, "FAILED", mins))

        report_memory(f"after:  {desc[:30]}")
        log("")   # blank line between steps

    log("=" * 60)
    log("LAPTOP SESSION FINISHED")
    log("=" * 60)
    total = 0.0
    for desc, status, mins in results:
        total += mins
        log(f"  {status:<8} {desc}  ({mins:.1f} min)")
    log(f"Total wall time: {total:.1f} min")

    failed = [d for d, s, _ in results if s == "FAILED"]
    if failed:
        log(f"ATTENTION: {len(failed)} step(s) failed — re-run individually: {failed}")
    else:
        log("All laptop steps completed successfully.")

    log("")
    log("NEXT STEPS (run simultaneously):")
    log("  Kaggle : python run_kaggle.py  — MARBERT only (~2.5-3 hrs)")
    log("  Server : copy results/ there, run run_server.py overnight")


if __name__ == "__main__":
    main()
