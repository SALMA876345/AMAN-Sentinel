"""
SESSION 2 — KAGGLE  (T4 GPU, 9–12 hour session limit)
Estimated time: ~4.5–6 hours.

Steps:
  1. MARBERT before + after fine-tuning  (~2.5-3 hrs on T4)
  2. FAAV-Full pipeline                  (~2-3 hrs on T4)

Both are heavy GPU transformer jobs — ideal for Kaggle running
in parallel while the laptop session finishes.

HOW TO USE ON KAGGLE
─────────────────────
1. Create a Kaggle dataset called "faav-arabic-fake-news" and upload:
     data/four_dataset_training.csv   (284 MB)
     data/uae_test_dataset_final.xlsx  (34 MB)

2. In your Kaggle notebook, attach that dataset. Files appear at:
     /kaggle/input/faav-arabic-fake-news/four_dataset_training.csv
     /kaggle/input/faav-arabic-fake-news/uae_test_dataset_final.xlsx

3. Upload the project scripts as a second dataset, then in a notebook cell:
       !python run_kaggle.py

4. After the session, download from /kaggle/working/results/:
     marbert_*   (5 files)
     faav_full_*  (2 files)
   Copy into the university server results/ folder before run_server.py.

NOTE: Paths are patched automatically for Kaggle — do not edit config.py.
"""
import os, sys, subprocess, time
from datetime import datetime

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()   # Kaggle notebook cell — no __file__
LOG_FILE = os.path.join(ROOT, "results", "kaggle_run_log.txt")

# ── Kaggle path patch ─────────────────────────────────────────────────────────
_ON_KAGGLE = os.path.exists("/kaggle/input")
if _ON_KAGGLE:
    import config as _cfg
    _dataset                     = "/kaggle/input/faav-arabic-fake-news"
    _cfg.FOUR_DATASET_TRAIN_FILE = os.path.join(_dataset, "four_dataset_training.csv")
    _cfg.TRAIN_FILE              = _cfg.FOUR_DATASET_TRAIN_FILE
    _cfg.TEST_FILE               = os.path.join(_dataset, "uae_test_dataset_final.xlsx")
    _cfg.RESULTS_DIR             = "/kaggle/working/results"
    os.makedirs(_cfg.RESULTS_DIR, exist_ok=True)
    print("[Kaggle] Paths patched:")
    print(f"  TRAIN  -> {_cfg.TRAIN_FILE}")
    print(f"  TEST   -> {_cfg.TEST_FILE}")
    print(f"  RESULT -> {_cfg.RESULTS_DIR}")
# ─────────────────────────────────────────────────────────────────────────────

STEPS = [
    ("baselines/baseline_marbert.py",      "MARBERT before + after fine-tune (T4, ~2.5-3 hrs)"),
    ("pipeline/faav_pipeline_full.py",     "FAAV-Full pipeline (T4, ~2-3 hrs)"),
]

REQUIRED_PACKAGES = [
    "pandas", "numpy", "sklearn", "torch",
    "transformers", "rank_bm25", "sentence_transformers",
    "openpyxl",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ram_free_gb():
    """Works on Linux (Kaggle) and Windows."""
    try:
        import ctypes
        class _MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _MEM(); m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1024**3, m.ullTotalPhys / 1024**3
    except Exception:
        pass
    # Linux fallback (Kaggle)
    try:
        with open("/proc/meminfo") as f:
            lines = f.read()
        total = int([l for l in lines.split("\n") if "MemTotal" in l][0].split()[1]) / 1024**2
        avail = int([l for l in lines.split("\n") if "MemAvailable" in l][0].split()[1]) / 1024**2
        return avail, total
    except Exception:
        return 0.0, 0.0


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

    missing_pkg = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkg.append(pkg)
    if missing_pkg:
        log(f"  MISSING packages: {missing_pkg}")
        ok = False
    else:
        log(f"  Packages : all {len(REQUIRED_PACKAGES)} present")

    import config as cfg
    for label, fpath in [("TRAIN", cfg.TRAIN_FILE), ("TEST", cfg.TEST_FILE)]:
        if os.path.exists(fpath):
            mb = os.path.getsize(fpath) / 1024**2
            log(f"  File OK  : {label} — {os.path.basename(fpath)}  ({mb:.0f} MB)")
        else:
            log(f"  MISSING  : {label} — {fpath}")
            ok = False

    try:
        import torch
        if torch.cuda.is_available():
            log(f"  GPU      : {torch.cuda.get_device_name(0)}  "
                f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB VRAM)  CUDA OK")
        else:
            log("  GPU      : no CUDA — CPU only (MARBERT will be very slow!)")
    except Exception as e:
        log(f"  GPU      : check failed — {e}")

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
    log(f"KAGGLE SESSION STARTED  ({'on Kaggle' if _ON_KAGGLE else 'local mode'})")
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
            tail = (proc.stderr or proc.stdout or "")[-3000:]
            log(f"       Last output:\n{tail}")
            results.append((desc, "FAILED", mins))

        report_memory(f"after:  {desc[:30]}")
        log("")

    log("=" * 60)
    log("KAGGLE SESSION FINISHED")
    log("=" * 60)
    total = 0.0
    for desc, status, mins in results:
        total += mins
        log(f"  {status:<8} {desc}  ({mins:.1f} min)")
    log(f"Total wall time: {total/60:.1f} hours")

    failed = [d for d, s, _ in results if s == "FAILED"]
    if failed:
        log(f"ATTENTION: {len(failed)} step(s) failed: {failed}")
    else:
        log("All Kaggle steps completed successfully.")

    log("")
    if _ON_KAGGLE:
        log("DOWNLOAD from /kaggle/working/results/ and copy to server:")
        log("  marbert_claim_results.csv         marbert_metrics.txt")
        log("  marbert_before_finetune_results.csv  marbert_before_finetune_metrics.txt")
        log("  marbert_training_loss.csv")
        log("  faav_full_results.csv             faav_full_metrics.txt")


if __name__ == "__main__":
    main()
