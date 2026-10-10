"""
LAPTOP — EXTRA MODELS (run immediately after run_laptop.py finishes)
Estimated time: ~10–20 minutes total.

Steps:
  1. FAAV-Light (rule-based)   (~5 min  — keyword faithfulness, no training)
  2. SBERT-RAG (dense retrieval) (~5-15 min — MiniLM encoding, GPU if available)

Why these matter:
  FAAV-Light  — rule-based faithfulness scoring. Compared against FAAV-Full
                (neural) in error analysis to show the value of the neural step.
  SBERT-RAG   — dense semantic retrieval vs BM25 keyword retrieval.
                Reveals whether semantic matching helps on dialectal Arabic.

Both are listed in error_analysis.py MODEL_REGISTRY. Without their result
files, the error analysis comparison table will be incomplete.

Usage (run after run_laptop.py finishes):
    python run_extra_laptop.py
"""
import os, sys, subprocess, time, ctypes
from datetime import datetime

ROOT     = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(ROOT, "results", "laptop_run_log.txt")

STEPS = [
    ("pipeline/faithfulness_aware_rag_experiment.py", "FAAV-Light rule-based (~5 min)"),
    ("pipeline/sbert_rag_experiment.py",              "SBERT-RAG dense retrieval (~5-15 min)"),
]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ram_free_gb():
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
        return 0.0, 0.0


def report_memory(label=""):
    avail, total = ram_free_gb()
    try:
        import torch
        if torch.cuda.is_available():
            free_v, total_v = torch.cuda.mem_get_info(0)
            vram_str = f"VRAM {free_v/1024**3:.1f}/{total_v/1024**3:.1f} GB free"
        else:
            vram_str = "VRAM N/A"
    except Exception:
        vram_str = "VRAM N/A"
    log(f"  [{label}] RAM {avail:.1f}/{total:.1f} GB free  |  {vram_str}")


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log("=" * 60)
    log("EXTRA MODELS STARTED  (FAAV-Light + SBERT-RAG)")
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
    log("EXTRA MODELS FINISHED")
    log("=" * 60)
    total = sum(m for _, _, m in results)
    for desc, status, mins in results:
        log(f"  {status:<8} {desc}  ({mins:.1f} min)")
    log(f"Total: {total:.1f} min")

    failed = [d for d, s, _ in results if s == "FAILED"]
    if failed:
        log(f"ATTENTION: failed — {failed}")
    else:
        log("Both extra models completed. Error analysis will now include all 9 models.")


if __name__ == "__main__":
    main()
