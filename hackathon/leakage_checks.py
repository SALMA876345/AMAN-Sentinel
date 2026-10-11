"""Phase 1 Step 4: automated leakage checks, run against BOTH the
HISTORICAL split (data/final_studies/source_*.csv, built by
utils/build_final_study_manifests.py, grouped by claim only) and the
CLEAN HACKATHON split (hackathon/data/clean_split/source_*.csv, built by
hackathon/build_clean_split.py, grouped by claim+evidence connected
components).

For each split family and each task (binary, three_class) this reports,
between every pair of {train, validation, test}:
  - exact claim-group overlap   (sha256 of normalised claim)
  - exact evidence-group overlap (sha256 of normalised evidence)
  - exact claim-text overlap     (raw normalised string match)
  - exact (claim, evidence) pair overlap
  - label distribution per split

Historical numbers here are expected to reproduce the audit's findings
(evidence_group overlap > 0 for the source family). Clean numbers are
expected to be exactly zero on every check. Nothing here modifies either
split; this is read + report only.

Output: hackathon/reports/leakage_check_results.json (+ printed table)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Code files/
HISTORICAL_DIR = ROOT / "data" / "final_studies"
CLEAN_DIR = ROOT / "hackathon" / "data" / "clean_split"
REPORT_DIR = ROOT / "hackathon" / "reports"

TASKS = ["binary", "three_class"]
PARTS = ["train", "validation", "test"]
PAIRS = [("train", "validation"), ("train", "test"), ("validation", "test")]

# Matches the clean-split builder's own grouping threshold
# (hackathon/build_clean_split.py MIN_EVIDENCE_CHARS_FOR_GROUPING, itself
# reused from utils/faav_common's faithfulness-gate default). Evidence
# shorter than this is a generic fragment ("age reaching", a bare name)
# carrying little information, and the clean split deliberately does not
# merge rows on it. Overlap checks below are split into "substantive"
# (>= threshold, a real leakage risk) and "trivial" (< threshold, expected
# and harmless) so a non-zero trivial count is not mistaken for a leak.
MIN_EVIDENCE_CHARS_FOR_GROUPING = 40


def load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False).fillna("")


def overlap(a: pd.DataFrame, b: pd.DataFrame, col: str) -> int:
    return len(set(a[col].astype(str)) & set(b[col].astype(str)))


def substantive_evidence_group_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Evidence-group overlap, split into substantive (real leak risk) vs
    trivial (short fragment, expected to repeat harmlessly)."""
    a_sub = a[a["evidence_norm"].str.len() >= MIN_EVIDENCE_CHARS_FOR_GROUPING]
    b_sub = b[b["evidence_norm"].str.len() >= MIN_EVIDENCE_CHARS_FOR_GROUPING]
    substantive = len(set(a_sub["evidence_group"]) & set(b_sub["evidence_group"]))
    total = overlap(a, b, "evidence_group")
    return {"substantive_overlap": substantive, "trivial_overlap": total - substantive, "total_overlap": total}


def substantive_evidence_text_overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Same split for raw exact evidence-text overlap (independent of the
    evidence_group hash column, as a second, redundant measurement)."""
    a_sub = a[a["evidence_norm"].str.len() >= MIN_EVIDENCE_CHARS_FOR_GROUPING]
    b_sub = b[b["evidence_norm"].str.len() >= MIN_EVIDENCE_CHARS_FOR_GROUPING]
    substantive = len(set(a_sub["evidence_norm"].astype(str)) & set(b_sub["evidence_norm"].astype(str)))
    total = overlap(a, b, "evidence_norm")
    return {"substantive_overlap": substantive, "trivial_overlap": total - substantive, "total_overlap": total}


def pair_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    pa = set(zip(a["claim_norm"].astype(str), a["evidence_norm"].astype(str)))
    pb = set(zip(b["claim_norm"].astype(str), b["evidence_norm"].astype(str)))
    return len(pa & pb)


def label_distribution(df: pd.DataFrame, task: str) -> dict:
    col = "binary_label" if task == "binary" else "three_class_label"
    return {str(k): int(v) for k, v in df[col].value_counts().items()}


def check_family(name: str, directory: Path, prefix: str) -> dict:
    result: dict = {"family": name, "tasks": {}}
    for task in TASKS:
        frames = {}
        for part in PARTS:
            path = directory / f"{prefix}_{task}_{part}.csv"
            frames[part] = load(path)
        if any(f is None for f in frames.values()):
            result["tasks"][task] = {"status": "FILES_MISSING", "missing": [
                part for part, f in frames.items() if f is None
            ]}
            continue

        task_result = {
            "status": "OK",
            "row_counts": {p: int(len(frames[p])) for p in PARTS},
            "label_distribution": {p: label_distribution(frames[p], task) for p in PARTS},
            "overlaps": {},
        }
        has_evidence_group = "evidence_group" in frames["train"].columns
        has_claim_group = "claim_group" in frames["train"].columns
        for a, b in PAIRS:
            key = f"{a}_vs_{b}"
            checks = {
                "exact_claim_text_overlap": overlap(frames[a], frames[b], "claim_norm"),
                "exact_evidence_text_overlap": substantive_evidence_text_overlap(frames[a], frames[b]),
                "claim_evidence_pair_overlap": pair_overlap(frames[a], frames[b]),
            }
            if has_claim_group:
                checks["claim_group_overlap"] = overlap(frames[a], frames[b], "claim_group")
            if has_evidence_group:
                checks["evidence_group_overlap"] = substantive_evidence_group_overlap(frames[a], frames[b])
            if "component_id" in frames[a].columns:
                checks["component_id_overlap"] = overlap(frames[a], frames[b], "component_id")
            task_result["overlaps"][key] = checks
        result["tasks"][task] = task_result
    return result


def print_summary(result: dict) -> None:
    print(f"\n{'=' * 70}\n{result['family']}\n{'=' * 70}")
    for task, task_result in result["tasks"].items():
        print(f"\n[{task}]")
        if task_result["status"] != "OK":
            print(f"  STATUS: {task_result['status']} (missing: {task_result['missing']})")
            continue
        print(f"  rows: {task_result['row_counts']}")
        for pair_key, checks in task_result["overlaps"].items():
            flagged = {}
            for k, v in checks.items():
                if k in ("evidence_group_overlap", "exact_evidence_text_overlap"):
                    if v["substantive_overlap"] != 0:
                        flagged[k] = v
                elif v != 0:
                    flagged[k] = v
            status = "CLEAN (substantive checks all zero)" if not flagged else f"LEAKAGE DETECTED: {flagged}"
            print(f"  {pair_key:<22} {checks}  -> {status}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    historical = check_family("HISTORICAL SPLIT (data/final_studies/, claim-grouped only)", HISTORICAL_DIR, "source")
    clean = check_family("CLEAN HACKATHON SPLIT (hackathon/data/clean_split/, claim+evidence component-grouped)", CLEAN_DIR, "source")

    print_summary(historical)
    print_summary(clean)

    out = {"historical": historical, "clean": clean}
    out_path = REPORT_DIR / "leakage_check_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
