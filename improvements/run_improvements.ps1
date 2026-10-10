# Reproducibility entry point for the improvement analyses.
#
# Everything here writes under results/improvements. No file in
# results/final_studies or data/final_studies is modified, so the submitted
# artifacts stay exactly as they were graded.
#
# Stage 1 is the only slow step: it retrains four AraBERT verifiers to recover
# the row-level probabilities the submission does not ship. Skip it once the
# prediction dumps exist.

param(
    [switch]$SkipTraining,
    [switch]$SkipNLI
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

try {
    # --- Stage 0: analyses that need no model -----------------------------
    # Provenance controls: how much of each label is recoverable from the
    # publisher alone, so reported gains can be stated net of that confound.
    python improvements/control_baselines.py

    # Cluster structure: effective sample size once repeated evidence
    # passages are treated as the clusters they are.
    python improvements/evidence_structure.py

    # Best-model table with ties named instead of silently resolved.
    python improvements/tie_aware_results.py

    # Blind annotation sheets, key and codebook for the missing IAA study.
    python improvements/annotation_sample.py

    # --- Stage 1: recover row-level predictions ---------------------------
    if (-not $SkipTraining) {
        python improvements/train_verifiers.py
    }

    # --- Stage 2: analyses that consume the predictions -------------------
    # Risk-coverage, AURC and matched-coverage comparison against the
    # confidence baselines the report omits.
    python improvements/selective_evaluation.py

    # Paired McNemar tests and cluster-aware bootstrap intervals.
    python improvements/significance.py

    # Gate-independent entailment faithfulness.
    if (-not $SkipNLI) {
        python improvements/faithfulness_nli.py
    }

    # Figures for the report.
    python improvements/make_figures.py

    Write-Host ""
    Write-Host "Outputs written to results/improvements:"
    Write-Host "  control_baselines.csv             provenance and majority controls"
    Write-Host "  evidence_structure.csv            effective N per split"
    Write-Host "  best_results_tie_aware.csv        ties named"
    Write-Host "  selective_aurc.csv                AURC and E-AURC per selector"
    Write-Host "  selective_at_matched_coverage.csv selective scores at 70/80/90%"
    Write-Host "  risk_coverage_curves.csv          curve data"
    Write-Host "  significance_tests.csv            paired McNemar"
    Write-Host "  bootstrap_intervals.csv           cluster-aware 95% CIs"
    Write-Host "  faithfulness_nli.csv              entailment faithfulness"
    Write-Host "  figures/                          report-ready PNGs"
    Write-Host "  annotation/                       blind sheets, key, codebook"
}
finally {
    Pop-Location
}
