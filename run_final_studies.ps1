param(
    [switch]$SkipManifestBuild,
    [switch]$SkipClassical,
    [switch]$SkipNeural,
    [switch]$SkipFAAV
)

$ErrorActionPreference = "Stop"

if (-not $SkipManifestBuild) {
    python utils/build_final_study_manifests.py
}

if (-not $SkipClassical) {
    python experiments/final_studies_classical.py
}

if (-not $SkipNeural) {
    python experiments/final_studies_neural.py
}

if (-not $SkipFAAV) {
    python experiments/final_studies_faav.py
    python experiments/final_studies_faav_binary.py
}

python utils/build_final_study_results.py
python utils/evaluate_final_verification_metrics.py
python utils/analyze_final_errors.py
python -m unittest utils.test_final_decision -v
python generate_final_report.py

Write-Host ""
Write-Host "Final comparison:"
Write-Host "  results/final_studies/final_model_comparison.csv"
Write-Host "Complete evaluation metrics:"
Write-Host "  results/final_studies/final_evaluation_metrics.csv"
Write-Host "Error analysis:"
Write-Host "  results/final_studies/final_error_analysis_by_setting.csv"
Write-Host "Final audit:"
Write-Host "  results/final_studies/final_results_audit.json"
Write-Host "Final report:"
Write-Host "  Assignment2_Final_Report_Full_UAE.docx"
