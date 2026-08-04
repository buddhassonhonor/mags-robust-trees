param(
    [string]$DataRoot = "data",
    [int]$Workers = 4,
    [switch]$SkipData
)

$ErrorActionPreference = "Stop"
$artifactRoot = Split-Path -Parent $PSScriptRoot
$metadataDir = Join-Path $artifactRoot "metadata"
New-Item -ItemType Directory -Force -Path $metadataDir | Out-Null
$logPath = Join-Path $metadataDir "clean_reproduction.log"

& {
    Set-Location $artifactRoot
    if (-not $SkipData) {
        python scripts/prepare_data.py --data-root $DataRoot
        if ($LASTEXITCODE -ne 0) { throw "Data preparation failed with exit code $LASTEXITCODE" }
    }
    $env:MAGS_DATA_ROOT = (Resolve-Path $DataRoot).Path
    python scripts/run_core.py --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Core experiment failed with exit code $LASTEXITCODE" }
    python scripts/run_baselines.py --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Baseline experiment failed with exit code $LASTEXITCODE" }
    python scripts/run_attacks.py --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Attack experiment failed with exit code $LASTEXITCODE" }
    python scripts/merge_stages.py
    if ($LASTEXITCODE -ne 0) { throw "Stage merge failed with exit code $LASTEXITCODE" }
    python exp/extract_threshold_case.py
    if ($LASTEXITCODE -ne 0) { throw "Threshold-case extraction failed with exit code $LASTEXITCODE" }
    New-Item -ItemType Directory -Force -Path results/summary | Out-Null
    Copy-Item exp/final_validation/threshold_case* results/summary/ -Force
    python scripts/build_statistics.py
    if ($LASTEXITCODE -ne 0) { throw "Statistics build failed with exit code $LASTEXITCODE" }
    python scripts/generate_figures.py
    if ($LASTEXITCODE -ne 0) { throw "Figure generation failed with exit code $LASTEXITCODE" }
    python scripts/validate_results.py
    if ($LASTEXITCODE -ne 0) { throw "Integrity validation failed with exit code $LASTEXITCODE" }
    python scripts/audit_consistency.py
    if ($LASTEXITCODE -ne 0) { throw "Consistency audit failed with exit code $LASTEXITCODE" }
    python -m pytest -q tests
    if ($LASTEXITCODE -ne 0) { throw "Tests failed with exit code $LASTEXITCODE" }
} *>&1 | Tee-Object -FilePath $logPath
