param(
    [string]$Output = "outputs/local-6gb-smoke",
    [string]$Config = "experiments/configs/local_6gb.json",
    [string]$Manifest = "data/smoke/manifest.csv",
    [switch]$Resume,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $env:HF_HOME = Join-Path $projectRoot "models/hf-cache"
    $env:TORCH_HOME = Join-Path $projectRoot "models/torch-cache"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $env:PYTHONFAULTHANDLER = "1"
    $env:PYTHONUNBUFFERED = "1"
    $env:TEMP = Join-Path $projectRoot ".cache/tmp"
    $env:TMP = $env:TEMP
    $python = Join-Path $projectRoot ".venv/Scripts/python.exe"
    $arguments = @("experiments/run_ablation.py", "--manifest", $Manifest,
        "--config", $Config, "--output", $Output, "--seeds", "42")
    if (-not $PrepareOnly) { $arguments += "--execute" }
    if ($Resume) { $arguments += "--resume" }
    & $python @arguments
    $runExit = $LASTEXITCODE
    if ($runExit -eq 0 -and -not $PrepareOnly) {
        & $python experiments/summarize_results.py $Output --require-complete
        $runExit = $LASTEXITCODE
    }
    exit $runExit
} finally {
    Pop-Location
}
