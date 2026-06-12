param(
    [string]$PythonExe = "D:\envs\nfdkf-gpu\Scripts\python.exe",
    [string]$OutDir = "results\phase1_gpu_p11_main",
    [string]$Mode = "main"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

& $PythonExe -u scripts\gpu_preflight_check.py --require-cuda

& $PythonExe -u scripts\run_phase1_nominal_benchmark.py `
  --methods $Mode `
  --device cuda `
  --force-regenerate-sims `
  --model-seeds 0,1,2,3,4 `
  --train-seed-range 10-69 `
  --val-seed-range 70-89 `
  --test-seed-range 90-109 `
  --epochs 80 `
  --out-dir $OutDir
