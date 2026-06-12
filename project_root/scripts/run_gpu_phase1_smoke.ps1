param(
    [string]$PythonExe = "D:\envs\nfdkf-gpu\Scripts\python.exe",
    [string]$OutDir = "results\phase1_gpu_p11_smoke"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

& $PythonExe -u scripts\gpu_preflight_check.py --require-cuda

& $PythonExe -u scripts\run_phase1_nominal_benchmark.py `
  --smoke `
  --methods main `
  --device cuda `
  --force-regenerate-sims `
  --out-dir $OutDir
