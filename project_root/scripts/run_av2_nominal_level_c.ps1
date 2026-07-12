param(
    [string]$PythonExe = "D:\envs\pefnet-av2-gpu\Scripts\python.exe",
    [string]$DataRoot = "E:\migration_packages\PEFNet_AV2\small_scene_feasibility_v1",
    [ValidateRange(30, 50)][int]$SceneCount = 30,
    [ValidateRange(1, 3)][int]$Epochs = 1,
    [ValidateRange(1, 4096)][int]$BatchSize = 128
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "GPU Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw "External AV2 root not found: $DataRoot"
}

Push-Location $ProjectRoot
try {
    & $PythonExe -u scripts\av2_nominal_gpu_preflight.py --root $DataRoot
    & $PythonExe -u scripts\av2_nominal_level_c.py --root $DataRoot --prepare-only --count $SceneCount --measurement-seed 100
    & $PythonExe -u scripts\av2_nominal_level_c.py --root $DataRoot --count $SceneCount --epochs $Epochs --batch-size $BatchSize --model-seed 0
}
finally {
    Pop-Location
}
