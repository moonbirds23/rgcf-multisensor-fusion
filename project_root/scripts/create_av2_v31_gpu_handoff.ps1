param(
    [Parameter(Mandatory = $true)]
    [string]$OutDir
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = (Resolve-Path $OutDir).Path
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TempRoot = Join-Path $OutputRoot "tmp_AV2_V3_1_GPU_HANDOFF_$Stamp"
$ZipPath = Join-Path $OutputRoot "AV2_V3_1_GPU_HANDOFF_$Stamp.zip"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

$TopLevel = @(
    "requirements_av2_level_c_gpu.txt",
    "docs\AV2_V3_1_GPU_HANDOFF_CN.md",
    "docs\AV2_NOMINAL_LEVEL_C_GPU_COMMANDS.md",
    "docs\AV2_1000_GPU_EXPERIMENT_COMMANDS_CN.md",
    "scripts\av2_nominal_gpu_preflight.py",
    "scripts\run_av2_nominal_level_c.ps1"
)
$Snapshot = @(
    "configs\av2_config.py",
    "configs\av2_level_b_plus_v11.py",
    "configs\av2_nominal_1000_v1.py",
    "configs\av2_nominal_1000_v1.yaml",
    "data\av2",
    "tools\av2_1000",
    "features\av2_shard_dataset.py",
    "simulation\av2_sensor_ekf.py",
    "simulation\measurement_models.py",
    "scripts\av2_small_level_b_plus_b1_b2.py",
    "scripts\av2_nominal_small_setup.py",
    "scripts\av2_nominal_small_integrity.py",
    "scripts\av2_nominal_small_diagnostics.py",
    "scripts\av2_nominal_small_report.py",
    "scripts\av2_nominal_level_c.py",
    "scripts\av2_nominal_gpu_preflight.py",
    "scripts\run_av2_nominal_level_c.ps1",
    "scripts\av2_1000_inventory.py",
    "scripts\av2_1000_build.py",
    "scripts\av2_1000_train.py",
    "scripts\av2_1000_evaluate.py",
    "scripts\av2_1000_aggregate.py",
    "scripts\av2_1000_validate.py"
)
function Copy-RelativeItem([string]$Relative, [string]$DestinationRoot) {
    $Source = Join-Path $ProjectRoot $Relative
    if (-not (Test-Path -LiteralPath $Source)) { throw "Missing package source: $Source" }
    $Destination = Join-Path $DestinationRoot $Relative
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
}
foreach ($Item in $TopLevel) { Copy-RelativeItem $Item $TempRoot }
foreach ($Item in $Snapshot) { Copy-RelativeItem $Item (Join-Path $TempRoot "SOURCE_SNAPSHOT") }
Get-ChildItem -LiteralPath (Join-Path $TempRoot "SOURCE_SNAPSHOT") -Directory -Recurse -Filter "__pycache__" |
    Remove-Item -Recurse -Force

$Manifest = [ordered]@{
    package = "AV2_V3_1_GPU_HANDOFF"
    created = (Get-Date).ToString("o")
    repository_root = $ProjectRoot
    included_snapshot_paths = $Snapshot
    excludes = @("raw AV2 data", "feature cache", "checkpoints", "results")
}
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $TempRoot "PACKAGE_MANIFEST.json") -Encoding UTF8
Compress-Archive -Path (Join-Path $TempRoot "*") -DestinationPath $ZipPath -Force
Remove-Item -LiteralPath $TempRoot -Recurse -Force
Write-Output $ZipPath
