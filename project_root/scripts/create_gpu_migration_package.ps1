param(
    [string]$OutDir = "migration_packages"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutRoot = Join-Path $ProjectRoot $OutDir
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PackagePath = Join-Path $OutRoot "project_root_gpu_migration_$Stamp.zip"
$TempRoot = Join-Path $OutRoot "tmp_gpu_migration_$Stamp"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

$IncludeItems = @(
    "configs",
    "core",
    "data",
    "experiments",
    "features",
    "models",
    "simulation",
    "training",
    "scripts",
    "rgcf_figures",
    "README.md",
    "EXPERIMENT_REDESIGN_CURRENT_CN.md",
    "GPU_MIGRATION_RUNBOOK_CN.md",
    "USB_TRANSFER_GUIDE_CN.md",
    "requirements_gpu_base.txt"
)

foreach ($item in $IncludeItems) {
    $src = Join-Path $ProjectRoot $item
    if (Test-Path -LiteralPath $src) {
        $dst = Join-Path $TempRoot $item
        if ((Get-Item -LiteralPath $src).PSIsContainer) {
            New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
            Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
        } else {
            New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
            Copy-Item -LiteralPath $src -Destination $dst -Force
        }
    }
}

Compress-Archive -Path (Join-Path $TempRoot "*") -DestinationPath $PackagePath -Force
Remove-Item -LiteralPath $TempRoot -Recurse -Force

Write-Output $PackagePath
