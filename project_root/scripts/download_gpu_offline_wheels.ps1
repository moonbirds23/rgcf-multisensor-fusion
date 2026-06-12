param(
    [string]$KitDir = "migration_packages\GPU_USB_KIT",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$PythonUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe",
    [string]$PythonTag = "310",
    [string]$PythonAbi = "cp310",
    [string]$PythonExe = "py"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$KitPath = Join-Path $ProjectRoot $KitDir
$InstallerDir = Join-Path $KitPath "installers"
$Wheelhouse = Join-Path $KitPath "wheelhouse"

New-Item -ItemType Directory -Force -Path $InstallerDir | Out-Null
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null

$pythonInstaller = Join-Path $InstallerDir (Split-Path $PythonUrl -Leaf)
if (-not (Test-Path -LiteralPath $pythonInstaller)) {
    Write-Host "Downloading Python installer..."
    Invoke-WebRequest -Uri $PythonUrl -OutFile $pythonInstaller
} else {
    Write-Host "Python installer already exists: $pythonInstaller"
}

Write-Host "Downloading base dependency wheels for Python $PythonTag win_amd64..."
$baseArgs = @(
    "-m", "pip", "download",
    "--dest", $Wheelhouse,
    "--only-binary=:all:",
    "--platform", "win_amd64",
    "--implementation", "cp",
    "--python-version", $PythonTag,
    "--abi", $PythonAbi,
    "-r", (Join-Path $ProjectRoot "requirements_gpu_base.txt")
)
& $PythonExe @baseArgs
if ($LASTEXITCODE -ne 0) {
    throw "pip download failed for base dependencies."
}

Write-Host "Downloading PyTorch GPU wheels from $TorchIndexUrl ..."
$torchArgs = @(
    "-m", "pip", "download",
    "--dest", $Wheelhouse,
    "--only-binary=:all:",
    "--platform", "win_amd64",
    "--implementation", "cp",
    "--python-version", $PythonTag,
    "--abi", $PythonAbi,
    "--extra-index-url", $TorchIndexUrl,
    "torch", "torchvision", "torchaudio"
)
& $PythonExe @torchArgs
if ($LASTEXITCODE -ne 0) {
    throw "pip download failed for PyTorch wheels."
}

$downloaded = Get-ChildItem -LiteralPath $Wheelhouse -File
if (-not ($downloaded | Where-Object { $_.Name -like "torch-*.whl" })) {
    throw "No torch wheel found in wheelhouse; download is incomplete."
}

<#
Legacy inline form kept as documentation only:
python -m pip download `
    --dest $Wheelhouse `
    --only-binary=:all: `
    --platform win_amd64 `
    --implementation cp `
    --python-version $PythonTag `
    --abi $PythonAbi `
    -r (Join-Path $ProjectRoot "requirements_gpu_base.txt")

python -m pip download `
    --dest $Wheelhouse `
    --only-binary=:all: `
    --platform win_amd64 `
    --implementation cp `
    --python-version $PythonTag `
    --abi $PythonAbi `
    --extra-index-url $TorchIndexUrl `
    torch torchvision torchaudio
#>

Write-Host "Wheelhouse ready:"
Write-Host $Wheelhouse
Get-ChildItem -LiteralPath $Wheelhouse | Sort-Object Name | Select-Object Name,Length | Format-Table -AutoSize
