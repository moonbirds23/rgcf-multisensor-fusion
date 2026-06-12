param(
    [string]$OutDir = "migration_packages\GPU_USB_KIT\installers",
    [string]$PythonUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutPath = Join-Path $ProjectRoot $OutDir
New-Item -ItemType Directory -Force -Path $OutPath | Out-Null

$pythonInstaller = Join-Path $OutPath (Split-Path $PythonUrl -Leaf)
Write-Host "Downloading Python installer:"
Write-Host $PythonUrl
Invoke-WebRequest -Uri $PythonUrl -OutFile $pythonInstaller

Write-Host "Saved:"
Write-Host $pythonInstaller
Write-Host ""
Write-Host "PyTorch CUDA wheels are not downloaded here because they must match the GPU machine driver."
Write-Host "Install PyTorch on the GPU machine with scripts\\install_gpu_env_online.ps1."
