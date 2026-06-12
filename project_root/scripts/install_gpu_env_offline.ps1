param(
    [string]$EnvDir = "D:\envs\nfdkf-gpu",
    [string]$Wheelhouse = "wheelhouse"
)

$ErrorActionPreference = "Stop"

Write-Host "== NF-DKF offline GPU environment installer =="
Write-Host "EnvDir: $EnvDir"
Write-Host "Wheelhouse: $Wheelhouse"

if (-not (Test-Path -LiteralPath $Wheelhouse)) {
    throw "Wheelhouse not found: $Wheelhouse"
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "python was not found. Install Python 3.10 x64 from installers first."
}

python -c "import sys; print(sys.version); assert sys.version_info[:2] >= (3, 10), 'Python 3.10+ is recommended'"

python -m venv $EnvDir
$envPython = Join-Path $EnvDir "Scripts\python.exe"
$envPip = Join-Path $EnvDir "Scripts\pip.exe"

& $envPip install --no-index --find-links $Wheelhouse `
    "torch==2.7.1+cu126" `
    "torchvision==0.22.1+cu126" `
    "torchaudio==2.7.1+cu126"
& $envPip install --no-index --find-links $Wheelhouse -r requirements_gpu_base.txt

& $envPython -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host "== done =="
Write-Host "Activate with:"
Write-Host "$EnvDir\Scripts\activate"
