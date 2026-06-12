param(
    [string]$EnvDir = "D:\envs\nfdkf-gpu",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"

Write-Host "== NF-DKF GPU environment installer =="
Write-Host "EnvDir: $EnvDir"
Write-Host "TorchIndexUrl: $TorchIndexUrl"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "python was not found. Install Python 3.10 x64 first, then rerun this script."
}

python -c "import sys; print(sys.version); assert sys.version_info[:2] >= (3, 10), 'Python 3.10+ is recommended'"

python -m venv $EnvDir
$envPython = Join-Path $EnvDir "Scripts\python.exe"
$envPip = Join-Path $EnvDir "Scripts\pip.exe"

& $envPython -m pip install --upgrade pip
& $envPip install torch torchvision torchaudio --index-url $TorchIndexUrl
& $envPip install -r requirements_gpu_base.txt

& $envPython -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

Write-Host "== done =="
Write-Host "Activate with:"
Write-Host "$EnvDir\Scripts\activate"
