param(
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [string]$PackageRoot = 'E:\migration_packages\PEFNet_AV2'
)

$ErrorActionPreference = 'Stop'
$expectedProtocol = 'AV2_NOMINAL_1000_V3.2'
$expectedHash = 'd5af0c4f2daf262e32cd45a188462542555eba56d5c80dfef49d870e53270b3f'
$dataRoot = Join-Path $PackageRoot 'av2_data'
$codeRoot = Join-Path $PackageRoot '00_GPU_WORKSPACE_V32\code_snapshot'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
foreach ($required in @(
    (Join-Path $PackageRoot '00_GPU_WORKSPACE_V32\ACTIVE_PACKAGE_MANIFEST_20260716.json'),
    (Join-Path $PackageRoot '00_GPU_WORKSPACE_V32\SOURCE_SNAPSHOT_MANIFEST_20260716.json'),
    (Join-Path $PackageRoot '90_ARCHIVE_READ_ONLY\00_DO_NOT_RUN_ARCHIVED_CONTENT.md'),
    (Join-Path $dataRoot 'data\sim_cache\av2_1000_v32\_SUCCESS'),
    (Join-Path $dataRoot 'data\feature_shards\av2_1000_v32\_SUCCESS'),
    (Join-Path $dataRoot 'reports\av2_1000_v32\data_integrity_audit.json'),
    (Join-Path $codeRoot 'scripts\av2_1000_train.py'),
    (Join-Path $codeRoot 'scripts\av2_1000_evaluate.py'),
    (Join-Path $codeRoot 'simulation\supplementary_baselines.py')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required V3.2 artifact is missing: $required"
    }
}

foreach ($forbidden in @(
    (Join-Path $dataRoot 'data\sim_cache\av2_1000'),
    (Join-Path $dataRoot 'data\feature_shards\av2_1000'),
    (Join-Path $dataRoot 'runs\av2_1000'),
    (Join-Path $dataRoot 'results\av2_1000')
)) {
    if (Test-Path -LiteralPath $forbidden) {
        throw "Forbidden historical path is present in the GPU workspace: $forbidden"
    }
}

$simMarker = Get-Content -LiteralPath (Join-Path $dataRoot 'data\sim_cache\av2_1000_v32\_SUCCESS') -Raw -Encoding UTF8 | ConvertFrom-Json
$featureMarker = Get-Content -LiteralPath (Join-Path $dataRoot 'data\feature_shards\av2_1000_v32\_SUCCESS') -Raw -Encoding UTF8 | ConvertFrom-Json
$audit = Get-Content -LiteralPath (Join-Path $dataRoot 'reports\av2_1000_v32\data_integrity_audit.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$activeManifest = Get-Content -LiteralPath (Join-Path $PackageRoot '00_GPU_WORKSPACE_V32\ACTIVE_PACKAGE_MANIFEST_20260716.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$sourceManifest = Get-Content -LiteralPath (Join-Path $PackageRoot '00_GPU_WORKSPACE_V32\SOURCE_SNAPSHOT_MANIFEST_20260716.json') -Raw -Encoding UTF8 | ConvertFrom-Json

foreach ($payload in @($simMarker, $featureMarker, $audit)) {
    if ($payload.protocol_name -ne $expectedProtocol) {
        throw "Protocol mismatch: $($payload.protocol_name)"
    }
    if ($payload.config_sha256 -ne $expectedHash) {
        throw "Config hash mismatch: $($payload.config_sha256)"
    }
}
if ($audit.status -ne 'PASS' -or $audit.unique_rng_streams -ne 1400) {
    throw 'Persisted V3.2 data audit did not pass'
}
if ($activeManifest.status -ne 'READY_FOR_GPU' -or $activeManifest.forbidden_active_paths_absent -ne $true) {
    throw 'Active package manifest is not READY_FOR_GPU'
}
foreach ($row in $sourceManifest.files) {
    $path = Join-Path $codeRoot ($row.path.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Source snapshot file is missing: $path"
    }
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $row.sha256) {
        throw "Source snapshot hash mismatch: $path"
    }
}

$simCount = (Get-ChildItem -LiteralPath (Join-Path $dataRoot 'data\sim_cache\av2_1000_v32') -Filter 'seed_*.npz' -File -Recurse).Count
$shardCount = (Get-ChildItem -LiteralPath (Join-Path $dataRoot 'data\feature_shards\av2_1000_v32') -Filter '_SUCCESS' -File -Recurse).Count - 1
if ($simCount -ne 1400) { throw "Expected 1400 sim caches, found $simCount" }
if ($shardCount -ne 14) { throw "Expected 14 feature shards, found $shardCount" }

& $Python -c "import torch; assert torch.cuda.is_available(); print('GPU=' + torch.cuda.get_device_name(0)); print('TORCH=' + torch.__version__); print('CUDA=' + str(torch.version.cuda))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA/Torch preflight failed' }

Write-Host 'AV2 V3.2 GPU PREFLIGHT: PASS' -ForegroundColor Green
Write-Host "DATA_ROOT=$dataRoot"
Write-Host "CODE_ROOT=$codeRoot"
Write-Host "SIM_COUNT=$simCount FEATURE_SHARDS=$shardCount"
Write-Host "SOURCE_HASH_FILES=$($sourceManifest.file_count)"
