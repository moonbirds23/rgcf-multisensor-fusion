# AV2 local pre-data test report

Date: 2026-07-10  
Scope: pre-data development only; no AV2 raw files were read, scanned, cached,
or used for training.

## Environment used for verification

```text
D:\code\python\env\env-NDKF - torch\Scripts\python.exe
```

## Checks completed

- Frozen AV2 Pilot Protocol v1.0 constants and document fingerprint.
- Portable external-root resolution and allow-listed output directories.
- Strict CUDA preflight helper without CPU fallback or raw-directory scanning.
- Timestamp-derived local coordinate conversion, one-second causal warmup, and
  immutable causal trajectory prefixes.
- Fixed AV2 sensor layout, causal independent EKFs, evidence-only nodes, and
  frozen Pilot fault semantics.
- `info_diag` fused covariance reporting without a fused-mean change.
- Diagonal NEES, coverage, and Gaussian NLL evaluation helpers.
- AV2 feature shape/mask/role/pair contracts.
- Versioned synthetic shard I/O with manifest/config hash verification.

## Command and result

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' `
  -m unittest discover -s tests -p 'test_*.py' -v
```

Result: 55 tests passed.  One symbolic-link escape test was skipped because
this Windows session cannot create directory symlinks; the test is expected to
run on a host where that permission is available.

## Explicitly deferred until data is mounted

- AV2 SDK reader and inventory scan;
- manifest generation from real scenes;
- truth cache and feature-shard generation;
- GPU path preflight against the mounted disk;
- Smoke/Pilot training, evaluation, and any formal experiment.
