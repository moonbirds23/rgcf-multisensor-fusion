# NF-DKF Multi-Sensor Fusion Experiments

This repository contains executable NF-DKF multi-sensor fusion experiments.
Active code lives in `project_root/`.

## Current Main Line

Updated: 2026-06-16

The current executable main line is:

```text
Phase1R RGCF baseline + ME-RGCF-A0 optional heterogeneous graph upgrade
```

Start here:

```text
project_root/docs/CURRENT_MAINLINE_CN.md
project_root/docs/PHASE1R_RGCF_GPU_EXPERIMENT_PLAN_CN.md
project_root/docs/ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md
```

Current defaults:

- `Phase1RRGCF` remains the stable baseline and default learned method.
- `ME-RGCF-A0` is the current upgrade, enabled only with `--methods me-a0` or `--methods rgcf,me-a0`.
- Clean scenarios only: `S1R` basic and `S2R` ordinary maneuver.
- Sensor roles are fixed as `3 track posterior sensors + 2 evidence measurement sensors`.
- Evidence sensors never output state-fusion weights.

## Main Commands

Dry-run:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_dryrun
```

CPU smoke for ME-A0:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cpu --methods me-a0 --out-dir project_root\results\phase2_me_a0_smoke_cpu
```

GPU comparison:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_compare
```

## Code Retention Policy

Legacy P0/P1/P11/P12/SNF/M4 code is retained for reproduction and diagnosis.
It should not be treated as the default research path unless a new task
explicitly reactivates it.

Current implementation entrypoints:

```text
project_root/scripts/run_phase1r_rgcf_benchmark.py
project_root/models/gnn_fusion.py
project_root/models/model_factory.py
project_root/configs/experiment_presets.py
project_root/features/
project_root/training/
```

Archived design notes are under:

```text
project_root/docs/_archive_design_notes/
```

Large raw datasets, simulation caches, local archives, and model checkpoints
are excluded by `.gitignore`.
