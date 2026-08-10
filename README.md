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

## Paper Writing and Experiment Result Index (2026-07-14)

The paper workspace and experiment supplements are intentionally separated. The
current paper does not discuss covariance estimation or uncertainty calibration;
NEES, NLL, coverage, covariance-audit tables, and related delivery packages are
frozen for traceability only.

### Active paper writing

| Location | Content | Status |
|---|---|---|
| `workshop_paper/01_论文初稿以及参考文献/PEFNet中文LaTeX初稿/main.tex` | Main Chinese LaTeX entry | Active |
| `workshop_paper/01_论文初稿以及参考文献/PEFNet中文LaTeX初稿/chapters/PEFNet_reformatted_chapters/PEFNet_combined.tex` | Chapters 1--3 and the Chapter 4 include point | Active |
| `workshop_paper/01_论文初稿以及参考文献/PEFNet中文LaTeX初稿/chapters/chapter4_experiments_standalone.tex` | Current complete Chapter 4, including position-error AV2 results | Active |
| `workshop_paper/01_论文初稿以及参考文献/PEFNet中文LaTeX初稿/figures/重新生成/` | Latest unified-style S1R/S2R and AV2 paper figures | Active |
| `workshop_paper/01_论文初稿以及参考文献/PEFNet中文LaTeX初稿/90_过期版本_按时间归档_20260714/` | Dated archive of old chapters, old figures, tests, and compiled outputs | Expired; traceability only |
| `workshop_paper/05_已完成实验结果/` | Frozen S1R/S2R main and ablation tables, figures, and plotting script | Paper result source |

The current Chapter 4 source is included from `PEFNet_combined.tex`; do not
maintain a second copied Chapter 4 in the combined file.

### Frozen experiment results

| Location | Experiment represented | Paper use |
|---|---|---|
| `workshop_paper/05_已完成实验结果/` | Phase1R S1R/S2R simulation: 3 posterior tracks + 2 evidence sources; main and ablation position metrics | Active position-error paper results |
| `workshop_paper/05_已完成实验结果/FROZEN_PAPER_ARTIFACTS_20260714.md` | Current paper data, figures, plotting programs, and SHA256 freeze list | Frozen paper artifact index |
| `workshop_paper/05_已完成实验结果/数据/第5章时间序列与轨迹/` | Frozen time-series and representative-trajectory exports used to generate paper figures | Paper figure input; not raw GPU data |
| `补充实验/00_封存实验结果/03_AV2_1000_实测轨迹_位置结果/` | AV2 nominal trajectory-driven position results: 200 test scenes, 3 model seeds, 3 measurement seeds, 100 evaluation steps | Frozen supplementary result; use only in the AV2 position section |
| `补充实验/00_封存实验结果/02_PHASE1R_协方差不确定性_不纳入当前论文/` | Phase1R NEES/NLL/coverage and seed-90 repair package | Archived; not used in current paper |
| `补充实验/00_封存实验结果/04_AV2_协方差不确定性_不纳入当前论文/` | AV2 uncertainty delivery and its audit output | Archived; not used in current paper |

### Plans, handoff, and audits

- `补充实验/01_执行计划与交接/` contains AV2 plans, GPU handoff packages,
  prompts, and small-scene execution records. These are planning/traceability
  documents, not paper result tables.
- `补充实验/90_审计与复核/` contains post-run audits and reproducibility
  checks. They do not change the active paper scope.
- `workshop_paper/90_交接归档/20260714_整理封存/论文历史草稿与临时产物/` contains
  historical LaTeX copies, audit packages, and review renders that are not active
  source files.
- `补充实验/实测数据集/` is the local index for external data. The large AV2
  source data remain on `E:\migration_packages\PEFNet_AV2` and are not copied
  into the writing workspace.
- The complete migration-disk freeze index is
  `E:\migration_packages\FREEZE_INDEX_20260714.md`. It distinguishes current
  Phase1R/AV2 sources from expired GPU results and legacy manifests.

### Temporary outputs

`tmp/` is now reserved for disposable outputs. Existing audits, smoke outputs,
figure reviews, and LaTeX verification products were moved into the indexed
archive locations above; the remaining `tmp/README.md` records this rule.
