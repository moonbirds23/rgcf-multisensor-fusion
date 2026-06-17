# NF-DKF Project Status

Updated: 2026-06-17

## Active Direction

The active direction is now split into three layers:

```text
Layer 1: Phase1R RGCF, stable clean benchmark baseline
Layer 2: ME-RGCF-A0, optional heterogeneous measurement-evaluated graph upgrade
Layer 3: ME-RGCF-A0D, directional M->P attention upgrade
```

`Phase1RRGCF` remains the stable baseline. `ME-RGCF-A0` and
`ME-RGCF-A0D` are available through the Phase1R benchmark script but are not
run unless requested with `--methods`.

## Active Documents

```text
docs/CURRENT_MAINLINE_CN.md
docs/NEXT_WINDOW_HANDOFF_CN.md
docs/PHASE1R_RGCF_GPU_EXPERIMENT_PLAN_CN.md
docs/ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md
docs/PHASE2_ME_A0D_GPU_EXECUTION_PLAN_CN.md
```

## Current Framing

The corrected task uses five sensors with explicit roles:

- `T1/T2/T3`: posterior track sensors. Each must independently support EKF tracking.
- `E1/E2`: evidence measurement sensors. They provide measurement-side evidence only.

Fusion weights and covariance calibration are produced only for `T1/T2/T3`.
Evidence nodes can affect reliability and covariance scale, but they do not
become pseudo-posterior nodes and do not enter the final explicit fusion
formula as weighted states.

## Implementation Status

Current executable methods:

- `RGCF`: default learned method.
- `ME-RGCF-A0`: optional Phase2 minimal heterogeneous graph method.
- `ME-RGCF-A0D`: optional Phase2 directional pair-aware graph method.

Current benchmark entry:

```text
scripts/run_phase1r_rgcf_benchmark.py
```

Current learned-method switch:

```text
--methods rgcf
--methods me-a0
--methods rgcf,me-a0
--methods me-a0-dir
--methods rgcf,me-a0,me-a0-dir
```

## Latest Result Snapshot

Latest migrated GPU results were inspected under:

```text
E:\migration_packages\results\phase2_me_a0_dir_only
E:\migration_packages\dataset_store\20260615_210428__phase1r_s1r_s2r_mixed_nominal__phase1r_s1r_s2r_mixed_n__53259407
```

Overall RMSE:

- `ME-RGCF-A0D`: 2.6722
- `RGCF`: 2.7013
- `ME-RGCF-A0`: 2.7056
- `CI-3T`: 3.1528

`ME-RGCF-A0D` is the current best learned result, with small but consistent
gains on both S1R and S2R. It also fixes the previous ME-A0 attention collapse:
M->P attention row standard deviation rises from about `5e-5`/`7e-5` to about
`0.065` on both scenes.

Open issue: the evidence residual correlation is positive but still weak
(`S1R=0.068`, `S2R=0.104`). This means directional evidence is being used, but
not strongly enough to treat the A0D mechanism as mature.

T2 diagnosis: T2 is consistently weaker than T1/T3 in the formal GPU dataset,
especially on S2R. The issue is mainly tangential radar error under maneuvering
geometry plus CV-EKF model mismatch, not a simple nominal noise typo.

## Retained Legacy Code

Older P0/P1/P11/P12/SNF/M4 code paths and documents are retained for
reproduction and diagnosis. They are not the current default plan.

Do not remove legacy model classes or old benchmark scripts casually: some old
results, presets, and archived diagnostics still depend on them. Treat them as
`reproduction-only` unless a task explicitly reopens that line.

## Archive

Historical design notes and old operational handoff documents are stored in:

```text
docs/_archive_design_notes/
```

The archive README explains which ideas have been retired and why.
