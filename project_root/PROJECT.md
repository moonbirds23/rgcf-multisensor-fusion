# NF-DKF Project Status

Updated: 2026-06-16

## Active Direction

The active direction is now split into two layers:

```text
Layer 1: Phase1R RGCF, stable clean benchmark baseline
Layer 2: ME-RGCF-A0, optional heterogeneous measurement-evaluated graph upgrade
```

`Phase1RRGCF` remains the default model. `ME-RGCF-A0` is available through the
Phase1R benchmark script but is not run unless requested with `--methods`.

## Active Documents

```text
docs/CURRENT_MAINLINE_CN.md
docs/PHASE1R_RGCF_GPU_EXPERIMENT_PLAN_CN.md
docs/ME_RGCF_HETEROGENEOUS_TEMPORAL_DESIGN_CN.md
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

Current benchmark entry:

```text
scripts/run_phase1r_rgcf_benchmark.py
```

Current learned-method switch:

```text
--methods rgcf
--methods me-a0
--methods rgcf,me-a0
```

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
