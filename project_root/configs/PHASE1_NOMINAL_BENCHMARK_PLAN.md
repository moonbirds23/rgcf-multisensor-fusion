# Phase 1 Nominal Fusion Benchmark Plan

Updated: 2026-06-09

This is the execution plan for the current Experiment 1. The main experiment
line is now the three-stage benchmark in `EXPERIMENT_REDESIGN_CURRENT_CN.md`,
not the old M4/P0-P12 tree.

## Scope

Phase 1 is a clean nominal fusion benchmark. It tests whether each method can
perform heterogeneous multi-sensor fusion without injected degradation or
hidden drift.

Paper protocol update: Phase 1 no longer trains one model per scene. It builds
one mixed S1/S2/S3 train/val dataset, trains each learned method once per model
seed on that mixed set, and then reports separate test metrics on S1, S2, and
S3. This keeps the paper story as one fusion model with cross-scene validation,
not three scene-specialized models.

Expected scenes, owned by Agent 1:

| ID | Label | Default preset placeholder | Purpose |
|---|---|---|---|
| S1 | S1-balanced | `phase1_s1_balanced_hetero_nominal` | Balanced heterogeneous coverage |
| S2 | S2-clustered | `phase1_s2_clustered_hetero_nominal` | One-sided/clustered sensor geometry |
| S3 | S3-maneuver | `phase1_s3_maneuver_hetero_nominal` | S1-like layout with maneuvering target |

The Phase 1 runner does not edit scenario files. If Agent 1 chooses different
preset names, pass them through `--scenario-presets`.

## Runner

Script:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py
```

Default learned methods:

| Method | Role |
|---|---|
| `P0_post_only_single_stream` | Post-only learned baseline |
| `P1_dual_stream_direct` | Dual-stream direct fusion |
| `P11_feature_stable_reliability_no_cov` | Current Phase 1 main model |

Method groups:

| CLI | Methods | Use |
|---|---|---|
| `--methods main` | P11 only | First formal main-version training |
| `--methods default` / `--methods compare` | P0, P1, P11 | Phase 1 paper comparison |
| `--methods extended` | P0, P1, P4, P11, P12 | Diagnostic/ablation pool |

P11 is selected as the current Phase 1 main version because it keeps the
reliability-gated dual-stream structure while removing the covariance path and
measurement-representation shortcut that previously amplified tail risk. P12 is
kept for Phase 3 recovery-aware stress testing, not as the Phase 1 default.

Rule baselines are evaluated on each scene test split: `AVG`, `CI-multi`,
`WAA-MM`, and `best-single`.

## Outputs

Default output root:

```text
results/phase1_nominal_benchmark
```

Main files:

| File | Meaning |
|---|---|
| `phase1_plan.json` / `phase1_plan.csv` | Dry-run and execution grid |
| `phase1_run_summary.json` / `phase1_run_summary.csv` | Mixed-trained method/model-seed results, expanded per S1/S2/S3 test scene |
| `phase1_aggregate_by_scene.json` / `.csv` | RMSE/p95/p99/max aggregated per S1/S2/S3 |
| `phase1_aggregate_overall.json` / `.csv` | Method-level aggregate across all scenes |
| `phase1_eval_details/` | Per-run test error details |

Metrics:

- `rmse`
- `p95`
- `p99`
- `max`

All metrics are position-error metrics over the test split.

## Commands

Dry-run with placeholder Agent-1 presets:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py --dry-run
```

Dry-run with explicit preset names:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py --dry-run `
  --scenario-presets "S1=phase1_s1_balanced_hetero_nominal,S2=phase1_s2_clustered_hetero_nominal,S3=phase1_s3_maneuver_hetero_nominal"
```

Smoke run, main model only:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py --smoke `
  --methods main `
  --scenario-presets "S1=phase1_s1_balanced_hetero_nominal,S2=phase1_s2_clustered_hetero_nominal,S3=phase1_s3_maneuver_hetero_nominal" `
  --out-dir results/phase1_nominal_benchmark_smoke
```

Formal run, current main model P11:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py `
  --methods main `
  --scenario-presets "S1=phase1_s1_balanced_hetero_nominal,S2=phase1_s2_clustered_hetero_nominal,S3=phase1_s3_maneuver_hetero_nominal" `
  --model-seeds 0,1,2,3,4 `
  --train-seed-range 10-69 `
  --val-seed-range 70-89 `
  --test-seed-range 90-109 `
  --epochs 80 `
  --out-dir results/phase1_nominal_benchmark
```

Formal comparison run, P0/P1/P11:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py `
  --methods compare `
  --scenario-presets "S1=phase1_s1_balanced_hetero_nominal,S2=phase1_s2_clustered_hetero_nominal,S3=phase1_s3_maneuver_hetero_nominal" `
  --model-seeds 0,1,2,3,4 `
  --train-seed-range 10-69 `
  --val-seed-range 70-89 `
  --test-seed-range 90-109 `
  --epochs 80 `
  --out-dir results/phase1_nominal_benchmark_compare
```

Resume an interrupted formal run:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py `
  --scenario-presets "S1=phase1_s1_balanced_hetero_nominal,S2=phase1_s2_clustered_hetero_nominal,S3=phase1_s3_maneuver_hetero_nominal" `
  --resume `
  --out-dir results/phase1_nominal_benchmark
```

If the mixed dataset has already been generated, reuse it explicitly:

```powershell
py -u scripts/run_phase1_nominal_benchmark.py `
  --mixed-dataset-dir "dataset_store/<mixed_phase1_dataset>"
```

## Current Main-Line Decisions

- The current main experiment is the three-stage benchmark:
  Phase 1 nominal fusion, Phase 2 observable degradation, Phase 3 hidden
  drift/recovery stress.
- Phase 1 is clean nominal only.
- Phase 1 uses one S1/S2/S3 mixed training set and reports separate S1/S2/S3
  test results.
- Do not train a separate special main model for each stage.
- Do not put drift into Phase 2.
- Hidden drift and recovery stress belong to Phase 3.
- P11 is the current Phase 1 main version.
- P12 is reserved for Phase 3 recovery-aware stress evidence unless later data
  proves it is also a cleaner nominal main model.
