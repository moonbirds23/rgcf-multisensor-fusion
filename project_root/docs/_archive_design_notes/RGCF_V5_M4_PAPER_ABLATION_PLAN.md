# RGCF V5 M4 Paper Ablation Plan

Status update, 2026-06-09: this document is archived. It records an older
M4-centered interpretation in which peer consistency was treated as the current
main version. The current experiment story has moved to the three-layer design
in `project_root/EXPERIMENT_REDESIGN_CURRENT_CN.md`. M4 results remain useful
as tail-risk diagnostics, not as the agreed main version.

This plan defines a focused ablation group around `M4_V5_peer_consistency`.
The goal is to support the paper claim that peer consistency mainly reduces
tail-risk fusion failures by suppressing high weights on faulty sensors,
especially under `bias_ramp`.

## Run Command

Dry-run the paper configuration:

```bash
python scripts/run_rgcf_v5_ablation.py --paper_config --dry_run
```

Run the full paper group on the fixed M4 dataset. This expands to 12 variants
and 5 model-initialization seeds per variant, with full tail diagnostics after
each run:

```bash
python scripts/run_rgcf_v5_ablation.py --paper_config
```

For the known local environment:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -u scripts/run_rgcf_v5_ablation.py --paper_config
```

For a shorter controlled run, override the seed list:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -u scripts/run_rgcf_v5_ablation.py ^
  --variants paper_m4 ^
  --train_seeds 0 ^
  --tail_diagnostics ^
  --out_dir results/rgcf_v5_m4_paper_ablation_seed0
```

## Variant Set

| Variant | Purpose | Key Change |
|---|---|---|
| `M3_V5_current` | Previous main model baseline | no peer consistency |
| `M4_V5_peer_consistency` | Full proposed model | peer + temp + mix + fault-weight loss |
| `M4A_no_peer_features` | Isolate peer feature contribution | disable peer features, keep other M4 stabilizers |
| `M4A_zero_peer_features` | Strict parameter-matched peer control | keep 15-D post input, zero the 6 peer channels |
| `M4A_no_fault_weight_loss` | Test whether explicit faulty-weight penalty matters | set `fault_weight_loss_weight=0` |
| `M4A_no_uniform_mix` | Test anti-collapse smoothing | set `weight_uniform_mix=0` |
| `M4A_no_logit_temperature` | Test logit softening | set `base_logit_temperature=1` |
| `M4A_no_cov_in_fusion` | Test using calibrated covariance inside information fusion | keep cov module, do not multiply covariance in fusion |
| `M4A_no_cov_module` | Test peer/gate weight path without covariance calibration | disable cov calibration, cov fusion, and cov losses |
| `M4A_no_gate_feature_path` | Test gate influence on representation | set `use_gate_on_meas_feature=False` |
| `M4A_no_gate_weight_bias` | Test gate as direct reliability bias | set `gate_weight_alpha=0` |
| `M4A_peer_loo_median` | Check peer-statistic implementation sensitivity | use leave-one-out peer median |

## Repeated Initializations

`--paper_config` uses:

```text
--train_seeds 0,1,2,3,4
```

Each seed controls Python, NumPy, PyTorch model initialization, and DataLoader
shuffle order through `TrainConfig.model_seed`. The dataset split remains fixed,
so the repeated runs measure training/initialization stability rather than data
resampling.

## Primary Metrics

Report these metrics for the main paper table:

- `quick_rmse_pos`
- `fault_window_rmse_pos`
- `fault_window_p95_error_pos`
- `fault_window_max_error_pos`
- `mean_weight_fault`
- `mean_weight_normal`
- `mean_gate_fault`
- `mean_gate_normal`
- `mean_cov_scale_fault`
- `mean_cov_scale_normal`

Report these metrics from max-error diagnostics for the tail-risk table:

- full-set `p95_error`
- full-set `p99_error`
- full-set `max_error`
- p99 mean fault sensor weight
- p99 fault top-rate
- per-scene `top max`, especially `bias_ramp`

The script writes these into:

```text
results/rgcf_v5_m4_paper_ablation/ablation_runs_summary.csv
results/rgcf_v5_m4_paper_ablation/ablation_runs_summary.json
results/rgcf_v5_m4_paper_ablation/tail_diagnostics/*.json
results/rgcf_v5_m4_paper_ablation/tail_diagnostics/*_top20.csv
```

## Interpretation Logic

The paper claim is supported if:

1. `M4_V5_peer_consistency` improves tail metrics over `M3_V5_current`.
2. `M4A_no_peer_features` regresses toward M3-like behavior more than other
   stabilizer removals.
3. `M4A_zero_peer_features` also regresses. This controls for the parameter-count
   change caused by replacing a 15-D post feature with a 9-D post feature.
4. Removing `fault_weight_loss`, `uniform_mix`, or temperature mainly affects
   weight concentration and tail stability, rather than explaining the entire
   M4 gain alone.
5. `M4A_no_cov_module` remains competitive on some tail metrics while cov
   diagnostics show residual failures, supporting the observation that current
   robustness is carried more strongly by the gate/weight path than the cov path.
6. `M4A_peer_loo_median` is close to or better than full M4. If it differs
   strongly, the paper should explicitly discuss peer-statistic construction.

## Suggested Next Diagnostic Pass

After training, run the existing grouped evaluation and max-error diagnosis on
the resulting run directories, then compare `bias_ramp` p99 composition and
fault-sensor top-weight rates.
