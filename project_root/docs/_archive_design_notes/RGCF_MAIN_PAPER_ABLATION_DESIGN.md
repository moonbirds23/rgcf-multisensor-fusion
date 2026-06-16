# RGCF Main Paper Ablation Design

Status update, 2026-06-09: this document is now a historical ablation record.
The current paper experiment design is defined in
`project_root/EXPERIMENT_REDESIGN_CURRENT_CN.md`. P0-P12 remain useful as
candidate variants and diagnostic evidence, but this document should not be
treated as the active paper plan.

This ablation is organized around the paper contributions rather than version
history. The three claimed components are:

1. Dual-stream input: post-filter state stream plus measurement-feature stream.
2. Reliability gate: learned gate for sensor reliability.
3. Fusion-formula modification: reliability-weighted AA versus covariance-aware
   information fusion.

Important: current diagnostics show that the covariance branch is unstable under
`bias_ramp`, so covariance-aware fusion should be treated as a diagnostic branch
rather than the default main method until it is proven reliable.

## Run Commands

Dry-run the main paper variants:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -u scripts/run_rgcf_v5_ablation.py --main_paper_config --dry_run
```

Run the full main paper ablation:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -u scripts/run_rgcf_v5_ablation.py --main_paper_config
```

Short 2-epoch trial:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' -u scripts/run_rgcf_v5_ablation.py ^
  --main_paper_config ^
  --epochs 2 ^
  --out_dir results/rgcf_main_paper_ablation_epoch2
```

## Main Variant Set

| Variant | Paper role | Question answered |
|---|---|---|
| `P0_post_only_single_stream` | Single-stream baseline | What happens without the measurement-feature stream? |
| `P1_dual_stream_direct` | Dual-stream, no gate | Does adding measurement features help before reliability gating? |
| `P2_gate_feature_path_only` | Gate ablation | Does the gate help by filtering measurement features? |
| `P3_gate_weight_path_only` | Gate ablation | Does the gate help by directly biasing fusion weights? |
| `P4_full_gate_no_cov_formula` | Full gate with information-style fusion | What is the contribution of the gate module before changing the output formula? |
| `P8_full_gate_aa_formula` | Full gate with AA output fusion | Does dropping covariance-dependent information fusion improve robustness? |

Covariance/formula diagnosis variants:

| Variant | Role |
|---|---|
| `P5_cov_fusion_formula_only` | Covariance scale enters the information matrix only. |
| `P6_cov_weight_formula_only` | Covariance scale enters the weight logits only. |
| `P7_full_gate_cov_formula` | Current full gate + covariance-aware information fusion. |
| `P9_gate_aa_cov_aux_only` | AA output fusion; covariance head is trained only as an auxiliary diagnostic head. |

Optional extension:

| Variant | Role |
|---|---|
| `P10_full_formula_peer_optional` | Later extension with peer consistency, not the main paper ablation unless peer is claimed as an additional contribution. |

## Table Structure

Recommended paper table columns:

| Component | P0 | P1 | P2 | P3 | P4 | P8 |
|---|---:|---:|---:|---:|---:|---:|
| post stream | yes | yes | yes | yes | yes | yes |
| measurement stream | no | yes | yes | yes | yes | yes |
| gate feature path | no | no | yes | no | yes | yes |
| gate weight path | no | no | no | yes | yes | yes |
| output fusion | info | info | info | info | info | AA |
| covariance in formula | no | no | no | no | no | no |

## Metrics

Primary accuracy metrics:

- `test_loss_pos`
- `quick_rmse_pos`
- `fault_window_rmse_pos`
- `fault_window_p95_error_pos`

Primary robustness metrics:

- `full_p95_error`
- `full_p99_error`
- `full_max_error`
- `bias_ramp_top_max_error`
- `p99_mean_fault_weight`
- `p99_fault_top_rate`

## Interpretation Logic

The paper story is supported if:

1. `P1_dual_stream_direct` improves over `P0_post_only_single_stream`, showing
   that measurement features add useful information.
2. `P4_full_gate_no_cov_formula` improves over `P1_dual_stream_direct`, showing
   that reliability gating matters beyond adding a second stream.
3. `P2_gate_feature_path_only` and `P3_gate_weight_path_only` reveal whether the
   gate contributes mainly through representation filtering or direct fusion
   weighting.
4. `P8_full_gate_aa_formula` improves over or matches `P4_full_gate_no_cov_formula`
   on tail metrics, supporting the move away from covariance-dependent output
   fusion.
5. `P5/P6/P7/P9` should be reported as a covariance diagnosis table. Covariance
   should only be promoted into the main formula if these variants improve
   `bias_ramp` and p99/max metrics consistently.

Peer consistency should be reported separately unless it is explicitly listed as
a fourth contribution. In that case, compare `P7_full_gate_cov_formula` against
`P8_full_formula_peer_optional` and the strict zero-peer control from the M4
ablation group.
