# EHGCF 当前实验审核说明

## 1. 当前主实验能支撑的论文叙述

当前正式结果中，最终版本 `ME-RGCF-A0D` 在论文中重命名为 `EHGCF`。已有结果覆盖两个 Phase1R 场景：

- `S1R`: basic-3track-2evidence
- `S2R`: maneuver-3track-2evidence

主实验可写成：在三条后验轨迹与两条外部测量证据共同存在的设置下，比较 `EHGCF` 与规则融合方法 `CI-3T`、`WAA-MM-3T`、`AVG-3T`、`single-T1/T2/T3`。评价指标使用 `RMSE`、`P95`、`P99`、`Max`。

已有最终结果来源：

```text
E:\migration_packages\results\phase2_me_a0_dir_only\phase1r_aggregate_overall.csv
E:\migration_packages\results\phase2_me_a0_dir_only\phase1r_aggregate_by_scene.csv
E:\migration_packages\results\phase2_me_a0_dir_only\phase1r_run_summary.csv
E:\migration_packages\results\phase2_me_a0_dir_only\eval_details\*.csv
```

该结果包含 5 个模型种子、S1R/S2R 共 10 个 learned evaluation runs、230200 个误差点。可以支撑主实验和分场景实验的基本论文叙述。

## 2. 当前需要谨慎说明的地方

现有 `RGCF`、`ME-RGCF-A0`、`ME-RGCF-A0D` 结果虽然 seeds 与规则基线一致，但来自不同 `dataset_store` 目录。用于阶段分析是可接受的；论文定稿建议在同一个 fixed mixed dataset 上重跑一遍主实验和精简消融。

`ME-RGCF-A0D-HS` 目前只作为小规模诊断结果使用，不进入主实验表，也不作为最终版本。

## 3. 论文建议实验组

主实验表：

- `EHGCF`
- `CI-3T`
- `WAA-MM-3T`
- `AVG-3T`
- `single-T1`
- `single-T2`
- `single-T3`

精简消融表：

- `Posterior-only`: 只使用后验轨迹，验证测量证据与后验信息共同引入的必要性。
- `RGCF`: 证据聚合/广播式图融合对照。
- `ME-RGCF-A0`: 异构 P/M 图但没有最终 directional pair evidence 的对照。
- `EHGCF w/o calibrated fusion`: 去掉可靠性门控与协方差校准融合，验证融合公式改良。
- `EHGCF`: 最终方法。

## 4. 结果数据预留

新增整理脚本会生成：

```text
table_main_overall.csv
table_by_scene.csv
table_ablation.csv
mechanism_plot_pool.csv
result_sources_manifest.csv
```

其中 `mechanism_plot_pool.csv` 预留以下机制图数据：

- `error_pos`
- `w_s1/w_s2/w_s3`
- `g_s1/g_s2/g_s3`
- `cov_scale_s1/cov_scale_s2/cov_scale_s3`
- `mp_attn_p*_m*`
- `mm_attn_m*_m*`
- `mp_pair_res_p*_m*`
- `mp_pair_rank_p*_m*`

这些列可用于后续画权重分布、gate/covariance scale 分布、attention 热力图、pair residual 与 attention 对应关系等图。

## 5. 仍需补充的信息

正式写论文前建议确认：

- GPU 端最终是否能使用同一个 `mixed_dataset_dir` 完成主实验和消融实验。
- 是否采用 `EHGCF` 作为英文方法名，中文名是否固定为“证据感知异构图校准融合方法”。
- 论文篇幅是否允许保留完整消融表；若篇幅不足，优先保留 `Posterior-only`、`EHGCF w/o calibrated fusion`、`EHGCF`，把 `RGCF` 和 `ME-RGCF-A0` 放入附录或补充说明。
