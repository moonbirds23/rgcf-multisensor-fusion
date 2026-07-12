# AV2 1000 场景 GPU 正式实验命令

本命令手册在 `AV2_NOMINAL_1000_V3.1` 的 Level C 报告为 `GO` 后执行。所有输出写入移动硬盘 `E:\migration_packages\PEFNet_AV2`；严禁 CPU fallback 或 fault/mixed-quality 数据。

```powershell
$pkg = 'E:\migration_packages\PEFNet_AV2'
$repo = 'D:\code\python\project-2\project_root'
$py = 'D:\envs\pefnet-av2-gpu\Scripts\python.exe'
Set-Location $repo
& $py -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# F1: inventory（2500 个只读候选 Parquet）
& $py scripts\av2_1000_inventory.py --root $pkg

# F2: 冻结分层 700/100/200 manifest v2
& $py scripts\av2_1000_build.py manifest --root $pkg
& $py scripts\av2_1000_validate.py --root $pkg manifest

# F3--F4: truth、nominal EKF sim、feature shards；仅 GPU 主机执行
& $py scripts\av2_1000_build.py truth --root $pkg
& $py scripts\av2_1000_build.py sim --root $pkg
& $py scripts\av2_1000_build.py features --root $pkg --shard-size 100
& $py scripts\av2_1000_validate.py --root $pkg all

# F5: 3 methods × 3 model seeds。九次训练均须完成。
foreach ($method in 'full_pefnet','pefnet_no_external_evidence','posterior_only') {
  foreach ($seed in 0,1,2) {
    & $py scripts\av2_1000_train.py --root $pkg --method $method --model-seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $method / seed $seed" }
  }
}

# F6: 评估固定 test 200 场景和 measurement seeds 100/101/102。
$checkpoints = @()
foreach ($method in 'full_pefnet','pefnet_no_external_evidence','posterior_only') {
  foreach ($seed in 0,1,2) { $checkpoints += "--checkpoint"; $checkpoints += "$method=$pkg\runs\av2_1000\$method\seed_$seed\best.pt" }
}
& $py scripts\av2_1000_evaluate.py --root $pkg @checkpoints
& $py scripts\av2_1000_aggregate.py --input "$pkg\results\av2_1000\scenario_level.csv" --output "$pkg\results\av2_1000"
```

训练和评估命令必须对三个 model seeds 都运行；聚合先在同一场景内平均三个 measurement seeds，再对学习方法平均三个 model seeds，最后以 200 个 scenario 为独立统计单位。
