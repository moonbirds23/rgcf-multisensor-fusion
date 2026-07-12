# AV2 V3.1 Level C GPU commands

Run these commands only after the V3.1 nominal-only report is `GO`.  They are
CUDA-only: do not substitute a CPU run.

```powershell
$py = 'D:\code\python\env\pefnet-av2\Scripts\python.exe'
$root = 'E:\migration_packages\PEFNet_AV2\small_scene_feasibility_v1'

& $py project_root\scripts\av2_nominal_level_c.py --root $root --prepare-only --count 30 --measurement-seed 100
& $py project_root\scripts\av2_nominal_level_c.py --root $root --count 30 --epochs 1 --batch-size 128 --model-seed 0
```

Expected artifacts are `metadata/av2_level_c_nominal_prepare.json`,
`logs/av2_gpu_smoke.log`, `checkpoints/av2_smoke_seed0.pt`, and
`reports/AV2_LEVEL_C_GPU_SMOKE_REPORT.md` under the selected external root.
