# AV2 small-scene Level C GPU command

Prerequisites:

- Pull the same repository revision and AV2 working changes on the Windows GPU host.
- Attach the mobile disk containing `small_scene_feasibility_v1`.
- Use the existing CUDA-enabled PEFNet environment; the GPU step does not require the AV2 API.

```powershell
$root = "E:\migration_packages\PEFNet_AV2\small_scene_feasibility_v1"
$python = "D:\envs\nfdkf-gpu\Scripts\python.exe"

& $python "D:\code\python\project-2\project_root\scripts\av2_small_level_c.py" `
  --root $root `
  --device cuda `
  --epochs 3 `
  --batch-size 128 `
  --model-seed 0
```

The command must fail when CUDA is unavailable. It writes the checkpoint, logs,
Level C report, and final feasibility decision only under the mobile-disk root.
