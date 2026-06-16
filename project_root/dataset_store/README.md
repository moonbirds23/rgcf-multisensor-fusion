# Dataset Store

Raw `*.pkl` dataset files are large and are intentionally excluded from GitHub.

Current Phase1R / ME-A0 benchmark scripts can generate or reuse dataset stores
automatically. Prefer the benchmark entrypoint instead of invoking older
dataset-generation presets by hand.

Dry-run:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --dry-run --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_dryrun
```

CPU/GPU smoke:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --smoke --device cuda --methods me-a0 --out-dir project_root\results\phase2_me_a0_smoke_cuda
```

Formal GPU comparison:

```powershell
& 'D:\code\python\env\env-NDKF - torch\Scripts\python.exe' project_root\scripts\run_phase1r_rgcf_benchmark.py --device cuda --resume --methods rgcf,me-a0 --out-dir project_root\results\phase2_me_a0_compare
```

Historical dataset-generation commands for P0/P11/M4/RGCF-V5 are retained only
inside `project_root/docs/_archive_design_notes/`.
