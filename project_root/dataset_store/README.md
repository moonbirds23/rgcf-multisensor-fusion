# Dataset Store

Raw `*.pkl` dataset files are large and are intentionally excluded from GitHub.

Regenerate the main dataset with:

```powershell
cd D:\code\python\project-2\project_root
$PY = "D:\code\python\env\env-NDKF\Scripts\python.exe"
& $PY -u main.py --mode generate_dataset --preset hetero_robust_matrix_mixed_v2_rgcf --dataset_note rgcf_robust_matrix_mixed
```
