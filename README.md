# NF-DKF / RGCF Multi-Sensor Fusion Experiments

This is the cleaned GitHub-ready snapshot of the NF-DKF / RGCF multi-sensor fusion project.

Active code lives in `project_root/`. The current main line is **V2-RGCF: Reliability-Guided Calibrated Fusion**.

2026-06-09 update: the current paper/code story is being reorganized around a
three-layer evaluation framework rather than a fixed M4/Px version line. Start
from:

```text
project_root/EXPERIMENT_REDESIGN_CURRENT_CN.md
```

Historical M4 and P0-P12 ablation documents are retained as diagnostic records,
not as the current default paper plan.

Start with:

```text
project_root/EXPERIMENT_REDESIGN_CURRENT_CN.md
WEB_HANDOFF_PROJECT_ANALYSIS_CN.md
GITHUB_UPLOAD_MANIFEST_CN.md
project_root/main.py
project_root/configs/experiment_presets.py
project_root/models/gnn_fusion.py
```

Large raw datasets, simulation caches, local archives, and model checkpoints are excluded by `.gitignore`.

## Development Workflow (双机协作约定)

- **本机（开发端）**：所有代码编写、重构、配置调整在本机完成。
- **GPU 端（实验端）**：另一台 GPU 机器仅用于执行实验，不做代码修改。
- **同步方式**：两台机器之间通过 `git` 进行代码同步。
- **铁律**：每轮版本迭代或发起实验前，**必须先 commit 并 push 本机改动**，GPU 端 `git pull` 后再执行实验。禁止在未提交状态下直接运行 GPU 端实验，避免结果无法追溯对应代码版本。
- **任务清单**：实验任务通过 `project_root/TASK_<批次名>.md` 文件交接，格式参考 `project_root/TASK_CHECKLIST_TEMPLATE.md`。GPU 端按清单顺序执行并回填结果，完成后 commit 通知开发端验收。
