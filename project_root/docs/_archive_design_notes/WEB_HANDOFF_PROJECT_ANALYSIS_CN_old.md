# 当前项目交接分析：面向网页版阅读

更新时间：2026-05-29

当前项目是一个围绕 **NF-DKF / 多传感器融合 / RGCF 可靠性引导融合** 的实验型 Python 工程。主干代码位于 `project_root/`。

## 主线

当前推荐研究主线：

```text
V2-RGCF: Reliability-Guided Calibrated Fusion
```

执行链路：

```text
main.py
  -> core.config_loader.load_experiment_bundle()
  -> configs.experiment_presets.build_experiment_config()
  -> experiments.<mode>
  -> simulation.runner.run_single_simulation()
  -> features.dataset / features.builders
  -> models.model_factory.build_model_from_bundle()
  -> training.trainer.train_fusion_model()
  -> core.result_manager.ResultManager
```

## 重要程序清单

```text
project_root/main.py
project_root/configs/experiment_presets.py
project_root/core/types.py
project_root/core/config_loader.py
project_root/core/result_manager.py
project_root/simulation/runner.py
project_root/features/post_features.py
project_root/features/meas_features.py
project_root/features/dataset.py
project_root/models/gnn_fusion.py
project_root/training/trainer.py
project_root/training/evaluator.py
project_root/data/dataset_store.py
```

## 主要结果

保留最新 full RGCF、两个消融、固定窗口 RGCF、clean V2 和主消融汇总结果。详见 `GITHUB_UPLOAD_MANIFEST_CN.md`。
