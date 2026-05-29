from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.io_utils import build_run_dir_name, ensure_dir, now_timestamp_str, save_csv_rows, save_json, save_numpy_npz, save_torch_checkpoint, to_plain
from core.types import ExperimentBundle


@dataclass
class ResultPaths:
    root: Path
    summary_dir: Path
    history_dir: Path
    metrics_dir: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


class ResultManager:
    def __init__(self, bundle: ExperimentBundle, *, mode: str, root_dir: str | Path | None = None, experiment_name_override: str | None = None):
        self.bundle = bundle
        self.mode = mode
        root = Path(root_dir or bundle.base.results.result_root)
        name = build_run_dir_name(
            mode=mode,
            preset_name=bundle.identity.preset_name,
            scene_name=bundle.identity.scene_name,
            model_name=bundle.identity.model_name,
            timestamp=now_timestamp_str(),
        )
        self.run_dir = ensure_dir(root / name)
        self.paths = ResultPaths(
            self.run_dir,
            ensure_dir(self.run_dir / "summary"),
            ensure_dir(self.run_dir / "history"),
            ensure_dir(self.run_dir / "metrics"),
            ensure_dir(self.run_dir / "artifacts"),
            ensure_dir(self.run_dir / "checkpoints"),
            ensure_dir(self.run_dir / "logs"),
        )
        save_json(self.run_dir / "meta.json", {
            "mode": mode,
            "run_dir": str(self.run_dir),
            "preset_name": bundle.identity.preset_name,
            "experiment_name": experiment_name_override or bundle.identity.experiment_name,
            "scene_name": bundle.identity.scene_name,
            "model_name": bundle.identity.model_name,
            "fault_mode": bundle.identity.fault_mode,
            "device": bundle.base.runtime.device,
        })
        save_json(self.run_dir / "run_config.json", {
            "identity": bundle.identity,
            "base": bundle.base,
            "scenario": bundle.scenario,
            "fault": bundle.fault,
            "model": bundle.model,
            "train": bundle.train,
        })

    def _save_timeseries_rows(self, rows: List[Dict[str, Any]] | None, *, stem: str):
        if rows:
            save_csv_rows(self.paths.artifacts_dir / f"{stem}.csv", to_plain(rows))
            save_json(self.paths.artifacts_dir / f"{stem}.json", rows)

    def _save_reliability_timeseries_rows(self, rows: List[Dict[str, Any]] | None):
        self._save_timeseries_rows(rows, stem="gate_timeseries")
        if rows:
            self._save_timeseries_rows(rows, stem="reliability_timeseries")
            self._save_timeseries_rows(rows, stem="weight_timeseries")
            self._save_timeseries_rows(rows, stem="cov_scale_timeseries")
            self._save_timeseries_rows(rows, stem="gate_target_timeseries")

    def save_checkpoint(self, model, filename: str = "best_model.pt"):
        return save_torch_checkpoint(self.paths.checkpoints_dir / filename, model)

    def save_simulate_result(self, *, sim: Dict, baseline_metrics: Dict[str, float], fault_logs: List[Dict] | None = None):
        save_json(self.paths.metrics_dir / "baseline_metrics.json", baseline_metrics)
        if fault_logs is not None:
            save_json(self.paths.logs_dir / "fault_logs.json", fault_logs)
        save_json(self.paths.summary_dir / "sim_summary.json", {
            "scene_name": sim.get("scene_name"),
            "fault_mode": sim.get("fault_mode"),
            "t_shape": list(np.asarray(sim["t"]).shape),
            "x_truth_4d_shape": list(np.asarray(sim["x_truth_4d"]).shape),
            "xhat_shape": list(np.asarray(sim["xhat"]).shape),
            "Phat_shape": list(np.asarray(sim["Phat"]).shape),
        })
        arrays = {k: np.asarray(sim[k]) for k in ("t", "x_truth_4d", "x_truth_5d", "xhat", "Phat", "valid_mask") if k in sim}
        if "fault_active_mask" in sim:
            arrays["fault_active_mask"] = np.asarray(sim["fault_active_mask"])
        save_numpy_npz(self.paths.artifacts_dir / "sim_arrays.npz", arrays)

    def save_train_result(self, *, train_info: Dict, history: List[Dict], quick_baseline_metrics: Dict, quick_gnn_metrics: Dict, quick_sim=None, model=None, quick_gnn_timeseries=None):
        save_json(self.paths.summary_dir / "training_summary.json", train_info)
        save_csv_rows(self.paths.history_dir / "train_history.csv", history)
        save_json(self.paths.metrics_dir / "quick_baseline_metrics.json", quick_baseline_metrics)
        save_json(self.paths.metrics_dir / "quick_gnn_metrics.json", quick_gnn_metrics)
        self._save_reliability_timeseries_rows(quick_gnn_timeseries)
        if quick_sim is not None:
            self.save_simulate_result(sim=quick_sim, baseline_metrics=quick_baseline_metrics)
        if model is not None and self.bundle.base.results.save_model_ckpt:
            self.save_checkpoint(model)

    def save_repeat_result(self, *, repeat_summary: Dict, all_results: List[Dict]):
        save_json(self.paths.summary_dir / "repeat_summary.json", repeat_summary)
        save_csv_rows(self.paths.history_dir / "repeat_runs.csv", all_results)

    def save_fault_test_result(self, **kwargs):
        save_json(self.paths.summary_dir / "fault_test_summary.json", kwargs)

    def save_generalization_result(self, **kwargs):
        save_json(self.paths.summary_dir / "generalization_summary.json", kwargs)
