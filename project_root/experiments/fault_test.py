from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from core.types import ExperimentBundle
from experiments.train import run_train_experiment


@dataclass
class FaultTestResult:
    train_info: Dict
    history: List[Dict]
    clean_quick_metrics: Dict
    fault_quick_metrics: Dict
    clean_baseline_metrics: Dict
    fault_baseline_metrics: Dict
    clean_quick_sim: Dict
    fault_quick_sim: Dict
    model: object
    dataset_store_dir: str = ""


def run_clean_train_fault_test(train_bundle: ExperimentBundle, test_bundle: ExperimentBundle, **kwargs) -> FaultTestResult:
    res = run_train_experiment(train_bundle, **kwargs)
    return FaultTestResult(res.train_info, res.history, res.quick_gnn_metrics, res.quick_gnn_metrics, res.quick_baseline_metrics, res.quick_baseline_metrics, res.quick_sim, res.quick_sim, res.model, res.dataset_store_dir)
