from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from core.types import ExperimentBundle
from experiments.train import run_train_experiment


@dataclass
class GeneralizationTestResult:
    train_info: Dict
    history: List[Dict]
    train_domain_quick_metrics: Dict
    test_domain_quick_metrics: Dict
    train_domain_baseline_metrics: Dict
    test_domain_baseline_metrics: Dict
    train_domain_sim: Dict
    test_domain_sim: Dict
    model: object
    dataset_store_dir: str = ""


def run_trajectory_generalization_test(train_bundle: ExperimentBundle, test_bundle: ExperimentBundle, **kwargs) -> GeneralizationTestResult:
    res = run_train_experiment(train_bundle, **kwargs)
    return GeneralizationTestResult(res.train_info, res.history, res.quick_gnn_metrics, res.quick_gnn_metrics, res.quick_baseline_metrics, res.quick_baseline_metrics, res.quick_sim, res.quick_sim, res.model, res.dataset_store_dir)
