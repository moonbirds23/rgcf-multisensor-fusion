from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from core.types import ExperimentBundle
from simulation.runner import run_single_simulation
from training.evaluator import evaluate_single_sim_fusion_with_timeseries
from training.trainer import train_fusion_model


@dataclass
class TrainExperimentResult:
    model: torch.nn.Module
    train_info: Dict
    history: List[Dict]
    quick_baseline_metrics: Dict
    quick_gnn_metrics: Dict
    quick_gnn_timeseries: List[Dict]
    quick_sim: Dict
    dataset_store_dir: str = ""


def run_train_experiment(bundle: ExperimentBundle, *, epochs=None, lr=None, batch_size=None, dataset_store_root="dataset_store", dataset_id=None, dataset_dir=None, use_latest_matching_dataset=False) -> TrainExperimentResult:
    tr = train_fusion_model(bundle, epochs=epochs, lr=lr, batch_size=batch_size, dataset_store_root=dataset_store_root, dataset_id=dataset_id, dataset_dir=dataset_dir, use_latest_matching_dataset=use_latest_matching_dataset)
    quick = run_single_simulation(bundle)
    ev = evaluate_single_sim_fusion_with_timeseries(quick.sim, tr.model, bundle, torch.device(bundle.base.runtime.device))
    return TrainExperimentResult(tr.model, tr.train_info, tr.history, quick.baseline_metrics, ev["summary"], ev["timeseries"], quick.sim, tr.dataset_store_dir)
