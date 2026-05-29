from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from core.types import ExperimentBundle
from simulation.runner import run_single_simulation


@dataclass
class SimulateResult:
    sim: Dict
    baseline_metrics: Dict[str, float]
    fault_logs: List[Dict]


def run_simulate_experiment(bundle: ExperimentBundle) -> SimulateResult:
    out = run_single_simulation(bundle)
    return SimulateResult(out.sim, out.baseline_metrics, out.fault_logs)
