from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from core.types import ExperimentBundle


@dataclass
class MeasFeatureOutput:
    node_feat: np.ndarray
    mask: np.ndarray
    t: np.ndarray
    meta: Dict


def _two(x):
    out = np.zeros(2, dtype=np.float32)
    if x is not None:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        out[: min(2, len(arr))] = arr[:2]
    return out


def build_meas_node_features_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> MeasFeatureOutput:
    valid = np.asarray(sim["valid_mask"], dtype=np.float32)
    xpred = np.asarray(sim["xpred"], dtype=np.float64)
    k, n = valid.shape
    node = np.zeros((k, n, 14), dtype=np.float32)
    sensor_pos = np.asarray(sim.get("sensor_pos", np.zeros((n, 2))), dtype=np.float64)
    names = list(sim.get("sensor_type_names", ["gps2d"] * n))
    order = ["gps2d", "radar_rb", "aoa_only", "uwb_range_only"]
    for i, name in enumerate(names):
        if str(name) in order:
            node[:, i, 8 + order.index(str(name))] = 1.0
    for ti in range(k):
        for i in range(n):
            if valid[ti, i] < 0.5:
                continue
            innov = sim.get("innovation_store", [[None]])[ti, i]
            r = sim.get("R_store", [[None]])[ti, i]
            nis = sim.get("nis_store", np.zeros((k, n)))[ti, i]
            node[ti, i, 0:2] = _two(innov)
            node[ti, i, 2] = np.log1p(max(float(nis) if np.isfinite(nis) else 0.0, 0.0))
            if r is not None:
                diag = np.diag(np.asarray(r, dtype=np.float64)) if np.asarray(r).ndim == 2 else np.asarray(r, dtype=np.float64).reshape(-1)
                node[ti, i, 3:5] = np.log1p(np.clip(_two(diag), 1e-9, None))
            dx = (xpred[ti, i, 0] - sensor_pos[i, 0]) / float(bundle.scenario.pos_scale)
            dy = (xpred[ti, i, 1] - sensor_pos[i, 1]) / float(bundle.scenario.pos_scale)
            node[ti, i, 5:8] = [dx, dy, (dx * dx + dy * dy) ** 0.5]
            node[ti, i, 12] = valid[ti, i]
            node[ti, i, 13] = 1.0
    return MeasFeatureOutput(node, valid.copy(), np.asarray(sim["t"], dtype=np.float32), {"feature_dim": 14})
