from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from core.types import ExperimentBundle


@dataclass
class PostFeatureOutput:
    node_feat: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    t: np.ndarray


def build_post_node_features_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> PostFeatureOutput:
    xhat = np.asarray(sim["xhat"], dtype=np.float64)
    phat = np.asarray(sim["Phat"], dtype=np.float64)
    valid = np.asarray(sim["valid_mask"], dtype=np.float64)
    k, n, _ = xhat.shape
    node = np.zeros((k, n, 9), dtype=np.float32)
    node[..., 0] = xhat[..., 0] / float(bundle.scenario.pos_scale)
    node[..., 1] = xhat[..., 1] / float(bundle.scenario.pos_scale)
    node[..., 2] = xhat[..., 2] / float(bundle.scenario.vel_scale)
    node[..., 3] = xhat[..., 3] / float(bundle.scenario.vel_scale)
    node[..., 4] = np.log1p(np.clip(phat[..., 0, 0], 1e-9, None))
    node[..., 5] = np.log1p(np.clip(phat[..., 1, 1], 1e-9, None))
    node[..., 6] = np.log1p(np.clip(phat[..., 2, 2], 1e-9, None))
    node[..., 7] = np.log1p(np.clip(phat[..., 3, 3], 1e-9, None))
    node[..., 8] = valid
    return PostFeatureOutput(node, valid.astype(np.float32), np.asarray(sim["x_truth_4d"], dtype=np.float32), np.asarray(sim["t"], dtype=np.float32))
