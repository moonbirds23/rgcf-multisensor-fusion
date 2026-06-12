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


def _peer_center(x: np.ndarray, valid_idx: np.ndarray, sensor_idx: int, mode: str) -> np.ndarray:
    if mode == "loo_median":
        peer_idx = valid_idx[valid_idx != sensor_idx]
        if peer_idx.size > 0:
            return np.median(x[peer_idx, 0:4], axis=0)
    return np.median(x[valid_idx, 0:4], axis=0)


def _peer_consistency_features(
    xhat: np.ndarray,
    valid: np.ndarray,
    pos_scale: float,
    vel_scale: float,
    mode: str = "median",
    zero_features: bool = False,
) -> np.ndarray:
    k, n, _ = xhat.shape
    out = np.zeros((k, n, 6), dtype=np.float32)
    if zero_features:
        return out
    mode = str(mode or "median").lower()
    if mode not in {"median", "loo_median"}:
        raise ValueError(f"Unknown peer_consistency_mode: {mode}")
    for ti in range(k):
        valid_idx = np.flatnonzero(valid[ti] > 0.5)
        if valid_idx.size == 0:
            continue
        if mode == "median":
            peer_center = np.median(xhat[ti, valid_idx, 0:4], axis=0)
            delta = xhat[ti, :, 0:4] - peer_center[None, :]
        else:
            delta = np.zeros((n, 4), dtype=np.float64)
            for si in valid_idx:
                peer_center = _peer_center(xhat[ti], valid_idx, int(si), mode)
                delta[si, :] = xhat[ti, si, 0:4] - peer_center
        out[ti, :, 0] = delta[:, 0] / pos_scale
        out[ti, :, 1] = delta[:, 1] / pos_scale
        out[ti, :, 2] = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2) / pos_scale
        out[ti, :, 3] = delta[:, 2] / vel_scale
        out[ti, :, 4] = delta[:, 3] / vel_scale
        out[ti, :, 5] = np.sqrt(delta[:, 2] ** 2 + delta[:, 3] ** 2) / vel_scale
        out[ti, valid[ti] <= 0.5, :] = 0.0
    return out


def build_post_node_features_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> PostFeatureOutput:
    xhat = np.asarray(sim["xhat"], dtype=np.float64)
    phat = np.asarray(sim["Phat"], dtype=np.float64)
    valid = np.asarray(sim["valid_mask"], dtype=np.float64)
    k, n, _ = xhat.shape
    use_peer = bool(getattr(bundle.model, "use_peer_consistency_features", False))
    zero_peer = bool(getattr(bundle.model, "zero_peer_consistency_features", False))
    peer_mode = str(getattr(bundle.model, "peer_consistency_mode", "median"))
    feature_dim = 15 if use_peer else 9
    node = np.zeros((k, n, feature_dim), dtype=np.float32)
    node[..., 0] = xhat[..., 0] / float(bundle.scenario.pos_scale)
    node[..., 1] = xhat[..., 1] / float(bundle.scenario.pos_scale)
    node[..., 2] = xhat[..., 2] / float(bundle.scenario.vel_scale)
    node[..., 3] = xhat[..., 3] / float(bundle.scenario.vel_scale)
    node[..., 4] = np.log1p(np.clip(phat[..., 0, 0], 1e-9, None))
    node[..., 5] = np.log1p(np.clip(phat[..., 1, 1], 1e-9, None))
    node[..., 6] = np.log1p(np.clip(phat[..., 2, 2], 1e-9, None))
    node[..., 7] = np.log1p(np.clip(phat[..., 3, 3], 1e-9, None))
    node[..., 8] = valid
    if use_peer:
        node[..., 9:15] = _peer_consistency_features(
            xhat,
            valid,
            pos_scale=float(bundle.scenario.pos_scale),
            vel_scale=float(bundle.scenario.vel_scale),
            mode=peer_mode,
            zero_features=zero_peer,
        )
    return PostFeatureOutput(node, valid.astype(np.float32), np.asarray(sim["x_truth_4d"], dtype=np.float32), np.asarray(sim["t"], dtype=np.float32))
