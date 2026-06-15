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


@dataclass
class EvidenceFeatureOutput:
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


def _rolling_mean(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    window = max(int(window), 1)
    for i in range(values.shape[1]):
        vals = np.where(valid[:, i] > 0.5, values[:, i], 0.0).astype(np.float64)
        cnt = (valid[:, i] > 0.5).astype(np.float64)
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        ccnt = np.concatenate([[0.0], np.cumsum(cnt)])
        for ti in range(values.shape[0]):
            lo = max(0, ti + 1 - window)
            denom = ccnt[ti + 1] - ccnt[lo]
            if denom > 0:
                out[ti, i] = float((csum[ti + 1] - csum[lo]) / denom)
    return out


def _peer_deviation_features(
    sim: Dict[str, np.ndarray],
    valid: np.ndarray,
    pos_scale: float,
    vel_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    xhat = np.asarray(sim.get("xhat", sim.get("xpred")), dtype=np.float64)
    k, n = valid.shape
    pos_dev = np.zeros((k, n), dtype=np.float32)
    vel_dev = np.zeros((k, n), dtype=np.float32)
    for ti in range(k):
        valid_idx = np.flatnonzero(valid[ti] > 0.5)
        if valid_idx.size <= 1:
            continue
        for si in valid_idx:
            peer_idx = valid_idx[valid_idx != si]
            peer_center = np.median(xhat[ti, peer_idx, 0:4], axis=0)
            delta = xhat[ti, si, 0:4] - peer_center
            pos_dev[ti, si] = float(np.linalg.norm(delta[0:2]) / max(float(pos_scale), 1e-6))
            vel_dev[ti, si] = float(np.linalg.norm(delta[2:4]) / max(float(vel_scale), 1e-6))
    return pos_dev, vel_dev


def build_meas_node_features_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> MeasFeatureOutput:
    valid = np.asarray(sim.get("track_valid_mask", sim["valid_mask"]), dtype=np.float32)
    xpred = np.asarray(sim.get("track_xpred", sim["xpred"]), dtype=np.float64)
    k, n = valid.shape
    feature_dim = 18
    node = np.zeros((k, n, feature_dim), dtype=np.float32)
    sensor_pos = np.asarray(sim.get("track_sensor_pos", sim.get("sensor_pos", np.zeros((n, 2)))), dtype=np.float64)
    names = list(sim.get("track_sensor_type_names", sim.get("sensor_type_names", ["gps2d"] * n)))
    order = ["gps2d", "radar_rb", "aoa_only", "uwb_range_only"]
    innovation_store = sim.get("track_innovation", sim.get("innovation_store", [[None]]))
    r_store = sim.get("track_R", sim.get("R_store", [[None]]))
    nis_store = sim.get("track_nis", sim.get("nis_store", np.zeros((k, n))))
    for i, name in enumerate(names):
        if str(name) in order:
            node[:, i, 8 + order.index(str(name))] = 1.0
    log_nis_all = np.zeros((k, n), dtype=np.float32)
    abs_white_innov_all = np.zeros((k, n), dtype=np.float32)
    for ti in range(k):
        for i in range(n):
            if valid[ti, i] < 0.5:
                continue
            innov = innovation_store[ti, i]
            r = r_store[ti, i]
            nis = nis_store[ti, i]
            innov2 = _two(innov)
            diag2 = np.ones(2, dtype=np.float32)
            if r is not None:
                diag = np.diag(np.asarray(r, dtype=np.float64)) if np.asarray(r).ndim == 2 else np.asarray(r, dtype=np.float64).reshape(-1)
                diag2 = _two(diag)
            log_nis = np.log1p(max(float(nis) if np.isfinite(nis) else 0.0, 0.0))
            white = innov2 / np.sqrt(np.clip(diag2, 1e-6, None))
            node[ti, i, 0:2] = np.tanh(white / 5.0)
            node[ti, i, 2] = np.clip(log_nis, 0.0, 5.0)
            node[ti, i, 3:5] = np.clip(np.log1p(np.clip(diag2, 1e-9, None)), 0.0, 8.0)
            abs_white_innov_all[ti, i] = float(np.linalg.norm(white))
            log_nis_all[ti, i] = float(node[ti, i, 2])
            dx = (xpred[ti, i, 0] - sensor_pos[i, 0]) / float(bundle.scenario.pos_scale)
            dy = (xpred[ti, i, 1] - sensor_pos[i, 1]) / float(bundle.scenario.pos_scale)
            node[ti, i, 5:8] = [dx, dy, (dx * dx + dy * dy) ** 0.5]
            node[ti, i, 12] = valid[ti, i]
    node[..., 13] = _rolling_mean(log_nis_all, valid, window=30)
    node[..., 14] = np.clip(_rolling_mean(abs_white_innov_all, valid, window=30), 0.0, 20.0)
    if bool(getattr(bundle.model, "use_peer_consistency_features", False)):
        pos_dev, vel_dev = _peer_deviation_features(
            {"xhat": sim.get("track_xhat", sim.get("xhat")), "xpred": xpred},
            valid,
            pos_scale=float(bundle.scenario.pos_scale),
            vel_scale=float(bundle.scenario.vel_scale),
        )
        node[..., 15] = np.clip(pos_dev, 0.0, 5.0)
        node[..., 16] = np.clip(vel_dev, 0.0, 5.0)
    node[..., 17] = 1.0
    node[valid <= 0.5, :] = 0.0
    return MeasFeatureOutput(node, valid.copy(), np.asarray(sim["t"], dtype=np.float32), {"feature_dim": feature_dim})


def build_evidence_node_features_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> EvidenceFeatureOutput | None:
    if "evidence_z" not in sim:
        return None
    valid = np.asarray(sim.get("evidence_valid_mask"), dtype=np.float32)
    k, e = valid.shape
    feature_dim = 16
    node = np.zeros((k, e, feature_dim), dtype=np.float32)
    sensor_pos = np.asarray(sim.get("evidence_sensor_pos", np.zeros((e, 2))), dtype=np.float64)
    names = list(sim.get("evidence_sensor_type_names", ["aoa_only"] * e))
    order = ["gps2d", "radar_rb", "aoa_only", "uwb_range_only"]
    z_store = sim.get("evidence_z")
    r_store = sim.get("evidence_R")
    residual_prior = np.asarray(sim.get("evidence_residual_to_prior", np.full((k, e), np.nan)), dtype=np.float64)
    residual_track = np.asarray(sim.get("evidence_residual_to_track", np.full((k, e, 1), np.nan)), dtype=np.float64)
    residual_track_mean = np.nanmean(residual_track, axis=2) if residual_track.ndim == 3 else np.zeros((k, e))
    residual_track_min = np.nanmin(residual_track, axis=2) if residual_track.ndim == 3 else np.zeros((k, e))

    for i, name in enumerate(names):
        if str(name) in order:
            node[:, i, 4 + order.index(str(name))] = 1.0
    for ti in range(k):
        for i in range(e):
            if valid[ti, i] < 0.5:
                continue
            z = _two(z_store[ti, i])
            r = r_store[ti, i]
            diag2 = np.ones(2, dtype=np.float32)
            if r is not None:
                diag = np.diag(np.asarray(r, dtype=np.float64)) if np.asarray(r).ndim == 2 else np.asarray(r, dtype=np.float64).reshape(-1)
                diag2 = _two(diag)
            node[ti, i, 0:2] = np.tanh(z / np.sqrt(np.clip(diag2, 1e-6, None)) / 10.0)
            node[ti, i, 2:4] = np.clip(np.log1p(np.clip(diag2, 1e-9, None)), 0.0, 8.0)
            node[ti, i, 8] = sensor_pos[i, 0] / float(bundle.scenario.pos_scale)
            node[ti, i, 9] = sensor_pos[i, 1] / float(bundle.scenario.pos_scale)
            node[ti, i, 10] = np.clip(np.log1p(max(float(residual_prior[ti, i]) if np.isfinite(residual_prior[ti, i]) else 0.0, 0.0)), 0.0, 6.0)
            node[ti, i, 11] = np.clip(np.log1p(max(float(residual_track_mean[ti, i]) if np.isfinite(residual_track_mean[ti, i]) else 0.0, 0.0)), 0.0, 6.0)
            node[ti, i, 12] = np.clip(np.log1p(max(float(residual_track_min[ti, i]) if np.isfinite(residual_track_min[ti, i]) else 0.0, 0.0)), 0.0, 6.0)
            node[ti, i, 13] = valid[ti, i]
            node[ti, i, 14] = 1.0
            node[ti, i, 15] = float(i + 1) / max(float(e), 1.0)
    node[valid <= 0.5, :] = 0.0
    return EvidenceFeatureOutput(node, valid.copy(), np.asarray(sim["t"], dtype=np.float32), {"feature_dim": feature_dim})
