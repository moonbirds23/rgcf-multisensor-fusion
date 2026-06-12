from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from core.types import ExperimentBundle
from .builders import FeatureBundle, build_feature_bundle_from_sim


class FusionTimeStepDataset(Dataset):
    """Per-timestep fusion dataset with pre-converted GPU-friendly tensors.

    All numpy arrays are converted to torch tensors once in __init__,
    and __getitem__ returns zero-copy views. This eliminates ~17 million
    torch.tensor() allocations during a typical Phase 1 full training run.
    """

    def __init__(self, feature_bundle: FeatureBundle, gate_target=None, gate_supervision_mask=None):
        self.feature_bundle = feature_bundle
        self._post_feat = torch.from_numpy(
            np.asarray(feature_bundle.post.node_feat, dtype=np.float32)
        )
        self._mask = torch.from_numpy(
            np.asarray(feature_bundle.post.mask, dtype=np.float32)
        )
        self._target = torch.from_numpy(
            np.asarray(feature_bundle.target, dtype=np.float32)
        )
        self.t = feature_bundle.t

        self._gate_target = None
        if gate_target is not None:
            self._gate_target = torch.from_numpy(
                np.asarray(gate_target, dtype=np.float32)
            )

        self._gate_supervision_mask = None
        if gate_supervision_mask is not None:
            self._gate_supervision_mask = torch.from_numpy(
                np.asarray(gate_supervision_mask, dtype=np.float32)
            )

        self._meas_feat = None
        if feature_bundle.meas is not None:
            self._meas_feat = torch.from_numpy(
                np.asarray(feature_bundle.meas.node_feat, dtype=np.float32)
            )

    def __len__(self):
        return self._post_feat.shape[0]

    def __getitem__(self, idx):
        out = {
            "post_feat": self._post_feat[idx],
            "mask": self._mask[idx],
            "target": self._target[idx],
        }
        if self._meas_feat is not None:
            out["meas_feat"] = self._meas_feat[idx]
        if self._gate_target is not None:
            out["gate_target"] = self._gate_target[idx]
            out["gate_supervision_mask"] = self._gate_supervision_mask[idx]
        return out


class FusionWindowDataset(FusionTimeStepDataset):
    """Windowed fusion dataset with pre-converted tensors and vectorized construction."""

    def __init__(self, feature_bundle: FeatureBundle, window_size=6, gate_target=None, gate_supervision_mask=None):
        super().__init__(feature_bundle, gate_target, gate_supervision_mask)
        if self._meas_feat is None:
            raise ValueError("FusionWindowDataset requires meas stream.")

        k, n, dp = self._post_feat.shape
        dm = self._meas_feat.shape[-1]
        window_size = int(window_size)

        # Vectorized sliding-window: for each lag, shift post_feat by
        # src_delta = window_size-1 - lag positions to the right (zero-fill left).
        # This replaces the original O(K * window_size) double for-loop.
        self._post_win = torch.zeros(k, n, window_size, dp, dtype=torch.float32)
        self._meas_win = torch.zeros(k, n, window_size, dm, dtype=torch.float32)

        for lag in range(window_size):
            src_delta = window_size - 1 - lag  # how many steps back to look
            # post_win[ti, :, lag, :] = post_feat[ti - src_delta] if ti >= src_delta else 0
            if src_delta < k:
                self._post_win[src_delta:, :, lag, :] = self._post_feat[:k - src_delta]
                self._meas_win[src_delta:, :, lag, :] = self._meas_feat[:k - src_delta]

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["post_win"] = self._post_win[idx]
        out["meas_win"] = self._meas_win[idx]
        return out


def _gate_arrays(sim: Dict[str, np.ndarray], feat: FeatureBundle, bundle: ExperimentBundle):
    if not bool(getattr(bundle.model, "use_gate_supervision", False)):
        return None, None
    valid = np.asarray(feat.post.mask, dtype=np.float32)
    fault = np.asarray(sim.get("fault_active_mask", np.zeros_like(valid)), dtype=np.float32)
    normal = float(getattr(bundle.model, "normal_gate_target", 0.8))
    bad = float(getattr(bundle.model, "fault_gate_target", 0.2))
    target = np.where(fault > 0.5, bad, normal).astype(np.float32)

    if bool(getattr(bundle.model, "use_error_aware_gate_target", False)):
        xhat = np.asarray(sim.get("xhat"), dtype=np.float32)
        truth = sim.get("x_truth_4d", sim.get("truth_4d", sim.get("x_true_4d")))
        if truth is None:
            raise KeyError("Error-aware gate target requires x_truth_4d/truth_4d/x_true_4d in sim.")
        truth = np.asarray(truth, dtype=np.float32)
        if xhat.ndim != 3 or truth.ndim != 2:
            raise ValueError(f"Unexpected xhat/truth shapes for gate target: {xhat.shape}, {truth.shape}")
        pos_err = np.linalg.norm(xhat[..., :2] - truth[:, None, :2], axis=-1)
        tau = max(float(getattr(bundle.model, "gate_error_tau", 5.0)), 1e-6)
        error_target = 1.0 / (1.0 + pos_err / tau)
        target = np.minimum(target, error_target.astype(np.float32))
        target_min = float(getattr(bundle.model, "gate_target_min", 0.1))
        target = np.clip(target, target_min, normal)

    return target.astype(np.float32) * valid, valid


def build_dataset_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle) -> FusionTimeStepDataset:
    feat = build_feature_bundle_from_sim(sim, bundle)
    gt, gm = _gate_arrays(sim, feat, bundle)
    return FusionTimeStepDataset(feat, gt, gm)


def build_window_dataset_from_sim(sim: Dict[str, np.ndarray], bundle: ExperimentBundle, window_size=6) -> FusionWindowDataset:
    feat = build_feature_bundle_from_sim(sim, bundle)
    gt, gm = _gate_arrays(sim, feat, bundle)
    return FusionWindowDataset(feat, window_size, gt, gm)


def build_dataset_list_from_sims(sims: List[Dict[str, np.ndarray]], bundle: ExperimentBundle) -> ConcatDataset:
    return ConcatDataset([build_dataset_from_sim(s, bundle) for s in sims])


def build_window_dataset_list_from_sims(sims: List[Dict[str, np.ndarray]], bundle: ExperimentBundle, window_size=6) -> ConcatDataset:
    return ConcatDataset([build_window_dataset_from_sim(s, bundle, window_size) for s in sims])
