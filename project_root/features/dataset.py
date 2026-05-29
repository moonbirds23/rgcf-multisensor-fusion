from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from core.types import ExperimentBundle
from .builders import FeatureBundle, build_feature_bundle_from_sim


class FusionTimeStepDataset(Dataset):
    def __init__(self, feature_bundle: FeatureBundle, gate_target=None, gate_supervision_mask=None):
        self.feature_bundle = feature_bundle
        self.post_node_feat = feature_bundle.post.node_feat
        self.mask = feature_bundle.post.mask
        self.target = feature_bundle.target
        self.t = feature_bundle.t
        self.gate_target = gate_target
        self.gate_supervision_mask = gate_supervision_mask
        self.meas_node_feat = feature_bundle.meas.node_feat if feature_bundle.meas is not None else None

    def __len__(self): return self.post_node_feat.shape[0]

    def __getitem__(self, idx):
        out = {"post_feat": torch.tensor(self.post_node_feat[idx], dtype=torch.float32), "mask": torch.tensor(self.mask[idx], dtype=torch.float32), "target": torch.tensor(self.target[idx], dtype=torch.float32)}
        if self.meas_node_feat is not None:
            out["meas_feat"] = torch.tensor(self.meas_node_feat[idx], dtype=torch.float32)
        if self.gate_target is not None:
            out["gate_target"] = torch.tensor(self.gate_target[idx], dtype=torch.float32)
            out["gate_supervision_mask"] = torch.tensor(self.gate_supervision_mask[idx], dtype=torch.float32)
        return out


class FusionWindowDataset(FusionTimeStepDataset):
    def __init__(self, feature_bundle: FeatureBundle, window_size=6, gate_target=None, gate_supervision_mask=None):
        super().__init__(feature_bundle, gate_target, gate_supervision_mask)
        if self.meas_node_feat is None:
            raise ValueError("FusionWindowDataset requires meas stream.")
        k, n, dp = self.post_node_feat.shape
        dm = self.meas_node_feat.shape[-1]
        self.post_win = np.zeros((k, n, window_size, dp), dtype=np.float32)
        self.meas_win = np.zeros((k, n, window_size, dm), dtype=np.float32)
        for ti in range(k):
            for li in range(window_size):
                src = ti - (window_size - 1 - li)
                if src >= 0:
                    self.post_win[ti, :, li] = self.post_node_feat[src]
                    self.meas_win[ti, :, li] = self.meas_node_feat[src]

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["post_win"] = torch.tensor(self.post_win[idx], dtype=torch.float32)
        out["meas_win"] = torch.tensor(self.meas_win[idx], dtype=torch.float32)
        return out


def _gate_arrays(sim: Dict[str, np.ndarray], feat: FeatureBundle, bundle: ExperimentBundle):
    if not bool(getattr(bundle.model, "use_gate_supervision", False)):
        return None, None
    valid = np.asarray(feat.post.mask, dtype=np.float32)
    fault = np.asarray(sim.get("fault_active_mask", np.zeros_like(valid)), dtype=np.float32)
    normal = float(getattr(bundle.model, "normal_gate_target", 0.8))
    bad = float(getattr(bundle.model, "fault_gate_target", 0.2))
    return np.where(fault > 0.5, bad, normal).astype(np.float32) * valid, valid


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
