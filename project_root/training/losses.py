from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_fusion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vel_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    当前 clean baseline 对应的监督损失：

    loss_pos = MSE(pred_xy, target_xy)
    loss_vel = MSE(pred_v,  target_v)
    total    = loss_pos + vel_weight * loss_vel

    默认保持与当前 clean 版本一致：
      vel_weight = 0.2
    """
    loss_pos = ((pred[:, 0:2] - target[:, 0:2]) ** 2).mean()
    loss_vel = ((pred[:, 2:4] - target[:, 2:4]) ** 2).mean()
    loss = loss_pos + vel_weight * loss_vel

    return loss, {
        "loss_pos": float(loss_pos.detach().cpu().item()),
        "loss_vel": float(loss_vel.detach().cpu().item()),
        "loss_total": float(loss.detach().cpu().item()),
    }


def compute_fusion_loss_with_gate(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    gate_target: torch.Tensor | None = None,
    gate_mask: torch.Tensor | None = None,
    vel_weight: float = 0.2,
    gate_weight: float = 0.05,
    gate_prior_weight: float = 0.005,
    gate_prior_mean: float = 0.75,
    balanced_gate_loss: bool = True,
    fault_gate_threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    track_loss, info = compute_fusion_loss(
        pred,
        target,
        vel_weight=vel_weight,
    )

    gate_loss = pred.new_tensor(0.0)
    gate_loss_fault = pred.new_tensor(0.0)
    gate_loss_normal = pred.new_tensor(0.0)
    if gate is not None and gate_target is not None and gate_mask is not None:
        gate_clamped = gate.clamp(1e-4, 1.0 - 1e-4)
        bce = F.binary_cross_entropy(
            gate_clamped,
            gate_target,
            reduction="none",
        )
        supervised = gate_mask > 0.5
        fault_mask = supervised & (gate_target < fault_gate_threshold)
        normal_mask = supervised & (gate_target >= fault_gate_threshold)

        if balanced_gate_loss and torch.any(fault_mask) and torch.any(normal_mask):
            gate_loss_fault = bce[fault_mask].mean()
            gate_loss_normal = bce[normal_mask].mean()
            gate_loss = 0.5 * (gate_loss_fault + gate_loss_normal)
        else:
            denom = gate_mask.sum().clamp_min(1.0)
            gate_loss = (bce * gate_mask).sum() / denom
            if torch.any(fault_mask):
                gate_loss_fault = bce[fault_mask].mean()
            if torch.any(normal_mask):
                gate_loss_normal = bce[normal_mask].mean()

    prior_loss = pred.new_tensor(0.0)
    mean_gate = pred.new_tensor(0.0)
    if gate is not None:
        if gate_mask is not None:
            denom = gate_mask.sum().clamp_min(1.0)
            mean_gate = (gate * gate_mask).sum() / denom
        else:
            mean_gate = gate.mean()
        prior_loss = (mean_gate - gate_prior_mean) ** 2

    total = track_loss + gate_weight * gate_loss + gate_prior_weight * prior_loss

    info.update({
        "loss_total": float(total.detach().cpu().item()),
        "loss_track": float(track_loss.detach().cpu().item()),
        "loss_gate": float(gate_loss.detach().cpu().item()),
        "loss_gate_fault": float(gate_loss_fault.detach().cpu().item()),
        "loss_gate_normal": float(gate_loss_normal.detach().cpu().item()),
        "loss_gate_prior": float(prior_loss.detach().cpu().item()),
        "mean_gate": float(mean_gate.detach().cpu().item()),
    })

    return total, info
