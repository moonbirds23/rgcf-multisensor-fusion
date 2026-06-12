from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_fusion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vel_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervised position/velocity MSE loss.

    loss_pos = MSE(pred_xy, target_xy)
    loss_vel = MSE(pred_v,  target_v)
    total    = loss_pos + vel_weight * loss_vel

    Returns values as Python floats to avoid GPU→CPU syncs in the training loop.
    Uses a single .item() call per loss component (no redundant .cpu()).
    """
    loss_pos = ((pred[:, 0:2] - target[:, 0:2]) ** 2).mean()
    loss_vel = ((pred[:, 2:4] - target[:, 2:4]) ** 2).mean()
    loss = loss_pos + vel_weight * loss_vel

    # Detach + item: .item() already moves to CPU, no need for .cpu() before it
    return loss, {
        "loss_pos": loss_pos.detach().item(),
        "loss_vel": loss_vel.detach().item(),
        "loss_total": loss.detach().item(),
    }


def compute_fusion_loss_with_gate(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    gate_target: torch.Tensor | None = None,
    gate_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    cov_scale: torch.Tensor | None = None,
    vel_weight: float = 0.2,
    gate_weight: float = 0.05,
    gate_prior_weight: float = 0.005,
    gate_prior_mean: float = 0.75,
    cov_prior_weight: float = 0.0,
    cov_sep_weight: float = 0.0,
    cov_fault_normal_margin: float = 1.0,
    fault_weight_loss_weight: float = 0.0,
    fault_weight_margin: float = 0.1,
    balanced_gate_loss: bool = True,
    fault_gate_threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Full fusion loss with gate supervision, priors, and cov penalties."""
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

    cov_prior_loss = pred.new_tensor(0.0)
    cov_sep_loss = pred.new_tensor(0.0)
    mean_cov_scale = pred.new_tensor(0.0)
    mean_cov_scale_fault = pred.new_tensor(0.0)
    mean_cov_scale_normal = pred.new_tensor(0.0)
    if cov_scale is not None:
        if gate_mask is not None:
            supervised_cov = gate_mask > 0.5
        else:
            supervised_cov = torch.ones_like(cov_scale, dtype=torch.bool)

        if torch.any(supervised_cov):
            mean_cov_scale = cov_scale[supervised_cov].mean()
        else:
            mean_cov_scale = cov_scale.mean()

        if gate_target is not None:
            fault_mask = supervised_cov & (gate_target < fault_gate_threshold)
            normal_mask = supervised_cov & (gate_target >= fault_gate_threshold)
            if torch.any(normal_mask):
                normal_cov = cov_scale[normal_mask]
                mean_cov_scale_normal = normal_cov.mean()
                cov_prior_loss = ((normal_cov - 1.0) ** 2).mean()
            if torch.any(fault_mask):
                mean_cov_scale_fault = cov_scale[fault_mask].mean()
            if torch.any(fault_mask) and torch.any(normal_mask):
                cov_sep_loss = F.relu(
                    cov_fault_normal_margin - (mean_cov_scale_fault - mean_cov_scale_normal)
                )
        else:
            cov_prior_loss = ((cov_scale - 1.0) ** 2).mean()

    fault_weight_loss = pred.new_tensor(0.0)
    mean_fault_weight = pred.new_tensor(0.0)
    fault_weight_above_margin_rate = pred.new_tensor(0.0)
    if (
        weights is not None
        and gate_target is not None
        and gate_mask is not None
        and float(fault_weight_loss_weight) > 0.0
    ):
        fault_mask = (gate_mask > 0.5) & (gate_target < fault_gate_threshold)
        if torch.any(fault_mask):
            fault_weights = weights[fault_mask]
            mean_fault_weight = fault_weights.mean()
            excess = F.relu(fault_weights - float(fault_weight_margin))
            fault_weight_loss = (excess ** 2).mean()
            fault_weight_above_margin_rate = (fault_weights > float(fault_weight_margin)).float().mean()

    total = (
        track_loss
        + gate_weight * gate_loss
        + gate_prior_weight * prior_loss
        + cov_prior_weight * cov_prior_loss
        + cov_sep_weight * cov_sep_loss
        + fault_weight_loss_weight * fault_weight_loss
    )

    # Batch all .item() calls (no redundant .cpu() calls)
    info.update({
        "loss_total": total.detach().item(),
        "loss_track": track_loss.detach().item(),
        "loss_gate": gate_loss.detach().item(),
        "loss_gate_fault": gate_loss_fault.detach().item(),
        "loss_gate_normal": gate_loss_normal.detach().item(),
        "loss_gate_prior": prior_loss.detach().item(),
        "mean_gate": mean_gate.detach().item(),
        "loss_cov_prior": cov_prior_loss.detach().item(),
        "loss_cov_sep": cov_sep_loss.detach().item(),
        "loss_fault_weight": fault_weight_loss.detach().item(),
        "mean_fault_weight": mean_fault_weight.detach().item(),
        "fault_weight_above_margin_rate": fault_weight_above_margin_rate.detach().item(),
        "mean_cov_scale": mean_cov_scale.detach().item(),
        "mean_cov_scale_fault": mean_cov_scale_fault.detach().item(),
        "mean_cov_scale_normal": mean_cov_scale_normal.detach().item(),
    })

    return total, info
