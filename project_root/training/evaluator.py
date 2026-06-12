from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.types import ExperimentBundle
from features.dataset import build_dataset_from_sim, build_window_dataset_from_sim
from models.base_fusion import FusionModelBase
from .losses import compute_fusion_loss


def evaluate_loader(
    model: FusionModelBase,
    loader: DataLoader,
    device: torch.device,
    vel_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Evaluate model on a DataLoader with batched inference.

    Returns:
    - loss
    - loss_pos
    - loss_vel
    """
    model.eval()
    total_loss = 0.0
    total_pos = 0.0
    total_vel = 0.0
    n = 0

    with torch.inference_mode():
        for batch in loader:
            post_feat = batch["post_feat"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            meas_feat = None
            if "meas_feat" in batch:
                meas_feat = batch["meas_feat"].to(device, non_blocking=True)

            post_win = None
            meas_win = None
            if "post_win" in batch:
                post_win = batch["post_win"].to(device, non_blocking=True)
                meas_win = batch["meas_win"].to(device, non_blocking=True)

            out = model(
                post_feat=post_feat,
                mask=mask,
                meas_feat=meas_feat,
                return_weights=False,
                post_win=post_win,
                meas_win=meas_win,
            )
            pred = out.pred

            loss, info = compute_fusion_loss(pred, target, vel_weight=vel_weight)

            bs = target.size(0)
            total_loss += info["loss_total"] * bs
            total_pos += info["loss_pos"] * bs
            total_vel += info["loss_vel"] * bs
            n += bs

    return {
        "loss": total_loss / max(n, 1),
        "loss_pos": total_pos / max(n, 1),
        "loss_vel": total_vel / max(n, 1),
    }


def evaluate_single_sim_fusion(
    sim: Dict[str, np.ndarray],
    model: FusionModelBase,
    bundle: ExperimentBundle,
    device: torch.device,
) -> Dict[str, float]:
    """
    Quick evaluation: return summary metrics only.
    """
    res = evaluate_single_sim_fusion_with_timeseries(
        sim=sim,
        model=model,
        bundle=bundle,
        device=device,
    )
    return res["summary"]


def evaluate_single_sim_fusion_with_timeseries(
    sim: Dict[str, np.ndarray],
    model: FusionModelBase,
    bundle: ExperimentBundle,
    device: torch.device,
) -> Dict[str, object]:
    """
    Evaluate a single simulation with batched inference, returning timeseries.

    Uses DataLoader with batching instead of iterating one timestep at a time,
    which eliminates ~1200 individual GPU kernel launches per sim.

    Returns:
    {
        "summary": {...},
        "timeseries": [...]
    }
    """
    model.eval()

    use_temporal = bool(getattr(bundle.model, "use_temporal", False))
    window_size = int(getattr(bundle.model, "window_size", 6))
    eval_batch_size = 256  # Large batch for evaluation throughput

    if use_temporal:
        ds = build_window_dataset_from_sim(sim, bundle, window_size=window_size)
    else:
        ds = build_dataset_from_sim(sim, bundle)

    loader = DataLoader(
        ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    weights_all: List[np.ndarray] = []
    gates_all: List[np.ndarray] = []
    cov_scale_all: List[np.ndarray] = []
    gate_target_all: List[np.ndarray] = []
    gate_mask_all: List[np.ndarray] = []
    valid_all: List[np.ndarray] = []
    ts_indices: List[int] = []

    with torch.inference_mode():
        sample_idx = 0
        for batch in loader:
            post_feat = batch["post_feat"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            meas_feat = None
            if "meas_feat" in batch:
                meas_feat = batch["meas_feat"].to(device, non_blocking=True)

            post_win = None
            meas_win = None
            if "post_win" in batch:
                post_win = batch["post_win"].to(device, non_blocking=True)
                meas_win = batch["meas_win"].to(device, non_blocking=True)

            out = model(
                post_feat=post_feat,
                mask=mask,
                meas_feat=meas_feat,
                return_weights=True,
                post_win=post_win,
                meas_win=meas_win,
            )

            pred = out.pred.detach().cpu().numpy()
            y = target.detach().cpu().numpy()

            preds.append(pred)
            trues.append(y)

            if out.weights is not None:
                weights_all.append(out.weights.detach().cpu().numpy())

            gate = out.aux.get("gate", None)
            if gate is not None:
                gates_all.append(gate.detach().cpu().numpy())

            cov_scale = out.aux.get("cov_scale", None)
            if cov_scale is not None:
                cov_scale_all.append(cov_scale.detach().cpu().numpy())

            if "gate_target" in batch:
                gate_target_all.append(batch["gate_target"].cpu().numpy())
            if "gate_supervision_mask" in batch:
                gate_mask_all.append(batch["gate_supervision_mask"].cpu().numpy())

            valid_all.append(batch["mask"].cpu().numpy())

            bs = y.shape[0]
            for i in range(bs):
                ts_indices.append(sample_idx)
                sample_idx += 1

    # Concatenate batched results
    pred_arr = np.concatenate(preds, axis=0)   # [K,4]
    true_arr = np.concatenate(trues, axis=0)   # [K,4]

    err_pos = pred_arr[:, 0:2] - true_arr[:, 0:2]
    err_full = pred_arr - true_arr

    rmse_pos = float(np.sqrt(np.mean(np.sum(err_pos ** 2, axis=1))))
    rmse_full = float(np.sqrt(np.mean(np.sum(err_full ** 2, axis=1))))

    out_dict: Dict[str, float] = {
        "rmse_gnn_pos": rmse_pos,
        "rmse_gnn_full": rmse_full,
    }

    w_arr = None
    if len(weights_all) > 0:
        w_arr = np.concatenate(weights_all, axis=0)   # [K,N]
        N = w_arr.shape[1]
        for i in range(N):
            out_dict[f"mean_w_s{i+1}"] = float(np.mean(w_arr[:, i]))

    g_arr = None
    if len(gates_all) > 0:
        g_arr = np.concatenate(gates_all, axis=0)     # [K,N]
        v_arr = np.concatenate(valid_all, axis=0)     # [K,N]

        out_dict["mean_gate"] = float(np.mean(g_arr))
        out_dict["std_gate"] = float(np.std(g_arr))

        valid_mask = v_arr > 0.5
        if np.any(valid_mask):
            out_dict["mean_gate_valid_only"] = float(np.mean(g_arr[valid_mask]))
        else:
            out_dict["mean_gate_valid_only"] = 0.0

        N = g_arr.shape[1]
        for i in range(N):
            out_dict[f"mean_g_s{i+1}"] = float(np.mean(g_arr[:, i]))

    c_arr = None
    if len(cov_scale_all) > 0:
        c_arr = np.concatenate(cov_scale_all, axis=0)
        out_dict["mean_cov_scale"] = float(np.mean(c_arr))
        out_dict["std_cov_scale"] = float(np.std(c_arr))
        N = c_arr.shape[1]
        for i in range(N):
            out_dict[f"mean_cov_scale_s{i+1}"] = float(np.mean(c_arr[:, i]))

    gt_arr = None
    gm_arr = None
    if len(gate_target_all) > 0:
        gt_arr = np.concatenate(gate_target_all, axis=0)
    if len(gate_mask_all) > 0:
        gm_arr = np.concatenate(gate_mask_all, axis=0)

    if gt_arr is not None:
        fault_gate_threshold = 0.5 * (
            float(getattr(bundle.model, "normal_gate_target", 0.8))
            + float(getattr(bundle.model, "fault_gate_target", 0.2))
        )
        fault_mask = gt_arr < fault_gate_threshold
        normal_mask = gt_arr >= fault_gate_threshold
        if gm_arr is not None:
            supervised = gm_arr > 0.5
            fault_mask = fault_mask & supervised
            normal_mask = normal_mask & supervised

        if g_arr is not None:
            if np.any(fault_mask):
                out_dict["mean_gate_fault"] = float(np.mean(g_arr[fault_mask]))
            if np.any(normal_mask):
                out_dict["mean_gate_normal"] = float(np.mean(g_arr[normal_mask]))

        if w_arr is not None:
            if np.any(fault_mask):
                out_dict["mean_weight_fault"] = float(np.mean(w_arr[fault_mask]))
            if np.any(normal_mask):
                out_dict["mean_weight_normal"] = float(np.mean(w_arr[normal_mask]))

        if c_arr is not None:
            if np.any(fault_mask):
                out_dict["mean_cov_scale_fault"] = float(np.mean(c_arr[fault_mask]))
            if np.any(normal_mask):
                out_dict["mean_cov_scale_normal"] = float(np.mean(c_arr[normal_mask]))

        if "mean_gate_fault" in out_dict and "mean_gate_normal" in out_dict:
            out_dict["gate_separation_normal_minus_fault"] = float(
                out_dict["mean_gate_normal"] - out_dict["mean_gate_fault"]
            )
        if "mean_weight_fault" in out_dict and "mean_weight_normal" in out_dict:
            out_dict["weight_separation_normal_minus_fault"] = float(
                out_dict["mean_weight_normal"] - out_dict["mean_weight_fault"]
            )
        if "mean_cov_scale_fault" in out_dict and "mean_cov_scale_normal" in out_dict:
            denom = max(float(out_dict["mean_cov_scale_normal"]), 1e-12)
            out_dict["cov_scale_fault_to_normal_ratio"] = float(
                out_dict["mean_cov_scale_fault"] / denom
            )

    def _add_region_stats(prefix: str, region_mask: np.ndarray):
        if not np.any(region_mask):
            return
        if g_arr is not None:
            vals = g_arr[region_mask]
            if vals.size > 0:
                out_dict[f"{prefix}_mean_gate"] = float(np.mean(vals))
        if w_arr is not None:
            vals = w_arr[region_mask]
            if vals.size > 0:
                out_dict[f"{prefix}_mean_weight"] = float(np.mean(vals))
        if c_arr is not None:
            vals = c_arr[region_mask]
            if vals.size > 0:
                out_dict[f"{prefix}_mean_cov_scale"] = float(np.mean(vals))

    fault_active_arr = None
    if "fault_active_mask" in sim:
        fault_active_arr = np.asarray(sim["fault_active_mask"], dtype=np.float32)
        active_any = np.any(fault_active_arr > 0.5, axis=1)
        active_idx = np.flatnonzero(active_any)
        if active_idx.size > 0:
            k0 = int(active_idx[0])
            k1 = int(active_idx[-1])
            valid_arr = np.concatenate(valid_all, axis=0) > 0.5
            pre_mask = np.zeros_like(valid_arr, dtype=bool)
            fault_region_mask = (fault_active_arr > 0.5) & valid_arr
            post_mask = np.zeros_like(valid_arr, dtype=bool)
            pre_mask[:k0, :] = valid_arr[:k0, :]
            post_mask[k1 + 1:, :] = valid_arr[k1 + 1:, :]

            out_dict["fault_window_t0"] = float(ds.t[k0])
            out_dict["fault_window_t1"] = float(ds.t[k1])
            out_dict["fault_window_steps"] = int(active_idx.size)
            _add_region_stats("pre_fault", pre_mask)
            _add_region_stats("fault_window", fault_region_mask)
            _add_region_stats("post_fault", post_mask)

    timeseries: List[Dict[str, float]] = []
    K = len(ds)
    n_nodes = 0
    if w_arr is not None:
        n_nodes = w_arr.shape[1]
    elif g_arr is not None:
        n_nodes = g_arr.shape[1]
    elif c_arr is not None:
        n_nodes = c_arr.shape[1]

    # per-timestep position error
    error_pos_arr = np.sqrt(np.sum((pred_arr[:, 0:2] - true_arr[:, 0:2]) ** 2, axis=1))
    error_full_arr = np.sqrt(np.sum((pred_arr - true_arr) ** 2, axis=1))

    fault_active_any_arr = np.zeros(K, dtype=np.float64)
    if "fault_active_mask" in sim:
        fa = np.asarray(sim["fault_active_mask"])
        fault_active_any_arr = (fa.max(axis=1) > 0.5).astype(np.float64)

    # weight behavior metrics
    if w_arr is not None and fault_active_any_arr.sum() > 0:
        fault_window = fault_active_any_arr > 0.5
        if fault_window.any():
            w_fault = w_arr[fault_window]
            if "fault_active_mask" in sim:
                fa_arr = np.asarray(sim["fault_active_mask"])
                fa_fault = fa_arr[fault_window]
                fa_fault_bool = fa_fault > 0.5

                w_max_idx = np.argmax(w_fault, axis=1)
                is_fault_top = fa_fault_bool[np.arange(w_fault.shape[0]), w_max_idx]
                out_dict["fault_top1_weight_rate"] = float(np.mean(is_fault_top))

                w_fault_sensors = w_fault[fa_fault_bool]
                if w_fault_sensors.size > 0:
                    out_dict["fault_weight_below_equal_rate"] = float(np.mean(w_fault_sensors < 0.25))
                    out_dict["fault_weight_below_010_rate"] = float(np.mean(w_fault_sensors < 0.10))

    # error window metrics
    def _error_window_metrics(err_arr, fault_any_arr, t_arr, t0, t1):
        metrics = {}
        metrics["overall_rmse_pos"] = float(np.sqrt(np.mean(err_arr ** 2)))
        metrics["overall_p95_error_pos"] = float(np.percentile(err_arr, 95))
        metrics["overall_max_error_pos"] = float(np.max(err_arr))

        fault_sel = fault_any_arr > 0.5
        if fault_sel.any():
            metrics["fault_window_rmse_pos"] = float(np.sqrt(np.mean(err_arr[fault_sel] ** 2)))
            metrics["fault_window_p95_error_pos"] = float(np.percentile(err_arr[fault_sel], 95))
            metrics["fault_window_max_error_pos"] = float(np.max(err_arr[fault_sel]))
        else:
            metrics["fault_window_rmse_pos"] = float("nan")
            metrics["fault_window_p95_error_pos"] = float("nan")
            metrics["fault_window_max_error_pos"] = float("nan")

        pre_sel = t_arr < t0
        post_sel = t_arr > t1
        if pre_sel.any():
            metrics["pre_fault_rmse_pos"] = float(np.sqrt(np.mean(err_arr[pre_sel] ** 2)))
        else:
            metrics["pre_fault_rmse_pos"] = float("nan")
        if post_sel.any():
            metrics["post_fault_rmse_pos"] = float(np.sqrt(np.mean(err_arr[post_sel] ** 2)))
        else:
            metrics["post_fault_rmse_pos"] = float("nan")

        if post_sel.any() and pre_sel.any():
            pre_median = float(np.median(err_arr[pre_sel]))
            post_idx = np.flatnonzero(post_sel)
            post_t = t_arr[post_sel]
            post_err = err_arr[post_sel]
            recovered = np.flatnonzero(post_err <= pre_median)
            if recovered.size > 0:
                metrics["recovery_time_error"] = float(post_t[recovered[0]] - t1)
            else:
                metrics["recovery_time_error"] = float("nan")
        else:
            metrics["recovery_time_error"] = float("nan")

        return metrics

    t_arr = np.array([float(ds.t[i]) for i in range(K)], dtype=np.float64)
    t0, t1 = 30.0, 70.0
    if "fault_window_t0" in out_dict:
        t0 = float(out_dict["fault_window_t0"])
        t1 = float(out_dict["fault_window_t1"])

    err_win = _error_window_metrics(error_pos_arr, fault_active_any_arr, t_arr, t0, t1)
    for k, v in err_win.items():
        out_dict[k] = v

    # Build per-timestep timeseries rows
    for k in range(K):
        row: Dict[str, float] = {
            "k": int(k),
            "t": float(ds.t[k]),
            "truth_px": float(true_arr[k, 0]),
            "truth_py": float(true_arr[k, 1]),
            "truth_vx": float(true_arr[k, 2]),
            "truth_vy": float(true_arr[k, 3]),
            "pred_px": float(pred_arr[k, 0]),
            "pred_py": float(pred_arr[k, 1]),
            "pred_vx": float(pred_arr[k, 2]),
            "pred_vy": float(pred_arr[k, 3]),
            "error_pos": float(error_pos_arr[k]),
            "error_full": float(error_full_arr[k]),
            "fault_active_any": float(fault_active_any_arr[k]),
        }

        if g_arr is not None:
            for i in range(n_nodes):
                row[f"g_s{i+1}"] = float(g_arr[k, i])

        if w_arr is not None:
            for i in range(n_nodes):
                row[f"w_s{i+1}"] = float(w_arr[k, i])

        if c_arr is not None:
            for i in range(n_nodes):
                row[f"cov_scale_s{i+1}"] = float(c_arr[k, i])

        if gt_arr is not None:
            for i in range(n_nodes):
                row[f"gate_target_s{i+1}"] = float(gt_arr[k, i])

        if gm_arr is not None:
            for i in range(n_nodes):
                row[f"gate_supervision_mask_s{i+1}"] = float(gm_arr[k, i])

        if "fault_active_mask" in sim:
            fault_active = np.asarray(sim["fault_active_mask"])
            for i in range(n_nodes):
                row[f"fault_active_s{i+1}"] = float(fault_active[k, i])

        if len(valid_all) > 0:
            v_arr = np.concatenate(valid_all, axis=0)
            for i in range(n_nodes):
                row[f"valid_s{i+1}"] = float(v_arr[k, i])

        timeseries.append(row)

    return {
        "summary": out_dict,
        "timeseries": timeseries,
    }
