from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import copy
import numpy as np
import torch

from core.types import ExperimentBundle
from simulation.runner import run_single_simulation
from .evaluator import evaluate_single_sim_fusion
from .trainer import train_fusion_model


@dataclass
class RepeatSummary:
    all_results: List[Dict]
    summary: Dict[str, Dict[str, float]]


def shift_bundle_seed_ranges(
    bundle: ExperimentBundle,
    run_seed_offset: int,
    stride: int = 100,
) -> ExperimentBundle:
    """
    将 train/val/test seed 区间整体平移。
    仅在“每轮使用不同数据集”的 repeat 语义下使用。
    """
    out = copy.deepcopy(bundle)

    shift = int(stride * run_seed_offset)

    out.train.train_seed_start = int(bundle.train.train_seed_start + shift)
    out.train.train_seed_end = int(bundle.train.train_seed_end + shift)

    out.train.val_seed_start = int(bundle.train.val_seed_start + shift)
    out.train.val_seed_end = int(bundle.train.val_seed_end + shift)

    out.train.test_seed_start = int(bundle.train.test_seed_start + shift)
    out.train.test_seed_end = int(bundle.train.test_seed_end + shift)

    return out


def _summarize_numeric_fields(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    对 repeat 结果中的数值字段做 mean/std/min/max 汇总。
    """
    if len(results) == 0:
        return {}

    keys = results[0].keys()
    summary = {}

    for k in keys:
        vals = [r[k] for r in results if isinstance(r.get(k, None), (int, float))]
        if len(vals) == 0:
            continue

        arr = np.asarray(vals, dtype=np.float64)
        summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return summary


def repeat_training_evaluation(
    bundle: ExperimentBundle,
    *,
    repeat_runs: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    early_stop_patience: int = 12,
    vel_weight: float = 0.2,
    dataset_store_root: str = "dataset_store",
    dataset_id: str | None = None,
    dataset_dir: str | None = None,
    use_latest_matching_dataset: bool = False,
) -> RepeatSummary:
    """
    多次重复训练 + quick eval。

    当前兼容两种 repeat 语义：

    1) 固定数据集重复训练
       - 当显式给了 dataset_dir 或 dataset_id 时启用
       - 所有轮使用同一份 dataset
       - 不再做 seed range shift
       - 测试训练稳定性 / 初始化波动

    2) shifted dataset repeat
       - 当使用 use_latest_matching_dataset=True 时启用
       - 每轮对 train/val/test seed ranges 整体平移
       - 并基于 shifted bundle 去匹配对应的数据集仓库
       - 测试多 seed 数据扰动下的稳定性

    quick eval 语义保持不变：
    - 每次训练结束后，都在固定展示 sim（原始 bundle.base.runtime.seed）上做一次 quick eval
    """
    repeat_runs = int(bundle.train.repeat_runs if repeat_runs is None else repeat_runs)

    # 显式给定 dataset_dir / dataset_id 时，认为用户要“固定数据集重复训练”
    fixed_dataset_mode = (dataset_dir is not None) or (dataset_id is not None)

    all_results: List[Dict] = []

    for r in range(repeat_runs):
        print(f"\n----- Repeat Run {r+1}/{repeat_runs} -----")

        if fixed_dataset_mode:
            # 固定同一份 dataset，不再 shift，避免“显示 seed 变了但实际数据没变”
            bundle_run = copy.deepcopy(bundle)
            seed_offset = 0
        else:
            # 每轮使用不同 seed ranges，对应不同 dataset store
            bundle_run = shift_bundle_seed_ranges(
                bundle,
                run_seed_offset=r,
                stride=bundle.train.repeat_seed_stride,
            )
            seed_offset = r

        train_res = train_fusion_model(
            bundle=bundle_run,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            early_stop_patience=early_stop_patience,
            vel_weight=vel_weight,
            dataset_store_root=dataset_store_root,
            dataset_id=dataset_id,
            dataset_dir=dataset_dir,
            use_latest_matching_dataset=use_latest_matching_dataset,
        )

        # quick evaluation on fixed display sim (using original bundle runtime seed)
        sim_quick = run_single_simulation(bundle).sim
        metrics_gnn = evaluate_single_sim_fusion(
            sim=sim_quick,
            model=train_res.model,
            bundle=bundle,
            device=torch.device(bundle.base.runtime.device),
        )

        out = {
            "run_idx": r + 1,
            "seed_offset": seed_offset,

            "train_seed_start": int(bundle_run.train.train_seed_start),
            "train_seed_end": int(bundle_run.train.train_seed_end),
            "val_seed_start": int(bundle_run.train.val_seed_start),
            "val_seed_end": int(bundle_run.train.val_seed_end),
            "test_seed_start": int(bundle_run.train.test_seed_start),
            "test_seed_end": int(bundle_run.train.test_seed_end),

            "best_val_loss": float(train_res.train_info["best_val_loss"]),
            "test_loss": float(train_res.train_info["test_loss"]),
            "best_epoch": float(train_res.train_info["best_epoch"]),
            "test_loss_pos": float(train_res.train_info.get("test_loss_pos", 0.0)),
            "test_loss_vel": float(train_res.train_info.get("test_loss_vel", 0.0)),

            "rmse_gnn_pos": float(metrics_gnn["rmse_gnn_pos"]),
            "rmse_gnn_full": float(metrics_gnn["rmse_gnn_full"]),
        }

        if getattr(train_res, "dataset_store_dir", ""):
            out["dataset_store_dir"] = str(train_res.dataset_store_dir)

        # 节点均值权重（若有）
        for k, v in metrics_gnn.items():
            if k.startswith("mean_w_s"):
                out[k] = float(v)

        all_results.append(out)

        print("Run result:")
        for k, v in out.items():
            if isinstance(v, float):
                print(f"{k:>20s}: {v:.6f}")
            else:
                print(f"{k:>20s}: {v}")

    summary = _summarize_numeric_fields(all_results)
    return RepeatSummary(all_results=all_results, summary=summary)