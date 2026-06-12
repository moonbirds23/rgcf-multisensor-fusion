from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from data.dataset_store import save_raw_dataset_store
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.types import ExperimentBundle
from features.dataset import build_dataset_list_from_sims, build_window_dataset_list_from_sims
from models.model_factory import build_model_from_bundle
from simulation.runner import run_single_simulation
from data.sim_cache import get_or_run_cached_sim
from .evaluator import evaluate_loader
from .losses import compute_fusion_loss, compute_fusion_loss_with_gate

from data.dataset_store import (
    load_raw_dataset_store,
    find_dataset_store_by_id,
    find_latest_matching_dataset_store,
)

@dataclass
class TrainHistoryItem:
    epoch: int
    train_loss: float
    train_loss_pos: float
    train_loss_vel: float
    val_loss: float
    val_loss_pos: float
    val_loss_vel: float
    lr: float
    train_loss_gate: float = 0.0
    train_loss_gate_prior: float = 0.0
    train_mean_gate: float = 0.0
    train_loss_cov_prior: float = 0.0
    train_loss_cov_sep: float = 0.0
    train_mean_cov_scale: float = 0.0
    train_loss_fault_weight: float = 0.0
    train_mean_fault_weight: float = 0.0


@dataclass
class TrainResult:
    model: torch.nn.Module
    train_info: Dict[str, float]
    history: List[Dict]
    train_sims: List[Dict]
    val_sims: List[Dict]
    test_sims: List[Dict]
    dataset_store_dir: str = ""


def _build_seed_list(start: int, end: int) -> List[int]:
    return list(range(int(start), int(end) + 1))


def _set_model_seed(seed: int | None):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clone_bundle_with_runtime_seed(bundle: ExperimentBundle, seed: int) -> ExperimentBundle:
    """
    基于同一个实验 bundle，仅替换 runtime seed。
    用于构造多条独立 rollout。
    """
    out = copy.deepcopy(bundle)
    out.base.runtime.seed = int(seed)
    return out


def build_sim_list_from_seed_range(
    bundle: ExperimentBundle,
    seed_start: int,
    seed_end: int,
    split_name: str = "train",
    *,
    use_sim_cache: bool = False,
    sim_cache_root: str = "sim_cache",
    force_regenerate_sims: bool = False,
) -> List[Dict]:
    seeds = _build_seed_list(seed_start, seed_end)
    sims = []

    print(f"[{split_name}] start building sims, total={len(seeds)}")

    for i, sd in enumerate(seeds, start=1):
        print(f"[{split_name}] sim {i}/{len(seeds)} | seed={sd}")
        if use_sim_cache:
            sim = get_or_run_cached_sim(
                bundle,
                sd,
                root_dir=sim_cache_root,
                force=force_regenerate_sims,
            )
        else:
            b_i = clone_bundle_with_runtime_seed(bundle, sd)
            run_out = run_single_simulation(b_i)
            sim = run_out.sim

        sim["split_name"] = split_name
        sim["seed"] = int(sd)
        sims.append(sim)

    print(f"[{split_name}] sims ready, total={len(sims)}")
    return sims


def train_fusion_model(
    bundle: ExperimentBundle,
    *,
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
) -> TrainResult:
    """
    Step 8 统一训练入口。

    默认尽量对齐当前 clean baseline 的训练行为：
    - Adam
    - lr 默认取 bundle.train.lr
    - weight_decay=1e-5
    - ReduceLROnPlateau(factor=0.5, patience=4)
    - grad_clip=1.0
    - early_stop_patience=12
    - test 评估基于 best val model
    """
    device = torch.device(bundle.base.runtime.device)
    _set_model_seed(getattr(bundle.train, "model_seed", None))

    epochs = int(bundle.train.epochs if epochs is None else epochs)
    lr = float(bundle.train.lr if lr is None else lr)
    batch_size = int(bundle.train.batch_size if batch_size is None else batch_size)

    train_seeds = _build_seed_list(bundle.train.train_seed_start, bundle.train.train_seed_end)
    val_seeds = _build_seed_list(bundle.train.val_seed_start, bundle.train.val_seed_end)
    test_seeds = _build_seed_list(bundle.train.test_seed_start, bundle.train.test_seed_end)

    print(f"Train seeds: {train_seeds}")
    print(f"Val   seeds: {val_seeds}")
    print(f"Test  seeds: {test_seeds}")

    loaded_dataset_dir = ""

    if dataset_dir is not None:
        loaded_dataset_dir = str(dataset_dir)
    elif dataset_id is not None:
        loaded_dataset_dir = str(find_dataset_store_by_id(dataset_id, root_dir=dataset_store_root))
    elif use_latest_matching_dataset:
        matched = find_latest_matching_dataset_store(bundle, root_dir=dataset_store_root)
        if matched is None:
            raise FileNotFoundError(
                "No matching dataset store found. "
                "Please run --mode generate_dataset first, or specify --dataset_id / --dataset_dir."
            )
        loaded_dataset_dir = str(matched)
    else:
        raise ValueError(
            "Train mode no longer builds sims directly. "
            "Please provide one of: "
            "--dataset_dir, --dataset_id, or --use_latest_matching_dataset."
        )

    print(f"[dataset_store] loading from: {loaded_dataset_dir}")
    ds_obj = load_raw_dataset_store(loaded_dataset_dir)

    train_sims = ds_obj["train_sims"]
    val_sims = ds_obj["val_sims"]
    test_sims = ds_obj["test_sims"]

    use_temporal = bool(getattr(bundle.model, "use_temporal", False))
    window_size = int(getattr(bundle.model, "window_size", 6))
    use_gate_supervision = bool(getattr(bundle.model, "use_gate_supervision", False))
    use_fault_weight_loss = float(getattr(bundle.model, "fault_weight_loss_weight", 0.0)) > 0.0

    def _build_ds(sims):
        if use_temporal:
            return build_window_dataset_list_from_sims(sims, bundle, window_size=window_size)
        return build_dataset_list_from_sims(sims, bundle)

    print("[dataset] building train dataset...")
    train_ds = _build_ds(train_sims)
    print(f"[dataset] train dataset ready, size={len(train_ds)}")

    print("[dataset] building val dataset...")
    val_ds = _build_ds(val_sims)
    print(f"[dataset] val dataset ready, size={len(val_ds)}")

    print("[dataset] building test dataset...")
    test_ds = _build_ds(test_sims)
    print(f"[dataset] test dataset ready, size={len(test_ds)}")

    loader_generator = None
    if getattr(bundle.train, "model_seed", None) is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(bundle.train.model_seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = build_model_from_bundle(bundle).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    history: List[Dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_pos_sum = 0.0
        train_vel_sum = 0.0
        train_gate_sum = 0.0
        train_gate_prior_sum = 0.0
        train_mean_gate_sum = 0.0
        train_cov_prior_sum = 0.0
        train_cov_sep_sum = 0.0
        train_mean_cov_scale_sum = 0.0
        train_fault_weight_sum = 0.0
        train_mean_fault_weight_sum = 0.0
        n = 0

        for batch in train_loader:
            post_feat = batch["post_feat"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            meas_feat = None
            if "meas_feat" in batch:
                meas_feat = batch["meas_feat"].to(device)

            post_win = None
            meas_win = None
            if "post_win" in batch:
                post_win = batch["post_win"].to(device)
                meas_win = batch["meas_win"].to(device)

            out = model(
                post_feat=post_feat,
                mask=mask,
                meas_feat=meas_feat,
                return_weights=use_gate_supervision or use_fault_weight_loss,
                post_win=post_win,
                meas_win=meas_win,
            )
            pred = out.pred

            if use_gate_supervision:
                gate_target = batch.get("gate_target", None)
                gate_mask = batch.get("gate_supervision_mask", None)
                if gate_target is not None:
                    gate_target = gate_target.to(device)
                if gate_mask is not None:
                    gate_mask = gate_mask.to(device)

                loss, info = compute_fusion_loss_with_gate(
                    pred,
                    target,
                    gate=out.aux.get("gate", None),
                    gate_target=gate_target,
                    gate_mask=gate_mask,
                    weights=out.weights,
                    cov_scale=out.aux.get("cov_scale", None),
                    vel_weight=vel_weight,
                    gate_weight=float(getattr(bundle.model, "gate_supervision_weight", 0.05)),
                    gate_prior_weight=float(getattr(bundle.model, "gate_prior_weight", 0.005)),
                    gate_prior_mean=float(getattr(bundle.model, "gate_prior_mean", 0.75)),
                    cov_prior_weight=float(getattr(bundle.model, "cov_prior_weight", 0.0)),
                    cov_sep_weight=float(getattr(bundle.model, "cov_sep_weight", 0.0)),
                    cov_fault_normal_margin=float(getattr(bundle.model, "cov_fault_normal_margin", 1.0)),
                    fault_weight_loss_weight=float(getattr(bundle.model, "fault_weight_loss_weight", 0.0)),
                    fault_weight_margin=float(getattr(bundle.model, "fault_weight_margin", 0.1)),
                    balanced_gate_loss=bool(getattr(bundle.model, "use_balanced_gate_loss", True)),
                    fault_gate_threshold=0.5 * (
                        float(getattr(bundle.model, "normal_gate_target", 0.8))
                        + float(getattr(bundle.model, "fault_gate_target", 0.2))
                    ),
                )
            else:
                loss, info = compute_fusion_loss(pred, target, vel_weight=vel_weight)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

            bs = target.size(0)
            train_loss_sum += info["loss_total"] * bs
            train_pos_sum += info["loss_pos"] * bs
            train_vel_sum += info["loss_vel"] * bs
            train_gate_sum += info.get("loss_gate", 0.0) * bs
            train_gate_prior_sum += info.get("loss_gate_prior", 0.0) * bs
            train_mean_gate_sum += info.get("mean_gate", 0.0) * bs
            train_cov_prior_sum += info.get("loss_cov_prior", 0.0) * bs
            train_cov_sep_sum += info.get("loss_cov_sep", 0.0) * bs
            train_mean_cov_scale_sum += info.get("mean_cov_scale", 0.0) * bs
            train_fault_weight_sum += info.get("loss_fault_weight", 0.0) * bs
            train_mean_fault_weight_sum += info.get("mean_fault_weight", 0.0) * bs
            n += bs

        train_metrics = {
            "loss": train_loss_sum / max(n, 1),
            "loss_pos": train_pos_sum / max(n, 1),
            "loss_vel": train_vel_sum / max(n, 1),
            "loss_gate": train_gate_sum / max(n, 1),
            "loss_gate_prior": train_gate_prior_sum / max(n, 1),
            "mean_gate": train_mean_gate_sum / max(n, 1),
            "loss_cov_prior": train_cov_prior_sum / max(n, 1),
            "loss_cov_sep": train_cov_sep_sum / max(n, 1),
            "mean_cov_scale": train_mean_cov_scale_sum / max(n, 1),
            "loss_fault_weight": train_fault_weight_sum / max(n, 1),
            "mean_fault_weight": train_mean_fault_weight_sum / max(n, 1),
        }

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            vel_weight=vel_weight,
        )
        scheduler.step(val_metrics["loss"])

        current_lr = float(opt.param_groups[0]["lr"])

        history_item = TrainHistoryItem(
            epoch=ep,
            train_loss=float(train_metrics["loss"]),
            train_loss_pos=float(train_metrics["loss_pos"]),
            train_loss_vel=float(train_metrics["loss_vel"]),
            val_loss=float(val_metrics["loss"]),
            val_loss_pos=float(val_metrics["loss_pos"]),
            val_loss_vel=float(val_metrics["loss_vel"]),
            lr=current_lr,
            train_loss_gate=float(train_metrics["loss_gate"]),
            train_loss_gate_prior=float(train_metrics["loss_gate_prior"]),
            train_mean_gate=float(train_metrics["mean_gate"]),
            train_loss_cov_prior=float(train_metrics["loss_cov_prior"]),
            train_loss_cov_sep=float(train_metrics["loss_cov_sep"]),
            train_mean_cov_scale=float(train_metrics["mean_cov_scale"]),
            train_loss_fault_weight=float(train_metrics["loss_fault_weight"]),
            train_mean_fault_weight=float(train_metrics["mean_fault_weight"]),
        )
        history.append(asdict(history_item))

        gate_log = ""
        if use_gate_supervision:
            gate_log = (
                f", gate={train_metrics['loss_gate']:.6f}, "
                f"prior={train_metrics['loss_gate_prior']:.6f}, "
                f"mean_gate={train_metrics['mean_gate']:.4f}, "
                f"cov_prior={train_metrics['loss_cov_prior']:.6f}, "
                f"cov_sep={train_metrics['loss_cov_sep']:.6f}, "
                f"mean_cov={train_metrics['mean_cov_scale']:.4f}, "
                f"fault_w={train_metrics['loss_fault_weight']:.6f}"
            )
        print(
            f"[Epoch {ep:03d}] "
            f"train={train_metrics['loss']:.6f} "
            f"(pos={train_metrics['loss_pos']:.6f}, vel={train_metrics['loss_vel']:.6f}{gate_log}) | "
            f"val={val_metrics['loss']:.6f} "
            f"(pos={val_metrics['loss_pos']:.6f}, vel={val_metrics['loss_vel']:.6f}) | "
            f"lr={current_lr:.2e}"
        )

        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if patience >= early_stop_patience:
            print(f"Early stopping at epoch {ep}, best val loss = {best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        device=device,
        vel_weight=vel_weight,
    )

    print(
        f"[Best Model Test] loss={test_metrics['loss']:.6f}, "
        f"pos={test_metrics['loss_pos']:.6f}, vel={test_metrics['loss_vel']:.6f}"
    )

    train_info = {
        "best_val_loss": float(best_val),
        "test_loss": float(test_metrics["loss"]),
        "best_epoch": float(best_epoch),
        "test_loss_pos": float(test_metrics["loss_pos"]),
        "test_loss_vel": float(test_metrics["loss_vel"]),
    }
    dataset_store_dir = loaded_dataset_dir

    return TrainResult(
        model=model,
        train_info=train_info,
        history=history,
        train_sims=train_sims,
        val_sims=val_sims,
        test_sims=test_sims,
        dataset_store_dir=dataset_store_dir,
    )
