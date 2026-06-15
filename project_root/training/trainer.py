from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from data.dataset_store import save_raw_dataset_store
import copy
import os
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

# ---------------------------------------------------------------------------
# GPU-optimized defaults (tuned for dual RTX 3090 Ti, 24 GB VRAM each)
# ---------------------------------------------------------------------------
_DEFAULT_NUM_WORKERS = int(os.environ.get("RGCF_NUM_WORKERS", "2"))
_DEFAULT_EVAL_BATCH_SIZE = int(os.environ.get("RGCF_EVAL_BATCH_SIZE", "256"))
_USE_TORCH_COMPILE = os.environ.get("RGCF_DISABLE_COMPILE", "0") != "1"

# ---------------------------------------------------------------------------
# Parallel simulation helpers
# ---------------------------------------------------------------------------

def _run_one_sim_worker(args: Tuple[ExperimentBundle, int, str, str, bool, str]):
    """Pickleable worker for parallel simulation generation."""
    bundle, seed, split_name, sim_cache_root, force_regenerate, _ = args
    # Re-seed numpy in the worker process
    np.random.seed(None)
    random.seed()

    if sim_cache_root:
        sim = get_or_run_cached_sim(
            bundle, seed, root_dir=sim_cache_root, force=force_regenerate,
        )
    else:
        b_i = clone_bundle_with_runtime_seed(bundle, seed)
        run_out = run_single_simulation(b_i)
        sim = run_out.sim

    sim["split_name"] = split_name
    sim["seed"] = int(seed)
    return sim


def _build_sims_parallel(
    bundle: ExperimentBundle,
    seeds: List[int],
    split_name: str,
    *,
    use_sim_cache: bool = False,
    sim_cache_root: str = "sim_cache",
    force_regenerate_sims: bool = False,
    max_workers: int | None = None,
) -> List[Dict]:
    """Build simulations in parallel using ProcessPoolExecutor.

    Falls back to sequential execution on pickling errors (e.g. complex bundle
    objects that can't cross process boundaries).
    """
    if max_workers is None:
        max_workers = min(int(os.environ.get("RGCF_SIM_WORKERS", "4")), len(seeds), os.cpu_count() or 4)

    if max_workers <= 1 or len(seeds) <= 1:
        return _build_sims_sequential(
            bundle, seeds, split_name,
            use_sim_cache=use_sim_cache,
            sim_cache_root=sim_cache_root,
            force_regenerate_sims=force_regenerate_sims,
        )

    cache_root = sim_cache_root if use_sim_cache else ""
    task_args = [
        (bundle, int(sd), split_name, cache_root, bool(force_regenerate_sims), "")
        for sd in seeds
    ]

    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        sims_by_seed: Dict[int, Dict] = {}
        # Use 'spawn' on Windows to avoid CUDA fork issues in worker children
        ctx = None
        if os.name == "nt":
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = {ex.submit(_run_one_sim_worker, args): args[1] for args in task_args}
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    sims_by_seed[seed] = fut.result()
                except Exception as exc:
                    print(f"[{split_name}] sim seed={seed} FAILED in worker: {exc}")
                    raise

        # Return in original seed order
        return [sims_by_seed[sd] for sd in seeds]

    except Exception as exc:
        print(f"[{split_name}] parallel sim generation failed ({exc}), falling back to sequential")
        return _build_sims_sequential(
            bundle, seeds, split_name,
            use_sim_cache=use_sim_cache,
            sim_cache_root=sim_cache_root,
            force_regenerate_sims=force_regenerate_sims,
        )


def _build_sims_sequential(
    bundle: ExperimentBundle,
    seeds: List[int],
    split_name: str,
    *,
    use_sim_cache: bool = False,
    sim_cache_root: str = "sim_cache",
    force_regenerate_sims: bool = False,
) -> List[Dict]:
    """Original sequential simulation builder (fallback)."""
    sims = []
    for i, sd in enumerate(seeds, start=1):
        print(f"[{split_name}] sim {i}/{len(seeds)} | seed={sd}")
        if use_sim_cache:
            sim = get_or_run_cached_sim(
                bundle, sd, root_dir=sim_cache_root, force=force_regenerate_sims,
            )
        else:
            b_i = clone_bundle_with_runtime_seed(bundle, sd)
            run_out = run_single_simulation(b_i)
            sim = run_out.sim

        sim["split_name"] = split_name
        sim["seed"] = int(sd)
        sims.append(sim)
    return sims


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
    train_loss_risk: float = 0.0
    train_loss_overconf: float = 0.0
    train_loss_underconf: float = 0.0
    train_loss_quarantine: float = 0.0
    train_loss_cap: float = 0.0
    train_loss_fused_nll: float = 0.0
    train_valid_drift_aug_count: float = 0.0


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
    """Clone experiment bundle with a new runtime seed for independent rollouts."""
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
    parallel: bool = True,
    max_workers: int | None = None,
) -> List[Dict]:
    """Build simulation rollouts from a seed range.

    Args:
        parallel: Use ProcessPoolExecutor for parallel generation (default True).
        max_workers: Max parallel workers. Defaults to RGCF_SIM_WORKERS env or 4.
    """
    seeds = _build_seed_list(seed_start, seed_end)
    print(f"[{split_name}] start building sims, total={len(seeds)}")

    if parallel and len(seeds) > 1:
        sims = _build_sims_parallel(
            bundle, seeds, split_name,
            use_sim_cache=use_sim_cache,
            sim_cache_root=sim_cache_root,
            force_regenerate_sims=force_regenerate_sims,
            max_workers=max_workers,
        )
    else:
        sims = _build_sims_sequential(
            bundle, seeds, split_name,
            use_sim_cache=use_sim_cache,
            sim_cache_root=sim_cache_root,
            force_regenerate_sims=force_regenerate_sims,
        )

    print(f"[{split_name}] sims ready, total={len(sims)}")
    return sims


def _collate_to_device(batch: List[Dict], device: torch.device) -> Dict:
    """Custom collate that stacks and transfers to GPU with non_blocking.

    Uses default_collate for stacking, then async transfers.
    """
    from torch.utils.data.dataloader import default_collate

    collated = default_collate(batch)
    out = {}
    for key, val in collated.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device, non_blocking=True)
        else:
            out[key] = val
    return out


def _apply_valid_but_drift_augmentation(
    post_feat: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    gate_target: torch.Tensor | None,
    gate_mask: torch.Tensor | None,
    bundle: ExperimentBundle,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int]:
    if not bool(getattr(bundle.model, "snf_use_valid_drift_aug", False)):
        return post_feat, gate_target, gate_mask, 0

    prob = float(getattr(bundle.model, "snf_aug_prob", 0.0))
    if prob <= 0.0 or post_feat.dim() != 3:
        return post_feat, gate_target, gate_mask, 0

    valid_idx = int(getattr(bundle.model, "post_valid_idx", 8))
    valid = (mask > 0.5) & (post_feat[..., valid_idx] > 0.5)
    aug_mask = (torch.rand(valid.shape, device=post_feat.device) < prob) & valid
    aug_count = int(aug_mask.sum().detach().item())
    if aug_count == 0:
        return post_feat, gate_target, gate_mask, 0

    post_feat = post_feat.clone()
    pos_scale = max(float(bundle.scenario.pos_scale), 1e-6)
    vel_scale = max(float(bundle.scenario.vel_scale), 1e-6)
    pos_drift = float(getattr(bundle.model, "snf_aug_pos_drift", 0.08))
    vel_drift = float(getattr(bundle.model, "snf_aug_vel_drift", 0.12))
    false_cov_log = float(getattr(bundle.model, "snf_aug_false_cov_log", 0.25))

    true_pos = target[:, None, 0:2] / pos_scale
    true_vel = target[:, None, 2:4] / vel_scale

    pos_dir = post_feat[..., 0:2].detach() - true_pos
    pos_dir = pos_dir + 0.05 * torch.randn_like(pos_dir)
    pos_dir = pos_dir / pos_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pos_mag = pos_drift * (0.5 + torch.rand((*valid.shape, 1), device=post_feat.device))

    vel_dir = post_feat[..., 2:4].detach() - true_vel
    vel_dir = vel_dir + 0.05 * torch.randn_like(vel_dir)
    vel_dir = vel_dir / vel_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vel_mag = vel_drift * (0.5 + torch.rand((*valid.shape, 1), device=post_feat.device))

    aug_f = aug_mask.unsqueeze(-1).to(post_feat.dtype)
    post_feat[..., 0:2] = post_feat[..., 0:2] + aug_f * pos_dir * pos_mag
    post_feat[..., 2:4] = post_feat[..., 2:4] + aug_f * vel_dir * vel_mag

    confident_cov = torch.full_like(post_feat[..., 4:8], false_cov_log)
    post_feat[..., 4:8] = torch.where(
        aug_f.bool(),
        torch.minimum(post_feat[..., 4:8], confident_cov),
        post_feat[..., 4:8],
    )

    if gate_target is not None:
        gate_target = gate_target.clone()
        bad_target = float(getattr(bundle.model, "fault_gate_target", 0.1))
        gate_target = torch.where(
            aug_mask,
            torch.full_like(gate_target, bad_target),
            gate_target,
        )
    if gate_mask is not None:
        gate_mask = gate_mask.clone()
        gate_mask = torch.where(aug_mask, torch.ones_like(gate_mask), gate_mask)

    return post_feat, gate_target, gate_mask, aug_count


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
    num_workers: int | None = None,
    use_compile: bool | None = None,
) -> TrainResult:
    """Unified training entry point with GPU optimizations.

    Key improvements over the baseline:
    - DataLoader with num_workers + pin_memory for async data loading
    - Optional torch.compile for kernel fusion
    - non_blocking GPU transfers
    - Reduced CPU-GPU sync frequency in loss reporting

    Args:
        num_workers: DataLoader workers. Defaults to RGCF_NUM_WORKERS env or 2.
        use_compile: Enable torch.compile. Defaults to True unless RGCF_DISABLE_COMPILE=1.
    """
    device = torch.device(bundle.base.runtime.device)
    _set_model_seed(getattr(bundle.train, "model_seed", None))

    epochs = int(bundle.train.epochs if epochs is None else epochs)
    lr = float(bundle.train.lr if lr is None else lr)
    batch_size = int(bundle.train.batch_size if batch_size is None else batch_size)

    if num_workers is None:
        num_workers = _DEFAULT_NUM_WORKERS
    if use_compile is None:
        use_compile = _USE_TORCH_COMPILE

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
    use_snf_aux_loss = (
        float(getattr(bundle.model, "snf_risk_loss_weight", 0.0)) > 0.0
        or float(getattr(bundle.model, "snf_overconf_loss_weight", 0.0)) > 0.0
        or float(getattr(bundle.model, "snf_quarantine_loss_weight", 0.0)) > 0.0
        or float(getattr(bundle.model, "snf_cap_loss_weight", 0.0)) > 0.0
        or float(getattr(bundle.model, "snf_fused_nll_weight", 0.0)) > 0.0
    )

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

    # GPU-optimized DataLoader configuration
    loader_kwargs = dict(
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=_DEFAULT_EVAL_BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=num_workers, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=_DEFAULT_EVAL_BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=num_workers, **loader_kwargs,
    )

    model = build_model_from_bundle(bundle).to(device)

    # torch.compile for kernel fusion on Ampere+ GPUs.
    # Uses mode="default" (not "reduce-overhead") to avoid Triton dependency
    # on Windows. The "reduce-overhead" mode requires CUDA graphs via Triton
    # which may not be installed. "default" still provides ~1.5x speedup
    # through kernel fusion without Triton.
    _compiled = False
    if use_compile and hasattr(torch, "compile"):
        for _mode in ("default",):
            try:
                model = torch.compile(model, mode=_mode)
                # Eagerly trigger compilation on a dummy batch to catch
                # lazy compilation errors (e.g. TritonMissing) early.
                dummy_nodes = 3 if str(getattr(bundle.model, "model_name", "")) == "phase1r_rgcf" else 4
                _dummy_post = torch.randn(1, dummy_nodes, int(bundle.model.post_in_dim), device=device)
                _dummy_mask = torch.ones(1, dummy_nodes, device=device)
                _dummy_meas = torch.randn(1, dummy_nodes, int(bundle.model.meas_in_dim), device=device)
                _dummy_evidence = None
                _dummy_evidence_mask = None
                if str(getattr(bundle.model, "model_name", "")) == "phase1r_rgcf":
                    _dummy_evidence = torch.randn(1, 2, int(getattr(bundle.model, "evidence_in_dim", 16)), device=device)
                    _dummy_evidence_mask = torch.ones(1, 2, device=device)
                _ = model(
                    post_feat=_dummy_post,
                    mask=_dummy_mask,
                    meas_feat=_dummy_meas,
                    evidence_feat=_dummy_evidence,
                    evidence_mask=_dummy_evidence_mask,
                    return_weights=False,
                )
                _compiled = True
                print(f"[compile] torch.compile enabled (mode={_mode})")
                # Enable TF32 for faster matmul on Ampere (safe for float32)
                torch.set_float32_matmul_precision('high')
                break
            except Exception as exc:
                print(f"[compile] torch.compile mode={_mode} failed ({exc})")
                # Reset model to uncompiled state for next attempt
                model = build_model_from_bundle(bundle).to(device)
        if not _compiled:
            print("[compile] torch.compile disabled, continuing without compile")

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    history: List[Dict] = []

    _log_interval = max(1, len(train_loader) // 10)  # Log ~10 times per epoch

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
        train_risk_sum = 0.0
        train_overconf_sum = 0.0
        train_underconf_sum = 0.0
        train_quarantine_sum = 0.0
        train_cap_sum = 0.0
        train_fused_nll_sum = 0.0
        train_aug_count = 0
        n = 0

        for batch_idx, batch in enumerate(train_loader):
            post_feat = batch["post_feat"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            gate_target = batch.get("gate_target", None)
            gate_mask = batch.get("gate_supervision_mask", None)
            if gate_target is not None:
                gate_target = gate_target.to(device, non_blocking=True)
            if gate_mask is not None:
                gate_mask = gate_mask.to(device, non_blocking=True)

            meas_feat = None
            if "meas_feat" in batch:
                meas_feat = batch["meas_feat"].to(device, non_blocking=True)

            evidence_feat = None
            evidence_mask = None
            if "evidence_feat" in batch:
                evidence_feat = batch["evidence_feat"].to(device, non_blocking=True)
                evidence_mask = batch["evidence_mask"].to(device, non_blocking=True)

            post_win = None
            meas_win = None
            if "post_win" in batch:
                post_win = batch["post_win"].to(device, non_blocking=True)
                meas_win = batch["meas_win"].to(device, non_blocking=True)

            post_feat, gate_target, gate_mask, aug_count = _apply_valid_but_drift_augmentation(
                post_feat,
                mask,
                target,
                gate_target,
                gate_mask,
                bundle,
            )
            train_aug_count += aug_count

            out = model(
                post_feat=post_feat,
                mask=mask,
                meas_feat=meas_feat,
                evidence_feat=evidence_feat,
                evidence_mask=evidence_mask,
                return_weights=use_gate_supervision or use_fault_weight_loss or use_snf_aux_loss,
                post_win=post_win,
                meas_win=meas_win,
            )
            pred = out.pred

            if use_gate_supervision:
                loss, info = compute_fusion_loss_with_gate(
                    pred,
                    target,
                    gate=out.aux.get("gate", None),
                    gate_target=gate_target,
                    gate_mask=gate_mask,
                    weights=out.weights,
                    cov_scale=out.aux.get("cov_scale", None),
                    risk=out.aux.get("risk", None),
                    quarantine=out.aux.get("quarantine", None),
                    weight_cap=out.aux.get("weight_cap", None),
                    fused_cov_diag=out.aux.get("fused_cov_diag", None),
                    vel_weight=vel_weight,
                    gate_weight=float(getattr(bundle.model, "gate_supervision_weight", 0.05)),
                    gate_prior_weight=float(getattr(bundle.model, "gate_prior_weight", 0.005)),
                    gate_prior_mean=float(getattr(bundle.model, "gate_prior_mean", 0.75)),
                    cov_prior_weight=float(getattr(bundle.model, "cov_prior_weight", 0.0)),
                    cov_sep_weight=float(getattr(bundle.model, "cov_sep_weight", 0.0)),
                    cov_fault_normal_margin=float(getattr(bundle.model, "cov_fault_normal_margin", 1.0)),
                    fault_weight_loss_weight=float(getattr(bundle.model, "fault_weight_loss_weight", 0.0)),
                    fault_weight_margin=float(getattr(bundle.model, "fault_weight_margin", 0.1)),
                    risk_loss_weight=float(getattr(bundle.model, "snf_risk_loss_weight", 0.0)),
                    overconf_loss_weight=float(getattr(bundle.model, "snf_overconf_loss_weight", 0.0)),
                    underconf_loss_weight=float(getattr(bundle.model, "snf_underconf_loss_weight", 0.0)),
                    risk_error_scale=float(getattr(bundle.model, "snf_risk_error_scale", 20.0)),
                    quarantine_loss_weight=float(getattr(bundle.model, "snf_quarantine_loss_weight", 0.0)),
                    cap_loss_weight=float(getattr(bundle.model, "snf_cap_loss_weight", 0.0)),
                    fused_nll_weight=float(getattr(bundle.model, "snf_fused_nll_weight", 0.0)),
                    tail_loss_weight=float(getattr(bundle.model, "rgcf_tail_loss_weight", 0.0)),
                    tail_error_scale=float(getattr(bundle.model, "rgcf_tail_error_scale", 25.0)),
                    balanced_gate_loss=bool(getattr(bundle.model, "use_balanced_gate_loss", True)),
                    fault_gate_threshold=0.5 * (
                        float(getattr(bundle.model, "normal_gate_target", 0.8))
                        + float(getattr(bundle.model, "fault_gate_target", 0.2))
                    ),
                )
            else:
                loss, info = compute_fusion_loss(pred, target, vel_weight=vel_weight)

            opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()
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
            train_risk_sum += info.get("loss_risk", 0.0) * bs
            train_overconf_sum += info.get("loss_overconf", 0.0) * bs
            train_underconf_sum += info.get("loss_underconf", 0.0) * bs
            train_quarantine_sum += info.get("loss_quarantine", 0.0) * bs
            train_cap_sum += info.get("loss_cap", 0.0) * bs
            train_fused_nll_sum += info.get("loss_fused_nll", 0.0) * bs
            n += bs

            # Progress logging every ~10% of epoch
            if (batch_idx + 1) % _log_interval == 0:
                print(f"[Epoch {ep:03d}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={info['loss_total']:.6f}")

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
            "loss_risk": train_risk_sum / max(n, 1),
            "loss_overconf": train_overconf_sum / max(n, 1),
            "loss_underconf": train_underconf_sum / max(n, 1),
            "loss_quarantine": train_quarantine_sum / max(n, 1),
            "loss_cap": train_cap_sum / max(n, 1),
            "loss_fused_nll": train_fused_nll_sum / max(n, 1),
            "valid_drift_aug_count": float(train_aug_count),
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
            train_loss_risk=float(train_metrics["loss_risk"]),
            train_loss_overconf=float(train_metrics["loss_overconf"]),
            train_loss_underconf=float(train_metrics["loss_underconf"]),
            train_loss_quarantine=float(train_metrics["loss_quarantine"]),
            train_loss_cap=float(train_metrics["loss_cap"]),
            train_loss_fused_nll=float(train_metrics["loss_fused_nll"]),
            train_valid_drift_aug_count=float(train_metrics["valid_drift_aug_count"]),
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
        if use_snf_aux_loss:
            gate_log += (
                f", risk={train_metrics['loss_risk']:.6f}, "
                f"overconf={train_metrics['loss_overconf']:.6f}, "
                f"q={train_metrics['loss_quarantine']:.6f}, "
                f"cap={train_metrics['loss_cap']:.6f}, "
                f"aug={int(train_metrics['valid_drift_aug_count'])}"
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
