"""
从 sim cache 重新组合 dataset store。

用途：已经通过 warmup_sim_cache.py 或 generate_dataset 生成过 seed cache 后，
可以用此脚本快速重新组合新的 dataset store（比如调整 train/val/test 划分后）。

运行：
    py -u scripts/resplit_from_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import RunRequest
from core.config_loader import load_experiment_bundle
from data.sim_cache import get_or_run_cached_sim
from data.dataset_store import save_raw_dataset_store


def load_sims(bundle, seeds, split_name, sim_cache_root="sim_cache"):
    sims = []
    for seed in seeds:
        sim = get_or_run_cached_sim(
            bundle, seed, root_dir=sim_cache_root, force=False
        )
        sim["seed"] = int(seed)
        sim["split_name"] = split_name
        sims.append(sim)
    return sims


def main():
    bundle = load_experiment_bundle(
        RunRequest(
            mode="generate_dataset",
            preset_name="hetero_robust_matrix_mixed_v2_rgcf",
        )
    )

    train_seeds = range(
        bundle.train.train_seed_start, bundle.train.train_seed_end + 1
    )
    val_seeds = range(
        bundle.train.val_seed_start, bundle.train.val_seed_end + 1
    )
    test_seeds = range(
        bundle.train.test_seed_start, bundle.train.test_seed_end + 1
    )

    print(f"[resplit] train seeds: {bundle.train.train_seed_start}..{bundle.train.train_seed_end}")
    print(f"[resplit] val   seeds: {bundle.train.val_seed_start}..{bundle.train.val_seed_end}")
    print(f"[resplit] test  seeds: {bundle.train.test_seed_start}..{bundle.train.test_seed_end}")

    train_sims = load_sims(bundle, train_seeds, "train")
    val_sims = load_sims(bundle, val_seeds, "val")
    test_sims = load_sims(bundle, test_seeds, "test")

    quick_sim = get_or_run_cached_sim(
        bundle, int(bundle.base.runtime.seed), root_dir="sim_cache"
    )

    ds_dir = save_raw_dataset_store(
        bundle=bundle,
        train_sims=train_sims,
        val_sims=val_sims,
        test_sims=test_sims,
        root_dir="dataset_store",
        note="from_sim_cache_balanced",
        quick_sim=quick_sim,
    )

    print(f"[dataset_store] saved to: {ds_dir}")


if __name__ == "__main__":
    main()
