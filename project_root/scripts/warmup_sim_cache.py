"""
预生成 sim cache（不生成 dataset store）。

用途：提前跑完所有 seed 的仿真，后续用 resplit_from_cache.py 快速重新组合 dataset store。

运行：
    py -u scripts/warmup_sim_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import RunRequest
from core.config_loader import load_experiment_bundle
from data.sim_cache import get_or_run_cached_sim


def main():
    bundle = load_experiment_bundle(
        RunRequest(
            mode="generate_dataset",
            preset_name="hetero_robust_matrix_mixed_v2_rgcf",
        )
    )

    seeds = list(range(
        bundle.train.train_seed_start,
        bundle.train.test_seed_end + 1,
    ))

    print(f"[sim_cache] total seeds to warm up: {len(seeds)}")
    print(f"[sim_cache] seed range: {seeds[0]} .. {seeds[-1]}")

    for seed in seeds:
        print(f"[sim_cache] seed={seed}")
        get_or_run_cached_sim(bundle, seed, root_dir="sim_cache", force=False)

    print("[done]")


if __name__ == "__main__":
    main()
