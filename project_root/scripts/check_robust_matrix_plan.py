"""
Dry-run 检查 seed → case / sensor 映射。

不跑仿真，只调用 materialize_robustness_fault_bundle() 获取每个 seed 的
effective_fault_mode 和 effective_fault_sensor_id，然后统计分布。

运行：
    py -u scripts/check_robust_matrix_plan.py
"""

from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import RunRequest
from core.config_loader import load_experiment_bundle
from simulation.robustness_faults import materialize_robustness_fault_bundle


def summarize(name: str, seeds, bundle):
    mode_counts = Counter()
    sensor_counts = Counter()
    joint_counts = Counter()

    for seed in seeds:
        b = copy.deepcopy(bundle)
        b.base.runtime.seed = int(seed)

        _, meta = materialize_robustness_fault_bundle(b)

        mode = meta.get("effective_fault_mode")
        sid = meta.get("effective_fault_sensor_id")

        mode_counts[str(mode)] += 1

        if sid is None:
            joint_key = f"{mode}_None"
        else:
            sid = int(sid)
            sensor_counts[f"S{sid}"] += 1
            joint_key = f"{mode}_S{sid}"

        joint_counts[joint_key] += 1

    print(f"\n===== {name} =====")
    print(f"mode_counts: {dict(mode_counts)}")
    print(f"sensor_counts_nonclean: {dict(sensor_counts)}")
    print(f"joint_counts: {dict(joint_counts)}")


def main():
    bundle = load_experiment_bundle(
        RunRequest(
            mode="generate_dataset",
            preset_name="hetero_robust_matrix_mixed_v2_rgcf",
        )
    )

    summarize(
        "train",
        range(bundle.train.train_seed_start, bundle.train.train_seed_end + 1),
        bundle,
    )
    summarize(
        "val",
        range(bundle.train.val_seed_start, bundle.train.val_seed_end + 1),
        bundle,
    )
    summarize(
        "test",
        range(bundle.train.test_seed_start, bundle.train.test_seed_end + 1),
        bundle,
    )


if __name__ == "__main__":
    main()
