from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from core.types import ExperimentBundle
from data.dataset_store import save_raw_dataset_store
from training.trainer import build_sim_list_from_seed_range


@dataclass
class GenerateDatasetResult:
    dataset_store_dir: str
    counts: Dict[str, int]


def run_generate_dataset_experiment(bundle: ExperimentBundle, *, dataset_store_root="dataset_store", dataset_note="") -> GenerateDatasetResult:
    train = build_sim_list_from_seed_range(bundle, bundle.train.train_seed_start, bundle.train.train_seed_end, "train", use_sim_cache=True)
    val = build_sim_list_from_seed_range(bundle, bundle.train.val_seed_start, bundle.train.val_seed_end, "val", use_sim_cache=True)
    test = build_sim_list_from_seed_range(bundle, bundle.train.test_seed_start, bundle.train.test_seed_end, "test", use_sim_cache=True)
    quick = train[0] if train else None
    ds = save_raw_dataset_store(bundle, train, val, test, root_dir=dataset_store_root, note=dataset_note, quick_sim=quick)
    return GenerateDatasetResult(str(ds), {"train": len(train), "val": len(val), "test": len(test)})
