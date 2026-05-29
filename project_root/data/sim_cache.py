from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

SIM_SCHEMA_VERSION = "raw_sim_v3_balanced_rgcf"


def _safe(x): return str(x).replace("\\", "_").replace("/", "_").replace(" ", "_").replace(":", "_")
def _hash(obj, n=8): return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def get_sim_cache_namespace(bundle, *, sim_schema_version: str = SIM_SCHEMA_VERSION) -> str:
    key = {
        "scene": bundle.identity.scene_name,
        "fault": bundle.identity.fault_mode,
        "preset": bundle.identity.preset_name,
        "window": getattr(bundle.fault, "window", None),
        "matrix": getattr(bundle.fault, "matrix_cases", []),
    }
    return f"{_safe(bundle.identity.scene_name)}__{_safe(bundle.identity.fault_mode)}__{sim_schema_version}__{_hash(key)}"


def get_seed_cache_path(bundle, seed: int, root_dir="sim_cache") -> Path:
    return Path(root_dir) / get_sim_cache_namespace(bundle) / f"seed_{int(seed):06d}.pkl"


def get_or_run_cached_sim(bundle, seed: int, root_dir="sim_cache", force=False):
    from core.config_loader import clone_bundle
    from simulation.runner import run_single_simulation
    path = get_seed_cache_path(bundle, seed, root_dir)
    if path.exists() and not force:
        return pickle.load(open(path, "rb"))
    b = clone_bundle(bundle)
    b.base.runtime.seed = int(seed)
    sim = run_single_simulation(b).sim
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(sim, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return sim
