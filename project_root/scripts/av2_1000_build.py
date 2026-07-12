"""Build formal manifests, truth cache, nominal sim cache, and feature shards."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.av2_1000.cache import build_feature_shards, build_sim, build_truth
from tools.av2_1000.common import formal_paths
from tools.av2_1000.manifest import build_manifests

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("manifest", "truth", "sim", "features", "all")); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--shard-size", type=int, default=100)
    args = parser.parse_args(); p = formal_paths(args.root)
    if args.stage in ("manifest", "all"): print(build_manifests(p["inventory"] / "inventory_train.csv", p["inventory"] / "inventory_val.csv", p["manifests"]))
    if args.stage in ("truth", "all"): print(build_truth(p["manifests"], p["truth"]))
    if args.stage in ("sim", "all"): print(build_sim(p["manifests"], p["truth"], p["sim"]))
    if args.stage in ("features", "all"): print(build_feature_shards(p["manifests"], p["sim"], p["features"], args.shard_size))
if __name__ == "__main__": main()
