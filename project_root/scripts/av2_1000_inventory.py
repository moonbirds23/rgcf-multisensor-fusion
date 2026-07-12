"""Build deterministic inventories for downloaded AV2 candidate parquet files."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.av2_1000.common import formal_paths, marker
from tools.av2_1000.inventory import build_inventory

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(); paths = formal_paths(args.root)
    summaries = {}
    for split, raw, name in (("train", paths["train_raw"], "inventory_train.csv"), ("val", paths["val_raw"], "inventory_val.csv")):
        if not raw.is_dir(): raise FileNotFoundError(raw)
        summaries[split] = build_inventory(raw, split, paths["inventory"] / name, Path(__file__).resolve().parent.parent)
    marker(paths["inventory"], {"stage": "inventory", "summaries": summaries}); print(summaries)
if __name__ == "__main__": main()
