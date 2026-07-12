"""Memory-mapped time-step dataset retaining AV2 scenario provenance."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class Av2TimeStepShardDataset(Dataset):
    fields = ("post_feat", "meas_feat", "evidence_feat", "evidence_mask", "mp_pair_feat", "mask", "target")
    def __init__(self, root: str | Path, eval_only: bool = True):
        self.arrays, self.items = [], []
        for shard in sorted(Path(root).glob("shard_*")):
            if not (shard / "_SUCCESS").exists(): raise RuntimeError("incomplete shard: " + str(shard))
            values = {name: np.load(shard / (name + ".npy"), mmap_mode="r") for name in (*self.fields, "eval_mask")}
            index = [json.loads(line) for line in (shard / "index.jsonl").read_text(encoding="utf-8").splitlines()]
            shard_index = len(self.arrays); self.arrays.append((values, index))
            for scenario_index, row in enumerate(index):
                for local in range(row["start"], row["stop"]):
                    if not eval_only or bool(values["eval_mask"][local]): self.items.append((shard_index, scenario_index, local, row))
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        shard_index, scenario_index, local, row = self.items[index]; values, _ = self.arrays[shard_index]
        result = {name: torch.from_numpy(np.asarray(values[name][local], dtype=np.float32)) for name in self.fields}
        result.update({"scenario_index": scenario_index, "measurement_seed": int(row["measurement_seed"]), "timestep": local - row["start"]})
        return result
