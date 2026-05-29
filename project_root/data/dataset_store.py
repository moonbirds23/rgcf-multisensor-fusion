from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetStoreMeta:
    dataset_id: str
    created_at: str
    preset_name: str
    experiment_name: str
    scene_name: str
    fault_mode: str
    train_seed_start: int
    train_seed_end: int
    val_seed_start: int
    val_seed_end: int
    test_seed_start: int
    test_seed_end: int
    feature_version: str = "raw_sim_v2_with_meas_stats"
    note: str = ""
    files: Dict[str, str] | None = None
    fault_detail: Dict[str, Any] | None = None
    split_case_summary: Dict[str, Any] | None = None


def _safe(x): return str(x).replace("\\", "_").replace("/", "_").replace(" ", "_").replace(":", "_")
def _ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def _id(bundle): return bundle.identity


def _fault_detail(bundle) -> Dict[str, Any]:
    f = bundle.fault
    return {
        "mode": f.mode,
        "dropout": {"enabled": f.dropout.enabled, "windows": [asdict(w) for w in f.dropout.windows]},
        "pollution": asdict(f.pollution),
        "matrix_cases": list(f.matrix_cases),
        "candidate_sensor_ids": list(f.candidate_sensor_ids),
        "random_target_by_seed": bool(f.random_target_by_seed),
        "window": f.window,
    }


def make_dataset_id(bundle, note="") -> str:
    parts = [_ts(), _safe(_id(bundle).preset_name), _safe(_id(bundle).experiment_name), _safe(_id(bundle).scene_name), _safe(_id(bundle).fault_mode), f"tr{bundle.train.train_seed_start}-{bundle.train.train_seed_end}", f"va{bundle.train.val_seed_start}-{bundle.train.val_seed_end}", f"te{bundle.train.test_seed_start}-{bundle.train.test_seed_end}"]
    if note:
        parts.append(_safe(note))
    s = "__".join(parts)
    return s[:71] + "__" + str(abs(hash(s)))[:8] if len(s) > 80 else s


def save_pickle(path, obj): Path(path).parent.mkdir(parents=True, exist_ok=True); pickle.dump(obj, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
def load_pickle(path): return pickle.load(open(path, "rb"))
def save_json(path, obj): Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_split_sims(sims: List[Dict[str, Any]]) -> Dict[str, Any]:
    mode_counter, sensor_counter, joint_counter = Counter(), Counter(), Counter()
    table = defaultdict(Counter)
    for sim in sims:
        mode = str(sim.get("effective_fault_mode", "unknown"))
        sid = sim.get("effective_fault_sensor_id", None)
        label = "None" if sid is None else f"S{int(sid)}"
        mode_counter[mode] += 1
        if mode != "clean" and label != "None":
            sensor_counter[label] += 1
        joint_counter[f"{mode}_{label}"] += 1
        table[mode][label] += 1
    return {"num_sims": len(sims), "mode_counts": dict(mode_counter), "sensor_counts_nonclean": dict(sensor_counter), "joint_counts": dict(joint_counter), "joint_table": {k: dict(v) for k, v in table.items()}}


def save_raw_dataset_store(bundle, train_sims, val_sims, test_sims, *, root_dir="dataset_store", note="", quick_sim=None) -> Path:
    root = Path(root_dir); root.mkdir(parents=True, exist_ok=True)
    ds_dir = root / make_dataset_id(bundle, note)
    ds_dir.mkdir(parents=True, exist_ok=False)
    files = {"train_sims": "train_sims.pkl", "val_sims": "val_sims.pkl", "test_sims": "test_sims.pkl"}
    if quick_sim is not None:
        files["quick_sim"] = "quick_sim.pkl"
    save_pickle(ds_dir / files["train_sims"], train_sims)
    save_pickle(ds_dir / files["val_sims"], val_sims)
    save_pickle(ds_dir / files["test_sims"], test_sims)
    if quick_sim is not None:
        save_pickle(ds_dir / files["quick_sim"], quick_sim)
    meta = DatasetStoreMeta(ds_dir.name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), bundle.identity.preset_name, bundle.identity.experiment_name, bundle.identity.scene_name, bundle.identity.fault_mode, bundle.train.train_seed_start, bundle.train.train_seed_end, bundle.train.val_seed_start, bundle.train.val_seed_end, bundle.train.test_seed_start, bundle.train.test_seed_end, note=note, files=files, fault_detail=_fault_detail(bundle), split_case_summary={"train": summarize_split_sims(train_sims), "val": summarize_split_sims(val_sims), "test": summarize_split_sims(test_sims)})
    save_json(ds_dir / "meta.json", asdict(meta))
    return ds_dir


def load_raw_dataset_store(dataset_dir) -> Dict[str, Any]:
    d = Path(dataset_dir)
    meta = load_json(d / "meta.json")
    files = meta["files"]
    out = {"meta": meta, "train_sims": load_pickle(d / files["train_sims"]), "val_sims": load_pickle(d / files["val_sims"]), "test_sims": load_pickle(d / files["test_sims"])}
    if "quick_sim" in files:
        out["quick_sim"] = load_pickle(d / files["quick_sim"])
    return out


def find_dataset_store_by_id(dataset_id: str, root_dir="dataset_store") -> Path:
    p = Path(root_dir) / dataset_id
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def find_latest_matching_dataset_store(bundle, root_dir="dataset_store") -> Optional[Path]:
    root = Path(root_dir)
    if not root.exists():
        return None
    matches = []
    for sub in root.iterdir():
        meta = sub / "meta.json"
        if not meta.exists():
            continue
        try:
            m = load_json(meta)
            if m.get("preset_name") == bundle.identity.preset_name and m.get("scene_name") == bundle.identity.scene_name and m.get("fault_mode") == bundle.identity.fault_mode:
                matches.append(sub)
        except Exception:
            pass
    return sorted(matches, key=lambda p: p.name)[-1] if matches else None
