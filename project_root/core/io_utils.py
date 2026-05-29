from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(text: str) -> str:
    return str(text).replace("\\", "_").replace("/", "_").replace(" ", "_").replace(":", "_")


def build_run_dir_name(*, mode: str, preset_name: str, scene_name: str, model_name: str, timestamp: str) -> str:
    return "__".join(map(safe_name, [timestamp, mode, preset_name, scene_name, model_name]))


def to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_json(path: str | Path, data: Any, *, ensure_parent: bool = True, indent: int = 2) -> Path:
    p = Path(path)
    if ensure_parent:
        ensure_dir(p.parent)
    p.write_text(json.dumps(to_plain(data), ensure_ascii=False, indent=indent), encoding="utf-8")
    return p


def save_text(path: str | Path, text: str, *, ensure_parent: bool = True) -> Path:
    p = Path(path)
    if ensure_parent:
        ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def save_csv_rows(path: str | Path, rows: List[Dict[str, Any]], *, ensure_parent: bool = True) -> Path:
    p = Path(path)
    if ensure_parent:
        ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(to_plain(rows))
    return p


def save_dict_csv(path: str | Path, data: Dict[str, Any], *, ensure_parent: bool = True) -> Path:
    return save_csv_rows(path, [{"key": k, "value": v} for k, v in data.items()], ensure_parent=ensure_parent)


def save_numpy_npz(path: str | Path, arrays: Dict[str, np.ndarray], *, ensure_parent: bool = True) -> Path:
    p = Path(path)
    if ensure_parent:
        ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)
    return p


def save_torch_checkpoint(path: str | Path, model, *, ensure_parent: bool = True) -> Path:
    import torch
    p = Path(path)
    if ensure_parent:
        ensure_dir(p.parent)
    torch.save(model.state_dict(), p)
    return p
