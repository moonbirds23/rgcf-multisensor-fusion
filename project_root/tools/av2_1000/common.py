"""Shared, side-effect-safe helpers for the AV2 formal pipeline."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import PROTOCOL_NAME, SCHEMA_VERSION


class Av2ExperimentError(RuntimeError):
    pass


class ArtifactConflictError(Av2ExperimentError):
    pass


class CudaRequiredError(Av2ExperimentError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "little")


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=str(path.parent), delete=False) as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno()); temporary = Path(stream.name)
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(str(temporary), **dict(arrays))
    os.replace(str(temporary), str(path))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=str(path.parent), delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader(); writer.writerows(materialized); stream.flush(); os.fsync(stream.fileno()); temporary = Path(stream.name)
    os.replace(str(temporary), str(path))


def marker(directory: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(directory / "_SUCCESS", {"protocol_name": PROTOCOL_NAME, "schema_version": SCHEMA_VERSION, **dict(payload)})


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise CudaRequiredError("AV2 formal training/evaluation requires CUDA; CPU fallback is forbidden")
    return torch.device("cuda")


def formal_paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    raw = root / "data" / "raw" / "av2_motion_forecasting"
    return {
        "root": root, "raw": raw, "train_raw": raw / "train_candidates", "val_raw": raw / "val_candidates",
        "inventory": root / "data" / "inventory" / "av2_1000",
        "manifests": root / "data" / "manifests" / "av2_1000_v2",
        "truth": root / "data" / "truth_cache" / "av2_1000",
        "sim": root / "data" / "sim_cache" / "av2_1000",
        "features": root / "data" / "feature_shards" / "av2_1000",
        "runs": root / "runs" / "av2_1000", "results": root / "results" / "av2_1000",
    }
