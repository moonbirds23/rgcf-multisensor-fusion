"""Frozen, stratified 700/100/200 manifest construction."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import PROTOCOL_NAME, atomic_json, canonical_json, read_csv, sha256_bytes, sha256_file, stable_seed, write_csv

FIELDS = ["scenario_id", "official_split", "experiment_split", "parquet_path", "city_name", "motion_type", "stratum_key", "selection_rank", "selection_seed", "inventory_sha256", "selection_config_sha256", "manifest_version"]


def _rank(row: dict[str, str], seed: int, namespace: str) -> tuple[int, str]:
    return stable_seed(seed, namespace, row["scenario_id"]), row["scenario_id"]


def _select(rows: list[dict[str, str]], count: int, seed: int, namespace: str) -> list[dict[str, str]]:
    if count > len(rows): raise ValueError("insufficient eligible scenarios")
    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows: strata[row["city_name"] + "::" + row["motion_type"]].append(row)
    total = len(rows); base = {key: int(count * len(value) / total) for key, value in strata.items()}
    remaining = count - sum(base.values())
    ranked = sorted(strata, key=lambda key: (-(count * len(strata[key]) / total - base[key]), key))
    for key in ranked[:remaining]: base[key] += 1
    selected: list[dict[str, str]] = []
    for key in sorted(strata): selected.extend(sorted(strata[key], key=lambda row: _rank(row, seed, namespace))[:base[key]])
    return sorted(selected, key=lambda row: _rank(row, seed, namespace))


def build_manifests(train_inventory: Path, val_inventory: Path, output_dir: Path, seed: int = 20260711) -> dict[str, Any]:
    train = [row for row in read_csv(train_inventory) if row["eligible"] == "true"]
    val = [row for row in read_csv(val_inventory) if row["eligible"] == "true"]
    pool = _select(train, 800, seed, "train-validation-pool")
    validation = _select(pool, 100, seed, "validation")
    validation_ids = {row["scenario_id"] for row in validation}
    training = [row for row in pool if row["scenario_id"] not in validation_ids]
    testing = _select(val, 200, seed, "test")
    if len(training) != 700 or len(validation) != 100 or len(testing) != 200: raise RuntimeError("manifest count invariant failed")
    all_ids = [row["scenario_id"] for group in (training, validation, testing) for row in group]
    if len(all_ids) != len(set(all_ids)): raise RuntimeError("split leakage detected")
    inventory_hash = sha256_bytes((sha256_file(train_inventory) + sha256_file(val_inventory)).encode())
    config = {"protocol_name": PROTOCOL_NAME, "selection_seed": seed, "train": 700, "validation": 100, "test": 200, "eligible_only": True, "stratification": "city_name::motion_type"}
    config_hash = sha256_bytes(canonical_json(config).encode())
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, split, rows in (("train_700.csv", "train", training), ("val_100.csv", "validation", validation), ("test_200.csv", "test", testing)):
        rendered = []
        for rank, row in enumerate(rows):
            rendered.append({"scenario_id": row["scenario_id"], "official_split": row["official_split"], "experiment_split": split, "parquet_path": row["parquet_path"], "city_name": row["city_name"], "motion_type": row["motion_type"], "stratum_key": row["city_name"] + "::" + row["motion_type"], "selection_rank": rank, "selection_seed": seed, "inventory_sha256": inventory_hash, "selection_config_sha256": config_hash, "manifest_version": "AV2_1000_MANIFEST_V2"})
        target = output_dir / name; write_csv(target, rendered, FIELDS); hashes[name] = "sha256:" + sha256_file(target)
    summary = {**config, "inventory_sha256": inventory_hash, "selection_config_sha256": config_hash, "hashes": hashes}
    atomic_json(output_dir / "manifest_hashes.json", summary); atomic_json(output_dir / "manifest_summary.json", summary)
    return summary
