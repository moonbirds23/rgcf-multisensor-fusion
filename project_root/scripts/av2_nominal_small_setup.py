"""Build and audit the isolated 20-scene nominal-only AV2 feature cache.

This is the A1--A6 boundary from the V3.1 plan.  It only reads the frozen
small-scene manifest and writes generated data below the supplied external
AV2 root.  Fault constructors are deliberately not imported or called.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v1 import (
    MEASUREMENT_SEEDS,
    NOMINAL_CACHE_SCHEMA_VERSION,
    NOMINAL_CONDITION,
    NOMINAL_CONDITIONS,
    NOMINAL_PROTOCOL,
    NOMINAL_PROTOCOL_NAME,
    WARMUP_SECONDS,
    canonical_config,
    config_sha256,
)
from data.av2.feature_builder import build_av2_feature_arrays
from scripts.av2_small_level_b_plus_b1_b2 import _load_local_truth
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent.parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise RuntimeError(f"Frozen nominal manifest must contain exactly 20 scenes, got {len(rows)}")
    ids = [row.get("scenario_id", "") for row in rows]
    if len(set(ids)) != 20 or not all(ids):
        raise RuntimeError("Frozen nominal manifest has missing or duplicate scenario IDs")
    return rows


def _archive_old_report(root: Path) -> Path:
    source = root / "reports" / "LEVEL_B_PLUS_REPORT.md"
    if not source.is_file():
        raise RuntimeError(f"Old B+ report required for archival is missing: {source}")
    destination = root / "reports" / "archive" / "LEVEL_B_PLUS_REPORT_V1.1_ARCHIVED.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = source.read_bytes()
    if destination.exists() and destination.read_bytes() != source_bytes:
        raise RuntimeError(f"Archive already exists with different content: {destination}")
    if not destination.exists():
        destination.write_bytes(source_bytes)
    return destination


def _cache_contract(manifest_hash: str, measurement_seed: int, scenario_id: str, code_commit: str) -> dict[str, Any]:
    return {
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "condition": NOMINAL_CONDITION,
        "faults_enabled": False,
        "conditions": NOMINAL_CONDITIONS,
        "sensor_config_hash": config_sha256(),
        "manifest_hash": manifest_hash,
        "measurement_seed": measurement_seed,
        "scenario_id": scenario_id,
        "code_commit": code_commit,
        "schema_version": NOMINAL_CACHE_SCHEMA_VERSION,
        "feature_schema_version": "av2_pilot_feature_v1",
    }


def _cache_paths(cache_dir: Path, scenario_id: str, seed: int) -> tuple[Path, Path]:
    stem = f"{scenario_id}_seed{seed}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.metadata.json"


def _validate_existing(payload_path: Path, metadata_path: Path, expected: dict[str, Any]) -> bool:
    if not payload_path.exists() and not metadata_path.exists():
        return False
    if not payload_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Incomplete nominal cache entry: {payload_path.stem}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata != expected:
        raise RuntimeError(f"Nominal cache contract mismatch: {metadata_path}")
    with np.load(payload_path, allow_pickle=False) as archive:
        required = {"post_feat", "post_mask", "meas_feat", "meas_mask", "evidence_feat", "evidence_mask", "mp_pair_feat", "mask", "target", "reported_pdiag", "true_post_error"}
        if set(archive.files) != required:
            raise RuntimeError(f"Nominal cache schema mismatch: {payload_path}")
        if not all(np.isfinite(archive[name]).all() for name in archive.files):
            raise RuntimeError(f"Non-finite nominal cache array: {payload_path}")
    return True


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifests" / "level_b_20.csv"
    rows = _read_manifest(manifest_path)
    manifest_hash = _sha256_file(manifest_path)
    archive_path = _archive_old_report(root)
    cache_dir = root / "cache" / "nominal_v1"
    metadata_dir = root / "metadata"
    code_commit = _head_commit()

    covariance_total = 0
    covariance_positive_definite = 0
    finite_feature_runs = 0
    generated_runs = 0
    reused_runs = 0
    for row in rows:
        scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
        if scenario.scenario_id != row["scenario_id"]:
            raise RuntimeError("Manifest scenario ID does not match parquet payload")
        for seed in MEASUREMENT_SEEDS:
            expected = _cache_contract(manifest_hash, seed, scenario.scenario_id, code_commit)
            payload_path, entry_metadata_path = _cache_paths(cache_dir, scenario.scenario_id, seed)
            if _validate_existing(payload_path, entry_metadata_path, expected):
                reused_runs += 1
                continue
            outputs = run_av2_sensor_ekfs(
                local.timestamps_ns, target, rng=np.random.default_rng(seed), protocol=NOMINAL_PROTOCOL
            )
            features = build_av2_feature_arrays(
                outputs, target, warmup_seconds=WARMUP_SECONDS, protocol=NOMINAL_PROTOCOL
            )
            covariance = np.asarray(outputs.posterior_covariance_internal, dtype=np.float64)
            available = np.asarray(outputs.posterior_available, dtype=bool)
            for matrix in covariance[available]:
                covariance_total += 1
                covariance_positive_definite += int(np.all(np.linalg.eigvalsh(matrix) > 0.0))
            arrays = {
                "post_feat": np.asarray(features.post_feat),
                "post_mask": np.asarray(features.post_mask),
                "meas_feat": np.asarray(features.meas_feat),
                "meas_mask": np.asarray(features.meas_mask),
                "evidence_feat": np.asarray(features.evidence_feat),
                "evidence_mask": np.asarray(features.evidence_mask),
                "mp_pair_feat": np.asarray(features.mp_pair_feat),
                "mask": np.asarray(features.eval_mask),
                "target": np.asarray(features.target),
                "reported_pdiag": np.diagonal(np.asarray(outputs.posterior_covariance_reported), axis1=2, axis2=3),
                "true_post_error": np.linalg.norm(np.asarray(outputs.posterior_mean)[..., :2] - target[:, None, :2], axis=2),
            }
            if not all(np.isfinite(array).all() for array in arrays.values()):
                raise RuntimeError(f"Non-finite nominal feature output: {scenario.scenario_id}, seed {seed}")
            finite_feature_runs += 1
            _write_npz_atomic(payload_path, arrays)
            _write_json_atomic(entry_metadata_path, expected)
            generated_runs += 1

    config = canonical_config()
    config_record = {"config": config, "sensor_config_hash": config_sha256(), "protocol_name": NOMINAL_PROTOCOL_NAME}
    _write_json_atomic(metadata_dir / "av2_nominal_config_hash.json", config_record)
    (metadata_dir / "level_b_20_sha256.txt").write_text(manifest_hash + "\n", encoding="utf-8")
    audit = {
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "condition": NOMINAL_CONDITION,
        "fault_variants_generated": 0,
        "faults_enabled": False,
        "scenes": len(rows),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "cache_dir": str(cache_dir),
        "manifest_sha256": manifest_hash,
        "sensor_config_hash": config_sha256(),
        "code_commit": code_commit,
        "generated_runs": generated_runs,
        "reused_runs": reused_runs,
        "finite_feature_runs": finite_feature_runs,
        "positive_definite_covariance_rate_generated_runs": (
            covariance_positive_definite / covariance_total if covariance_total else None
        ),
        "archived_old_level_b_plus_report": str(archive_path),
    }
    _write_json_atomic(metadata_dir / "av2_nominal_cache_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated AV2 V3.1 nominal-only small-scene cache.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
