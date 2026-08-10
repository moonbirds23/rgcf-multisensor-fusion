"""Fail-fast numerical audit for regenerated AV2 V3.2 simulation caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v32 import (
    ARTIFACT_KEY,
    NOMINAL_PROTOCOL,
    NOMINAL_PROTOCOL_NAME,
    config_sha256,
)
from tools.av2_1000.common import atomic_json, formal_paths, stable_seed


EXPECTED_COUNTS = {"train": 700, "validation": 100, "test": 600}
SENSOR_NAMES = ("T1", "T2", "T3", "E1", "E2")
EXPECTED_EMISSIONS = (55, 55, 55, 55, 110)
EXPECTED_FEATURES = {
    "train": (7, 700, 77_000, 70_000),
    "validation": (1, 100, 11_000, 10_000),
    "test": (6, 600, 66_000, 60_000),
}
FEATURE_SHAPES = {
    "post_feat": (3, 9),
    "meas_feat": (3, 18),
    "evidence_feat": (2, 16),
    "evidence_mask": (2,),
    "mp_pair_feat": (3, 5, 8),
    "mask": (3,),
    "target": (4,),
    "eval_mask": (),
    "true_post_error": (3,),
    "reported_pdiag": (3, 4),
}


def _native_residual(sensor_index: int, z: np.ndarray, target: np.ndarray) -> np.ndarray:
    spec = NOMINAL_PROTOCOL.sensors[sensor_index]
    if spec.measurement_model == "gps_2d":
        return z[:, :2] - target[:, :2]
    sx, sy = spec.local_position_xy_m  # type: ignore[misc]
    dx, dy = target[:, 0] - sx, target[:, 1] - sy
    if spec.measurement_model == "range_bearing":
        prediction = np.column_stack((np.hypot(dx, dy), np.arctan2(dy, dx)))
        residual = z[:, :2] - prediction
        residual[:, 1] = (residual[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
        return residual
    if spec.measurement_model == "aoa_bearing":
        residual = z[:, 0] - np.arctan2(dy, dx)
        return (((residual + np.pi) % (2.0 * np.pi) - np.pi))[:, None]
    return (z[:, 0] - np.hypot(dx, dy))[:, None]


def _audit_features(feature_root: Path, digest: str) -> dict[str, object]:
    report: dict[str, object] = {}
    for split, (expected_shards, expected_records, expected_rows, expected_eval_rows) in EXPECTED_FEATURES.items():
        shards = sorted((feature_root / split).glob("shard_*"))
        if len(shards) != expected_shards:
            raise RuntimeError(f"{split}: expected {expected_shards} shards, got {len(shards)}")
        records = rows = eval_rows = 0
        scenario_ids: list[str] = []
        measurement_seeds: list[int] = []
        for shard in shards:
            success = json.loads((shard / "_SUCCESS").read_text(encoding="utf-8"))
            metadata = json.loads((shard / "metadata.json").read_text(encoding="utf-8"))
            for payload in (success, metadata):
                if payload.get("protocol_name") != NOMINAL_PROTOCOL_NAME or payload.get("config_sha256") != digest:
                    raise RuntimeError(f"feature metadata mismatch: {shard}")
            index = [
                json.loads(line)
                for line in (shard / "index.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            local_rows = int(metadata["rows"])
            records += len(index)
            rows += local_rows
            scenario_ids.extend(str(value["scenario_id"]) for value in index)
            measurement_seeds.extend(int(value["measurement_seed"]) for value in index)
            for name, tail in FEATURE_SHAPES.items():
                array = np.load(shard / f"{name}.npy", mmap_mode="r")
                if array.shape != (local_rows, *tail):
                    raise RuntimeError(f"feature shape mismatch: {shard} / {name} / {array.shape}")
                if not np.all(np.isfinite(array)):
                    raise RuntimeError(f"non-finite feature values: {shard} / {name}")
                if name == "eval_mask":
                    eval_rows += int(array.sum())
        if (records, rows, eval_rows) != (expected_records, expected_rows, expected_eval_rows):
            raise RuntimeError(
                f"{split}: feature totals mismatch: {(records, rows, eval_rows)}"
            )
        report[split] = {
            "shards": len(shards),
            "records": records,
            "rows": rows,
            "eval_rows": eval_rows,
            "unique_scenarios": len(set(scenario_ids)),
            "unique_measurement_seeds": len(set(measurement_seeds)),
        }
    return report


def run(root: Path) -> dict[str, object]:
    paths = formal_paths(root, artifact_key=ARTIFACT_KEY)
    digest = config_sha256()
    residual_parts: list[list[np.ndarray]] = [[] for _ in SENSOR_NAMES]
    rng_seeds: set[int] = set()
    counts: dict[str, int] = {}
    min_posterior_eigenvalue = float("inf")

    for split, expected_count in EXPECTED_COUNTS.items():
        files = sorted((paths["sim"] / split).rglob("seed_*.npz"))
        counts[split] = len(files)
        if len(files) != expected_count:
            raise RuntimeError(f"{split}: expected {expected_count} caches, got {len(files)}")
        for path in files:
            with np.load(path, allow_pickle=False) as raw:
                if str(raw["protocol_name"]) != NOMINAL_PROTOCOL_NAME:
                    raise RuntimeError(f"protocol mismatch: {path}")
                if str(raw["config_sha256"]) != digest:
                    raise RuntimeError(f"config hash mismatch: {path}")
                scenario_id = str(raw["scenario_id"])
                measurement_seed = int(raw["measurement_seed"])
                rng_seed = int(raw["rng_seed"])
                expected_rng_seed = stable_seed(
                    NOMINAL_PROTOCOL_NAME, scenario_id, measurement_seed, "measurement"
                )
                if rng_seed != expected_rng_seed:
                    raise RuntimeError(f"RNG derivation mismatch: {path}")
                if rng_seed in rng_seeds:
                    raise RuntimeError(f"duplicate scenario-scoped RNG seed: {rng_seed}")
                rng_seeds.add(rng_seed)

                target = np.asarray(raw["target"], dtype=np.float64)
                z = np.asarray(raw["measurement_z"], dtype=np.float64)
                covariance = np.asarray(raw["measurement_r"], dtype=np.float64)
                valid = np.asarray(raw["measurement_valid"], dtype=bool)
                emitted = np.asarray(raw["measurement_emitted"], dtype=bool)
                if target.shape != (110, 4) or z.shape != (110, 5, 2):
                    raise RuntimeError(f"unexpected cache shape: {path}")
                for sensor_index, expected_emissions in enumerate(EXPECTED_EMISSIONS):
                    if int(emitted[:, sensor_index].sum()) != expected_emissions:
                        raise RuntimeError(f"emission cadence mismatch: {path}")
                    if not np.array_equal(valid[:, sensor_index], emitted[:, sensor_index]):
                        raise RuntimeError(f"nominal valid/emitted mismatch: {path}")
                    dimensions = len(NOMINAL_PROTOCOL.sensors[sensor_index].noise_standard_deviations)
                    expected_diag = np.square(
                        [
                            value
                            for _, value in NOMINAL_PROTOCOL.sensors[
                                sensor_index
                            ].noise_standard_deviations
                        ]
                    )
                    actual_diag = np.diagonal(
                        covariance[valid[:, sensor_index], sensor_index, :dimensions, :dimensions],
                        axis1=1,
                        axis2=2,
                    )
                    if not np.allclose(actual_diag, expected_diag, rtol=2e-6, atol=1e-10):
                        raise RuntimeError(f"measurement covariance mismatch: {path}")
                    residual_parts[sensor_index].append(
                        _native_residual(
                            sensor_index,
                            z[valid[:, sensor_index], sensor_index],
                            target[valid[:, sensor_index]],
                        )
                    )

                posterior = np.asarray(raw["posterior_covariance_internal"], dtype=np.float64)
                available = np.asarray(raw["posterior_available"], dtype=bool)
                eigenvalues = np.linalg.eigvalsh(posterior[available])
                if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues.min()) <= 0.0:
                    raise RuntimeError(f"posterior covariance is not SPD: {path}")
                min_posterior_eigenvalue = min(
                    min_posterior_eigenvalue, float(eigenvalues.min())
                )

    empirical: dict[str, object] = {}
    for sensor_index, name in enumerate(SENSOR_NAMES):
        values = np.concatenate(residual_parts[sensor_index], axis=0)
        expected_std = np.asarray(
            [value for _, value in NOMINAL_PROTOCOL.sensors[sensor_index].noise_standard_deviations]
        )
        observed_std = values.std(axis=0, ddof=1)
        if not np.allclose(observed_std, expected_std, rtol=0.08, atol=5e-4):
            raise RuntimeError(
                f"{name}: empirical std {observed_std} does not match {expected_std}"
            )
        empirical[name] = {
            "samples": int(values.shape[0]),
            "mean": values.mean(axis=0).tolist(),
            "std": observed_std.tolist(),
            "expected_std": expected_std.tolist(),
        }

    result = {
        "status": "PASS",
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "artifact_key": ARTIFACT_KEY,
        "config_sha256": digest,
        "counts": counts,
        "unique_rng_streams": len(rng_seeds),
        "minimum_posterior_covariance_eigenvalue": min_posterior_eigenvalue,
        "empirical_noise": empirical,
        "feature_integrity": _audit_features(paths["features"], digest),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root)
    output = args.output or root / "reports" / ARTIFACT_KEY / "data_integrity_audit.json"
    atomic_json(output, result)


if __name__ == "__main__":
    main()
