"""Truth, nominal EKF simulation, and feature-shard construction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from configs.av2_config import Av2PilotProtocol
from configs.av2_nominal_1000_v1 import NOMINAL_PROTOCOL, WARMUP_SECONDS
from data.av2.feature_builder import build_av2_feature_arrays
from simulation.av2_sensor_ekf import AV2SensorEkfOutputs, SensorMeasurement, run_av2_sensor_ekfs

from .common import ArtifactConflictError, PROTOCOL_NAME, atomic_json, atomic_npz, marker, read_csv, stable_seed


def _manifest_rows(manifest_dir: Path, split: str) -> list[dict[str, str]]:
    name = {"train": "train_700.csv", "validation": "val_100.csv", "test": "test_200.csv"}[split]
    return read_csv(manifest_dir / name)


def _truth_path(root: Path, split: str, scenario_id: str) -> Path: return root / split / (scenario_id + ".npz")
def _sim_path(root: Path, split: str, scenario_id: str, seed: int) -> Path: return root / split / scenario_id / ("seed_" + str(seed) + ".npz")


def build_truth(manifest_dir: Path, output_root: Path, splits: tuple[str, ...] = ("train", "validation", "test")) -> dict[str, int]:
    from scripts.av2_small_level_b_plus_b1_b2 import _load_local_truth
    counts: dict[str, int] = {}
    for split in splits:
        rows = _manifest_rows(manifest_dir, split); counts[split] = 0
        for row in rows:
            target_path = _truth_path(output_root, split, row["scenario_id"])
            if target_path.exists(): counts[split] += 1; continue
            scenario, local, truth = _load_local_truth(Path(row["parquet_path"]))
            if truth.shape != (110, 4) or not np.isfinite(truth).all(): raise ValueError("formal truth must be finite [110,4]")
            atomic_npz(target_path, {"timestamps_ns": local.timestamps_ns, "target": truth.astype(np.float32), "heading_rad": np.asarray(local.headings_rad, dtype=np.float32), "scenario_id": np.asarray(scenario.scenario_id), "city_name": np.asarray(row["city_name"]), "parquet_path": np.asarray(row["parquet_path"]), "experiment_split": np.asarray(split)})
            counts[split] += 1
    marker(output_root, {"stage": "truth", "counts": counts, "condition": "nominal"}); return counts


def _serialize(outputs: AV2SensorEkfOutputs, target: np.ndarray) -> dict[str, Any]:
    steps = len(outputs.timestamps_ns); z = np.full((steps, 5, 2), np.nan); r = np.full((steps, 5, 2, 2), np.nan); valid = np.zeros((steps, 5), bool); emitted = np.zeros((steps, 5), bool)
    for t, records in enumerate(outputs.measurements):
        for s, measurement in enumerate(records):
            valid[t, s] = measurement.valid; emitted[t, s] = measurement.emitted
            if measurement.valid:
                size = len(measurement.z); z[t, s, :size] = measurement.z; r[t, s, :size, :size] = measurement.R_reported
    return {"timestamps_ns": outputs.timestamps_ns, "target": target.astype(np.float32), "posterior_mean": outputs.posterior_mean.astype(np.float32), "posterior_covariance_internal": outputs.posterior_covariance_internal.astype(np.float32), "posterior_covariance_reported": outputs.posterior_covariance_reported.astype(np.float32), "posterior_available": outputs.posterior_available, "posterior_measurement_valid": outputs.posterior_measurement_valid, "evidence_valid": outputs.evidence_valid, "measurement_z": z.astype(np.float32), "measurement_r": r.astype(np.float32), "measurement_valid": valid, "measurement_emitted": emitted}


def load_outputs(path: Path) -> tuple[AV2SensorEkfOutputs, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw: data = {key: raw[key] for key in raw.files}
    names, roles, dims = ("T1", "T2", "T3", "E1", "E2"), ("posterior", "posterior", "posterior", "evidence", "evidence"), (2, 2, 2, 1, 1)
    records = []
    for t in range(len(data["timestamps_ns"])):
        row = []
        for s, (name, role, dim) in enumerate(zip(names, roles, dims)):
            yes = bool(data["measurement_valid"][t, s]); emitted = bool(data["measurement_emitted"][t, s])
            row.append(SensorMeasurement(name, role, emitted, yes, data["measurement_z"][t, s, :dim] if yes else None, data["measurement_r"][t, s, :dim, :dim] if yes else None, data["measurement_r"][t, s, :dim, :dim] if yes else None, int(data["timestamps_ns"][t]) if yes else None, False))
        records.append(tuple(row))
    return AV2SensorEkfOutputs(data["timestamps_ns"], data["posterior_mean"], data["posterior_covariance_internal"], data["posterior_covariance_reported"], data["posterior_available"], data["posterior_measurement_valid"], data["evidence_valid"], tuple(records)), data["target"]


def build_sim(
    manifest_dir: Path,
    truth_root: Path,
    output_root: Path,
    train_global_seed: int = 20260711,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    *,
    protocol: Av2PilotProtocol = NOMINAL_PROTOCOL,
    protocol_name: str = PROTOCOL_NAME,
    config_sha256: str = "legacy-unspecified",
    scenario_scoped_rng: bool = False,
) -> dict[str, int]:
    """Build nominal simulations without ever overwriting existing records.

    V3.1 keeps the historical seed behavior by default.  Corrected protocols
    set ``scenario_scoped_rng=True`` so a public measurement-seed label is
    deterministically namespaced by scenario before initializing NumPy's RNG.
    """
    counts: dict[str, int] = {}
    for split in splits:
        rows = _manifest_rows(manifest_dir, split)
        for row in rows:
            seed_list = [stable_seed(row["scenario_id"], train_global_seed)] if split == "train" else ([100] if split == "validation" else [100, 101, 102])
            for seed in seed_list:
                path = _sim_path(output_root, split, row["scenario_id"], seed)
                if path.exists():
                    if scenario_scoped_rng or protocol_name != PROTOCOL_NAME:
                        with np.load(path, allow_pickle=False) as existing:
                            expected_rng_seed = stable_seed(
                                protocol_name, row["scenario_id"], seed, "measurement"
                            )
                            compatible = (
                                "protocol_name" in existing.files
                                and "config_sha256" in existing.files
                                and "rng_seed" in existing.files
                                and str(existing["protocol_name"]) == protocol_name
                                and str(existing["config_sha256"]) == config_sha256
                                and int(existing["rng_seed"]) == expected_rng_seed
                            )
                        if not compatible:
                            raise ArtifactConflictError(
                                f"existing simulation cache violates requested protocol: {path}"
                            )
                    continue
                with np.load(_truth_path(truth_root, split, row["scenario_id"]), allow_pickle=False) as truth: timestamps, target = truth["timestamps_ns"], truth["target"]
                rng_seed = (
                    stable_seed(protocol_name, row["scenario_id"], seed, "measurement")
                    if scenario_scoped_rng
                    else seed
                )
                result = run_av2_sensor_ekfs(
                    timestamps,
                    target,
                    rng=np.random.default_rng(rng_seed),
                    protocol=protocol,
                )
                atomic_npz(
                    path,
                    {
                        **_serialize(result, target),
                        "scenario_id": np.asarray(row["scenario_id"]),
                        "experiment_split": np.asarray(split),
                        "measurement_seed": np.asarray(seed),
                        "rng_seed": np.asarray(rng_seed),
                        "protocol_name": np.asarray(protocol_name),
                        "config_sha256": np.asarray(config_sha256),
                    },
                )
        counts[split] = len(list((output_root / split).rglob("seed_*.npz")))
    marker(
        output_root,
        {
            "stage": "sim",
            "counts": counts,
            "condition": "nominal",
            "fault_variants_generated": 0,
            "config_sha256": config_sha256,
            "scenario_scoped_rng": scenario_scoped_rng,
        },
        protocol_name=protocol_name,
    )
    return counts


def build_feature_shards(
    manifest_dir: Path,
    sim_root: Path,
    output_root: Path,
    scenarios_per_shard: int = 100,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    *,
    protocol: Av2PilotProtocol = NOMINAL_PROTOCOL,
    protocol_name: str = PROTOCOL_NAME,
    config_sha256: str = "legacy-unspecified",
    warmup_seconds: float = WARMUP_SECONDS,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for split in splits:
        refs = sorted((sim_root / split).rglob("seed_*.npz")); result[split] = 0
        for shard_number, group_start in enumerate(range(0, len(refs), scenarios_per_shard)):
            group = refs[group_start:group_start + scenarios_per_shard]; shard = output_root / split / ("shard_%03d" % shard_number)
            if (shard / "_SUCCESS").exists():
                if protocol_name != PROTOCOL_NAME:
                    success = json.loads((shard / "_SUCCESS").read_text(encoding="utf-8"))
                    if success.get("protocol_name") != protocol_name or success.get("config_sha256") != config_sha256:
                        raise ArtifactConflictError(
                            f"existing feature shard violates requested protocol: {shard}"
                        )
                result[split] += len(group); continue
            arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("post_feat", "meas_feat", "evidence_feat", "evidence_mask", "mp_pair_feat", "mask", "target", "eval_mask", "true_post_error", "reported_pdiag")}; offsets, index = [0], []
            for path in group:
                outputs, target = load_outputs(path); feature = build_av2_feature_arrays(outputs, target, warmup_seconds=warmup_seconds, protocol=protocol)
                post_error = np.linalg.norm(outputs.posterior_mean[:, :, :2] - target[:, None, :2], axis=2).copy(); post_error[~outputs.posterior_available] = 0.0
                pdiag = np.diagonal(outputs.posterior_covariance_reported, axis1=2, axis2=3).copy(); pdiag[~np.isfinite(pdiag)] = 0.0
                values = {"post_feat": feature.post_feat, "meas_feat": feature.meas_feat, "evidence_feat": feature.evidence_feat, "evidence_mask": feature.evidence_mask, "mp_pair_feat": feature.mp_pair_feat, "mask": feature.post_mask, "target": feature.target, "eval_mask": feature.eval_mask, "true_post_error": post_error, "reported_pdiag": pdiag}
                for key, value in values.items(): arrays[key].append(np.asarray(value, dtype=np.float32 if key != "eval_mask" else bool))
                with np.load(path, allow_pickle=False) as raw: index.append({"scenario_id": str(raw["scenario_id"]), "experiment_split": split, "measurement_seed": int(raw["measurement_seed"]), "start": offsets[-1], "stop": offsets[-1] + len(feature.target)})
                offsets.append(index[-1]["stop"])
            shard.mkdir(parents=True, exist_ok=True)
            for key, parts in arrays.items(): np.save(shard / (key + ".npy"), np.concatenate(parts, axis=0))
            np.save(shard / "scenario_offsets.npy", np.asarray(offsets, dtype=np.int64))
            (shard / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index), encoding="utf-8")
            atomic_json(shard / "metadata.json", {"protocol_name": protocol_name, "config_sha256": config_sha256, "feature_schema_version": 1, "split": split, "scenarios": len(index), "rows": offsets[-1]})
            marker(shard, {"stage": "feature_shard", "split": split, "scenarios": len(index), "config_sha256": config_sha256}, protocol_name=protocol_name); result[split] += len(group)
    marker(output_root, {"stage": "features", "counts": result, "condition": "nominal", "config_sha256": config_sha256}, protocol_name=protocol_name); return result
