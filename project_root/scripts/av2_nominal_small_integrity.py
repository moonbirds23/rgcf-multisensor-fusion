"""Nominal-only causality and feature-interface checks for AV2 V3.1.

The checks perturb only in-memory copies to demonstrate that prefix features
are causal and pair relations are local.  They do not construct, persist, or
score fault variants such as bias, dropout, or time delay.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v1 import MEASUREMENT_SEEDS, NOMINAL_PROTOCOL, NOMINAL_PROTOCOL_NAME
from data.av2.feature_builder import AV2FeatureArrays, build_av2_feature_arrays
from scripts.av2_small_level_b_plus_b1_b2 import _load_local_truth
from simulation.av2_sensor_ekf import AV2SensorEkfOutputs, SensorMeasurement, run_av2_sensor_ekfs


FIXED_SCENE_INDEXES = (0, 4, 8, 12, 16)
FORBIDDEN_METADATA_FIELDS = {
    "motion_type", "total_displacement", "complete_heading_change",
    "future_acceleration", "city_name", "scenario_end_position",
}


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.dtype == bool or right_array.dtype == bool:
        return 0.0 if np.array_equal(left_array, right_array) else 1.0
    if not np.array_equal(np.isnan(left_array), np.isnan(right_array)):
        return float("inf")
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    return float(np.max(np.abs(left_array[finite] - right_array[finite]))) if finite.any() else 0.0


def _output_prefix_difference(left: AV2SensorEkfOutputs, right: AV2SensorEkfOutputs, end: int) -> float:
    values = [
        _max_abs_difference(getattr(left, field)[:end], getattr(right, field)[:end])
        for field in (
            "posterior_mean", "posterior_covariance_internal", "posterior_covariance_reported",
            "posterior_available", "posterior_measurement_valid", "evidence_valid",
        )
    ]
    for step in range(end):
        for first, second in zip(left.measurements[step], right.measurements[step]):
            if (first.valid, first.emitted, first.source_timestamp_ns) != (second.valid, second.emitted, second.source_timestamp_ns):
                values.append(float("inf"))
                continue
            for field in ("z", "R_actual", "R_reported"):
                a, b = getattr(first, field), getattr(second, field)
                values.append(0.0 if a is None and b is None else _max_abs_difference(a, b) if a is not None and b is not None else float("inf"))
    return max(values, default=0.0)


def _feature_prefix_difference(left: AV2FeatureArrays, right: AV2FeatureArrays, end: int) -> float:
    return max(
        _max_abs_difference(getattr(left, field)[:end], getattr(right, field)[:end])
        for field in ("post_feat", "post_mask", "meas_feat", "meas_mask", "evidence_feat", "evidence_mask", "mp_pair_feat", "target")
    )


def _replace_measurements(outputs: AV2SensorEkfOutputs, measurements: tuple[tuple[SensorMeasurement, ...], ...], posterior_mean: np.ndarray | None = None) -> AV2SensorEkfOutputs:
    return AV2SensorEkfOutputs(
        timestamps_ns=outputs.timestamps_ns,
        posterior_mean=outputs.posterior_mean if posterior_mean is None else posterior_mean,
        posterior_covariance_internal=outputs.posterior_covariance_internal,
        posterior_covariance_reported=outputs.posterior_covariance_reported,
        posterior_available=outputs.posterior_available,
        posterior_measurement_valid=outputs.posterior_measurement_valid,
        evidence_valid=outputs.evidence_valid,
        measurements=measurements,
    )


def _perturb_e1(outputs: AV2SensorEkfOutputs, step: int) -> AV2SensorEkfOutputs:
    rows = [list(row) for row in outputs.measurements]
    original = rows[step][3]
    if not original.valid or original.z is None:
        raise RuntimeError("selected E1 step is invalid")
    changed = np.asarray(original.z, dtype=np.float64).copy()
    changed[0] += 0.1
    rows[step][3] = replace(original, z=changed)
    return _replace_measurements(outputs, tuple(tuple(row) for row in rows))


def _perturb_p2(outputs: AV2SensorEkfOutputs, step: int) -> AV2SensorEkfOutputs:
    changed = np.asarray(outputs.posterior_mean, dtype=np.float64).copy()
    changed[step, 1, 0] += 20.0
    return _replace_measurements(outputs, outputs.measurements, posterior_mean=changed)


def _selected_rows(root: Path) -> list[dict[str, str]]:
    with (root / "manifests" / "level_b_20.csv").open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["scenario_id"])
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 frozen nominal scenes, got {len(rows)}")
    return [rows[index] for index in FIXED_SCENE_INDEXES]


def run(root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    future_runs: list[dict[str, Any]] = []
    selected = _selected_rows(root)
    for row in selected:
        scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
        cut = local.num_steps // 2
        altered_target = target.copy()
        altered_target[cut + 1:, :2] += np.array([50_000.0, -50_000.0])
        altered_target[cut + 1:, 2:] += np.array([500.0, -500.0])
        for seed in MEASUREMENT_SEEDS:
            original = run_av2_sensor_ekfs(local.timestamps_ns, target, rng=np.random.default_rng(seed), protocol=NOMINAL_PROTOCOL)
            altered = run_av2_sensor_ekfs(local.timestamps_ns, altered_target, rng=np.random.default_rng(seed), protocol=NOMINAL_PROTOCOL)
            original_features = build_av2_feature_arrays(original, target, protocol=NOMINAL_PROTOCOL)
            altered_features = build_av2_feature_arrays(altered, altered_target, protocol=NOMINAL_PROTOCOL)
            output_difference = _output_prefix_difference(original, altered, cut + 1)
            feature_difference = _feature_prefix_difference(original_features, altered_features, cut + 1)
            future_runs.append({
                "scenario_id": scenario.scenario_id, "measurement_seed": seed, "cut_index": cut,
                "sensor_ekf_max_abs_difference_before_cut": output_difference,
                "feature_max_abs_difference_before_cut": feature_difference,
                "passed": max(output_difference, feature_difference) <= 1e-6,
            })

    row = selected[0]
    scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
    output = run_av2_sensor_ekfs(local.timestamps_ns, target, rng=np.random.default_rng(MEASUREMENT_SEEDS[0]), protocol=NOMINAL_PROTOCOL)
    baseline = build_av2_feature_arrays(output, target, protocol=NOMINAL_PROTOCOL)
    step = int(np.flatnonzero(np.asarray(baseline.evidence_mask[:, 0], dtype=bool))[len(np.flatnonzero(np.asarray(baseline.evidence_mask[:, 0], dtype=bool))) // 2])
    e1 = build_av2_feature_arrays(_perturb_e1(output, step), target, protocol=NOMINAL_PROTOCOL)
    p2 = build_av2_feature_arrays(_perturb_p2(output, step), target, protocol=NOMINAL_PROTOCOL)
    evidence_checks = {
        "e1_feature_changed": _max_abs_difference(baseline.evidence_feat[step, 0], e1.evidence_feat[step, 0]) > 0.0,
        "all_e1_pair_rows_changed": all(_max_abs_difference(baseline.mp_pair_feat[step, p, 3], e1.mp_pair_feat[step, p, 3]) > 0.0 for p in range(3)),
        "e2_feature_unchanged": _max_abs_difference(baseline.evidence_feat[step, 1], e1.evidence_feat[step, 1]) == 0.0,
        "e2_pair_column_unchanged": _max_abs_difference(baseline.mp_pair_feat[step, :, 4], e1.mp_pair_feat[step, :, 4]) == 0.0,
        "post_features_unchanged": _max_abs_difference(baseline.post_feat, e1.post_feat) == 0.0,
    }
    posterior_checks = {
        "p2_post_changed": _max_abs_difference(baseline.post_feat[step, 1], p2.post_feat[step, 1]) > 0.0,
        "p1_p3_post_unchanged": _max_abs_difference(baseline.post_feat[step, (0, 2)], p2.post_feat[step, (0, 2)]) == 0.0,
        "p2_pair_row_changed": _max_abs_difference(baseline.mp_pair_feat[step, 1], p2.mp_pair_feat[step, 1]) > 0.0,
        "p1_p3_pair_rows_unchanged": _max_abs_difference(baseline.mp_pair_feat[step, (0, 2)], p2.mp_pair_feat[step, (0, 2)]) == 0.0,
    }
    feature_keys = set(baseline.as_shard())
    report = {
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "condition": "nominal",
        "fault_variants_generated": 0,
        "future_perturbation_test": all(item["passed"] for item in future_runs),
        "future_perturbation_max_abs_difference": max(max(item["sensor_ekf_max_abs_difference_before_cut"], item["feature_max_abs_difference_before_cut"]) for item in future_runs),
        "metadata_isolation_test": not bool(feature_keys & FORBIDDEN_METADATA_FIELDS),
        "split_isolation_test": all(row.get("source_split") == "train" and "train_candidates" in row["parquet_path"] and "validation" not in row["parquet_path"].lower() for row in selected),
        "evidence_pair_column_locality": all(evidence_checks.values()),
        "posterior_pair_row_locality": all(posterior_checks.values()),
        "measurement_identity_alignment": bool(np.all(baseline.mp_pair_feat[:, 0, 0, 0] == 1.0) and np.all(baseline.mp_pair_feat[:, 1, 1, 0] == 1.0) and np.all(baseline.mp_pair_feat[:, 2, 2, 0] == 1.0) and np.all(baseline.mp_pair_feat[:, :, 3:, 2] == 1.0)),
        "feature_shape_test": baseline.validation_report.schema_version == "av2_pilot_feature_v1",
        "finite_feature_rate": float(np.mean([np.isfinite(np.asarray(getattr(baseline, name))).all() for name in ("post_feat", "post_mask", "meas_feat", "meas_mask", "evidence_feat", "evidence_mask", "mp_pair_feat", "target")])),
        "future_runs": future_runs,
        "pair_locality": {"scenario_id": scenario.scenario_id, "measurement_seed": MEASUREMENT_SEEDS[0], "step": step, "evidence_checks": evidence_checks, "posterior_checks": posterior_checks},
    }
    hard_pass = all((report["future_perturbation_test"], report["metadata_isolation_test"], report["split_isolation_test"], report["evidence_pair_column_locality"], report["posterior_pair_row_locality"], report["measurement_identity_alignment"], report["feature_shape_test"], report["finite_feature_rate"] == 1.0))
    report["pass"] = hard_pass
    destination = output_path or root / "metadata" / "nominal_v1" / "av2_nominal_integrity_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not hard_pass:
        raise RuntimeError("Nominal-only causality or interface test failed")
    print(json.dumps({key: value for key, value in report.items() if key not in {"future_runs", "pair_locality"}}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AV2 nominal-only integrity checks.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
