from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.av2.feature_builder import AV2FeatureArrays, build_av2_feature_arrays
from configs.av2_level_b_plus_v11 import AGGRESSIVE_PROTOCOL, LEVEL_B_PLUS_PROTOCOL_VERSION
from scripts.av2_small_level_b import _load_local_truth
from simulation.av2_sensor_ekf import (
    AV2SensorEkfOutputs,
    PrincipalFault,
    SensorMeasurement,
    run_av2_sensor_ekfs,
)


FIXED_SCENE_INDEXES = (0, 4, 8, 12, 16)
MEASUREMENT_SEEDS = (100, 101, 102)
FORBIDDEN_METADATA_FIELDS = {
    "motion_type",
    "total_displacement",
    "complete_heading_change",
    "future_acceleration",
    "city_name",
    "scenario_end_position",
}


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.dtype == bool or right_array.dtype == bool:
        return 0.0 if np.array_equal(left_array, right_array) else 1.0
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    if not np.array_equal(np.isnan(left_array), np.isnan(right_array)):
        return float("inf")
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left_array[finite] - right_array[finite])))


def _output_prefix_difference(
    left: AV2SensorEkfOutputs, right: AV2SensorEkfOutputs, end: int
) -> float:
    differences = [
        _max_abs_difference(getattr(left, field)[:end], getattr(right, field)[:end])
        for field in (
            "posterior_mean",
            "posterior_covariance_internal",
            "posterior_covariance_reported",
            "posterior_available",
            "posterior_measurement_valid",
            "evidence_valid",
        )
    ]
    for step in range(end):
        for left_measurement, right_measurement in zip(
            left.measurements[step], right.measurements[step]
        ):
            if (
                left_measurement.valid != right_measurement.valid
                or left_measurement.emitted != right_measurement.emitted
                or left_measurement.source_timestamp_ns
                != right_measurement.source_timestamp_ns
            ):
                differences.append(float("inf"))
                continue
            for field in ("z", "R_actual", "R_reported"):
                left_value = getattr(left_measurement, field)
                right_value = getattr(right_measurement, field)
                if left_value is None or right_value is None:
                    differences.append(0.0 if left_value is right_value else float("inf"))
                else:
                    differences.append(_max_abs_difference(left_value, right_value))
    return max(differences, default=0.0)


def _feature_prefix_difference(
    left: AV2FeatureArrays, right: AV2FeatureArrays, end: int
) -> float:
    return max(
        _max_abs_difference(getattr(left, field)[:end], getattr(right, field)[:end])
        for field in (
            "post_feat",
            "post_mask",
            "meas_feat",
            "meas_mask",
            "evidence_feat",
            "evidence_mask",
            "mp_pair_feat",
            "target",
        )
    )


def _replace_measurements(
    outputs: AV2SensorEkfOutputs,
    measurements: tuple[tuple[SensorMeasurement, ...], ...],
    posterior_mean: np.ndarray | None = None,
) -> AV2SensorEkfOutputs:
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


def _perturb_e1_measurement(
    outputs: AV2SensorEkfOutputs, step: int, delta_rad: float = 0.1
) -> AV2SensorEkfOutputs:
    rows = [list(row) for row in outputs.measurements]
    original = rows[step][3]
    if not original.valid or original.z is None:
        raise ValueError("Selected E1 perturbation step is invalid")
    changed_z = np.asarray(original.z, dtype=np.float64).copy()
    changed_z[0] += delta_rad
    rows[step][3] = replace(original, z=changed_z)
    return _replace_measurements(outputs, tuple(tuple(row) for row in rows))


def _perturb_p2_state(
    outputs: AV2SensorEkfOutputs, step: int, delta_x_m: float = 20.0
) -> AV2SensorEkfOutputs:
    posterior_mean = np.asarray(outputs.posterior_mean, dtype=np.float64).copy()
    posterior_mean[step, 1, 0] += delta_x_m
    return _replace_measurements(outputs, outputs.measurements, posterior_mean=posterior_mean)


def _all_feature_arrays(features: AV2FeatureArrays) -> Iterable[np.ndarray]:
    for field in (
        "post_feat",
        "post_mask",
        "meas_feat",
        "meas_mask",
        "evidence_feat",
        "evidence_mask",
        "mp_pair_feat",
        "target",
    ):
        yield np.asarray(getattr(features, field))


def _read_fixed_rows(root: Path) -> list[dict[str, str]]:
    with (root / "manifests" / "level_b_20.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["scenario_id"])
    if len(rows) != 20:
        raise RuntimeError(f"Expected frozen Level B manifest with 20 scenes, got {len(rows)}")
    return [rows[index] for index in FIXED_SCENE_INDEXES]


def run(root: Path) -> None:
    selected = _read_fixed_rows(root)
    future_results: list[dict[str, Any]] = []
    delay_results: list[dict[str, Any]] = []

    for row in selected:
        scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
        cut = local.num_steps // 2
        changed_target = target.copy()
        changed_target[cut + 1 :, :2] += np.array([50_000.0, -50_000.0])
        changed_target[cut + 1 :, 2:] += np.array([500.0, -500.0])
        for seed in MEASUREMENT_SEEDS:
            original = run_av2_sensor_ekfs(
                local.timestamps_ns, target, rng=np.random.default_rng(seed), protocol=AGGRESSIVE_PROTOCOL
            )
            changed = run_av2_sensor_ekfs(
                local.timestamps_ns, changed_target, rng=np.random.default_rng(seed), protocol=AGGRESSIVE_PROTOCOL
            )
            original_features = build_av2_feature_arrays(original, target, protocol=AGGRESSIVE_PROTOCOL)
            changed_features = build_av2_feature_arrays(changed, changed_target, protocol=AGGRESSIVE_PROTOCOL)
            output_difference = _output_prefix_difference(original, changed, cut + 1)
            feature_difference = _feature_prefix_difference(
                original_features, changed_features, cut + 1
            )
            future_results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "measurement_seed": seed,
                    "cut_index": cut,
                    "sensor_ekf_max_abs_difference_before_cut": output_difference,
                    "feature_max_abs_difference_before_cut": feature_difference,
                    "passed": max(output_difference, feature_difference) <= 1e-6,
                }
            )

            nominal = original
            delayed = run_av2_sensor_ekfs(
                local.timestamps_ns,
                target,
                rng=np.random.default_rng(seed),
                fault=PrincipalFault("time_delay", "E2", delay_steps=2),
                protocol=AGGRESSIVE_PROTOCOL,
            )
            wrong = 0
            future_access = 0
            for step, record in enumerate(delayed.measurements):
                measurement = record[4]
                if step < 2:
                    wrong += int(measurement.valid)
                    continue
                source = nominal.measurements[step - 2][4]
                if measurement.valid != source.valid:
                    wrong += 1
                elif measurement.valid and _max_abs_difference(measurement.z, source.z) > 1e-12:
                    wrong += 1
                if measurement.source_timestamp_ns is not None and measurement.source_timestamp_ns > int(local.timestamps_ns[step]):
                    future_access += 1
            delay_results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "measurement_seed": seed,
                    "wrong_delay_direction_count": wrong,
                    "future_measurement_access_count": future_access,
                    "initial_invalid_steps": int(
                        not delayed.measurements[0][4].valid
                        and not delayed.measurements[1][4].valid
                    )
                    * 2,
                    "passed": wrong == 0 and future_access == 0,
                }
            )

    # Pair-locality uses one deterministic scene and one seed.
    row = selected[0]
    scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
    outputs = run_av2_sensor_ekfs(
        local.timestamps_ns, target, rng=np.random.default_rng(MEASUREMENT_SEEDS[0]), protocol=AGGRESSIVE_PROTOCOL
    )
    baseline = build_av2_feature_arrays(outputs, target, protocol=AGGRESSIVE_PROTOCOL)
    valid_steps = np.flatnonzero(np.asarray(baseline.evidence_mask[:, 0], dtype=bool))
    step = int(valid_steps[len(valid_steps) // 2])
    e1_features = build_av2_feature_arrays(_perturb_e1_measurement(outputs, step), target, protocol=AGGRESSIVE_PROTOCOL)
    p2_features = build_av2_feature_arrays(_perturb_p2_state(outputs, step), target, protocol=AGGRESSIVE_PROTOCOL)

    e1_column_differences = [
        _max_abs_difference(
            baseline.mp_pair_feat[step, posterior, 3],
            e1_features.mp_pair_feat[step, posterior, 3],
        )
        for posterior in range(3)
    ]
    evidence_locality = {
        "e1_feature_changed": _max_abs_difference(
            baseline.evidence_feat[step, 0], e1_features.evidence_feat[step, 0]
        )
        > 0.0,
        "all_e1_pair_rows_changed": all(value > 0.0 for value in e1_column_differences),
        "e2_feature_unchanged": _max_abs_difference(
            baseline.evidence_feat[step, 1], e1_features.evidence_feat[step, 1]
        )
        == 0.0,
        "e2_pair_column_unchanged": _max_abs_difference(
            baseline.mp_pair_feat[step, :, 4], e1_features.mp_pair_feat[step, :, 4]
        )
        == 0.0,
        "post_features_unchanged": _max_abs_difference(
            baseline.post_feat, e1_features.post_feat
        )
        == 0.0,
    }
    posterior_locality = {
        "p2_post_changed": _max_abs_difference(
            baseline.post_feat[step, 1], p2_features.post_feat[step, 1]
        )
        > 0.0,
        "p1_p3_post_unchanged": _max_abs_difference(
            baseline.post_feat[step, (0, 2)], p2_features.post_feat[step, (0, 2)]
        )
        == 0.0,
        "p2_pair_row_changed": _max_abs_difference(
            baseline.mp_pair_feat[step, 1], p2_features.mp_pair_feat[step, 1]
        )
        > 0.0,
        "p1_p3_pair_rows_unchanged": _max_abs_difference(
            baseline.mp_pair_feat[step, (0, 2)], p2_features.mp_pair_feat[step, (0, 2)]
        )
        == 0.0,
    }
    pair = baseline.mp_pair_feat
    identity_alignment = bool(
        np.all(pair[:, 0, 0, 0] == 1.0)
        and np.all(pair[:, 1, 1, 0] == 1.0)
        and np.all(pair[:, 2, 2, 0] == 1.0)
        and np.all(pair[:, :, 3:, 2] == 1.0)
    )
    feature_keys = set(baseline.as_shard())
    metadata_isolation = not bool(feature_keys & FORBIDDEN_METADATA_FIELDS)
    finite_feature_rate = float(
        np.mean([np.isfinite(array).all() for array in _all_feature_arrays(baseline)])
    )
    split_isolation = all(
        row.get("source_split") == "train"
        and "train_candidates" in row["parquet_path"]
        and "validation" not in row["parquet_path"].lower()
        and "test" not in row["parquet_path"].lower()
        for row in selected
    )

    causality = {
        "protocol": LEVEL_B_PLUS_PROTOCOL_VERSION,
        "future_perturbation_test": all(item["passed"] for item in future_results),
        "future_perturbation_max_abs_difference": max(
            max(
                item["sensor_ekf_max_abs_difference_before_cut"],
                item["feature_max_abs_difference_before_cut"],
            )
            for item in future_results
        ),
        "delay_direction_test": all(item["passed"] for item in delay_results),
        "wrong_delay_direction_count": sum(
            item["wrong_delay_direction_count"] for item in delay_results
        ),
        "future_measurement_access_count": sum(
            item["future_measurement_access_count"] for item in delay_results
        ),
        "metadata_isolation_test": metadata_isolation,
        "split_isolation_test": split_isolation,
        "runs": len(future_results),
    }
    pair_locality = {
        "protocol": LEVEL_B_PLUS_PROTOCOL_VERSION,
        "scenario_id": scenario.scenario_id,
        "measurement_seed": MEASUREMENT_SEEDS[0],
        "step": step,
        "evidence_checks": evidence_locality,
        "posterior_checks": posterior_locality,
        "evidence_pair_column_locality": all(evidence_locality.values()),
        "posterior_pair_row_locality": all(posterior_locality.values()),
        "measurement_identity_alignment": identity_alignment,
        # The validation report is constructed only after the schema validator
        # has completed without raising.
        "feature_shape_test": baseline.validation_report.schema_version
        == "av2_pilot_feature_v1",
        "finite_feature_rate": finite_feature_rate,
    }
    metadata_dir = root / "metadata" / "level_b_plus"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "causality_test_results.json").write_text(
        json.dumps(
            {"summary": causality, "future_runs": future_results, "delay_runs": delay_results},
            indent=2,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "pair_feature_locality.json").write_text(
        json.dumps(pair_locality, indent=2), encoding="utf-8"
    )
    print(json.dumps({"causality": causality, "pair_locality": pair_locality}, indent=2))
    if not all(
        (
            causality["future_perturbation_test"],
            causality["delay_direction_test"],
            causality["metadata_isolation_test"],
            causality["split_isolation_test"],
            pair_locality["evidence_pair_column_locality"],
            pair_locality["posterior_pair_row_locality"],
            pair_locality["measurement_identity_alignment"],
            pair_locality["feature_shape_test"],
            pair_locality["finite_feature_rate"] == 1.0,
        )
    ):
        raise RuntimeError("Level B+ causality or pair-locality hard gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Level B+ B5/B6 integrity tests.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
