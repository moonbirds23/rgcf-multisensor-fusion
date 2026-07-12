"""Run the frozen AV2 Level B+ B4 single-fault directionality audit.

This is a CPU-only diagnostic.  It reads the already frozen Level B manifest,
selects sorted scene indexes 0/4/8/12/16, and writes results beside the AV2
data root.  It does not modify raw AV2 data or train PEFNet.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.av2.feature_builder import AV2FeatureArrays, build_av2_feature_arrays
from configs.av2_level_b_plus_v11 import (
    AGGRESSIVE_PROTOCOL,
    LEVEL_B_PLUS_PROTOCOL_VERSION,
    MEASUREMENT_SEEDS,
    T2_BIAS_ONSET_SECONDS,
    T2_BIAS_RAMP_DURATION_SECONDS,
    T2_BIAS_TERMINAL_RANGE_M,
)
from scripts.av2_small_level_b import _load_local_truth
from simulation.av2_sensor_ekf import PrincipalFault, run_av2_sensor_ekfs


PROTOCOL = LEVEL_B_PLUS_PROTOCOL_VERSION
SCENE_INDEXES = (0, 4, 8, 12, 16)
EPS = 1e-12


def _finite_features(features: AV2FeatureArrays) -> bool:
    return all(
        np.isfinite(np.asarray(getattr(features, name))).all()
        for name in (
            "post_feat",
            "post_mask",
            "meas_feat",
            "meas_mask",
            "evidence_feat",
            "evidence_mask",
            "mp_pair_feat",
        )
    )


def _same_optional(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _evidence_scores(
    nominal: AV2FeatureArrays, candidate: AV2FeatureArrays
) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal/candidate q using nominal-only evidence calibration.

    The B3 definition standardizes each evidence source before averaging.  We
    estimate each source's mean/std from nominal, evaluated, valid pairs only,
    then apply those frozen values to both conditions.  Raw residual norms are
    squared to obtain the normalized quadratic residual d.
    """
    nom_d = np.square(np.asarray(nominal.diagnostics.postfit_residual_norm, dtype=np.float64)[:, 3:5])
    can_d = np.square(np.asarray(candidate.diagnostics.postfit_residual_norm, dtype=np.float64)[:, 3:5])
    nom_q = np.full((nom_d.shape[0], 3), np.nan, dtype=np.float64)
    can_q = np.full_like(nom_q, np.nan)
    nom_z = np.full_like(nom_d, np.nan)
    can_z = np.full_like(can_d, np.nan)
    for evidence_index in range(2):
        calibration = nom_d[:, evidence_index]
        valid = (
            (np.asarray(nominal.eval_mask)[:, None] > 0.5)
            & (np.asarray(nominal.post_mask) > 0.5)
            & (np.asarray(nominal.evidence_mask)[:, evidence_index, None] > 0.5)
            & np.isfinite(calibration)
        )
        values = calibration[valid]
        mu = float(np.mean(values))
        sigma = max(float(np.std(values)), EPS)
        nom_z[:, evidence_index] = (nom_d[:, evidence_index] - mu) / sigma
        can_z[:, evidence_index] = (can_d[:, evidence_index] - mu) / sigma
    for step in range(nom_q.shape[0]):
        for posterior in range(3):
            n = nom_z[step, :, posterior]
            c = can_z[step, :, posterior]
            if np.isfinite(n).any():
                nom_q[step, posterior] = float(np.nanmean(n))
            if np.isfinite(c).any():
                can_q[step, posterior] = float(np.nanmean(c))
    return nom_q, can_q


def _median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _row_base(scenario_id: str, seed: int, condition: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "scenario_id": scenario_id,
        "measurement_seed": seed,
        "condition": condition,
    }


def _bias_row(
    scenario_id: str,
    seed: int,
    elapsed: np.ndarray,
    target: np.ndarray,
    nominal_out: Any,
    nominal_feat: AV2FeatureArrays,
) -> dict[str, Any]:
    fault = PrincipalFault(
        "bias_ramp",
        "T2",
        onset_seconds=T2_BIAS_ONSET_SECONDS,
        duration_seconds=T2_BIAS_RAMP_DURATION_SECONDS,
        terminal_bias=np.array([T2_BIAS_TERMINAL_RANGE_M, 0.0], dtype=np.float64),
    )
    output = run_av2_sensor_ekfs(
        nominal_out.timestamps_ns,
        target,
        rng=np.random.default_rng(seed),
        fault=fault,
        protocol=AGGRESSIVE_PROTOCOL,
    )
    features = build_av2_feature_arrays(
        output, target, warmup_seconds=1.0, protocol=AGGRESSIVE_PROTOCOL
    )
    window = (elapsed >= T2_BIAS_ONSET_SECONDS) & (np.asarray(features.eval_mask) > 0.5)
    nom_error = np.linalg.norm(np.asarray(nominal_out.posterior_mean)[:, 1, :2] - target[:, :2], axis=1)
    fault_error = np.linalg.norm(np.asarray(output.posterior_mean)[:, 1, :2] - target[:, :2], axis=1)
    nom_q, fault_q = _evidence_scores(nominal_feat, features)
    nom_error_median = _median(nom_error[window])
    fault_error_median = _median(fault_error[window])
    nom_score_median = _median(nom_q[window, 1])
    fault_score_median = _median(fault_q[window, 1])
    error_increase = (fault_error_median - nom_error_median) / (nom_error_median + EPS)
    # A standardized score can cross zero, so use |nominal| in the relative
    # increase denominator while also preserving the literal median ratio.
    score_increase = (fault_score_median - nom_score_median) / (abs(nom_score_median) + EPS)
    detected = np.nanargmax(fault_q[window], axis=1) == 1
    detection_accuracy = float(np.mean(detected)) if detected.size else 0.0
    success = error_increase >= 0.25 and score_increase >= 0.20 and detection_accuracy >= 0.70
    return {
        **_row_base(scenario_id, seed, "T2_range_bias_ramp"),
        "fault_onset_seconds": T2_BIAS_ONSET_SECONDS,
        "fault_ramp_duration_seconds": T2_BIAS_RAMP_DURATION_SECONDS,
        "fault_terminal_range_bias_m": T2_BIAS_TERMINAL_RANGE_M,
        "fault_window_steps": int(window.sum()),
        "posterior_error_nominal_median": nom_error_median,
        "posterior_error_fault_median": fault_error_median,
        "posterior_error_gain_ratio": fault_error_median / (nom_error_median + EPS),
        "posterior_error_relative_increase": error_increase,
        "evidence_score_nominal_median": nom_score_median,
        "evidence_score_fault_median": fault_score_median,
        "evidence_score_gain_ratio": fault_score_median / (nom_score_median + EPS),
        "evidence_score_relative_increase": score_increase,
        "fault_source_detection_accuracy": detection_accuracy,
        "finite_features": _finite_features(features),
        "success": bool(success and _finite_features(features)),
    }


def _dropout_row(
    scenario_id: str,
    seed: int,
    elapsed: np.ndarray,
    target: np.ndarray,
    nominal_out: Any,
) -> dict[str, Any]:
    fault = PrincipalFault("dropout", "E1", onset_seconds=4.0, duration_seconds=1.0)
    output = run_av2_sensor_ekfs(
        nominal_out.timestamps_ns,
        target,
        rng=np.random.default_rng(seed),
        fault=fault,
        protocol=AGGRESSIVE_PROTOCOL,
    )
    features = build_av2_feature_arrays(
        output, target, warmup_seconds=1.0, protocol=AGGRESSIVE_PROTOCOL
    )
    window = (elapsed >= 4.0) & (elapsed < 5.0)
    mask = np.asarray(features.evidence_mask)[:, 0] > 0.5
    # E1 is rate-gated at 5 Hz while the AV2 timeline is 10 Hz.  "Outside ==
    # 1" means every measurement that would be valid nominally stays valid;
    # non-emission timeline steps must remain zero in both conditions.
    expected = np.asarray(nominal_out.evidence_valid)[:, 0].copy()
    expected[window] = False
    exact = bool(np.array_equal(mask, expected))
    no_fill = all(output.measurements[k][3].z is None for k in np.flatnonzero(window))
    zero_features = bool(
        np.all(np.asarray(features.evidence_feat)[window, 0] == 0.0)
        and np.all(np.asarray(features.mp_pair_feat)[window, :, 3, 3:6] == 0.0)
    )
    e2_unchanged = all(
        output.evidence_valid[k, 1] == nominal_out.evidence_valid[k, 1]
        and _same_optional(output.measurements[k][4].z, nominal_out.measurements[k][4].z)
        and _same_optional(output.measurements[k][4].R_actual, nominal_out.measurements[k][4].R_actual)
        for k in range(elapsed.size)
    )
    finite = _finite_features(features)
    return {
        **_row_base(scenario_id, seed, "E1_dropout"),
        "fault_window_steps": int(window.sum()),
        "mask_exact_match_rate": float(np.mean(mask == expected)),
        "invalid_evidence_forward_fill_count": int(not no_fill),
        "masked_features_zero": zero_features,
        "unaffected_evidence_changed": not e2_unchanged,
        "finite_features": finite,
        "success": bool(exact and no_fill and zero_features and e2_unchanged and finite),
    }


def _delay_row(
    scenario_id: str,
    seed: int,
    target: np.ndarray,
    nominal_out: Any,
) -> dict[str, Any]:
    fault = PrincipalFault("time_delay", "E2", delay_steps=2)
    output = run_av2_sensor_ekfs(
        nominal_out.timestamps_ns,
        target,
        rng=np.random.default_rng(seed),
        fault=fault,
        protocol=AGGRESSIVE_PROTOCOL,
    )
    wrong = 0
    future = 0
    for k in range(2, len(output.measurements)):
        delayed = output.measurements[k][4]
        original = nominal_out.measurements[k - 2][4]
        if delayed.valid != original.valid or (delayed.valid and not _same_optional(delayed.z, original.z)):
            wrong += 1
        if delayed.source_timestamp_ns is not None and delayed.source_timestamp_ns > int(output.timestamps_ns[k]):
            future += 1
    initial_invalid = sum(not output.measurements[k][4].valid for k in range(2))
    features = build_av2_feature_arrays(
        output, target, warmup_seconds=1.0, protocol=AGGRESSIVE_PROTOCOL
    )
    finite = _finite_features(features)
    return {
        **_row_base(scenario_id, seed, "E2_delay_2_steps"),
        "wrong_delay_direction_count": wrong,
        "future_measurement_access_count": future,
        "initial_invalid_steps": initial_invalid,
        "finite_features": finite,
        "success": bool(wrong == 0 and future == 0 and initial_invalid == 2 and finite),
    }


def _mismatch_row(
    scenario_id: str,
    seed: int,
    alpha: float,
    target: np.ndarray,
    nominal_out: Any,
    nominal_feat: AV2FeatureArrays,
) -> dict[str, Any]:
    fault = PrincipalFault("covariance_mismatch", "T2", covariance_alpha=alpha)
    output = run_av2_sensor_ekfs(
        nominal_out.timestamps_ns,
        target,
        rng=np.random.default_rng(seed),
        fault=fault,
        protocol=AGGRESSIVE_PROTOCOL,
    )
    features = build_av2_feature_arrays(
        output, target, warmup_seconds=1.0, protocol=AGGRESSIVE_PROTOCOL
    )
    state_diff = float(np.nanmax(np.abs(np.asarray(output.posterior_mean) - np.asarray(nominal_out.posterior_mean))))
    cov_diff = float(
        np.nanmax(
            np.abs(
                np.asarray(output.posterior_covariance_internal)
                - np.asarray(nominal_out.posterior_covariance_internal)
            )
        )
    )
    valid = np.asarray(output.posterior_available)[:, 1]
    internal_diag = np.diagonal(np.asarray(output.posterior_covariance_internal)[:, 1], axis1=1, axis2=2)[valid]
    reported_diag = np.diagonal(np.asarray(output.posterior_covariance_reported)[:, 1], axis1=1, axis2=2)[valid]
    relative_scale_error = float(np.max(np.abs(reported_diag / internal_diag - alpha) / alpha))
    feature_delta = float(
        np.max(
            np.abs(
                np.asarray(features.post_feat)[valid, 1, 4:8]
                - np.asarray(nominal_feat.post_feat)[valid, 1, 4:8]
            )
        )
    )
    diagnostics_contract = bool(
        np.array_equal(
            np.asarray(features.diagnostics.posterior_covariance_reported),
            np.asarray(output.posterior_covariance_reported),
            equal_nan=True,
        )
    )
    finite = _finite_features(features)
    success = (
        state_diff <= 1e-10
        and cov_diff <= 1e-10
        and relative_scale_error <= 1e-6
        and feature_delta > 0.0
        and diagnostics_contract
        and finite
    )
    return {
        **_row_base(scenario_id, seed, f"T2_covariance_mismatch_{alpha:g}"),
        "covariance_alpha": alpha,
        "internal_state_max_abs_difference": state_diff,
        "internal_covariance_max_abs_difference": cov_diff,
        "reported_scale_relative_error": relative_scale_error,
        "feature_covariance_max_abs_difference": feature_delta,
        "feature_covariance_scale_visible": feature_delta > 0.0,
        "fusion_facing_reported_covariance_contract": diagnostics_contract,
        "finite_features": finite,
        "success": success,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifests" / "level_b_20.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = sorted(csv.DictReader(handle), key=lambda row: row["scenario_id"])
    if len(manifest) != 20:
        raise RuntimeError(f"Frozen Level B manifest must contain exactly 20 scenes, found {len(manifest)}")
    selected = [manifest[index] for index in SCENE_INDEXES]
    metadata_dir = root / "metadata" / "level_b_plus"
    log_dir = root / "logs" / "level_b_plus"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for scene in selected:
        scenario, local, target = _load_local_truth(Path(scene["parquet_path"]))
        for seed in MEASUREMENT_SEEDS:
            nominal_out = run_av2_sensor_ekfs(
                local.timestamps_ns,
                target,
                rng=np.random.default_rng(seed),
                protocol=AGGRESSIVE_PROTOCOL,
            )
            nominal_feat = build_av2_feature_arrays(
                nominal_out, target, warmup_seconds=1.0, protocol=AGGRESSIVE_PROTOCOL
            )
            rows.append(_bias_row(scenario.scenario_id, seed, local.elapsed_seconds, target, nominal_out, nominal_feat))
            rows.append(_dropout_row(scenario.scenario_id, seed, local.elapsed_seconds, target, nominal_out))
            rows.append(_delay_row(scenario.scenario_id, seed, target, nominal_out))
            for alpha in (0.25, 4.0):
                rows.append(_mismatch_row(scenario.scenario_id, seed, alpha, target, nominal_out, nominal_feat))

    output_csv = metadata_dir / "fault_directionality.csv"
    _write_csv(output_csv, rows)
    condition_summary: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        condition_summary[condition] = {
            "runs": len(group),
            "successful_runs": sum(bool(row["success"]) for row in group),
            "pass": all(bool(row["success"]) for row in group),
        }
    bias_success = condition_summary["T2_range_bias_ramp"]["successful_runs"]
    summary = {
        "protocol": PROTOCOL,
        "scene_indexes": list(SCENE_INDEXES),
        "scenario_ids": [row["scenario_id"] for row in selected],
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "total_condition_runs": len(rows),
        "conditions": condition_summary,
        "bias_successful_runs_gate": {"required": 12, "actual": bias_success, "pass": bias_success >= 12},
        "hard_interface_checks_pass": all(
            condition_summary[name]["pass"]
            for name in (
                "E1_dropout",
                "E2_delay_2_steps",
                "T2_covariance_mismatch_0.25",
                "T2_covariance_mismatch_4",
            )
        ),
    }
    summary["b4_pass"] = bool(summary["bias_successful_runs_gate"]["pass"] and summary["hard_interface_checks_pass"])
    summary_path = metadata_dir / "fault_directionality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (log_dir / "b4_completed.json").write_text(
        json.dumps({"csv": str(output_csv), "summary": str(summary_path), "b4_pass": summary["b4_pass"]}, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AV2 Level B+ B4 single-fault directionality checks.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.root.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
