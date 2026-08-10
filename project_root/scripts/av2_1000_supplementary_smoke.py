"""CPU smoke test for all deterministic AV2 V3.2 experiment groups."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v32 import (
    ARTIFACT_KEY,
    NOMINAL_PROTOCOL,
    NOMINAL_PROTOCOL_NAME,
    WARMUP_SECONDS,
    config_sha256,
)
from simulation.supplementary_baselines import (
    fuse_av2_ci_tracks,
    run_centralized_multisensor_ekf,
    run_ci_eu_sequence,
)
from tools.av2_1000.cache import load_outputs
from tools.av2_1000.common import atomic_json, formal_paths, write_csv


def _metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction)[mask] - target[mask]
    position = np.linalg.norm(error[:, :2], axis=1)
    velocity = np.linalg.norm(error[:, 2:], axis=1)
    return {
        "num_eval_steps": int(mask.sum()),
        "position_rmse": float(np.sqrt(np.mean(position**2))),
        "position_p95": float(np.percentile(position, 95)),
        "position_max": float(np.max(position)),
        "velocity_rmse": float(np.sqrt(np.mean(velocity**2))),
    }


def _existing_baselines(outputs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    states = np.asarray(outputs.posterior_mean, dtype=np.float64)
    covariances = np.asarray(outputs.posterior_covariance_reported, dtype=np.float64)
    valid = np.asarray(outputs.posterior_available, dtype=bool)
    steps = len(states)
    mean = np.full((steps, 4), np.nan, dtype=np.float64)
    weighted = np.full((steps, 4), np.nan, dtype=np.float64)
    ci = np.full((steps, 4), np.nan, dtype=np.float64)
    mean_covariance = np.full((steps, 4, 4), np.nan, dtype=np.float64)
    weighted_covariance = np.full((steps, 4, 4), np.nan, dtype=np.float64)
    ci_covariance = np.full((steps, 4, 4), np.nan, dtype=np.float64)
    for step in range(steps):
        indexes = np.flatnonzero(valid[step])
        if not indexes.size:
            continue
        mean[step] = states[step, indexes].mean(axis=0)
        mean_covariance[step] = covariances[step, indexes].mean(axis=0)
        weights = 1.0 / np.maximum(
            np.trace(covariances[step, indexes], axis1=1, axis2=2), 1e-8
        )
        weights /= weights.sum()
        weighted[step] = np.sum(states[step, indexes] * weights[:, None], axis=0)
        weighted_covariance[step] = np.sum(
            covariances[step, indexes] * weights[:, None, None], axis=0
        )
        ci_state, ci_P, _ = fuse_av2_ci_tracks(
            states[step], covariances[step], valid[step], n_grid=31
        )
        ci[step], ci_covariance[step] = ci_state, ci_P
    predictions = {
        "t1": states[:, 0],
        "t2": states[:, 1],
        "t3": states[:, 2],
        "mean_fusion": mean,
        "covariance_weighted": weighted,
        "covariance_intersection": ci,
    }
    uncertainty = {
        "t1": covariances[:, 0],
        "t2": covariances[:, 1],
        "t3": covariances[:, 2],
        "mean_fusion": mean_covariance,
        "covariance_weighted": weighted_covariance,
        "covariance_intersection": ci_covariance,
    }
    return predictions, uncertainty


def run(
    root: Path,
    *,
    count: int = 5,
    measurement_seed: int = 100,
    output: Path | None = None,
) -> dict[str, object]:
    if count < 1 or count > 200:
        raise ValueError("count must lie in [1, 200]")
    paths = formal_paths(root, artifact_key=ARTIFACT_KEY)
    files = sorted((paths["sim"] / "test").rglob(f"seed_{measurement_seed}.npz"))[:count]
    if len(files) != count:
        raise RuntimeError(f"expected {count} smoke caches, found {len(files)}")
    digest = config_sha256()
    rows: list[dict[str, object]] = []
    squared_errors: dict[str, list[np.ndarray]] = defaultdict(list)
    diagnostics: list[dict[str, object]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as raw:
            scenario_id = str(raw["scenario_id"])
            if str(raw["protocol_name"]) != NOMINAL_PROTOCOL_NAME:
                raise RuntimeError(f"protocol mismatch: {path}")
            if str(raw["config_sha256"]) != digest:
                raise RuntimeError(f"config hash mismatch: {path}")
        outputs, target = load_outputs(path)
        elapsed = (outputs.timestamps_ns - outputs.timestamps_ns[0]).astype(np.float64) * 1e-9
        eval_mask = elapsed >= WARMUP_SECONDS
        predictions, uncertainty = _existing_baselines(outputs)
        ci_eu = run_ci_eu_sequence(outputs, NOMINAL_PROTOCOL, ci_grid_points=31)
        cm_ekf = run_centralized_multisensor_ekf(outputs, NOMINAL_PROTOCOL)
        predictions.update(
            {
                "ci_eu": ci_eu.xhat,
                "centralized_multisensor_ekf": cm_ekf.xhat,
            }
        )
        uncertainty.update(
            {
                "ci_eu": ci_eu.Phat,
                "centralized_multisensor_ekf": cm_ekf.Phat,
            }
        )
        valid_by_method = {
            **{name: np.all(outputs.posterior_available, axis=1) for name in predictions},
            "ci_eu": ci_eu.valid_mask,
            "centralized_multisensor_ekf": cm_ekf.valid_mask,
        }
        for method, prediction in predictions.items():
            mask = eval_mask & valid_by_method[method]
            covariance = uncertainty[method][mask]
            minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
            if not np.all(np.isfinite(prediction[mask])) or minimum_eigenvalue <= 0.0:
                raise RuntimeError(f"{method}: non-finite prediction or non-SPD covariance")
            metric = _metrics(prediction, target, mask)
            rows.append(
                {
                    "method": method,
                    "model_seed": -1,
                    "scenario_id": scenario_id,
                    "measurement_seed": measurement_seed,
                    **metric,
                    "minimum_covariance_eigenvalue": minimum_eigenvalue,
                }
            )
            position_error = prediction[mask, :2] - target[mask, :2]
            squared_errors[method].append(np.sum(position_error**2, axis=1))
        diagnostics.append(
            {
                "scenario_id": scenario_id,
                "ci_effective_mean_weights_t1_t2_t3": ci_eu.ci_weights[eval_mask].mean(axis=0).tolist(),
                "ci_dominant_source_counts_t1_t2_t3": np.bincount(
                    np.argmax(ci_eu.ci_weights[eval_mask], axis=1), minlength=3
                ).astype(int).tolist(),
                "ci_eu_evidence_dim_counts": {
                    str(value): int(np.sum(ci_eu.evidence_dim[eval_mask] == value))
                    for value in np.unique(ci_eu.evidence_dim[eval_mask])
                },
                "ci_eu_mean_nis": float(np.nanmean(ci_eu.nis[eval_mask])),
                "cm_active_dim_counts": {
                    str(value): int(np.sum(cm_ekf.active_measurement_dims[eval_mask] == value))
                    for value in np.unique(cm_ekf.active_measurement_dims[eval_mask])
                },
                "cm_mean_nis": float(np.nanmean(cm_ekf.nis[eval_mask])),
            }
        )
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    pooled_rmse = {
        method: float(np.sqrt(np.mean(np.concatenate(values))))
        for method, values in squared_errors.items()
    }
    ci_rmse = pooled_rmse["covariance_intersection"]
    summary_rows = []
    for method in sorted(by_method):
        group = by_method[method]
        summary_rows.append(
            {
                "method": method,
                "scenarios": len(group),
                "mean_scene_position_rmse": float(
                    np.mean([float(row["position_rmse"]) for row in group])
                ),
                "pooled_position_rmse": pooled_rmse[method],
                "position_p95_mean": float(
                    np.mean([float(row["position_p95"]) for row in group])
                ),
                "relative_rmse_improvement_vs_ci": float(
                    (ci_rmse - pooled_rmse[method]) / ci_rmse
                ),
            }
        )
    result = {
        "status": "PASS",
        "scope": "CPU deterministic baselines; learned PEFNet groups require CUDA checkpoints",
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "config_sha256": digest,
        "measurement_seed": measurement_seed,
        "scenario_count": count,
        "methods": [row["method"] for row in summary_rows],
        "summary": summary_rows,
        "diagnostics": diagnostics,
    }
    target_dir = output or paths["results"] / f"smoke_supplementary_{count}_seed_{measurement_seed}"
    target_dir.mkdir(parents=True, exist_ok=True)
    write_csv(target_dir / "scenario_level.csv", rows, list(rows[0]))
    write_csv(target_dir / "summary.csv", summary_rows, list(summary_rows[0]))
    atomic_json(target_dir / "smoke_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--measurement-seed", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        count=args.count,
        measurement_seed=args.measurement_seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
