"""Audit the GPU-exported AV2 timestep artifact before paper plotting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {"full_pefnet", "covariance_intersection", "posterior_only", "pefnet_no_external_evidence", "t1"}
LEARNED = {"full_pefnet", "posterior_only", "pefnet_no_external_evidence"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.test_manifest)
    required = {
        "method", "scenario_id", "eval_step", "time_seconds", "position_squared_error",
        "model_seed_count", "measurement_seed_count", "position_rmse",
    }
    checks = {
        "metadata_status_pass": metadata.get("status") == "PASS",
        "required_columns_present": required.issubset(frame.columns),
        "rows_equal_100000": len(frame) == 100000,
        "methods_exact": set(frame["method"]) == METHODS,
        "unique_method_scene_step": not frame.duplicated(["method", "scenario_id", "eval_step"]).any(),
        "finite_numeric_values": bool(np.isfinite(frame[["eval_step", "time_seconds", "position_squared_error", "position_rmse"]].to_numpy(float)).all()),
        "nonnegative_squared_error": bool((frame["position_squared_error"] >= 0).all()),
        "rmse_matches_squared_error": bool(np.allclose(frame["position_rmse"].to_numpy(float) ** 2, frame["position_squared_error"].to_numpy(float), rtol=1e-10, atol=1e-10)),
        "all_measurement_seed_counts_equal_3": set(frame["measurement_seed_count"].astype(int)) == {3},
    }
    coverage = frame.groupby("method").agg(scenarios=("scenario_id", "nunique"), steps=("eval_step", "nunique"), rows=("eval_step", "size"), model_seed_count=("model_seed_count", "first"))
    checks["coverage_exact"] = bool((coverage["scenarios"] == 200).all() and (coverage["steps"] == 100).all() and (coverage["rows"] == 20000).all())
    expected_model_counts = coverage.index.to_series().map(lambda method: 3 if method in LEARNED else 1)
    checks["model_seed_counts_exact"] = bool((coverage["model_seed_count"].astype(int).to_numpy() == expected_model_counts.to_numpy()).all())
    scene_sets = frame.groupby("method")["scenario_id"].agg(lambda values: set(values))
    manifest_scenes = set(manifest["scenario_id"])
    checks["scene_sets_match_test_manifest"] = all(scenes == manifest_scenes for scenes in scene_sets)
    time_stats = frame.groupby("eval_step")["time_seconds"].agg(["min", "max", "mean"])
    expected_time = (time_stats.index.to_numpy(float) - 1.0) * 0.1
    checks["time_axis_consistent"] = bool(np.allclose(time_stats["min"], time_stats["max"], atol=1e-12) and np.allclose(time_stats["mean"], expected_time, atol=1e-12))
    actual_hash = sha256(args.data)
    checks["csv_hash_matches_metadata"] = actual_hash == metadata.get("output_sha256")
    checkpoint_results = []
    for item in metadata.get("checkpoints", []):
        path = args.checkpoint_root / item["method"] / f"seed_{int(item['model_seed'])}" / "best.pt"
        actual = sha256(path) if path.exists() else None
        checkpoint_results.append({"method": item["method"], "model_seed": int(item["model_seed"]), "exists": path.exists(), "expected_sha256": item["sha256"], "actual_sha256": actual, "matches": actual == item["sha256"]})
    checks["all_checkpoint_hashes_match"] = len(checkpoint_results) == 9 and all(item["matches"] for item in checkpoint_results)
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = frame.groupby("method", as_index=False).agg(
        global_position_rmse=("position_squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        mean_scene_step_rmse=("position_rmse", "mean"),
    ).sort_values("global_position_rmse")
    output = {
        "status": status,
        "source_csv": str(args.data.resolve()),
        "source_csv_sha256": actual_hash,
        "rows": int(len(frame)),
        "time_seconds": {"min": float(frame["time_seconds"].min()), "max": float(frame["time_seconds"].max()), "steps": int(frame["eval_step"].nunique())},
        "checks": checks,
        "coverage": {method: {key: int(value) for key, value in row.items()} for method, row in coverage.to_dict("index").items()},
        "checkpoint_audit": checkpoint_results,
        "descriptive_summary": summary.to_dict("records"),
        "aggregation_note": "The timestep curve is sqrt(mean squared position error across independent scenes) at each time step after the GPU export has averaged measurement seeds and then model seeds.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output), "checks": checks}, ensure_ascii=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
