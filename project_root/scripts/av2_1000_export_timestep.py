"""GPU-only export of seed-aggregated, scene-level AV2 timestep errors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v1 import NOMINAL_PROTOCOL, WARMUP_SECONDS
from data.av2.feature_builder import build_av2_feature_arrays
from scripts.av2_1000_evaluate import _baseline, _model, _predict
from tools.av2_1000.cache import load_outputs
from tools.av2_1000.common import atomic_json, formal_paths, require_cuda, sha256_file


SELECTED_METHODS = ("full_pefnet", "covariance_intersection", "posterior_only", "pefnet_no_external_evidence", "t1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-timestep position errors without retraining.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="method=path; provide all three model seeds for every learned method")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = formal_paths(root)
    output = args.output or (paths["results"] / "timestep_scene_level.csv")
    device = require_cuda()
    import torch

    models = []
    checkpoint_meta = []
    for entry in args.checkpoint:
        method, path_text = entry.split("=", 1)
        path = Path(path_text).resolve()
        if method not in {"full_pefnet", "posterior_only", "pefnet_no_external_evidence"}:
            raise ValueError(f"unsupported learned method: {method}")
        payload = torch.load(path, map_location=device)
        model_seed = int(payload.get("metadata", {}).get("model_seed", -1))
        model = _model().to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append((method, model_seed, model))
        checkpoint_meta.append({"method": method, "model_seed": model_seed, "path": str(path), "sha256": sha256_file(path)})
    expected = {(method, seed) for method in ("full_pefnet", "posterior_only", "pefnet_no_external_evidence") for seed in range(3)}
    actual = {(method, seed) for method, seed, _ in models}
    if actual != expected or len(models) != 9:
        raise RuntimeError(f"expected exactly nine learned checkpoints; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")

    rows: list[dict[str, object]] = []
    sim_paths = sorted((paths["sim"] / "test").rglob("seed_*.npz"))
    if len(sim_paths) != 600:
        raise RuntimeError(f"expected 600 test simulation records, found {len(sim_paths)}")
    for sim_path in sim_paths:
        outputs, target = load_outputs(sim_path)
        feature = build_av2_feature_arrays(outputs, target, warmup_seconds=WARMUP_SECONDS, protocol=NOMINAL_PROTOCOL)
        mask = np.asarray(feature.eval_mask, dtype=bool) & np.all(outputs.posterior_available, axis=1)
        if int(mask.sum()) != 100:
            raise RuntimeError(f"expected 100 common evaluation steps: {sim_path}")
        with np.load(sim_path, allow_pickle=False) as raw:
            scenario_id = str(raw["scenario_id"])
            measurement_seed = int(raw["measurement_seed"])
        timestamps = np.asarray(outputs.timestamps_ns)[mask]
        time_seconds = (timestamps - timestamps[0]).astype(np.float64) / 1e9
        predictions = {key: value for key, value in _baseline(outputs, target, mask).items() if key in {"covariance_intersection", "t1"}}
        for method, prediction in predictions.items():
            squared = np.sum((np.asarray(prediction)[mask, :2] - target[mask, :2]) ** 2, axis=1)
            for step, (elapsed, value) in enumerate(zip(time_seconds, squared), start=1):
                rows.append({"method": method, "model_seed": -1, "scenario_id": scenario_id, "measurement_seed": measurement_seed, "eval_step": step, "time_seconds": float(elapsed), "position_squared_error": float(value)})
        for method, model_seed, model in models:
            prediction = _predict(model, feature, method, device)
            squared = np.sum((prediction[mask, :2] - target[mask, :2]) ** 2, axis=1)
            for step, (elapsed, value) in enumerate(zip(time_seconds, squared), start=1):
                rows.append({"method": method, "model_seed": model_seed, "scenario_id": scenario_id, "measurement_seed": measurement_seed, "eval_step": step, "time_seconds": float(elapsed), "position_squared_error": float(value)})

    frame = pd.DataFrame.from_records(rows)
    measurement = frame.groupby(["method", "model_seed", "scenario_id", "eval_step"], as_index=False).agg(
        time_seconds=("time_seconds", "mean"),
        position_squared_error=("position_squared_error", "mean"),
        measurement_seed_count=("measurement_seed", "nunique"),
    )
    if set(measurement["measurement_seed_count"]) != {3}:
        raise RuntimeError("measurement seed aggregation is incomplete")
    scene = measurement.groupby(["method", "scenario_id", "eval_step"], as_index=False).agg(
        time_seconds=("time_seconds", "mean"),
        position_squared_error=("position_squared_error", "mean"),
        model_seed_count=("model_seed", "nunique"),
        measurement_seed_count=("measurement_seed_count", "first"),
    )
    expected_counts = scene["method"].map(lambda method: 3 if method in {"full_pefnet", "posterior_only", "pefnet_no_external_evidence"} else 1)
    if not (scene["model_seed_count"].to_numpy() == expected_counts.to_numpy()).all():
        raise RuntimeError("model seed aggregation is incomplete")
    if len(scene) != 5 * 200 * 100 or scene.duplicated(["method", "scenario_id", "eval_step"]).any():
        raise RuntimeError("unexpected timestep scene-level coverage")
    scene["position_rmse"] = np.sqrt(scene["position_squared_error"])
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.to_csv(output, index=False)
    metadata = {
        "status": "PASS",
        "methods": list(SELECTED_METHODS),
        "scenarios": 200,
        "evaluation_steps": 100,
        "measurement_seeds": [100, 101, 102],
        "learned_model_seeds": [0, 1, 2],
        "aggregation": "mean squared position error over measurement seeds, then model seeds; scene remains independent",
        "rows": int(len(scene)),
        "checkpoints": checkpoint_meta,
        "output_sha256": sha256_file(output),
    }
    atomic_json(output.with_suffix(".metadata.json"), metadata)
    print(json.dumps({"status": "PASS", "output": str(output), "rows": len(scene)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
