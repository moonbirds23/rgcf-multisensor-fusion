"""Derive grouped tables and a formal report from completed AV2 1000 results.

This command is read-only with respect to raw data, caches, checkpoints, and
existing evaluation rows.  It writes only reporting artifacts under results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _scenario_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Average measurement seeds first, then model seeds for learned methods."""
    metrics = ["position_rmse", "velocity_rmse", "position_p95"]
    seed_level = raw.groupby(["method", "model_seed", "scenario_id"], as_index=False)[metrics].mean()
    scenario_level = seed_level.groupby(["method", "scenario_id"], as_index=False)[metrics].mean()
    model_std = seed_level.groupby(["method", "scenario_id"], as_index=False)["position_rmse"].std(ddof=0)
    model_std = model_std.rename(columns={"position_rmse": "position_rmse_model_seed_std"})
    return scenario_level.merge(model_std, on=["method", "scenario_id"], how="left")


def _group_table(scenario: pd.DataFrame, metadata: pd.DataFrame, column: str) -> pd.DataFrame:
    joined = scenario.merge(metadata[["scenario_id", column]], on="scenario_id", how="left", validate="many_to_one")
    if joined[column].isna().any():
        raise RuntimeError("test manifest metadata is missing for one or more evaluated scenarios")
    return joined.groupby(["method", column], as_index=False).agg(
        scenarios=("scenario_id", "nunique"),
        position_rmse_mean=("position_rmse", "mean"),
        position_rmse_std=("position_rmse", "std"),
        velocity_rmse_mean=("velocity_rmse", "mean"),
        position_p95_mean=("position_p95", "mean"),
        model_seed_std_mean=("position_rmse_model_seed_std", "mean"),
    ).sort_values(["method", column])


def _bare_sha256(value: str) -> str:
    return str(value).removeprefix("sha256:")


def _efficiency_summary(
    path: Path,
    checkpoint_hashes: dict[str, str],
    expected_commit: str,
    expected_test_manifest_sha256: str,
) -> tuple[pd.DataFrame, dict]:
    """Validate the controlled GPU timing artifact and return method summaries."""
    frame = pd.read_csv(path)
    methods = ("full_pefnet", "pefnet_no_external_evidence", "posterior_only")
    required = {
        "method", "model_seed", "parameter_count", "batch_size", "warmup_batches", "timed_batches", "total_timesteps",
        "mean_inference_ms_per_step", "median_inference_ms_per_step", "p95_inference_ms_per_step", "inference_ms_per_scene",
        "peak_gpu_memory_mb", "gpu_name", "pytorch_version", "cuda_version", "frozen_git_commit", "checkpoint_sha256",
        "test_manifest_sha256",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"efficiency.csv is missing required columns: {missing}")
    expected_pairs = {(method, seed) for method in methods for seed in range(3)}
    actual_pairs = set(zip(frame["method"], frame["model_seed"].astype(int)))
    if len(frame) != 9 or actual_pairs != expected_pairs or frame.duplicated(["method", "model_seed"]).any():
        raise RuntimeError("efficiency.csv must contain exactly one row for each neural method and model seed 0/1/2")
    numeric = [
        "parameter_count", "batch_size", "warmup_batches", "timed_batches", "total_timesteps",
        "mean_inference_ms_per_step", "median_inference_ms_per_step", "p95_inference_ms_per_step",
        "inference_ms_per_scene", "peak_gpu_memory_mb",
    ]
    if not all(frame[column].map(lambda value: math.isfinite(float(value))).all() for column in numeric):
        raise RuntimeError("efficiency.csv contains a non-finite numeric value")
    if (frame[["mean_inference_ms_per_step", "median_inference_ms_per_step", "p95_inference_ms_per_step", "inference_ms_per_scene", "peak_gpu_memory_mb"]] <= 0).any().any():
        raise RuntimeError("efficiency.csv contains a non-positive timing or memory value")
    protocol_columns = {"batch_size": 110, "warmup_batches": 20, "timed_batches": 546, "total_timesteps": 60000}
    for column, expected in protocol_columns.items():
        if set(frame[column].astype(int)) != {expected}:
            raise RuntimeError(f"efficiency.csv protocol mismatch: {column} must be {expected}")
    scene_steps = frame["inference_ms_per_scene"] / frame["mean_inference_ms_per_step"]
    if not scene_steps.map(lambda value: math.isclose(float(value), 100.0, rel_tol=0.0, abs_tol=1e-6)).all():
        raise RuntimeError("efficiency.csv scene latency must represent 100 post-warmup evaluation steps")
    if set(frame["frozen_git_commit"]) != {expected_commit}:
        raise RuntimeError("efficiency.csv frozen_git_commit does not match the frozen formal experiment commit")
    if set(frame["test_manifest_sha256"].map(_bare_sha256)) != {_bare_sha256(expected_test_manifest_sha256)}:
        raise RuntimeError("efficiency.csv test manifest hash does not match v2 test_200.csv")
    for row in frame.itertuples(index=False):
        key = f"runs/av2_1000/{row.method}/seed_{int(row.model_seed)}/best.pt"
        if _bare_sha256(row.checkpoint_sha256) != _bare_sha256(checkpoint_hashes[key]):
            raise RuntimeError(f"efficiency.csv checkpoint hash does not match {key}")
    summary = frame.groupby("method", as_index=False).agg(
        model_seeds=("model_seed", "nunique"),
        parameter_count=("parameter_count", "first"),
        mean_inference_ms_per_step=("mean_inference_ms_per_step", "mean"),
        median_inference_ms_per_step=("median_inference_ms_per_step", "mean"),
        p95_inference_ms_per_step=("p95_inference_ms_per_step", "mean"),
        inference_ms_per_scene=("inference_ms_per_scene", "mean"),
        peak_gpu_memory_mb=("peak_gpu_memory_mb", "mean"),
    ).sort_values("mean_inference_ms_per_step")
    audit = {
        "status": "PASS",
        "efficiency_csv_sha256": _sha256(path),
        "rows": int(len(frame)),
        "method_seed_coverage": {method: [0, 1, 2] for method in methods},
        "protocol": {**protocol_columns, "reported_scene_steps": 100},
        "environment": {column: sorted(map(str, frame[column].unique())) for column in ("gpu_name", "pytorch_version", "cuda_version", "frozen_git_commit")},
        "checks": {
            "unique_method_seed_rows": True,
            "finite_positive_metrics": True,
            "checkpoint_hashes_match": True,
            "test_manifest_hash_matches": True,
            "reported_scene_latency_is_100_post_warmup_steps": True,
        },
        "reporting_boundary": "Controlled timing covers the three neural methods only; conventional baselines were not timed and no all-method speed ranking is supported.",
    }
    return summary, audit


def _write_report(
    path: Path,
    overall: pd.DataFrame,
    bootstrap: pd.DataFrame,
    by_motion: pd.DataFrame,
    by_city: pd.DataFrame,
    efficiency: pd.DataFrame,
    efficiency_audit: dict,
) -> None:
    def table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
        rows = ["| " + " | ".join(columns) + " |", "|" + "|".join([" --- "] * len(columns)) + "|"]
        for _, row in frame[columns].iterrows():
            values = []
            for col in columns:
                value = row[col]
                values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
            rows.append("| " + " | ".join(values) + " |")
        return rows

    full = float(overall.loc[overall.method == "full_pefnet", "position_rmse_mean"].iloc[0])
    lines = [
        "# AV2 1000-Scene Formal Experiment Report",
        "",
        "- Protocol: `AV2_NOMINAL_1000_V3.1`",
        "- Scope: fixed 700/100/200 AV2 subset; real-trajectory-driven nominal multi-sensor simulation.",
        "- Test aggregation: measurement seeds `[100,101,102]` are averaged per scenario before model-seed and cross-scenario aggregation.",
        "- Code gate: `16fb852bad13fd73946231114259d83286b5319d`.",
        "",
        "## Main results",
        "",
        *table(overall, ["method", "position_rmse_mean", "position_rmse_std", "scenarios"]),
        "",
        "## Paired bootstrap comparisons",
        "",
        *table(bootstrap, ["comparison", "scenarios", "mean_difference", "ci95_low", "ci95_high"]),
        "",
        "Negative differences mean Full PEFNet has lower position RMSE than the comparison method.",
        "",
        "## Grouped reporting",
        "",
        f"- Motion groups: {by_motion['motion_type'].nunique()} groups; detailed values are in `by_motion.csv`.",
        f"- Cities: {by_city['city_name'].nunique()} groups; detailed values are in `by_city.csv`.",
        f"- Full PEFNet aggregate position RMSE: {full:.3f} m.",
        "",
        "## Reporting boundary",
        "",
        "These results support claims on this fixed AV2 subset under nominal simulated measurements. They do not constitute a complete AV2 benchmark, raw multi-sensor hardware validation, or real sensor-fault validation.",
        "",
        "## Controlled GPU efficiency",
        "",
        *table(efficiency, ["method", "model_seeds", "parameter_count", "mean_inference_ms_per_step", "median_inference_ms_per_step", "p95_inference_ms_per_step", "inference_ms_per_scene", "peak_gpu_memory_mb"]),
        "",
        "- Environment: " + "; ".join(f"{key}={', '.join(value)}" for key, value in efficiency_audit["environment"].items()),
        "- Protocol: batch size 110; 20 warm-up batches; 546 timed batches; 60,000 timed steps per model seed.",
        "- `inference_ms_per_scene` is exactly 100 post-warm-up evaluation steps, rather than all 110 raw sequence steps.",
        "- Reporting boundary: only the three neural methods were timed. Conventional baselines have no controlled timing rows, so this report does not claim an all-method speed ranking or deployment real-time capability.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AV2 1000 grouped tables and reporting artifacts.")
    parser.add_argument("--root", type=Path, required=True, help="The av2_data directory")
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results" / "av2_1000"
    manifest_dir = root / "data" / "manifests" / "av2_1000_v2"
    raw = pd.read_csv(results / "scenario_level.csv")
    metadata = pd.read_csv(manifest_dir / "test_200.csv")
    if raw["scenario_id"].nunique() != 200 or len(metadata) != 200:
        raise RuntimeError("expected exactly 200 independent test scenarios")
    scenario = _scenario_metrics(raw)
    by_motion = _group_table(scenario, metadata, "motion_type")
    by_city = _group_table(scenario, metadata, "city_name")
    overall = scenario.groupby("method", as_index=False).agg(
        position_rmse_mean=("position_rmse", "mean"), position_rmse_std=("position_rmse", "std"), scenarios=("scenario_id", "nunique"),
    ).sort_values("position_rmse_mean")
    bootstrap = pd.read_csv(results / "bootstrap_confidence_intervals.csv")
    comparison = overall[["method", "position_rmse_mean", "position_rmse_std", "scenarios"]].copy()
    full = float(comparison.loc[comparison.method == "full_pefnet", "position_rmse_mean"].iloc[0])
    comparison["full_minus_method_rmse_m"] = full - comparison["position_rmse_mean"]
    comparison["relative_change_vs_method_pct"] = 100.0 * comparison["full_minus_method_rmse_m"] / comparison["position_rmse_mean"]
    by_motion.to_csv(results / "by_motion.csv", index=False)
    by_city.to_csv(results / "by_city.csv", index=False)
    comparison.to_csv(results / "ablation.csv", index=False)
    inventory = {
        split: json.loads((root / "data" / "inventory" / "av2_1000" / ("inventory_" + split + "_summary.json")).read_text(encoding="utf-8"))
        for split in ("train", "val")
    }
    checkpoint_hashes = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted((root / "runs" / "av2_1000").rglob("best.pt"))
    }
    manifest_hashes = json.loads((manifest_dir / "manifest_hashes.json").read_text(encoding="utf-8"))
    efficiency, efficiency_audit = _efficiency_summary(
        results / "efficiency.csv",
        checkpoint_hashes,
        inventory["train"].get("git_commit"),
        manifest_hashes["hashes"]["test_200.csv"],
    )
    (results / "efficiency_audit.json").write_text(json.dumps(efficiency_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(results / "AV2_1000_FORMAL_REPORT.md", overall, bootstrap, by_motion, by_city, efficiency, efficiency_audit)
    artifact_paths = [results / name for name in ("by_motion.csv", "by_city.csv", "ablation.csv", "efficiency_audit.json", "AV2_1000_FORMAL_REPORT.md")]
    reproducibility = {
        "protocol_name": "AV2_NOMINAL_1000_V3.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": inventory["train"].get("git_commit"),
        "manifest_hashes": manifest_hashes,
        "inventory_summaries": inventory,
        "evaluation": {"test_scenarios": 200, "measurement_seeds": sorted(map(int, raw.measurement_seed.unique())), "model_seeds": sorted(map(int, raw.loc[raw.model_seed >= 0, "model_seed"].unique())), "raw_rows": int(len(raw))},
        "checkpoint_sha256": checkpoint_hashes,
        "derived_artifact_sha256": {str(path.relative_to(root)): _sha256(path) for path in artifact_paths},
        "generator": {"python": platform.python_version(), "pandas": pd.__version__},
        "report_generator_sha256": _sha256(Path(__file__).resolve()),
        "efficiency": efficiency_audit,
    }
    (results / "REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(reproducibility, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "outputs": ["by_motion.csv", "by_city.csv", "ablation.csv", "efficiency_audit.json", "AV2_1000_FORMAL_REPORT.md", "REPRODUCIBILITY_MANIFEST.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
