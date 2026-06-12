from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_store import load_raw_dataset_store
from models.model_factory import build_model_from_bundle
from scripts.run_rgcf_v5_ablation import build_variant
from training.evaluator import evaluate_single_sim_fusion_with_timeseries


DEFAULT_DATASET_DIR = (
    "dataset_store/20260601_144533__hetero_robust_matrix_mixed_v2_rgcf__"
    "hetero_robust_matr__70722427"
)
DEFAULT_CHECKPOINT = (
    "results/20260601_170351__train__M3_V5_current__hetero_4sensor_scene__"
    "post_meas_soft_gate_fusion/checkpoints/best_model.pt"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_project_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return _project_root() / path


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _mean_metric(rows: list[dict], key: str):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        values.append(float(value))
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def _summarize_rows(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups["ALL"].append(row)
        groups[str(row["mode"])].append(row)
        sid = row.get("sid")
        if sid is not None:
            groups[f"S{int(sid)}"].append(row)
            groups[f"{row['mode']}_S{int(sid)}"].append(row)

    summary = {}
    for group, values in sorted(groups.items()):
        summary[group] = {
            "n": len(values),
            "overall": _mean_metric(values, "overall"),
            "fault": _mean_metric(values, "fault"),
            "p95": _mean_metric(values, "p95"),
            "mean_max": _mean_metric(values, "max"),
        }
    return summary


def _evaluate_combo(test_sims, state_dict, *, temp: float, mix: float) -> dict:
    bundle = build_variant("M3_V5_current", epochs=80, lr=1e-3, batch_size=64, hidden_dim=64)
    bundle.model.base_logit_temperature = float(temp)
    bundle.model.weight_uniform_mix = float(mix)

    model = build_model_from_bundle(bundle)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    worst_rows = []
    device = torch.device("cpu")

    for sim in test_sims:
        ev = evaluate_single_sim_fusion_with_timeseries(sim, model, bundle, device)
        metrics = ev["summary"]
        rows.append(
            {
                "seed": sim.get("seed"),
                "mode": sim.get("effective_fault_mode"),
                "sid": sim.get("effective_fault_sensor_id"),
                "overall": metrics.get("overall_rmse_pos"),
                "fault": metrics.get("fault_window_rmse_pos"),
                "p95": metrics.get("fault_window_p95_error_pos"),
                "max": metrics.get("fault_window_max_error_pos"),
            }
        )

        fault_ts = [r for r in ev["timeseries"] if r.get("fault_active_any", 0.0) > 0.5]
        if fault_ts:
            worst = max(fault_ts, key=lambda r: r["error_pos"])
            worst_rows.append(
                {
                    "seed": sim.get("seed"),
                    "mode": sim.get("effective_fault_mode"),
                    "sid": sim.get("effective_fault_sensor_id"),
                    "max": round(float(worst["error_pos"]), 4),
                    "t": round(float(worst["t"]), 2),
                }
            )

    summary = _summarize_rows(rows)
    worst_sorted = sorted(worst_rows, key=lambda r: r["max"], reverse=True)
    summary["worst_max"] = worst_sorted[0]["max"] if worst_sorted else None
    summary["worst_top3"] = worst_sorted[:3]
    return summary


def _rank_rows(results: dict) -> list[dict]:
    rows = []
    for name, item in results.items():
        all_metrics = item["metrics"]["ALL"]
        rows.append(
            {
                "variant": name,
                "base_logit_temperature": item["base_logit_temperature"],
                "weight_uniform_mix": item["weight_uniform_mix"],
                "overall": all_metrics["overall"],
                "fault": all_metrics["fault"],
                "p95": all_metrics["p95"],
                "mean_max": all_metrics["mean_max"],
                "worst_max": item["metrics"]["worst_max"],
            }
        )
    return sorted(rows, key=lambda r: (r["fault"], r["p95"], r["worst_max"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose posthoc RGCF-V5 logit calibration on test sims.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--temps", default="1,2,4")
    parser.add_argument("--mixes", default="0,0.02,0.05,0.10")
    parser.add_argument("--out", default="results/rgcf_v5_temp_alpha_sweep/M3_posthoc_temp_mix_grid.json")
    parser.add_argument("--rank_csv", default="results/rgcf_v5_temp_alpha_sweep/M3_posthoc_temp_mix_rank.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = _resolve_project_path(args.dataset_dir)
    checkpoint = _resolve_project_path(args.checkpoint)
    out_path = _resolve_project_path(args.out)
    rank_path = _resolve_project_path(args.rank_csv)

    ds = load_raw_dataset_store(dataset_dir)
    test_sims = ds["test_sims"]
    state_dict = torch.load(checkpoint, map_location="cpu")

    results = {}
    for temp in _parse_float_list(args.temps):
        for mix in _parse_float_list(args.mixes):
            name = f"T{temp:g}_mix{mix:g}"
            metrics = _evaluate_combo(test_sims, state_dict, temp=temp, mix=mix)
            results[name] = {
                "base_logit_temperature": temp,
                "weight_uniform_mix": mix,
                "metrics": metrics,
            }
            print(f"[done] {name} ALL={metrics['ALL']} worst={metrics['worst_max']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rank_rows = _rank_rows(results)
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    with rank_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rank_rows)

    print(f"[summary_json] {out_path}")
    print(f"[rank_csv] {rank_path}")


if __name__ == "__main__":
    main()
