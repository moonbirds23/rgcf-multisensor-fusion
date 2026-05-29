"""
Re-run quick eval on an existing trained model and save pred_timeseries.csv.

Usage:
    py -u scripts/rerun_quick_eval_artifacts.py \
      --result-dir "results/<dir>" \
      --dataset-dir "dataset_store/<dir>" \
      --preset "<preset_name>"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import RunRequest
from core.config_loader import load_experiment_bundle
from models.model_factory import build_model_from_bundle
from data.dataset_store import load_pickle
from training.evaluator import evaluate_single_sim_fusion_with_timeseries


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def save_csv(path: Path, rows: list) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=str, required=True)
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--preset", type=str, required=True)
    args = p.parse_args()

    result_dir = Path(args.result_dir)
    dataset_dir = Path(args.dataset_dir)
    artifacts_dir = result_dir / "artifacts"
    metrics_dir = result_dir / "metrics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rerun_eval] preset    : {args.preset}")
    print(f"[rerun_eval] result_dir: {result_dir}")
    print(f"[rerun_eval] dataset   : {dataset_dir}")

    # 1. Load bundle and build model
    bundle = load_experiment_bundle(
        RunRequest(mode="train", preset_name=args.preset)
    )
    model = build_model_from_bundle(bundle)
    device = torch.device("cpu")

    # 2. Load checkpoint
    ckpt_path = result_dir / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("[rerun_eval] model loaded from best_model.pt")

    # 3. Load quick_sim
    quick_sim_path = dataset_dir / "quick_sim.pkl"
    if not quick_sim_path.exists():
        raise FileNotFoundError(f"quick_sim.pkl not found: {quick_sim_path}")
    sim = load_pickle(quick_sim_path)
    print(f"[rerun_eval] quick_sim loaded: K={len(sim['t'])}")

    # 4. Run evaluation
    result = evaluate_single_sim_fusion_with_timeseries(
        sim=sim, model=model, bundle=bundle, device=device,
    )
    summary = result["summary"]
    ts = result["timeseries"]

    # 5. Save pred_timeseries.csv
    pred_rows = []
    for row in ts:
        pred_rows.append({
            "t": row["t"],
            "truth_px": row["truth_px"],
            "truth_py": row["truth_py"],
            "truth_vx": row["truth_vx"],
            "truth_vy": row["truth_vy"],
            "pred_px": row["pred_px"],
            "pred_py": row["pred_py"],
            "pred_vx": row["pred_vx"],
            "pred_vy": row["pred_vy"],
            "error_pos": row["error_pos"],
            "error_full": row["error_full"],
            "fault_active_any": row["fault_active_any"],
            "fault_active_s1": row.get("fault_active_s1", 0),
            "fault_active_s2": row.get("fault_active_s2", 0),
            "fault_active_s3": row.get("fault_active_s3", 0),
            "fault_active_s4": row.get("fault_active_s4", 0),
        })
    save_csv(artifacts_dir / "pred_timeseries.csv", pred_rows)
    print(f"[rerun_eval] saved pred_timeseries.csv ({len(pred_rows)} rows)")

    # 6. Save weight_timeseries.csv (always, for A1/A3/A4)
    w_rows = []
    for row in ts:
        w_rows.append({
            "t": row["t"],
            "w_s1": row.get("w_s1", float("nan")),
            "w_s2": row.get("w_s2", float("nan")),
            "w_s3": row.get("w_s3", float("nan")),
            "w_s4": row.get("w_s4", float("nan")),
            "fault_active_s1": row.get("fault_active_s1", 0),
            "fault_active_s2": row.get("fault_active_s2", 0),
            "fault_active_s3": row.get("fault_active_s3", 0),
            "fault_active_s4": row.get("fault_active_s4", 0),
        })
    save_csv(artifacts_dir / "weight_timeseries.csv", w_rows)
    print(f"[rerun_eval] saved weight_timeseries.csv ({len(w_rows)} rows)")

    # 7. Save gate_timeseries.csv (wide) if gate/cov_scale available
    has_gate = all(f"g_s{i}" in ts[0] for i in range(1, 5))
    has_cov = all(f"cov_scale_s{i}" in ts[0] for i in range(1, 5))
    if has_gate or has_cov:
        wide_rows = []
        for row in ts:
            r = {"t": row["t"], "k": int(row["k"])}
            for i in range(1, 5):
                r[f"g_s{i}"] = row.get(f"g_s{i}", float("nan"))
                r[f"w_s{i}"] = row.get(f"w_s{i}", float("nan"))
                r[f"cov_scale_s{i}"] = row.get(f"cov_scale_s{i}", float("nan"))
                r[f"fault_active_s{i}"] = row.get(f"fault_active_s{i}", 0)
                r[f"valid_s{i}"] = row.get(f"valid_s{i}", float("nan"))
                r[f"gate_target_s{i}"] = row.get(f"gate_target_s{i}", float("nan"))
                r[f"gate_supervision_mask_s{i}"] = row.get(f"gate_supervision_mask_s{i}", float("nan"))
            wide_rows.append(r)
        save_csv(artifacts_dir / "gate_timeseries.csv", wide_rows)
        print(f"[rerun_eval] saved gate_timeseries.csv ({len(wide_rows)} rows)")

    # 8. Save quick_error_window_metrics.json
    err_keys = [
        "overall_rmse_pos", "fault_window_rmse_pos",
        "pre_fault_rmse_pos", "post_fault_rmse_pos",
        "overall_p95_error_pos", "fault_window_p95_error_pos",
        "overall_max_error_pos", "fault_window_max_error_pos",
        "recovery_time_error",
    ]
    err_metrics = {k: summary.get(k, float("nan")) for k in err_keys}
    save_json(metrics_dir / "quick_error_window_metrics.json", err_metrics)
    print("[rerun_eval] saved quick_error_window_metrics.json")

    # 9. Update quick_gnn_metrics.json with weight behavior stats
    weight_keys = [
        "fault_top1_weight_rate",
        "fault_weight_below_equal_rate",
        "fault_weight_below_010_rate",
    ]
    gnn_path = metrics_dir / "quick_gnn_metrics.json"
    if gnn_path.exists():
        with open(gnn_path, "r", encoding="utf-8") as f:
            gnn = json.load(f)
    else:
        gnn = {}
    for k in weight_keys:
        if k in summary:
            gnn[k] = summary[k]
    save_json(gnn_path, gnn)
    print("[rerun_eval] updated quick_gnn_metrics.json with weight behavior stats")

    print("[rerun_eval] done.")


if __name__ == "__main__":
    main()
