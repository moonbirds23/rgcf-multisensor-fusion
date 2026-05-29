from __future__ import annotations
import argparse
from pathlib import Path
from .style import apply_paper_style
from .demo_data import generate_demo_data
from .data_loader import read_config, safe_read_csv, find_first_existing, standardize_pred_df, merge_reliability_tables, default_sensor_positions
from .fig01_scene import plot_scene
from .fig02_ablation import plot_ablation_summary
from .fig03_trajectory_error import plot_trajectory_error
from .fig04_reliability_chain import plot_reliability_chain
from .fig05_fault_normal_distribution import plot_fault_normal_distribution
from .fig06_anomaly_heatmap import plot_anomaly_heatmap


def load_inputs(data_root, cfg):
    root = Path(data_root)
    p_ab = find_first_existing(root, ["ablation_summary.csv", "metrics/ablation_summary.csv", "summary/ablation_summary.csv"])
    p_a3 = find_first_existing(root, ["A3_pred_timeseries.csv", "a3_pred_timeseries.csv", "artifacts/A3_pred_timeseries.csv", "artifacts/a3_pred_timeseries.csv"])
    p_a4 = find_first_existing(root, ["A4_pred_timeseries.csv", "a4_pred_timeseries.csv", "artifacts/A4_pred_timeseries.csv", "artifacts/a4_pred_timeseries.csv", "artifacts/pred_timeseries.csv"])
    p_rel = find_first_existing(root, ["reliability_timeseries.csv", "artifacts/reliability_timeseries.csv"])
    p_gate = find_first_existing(root, ["gate_timeseries.csv", "artifacts/gate_timeseries.csv"])
    p_weight = find_first_existing(root, ["weight_timeseries.csv", "artifacts/weight_timeseries.csv"])
    p_cov = find_first_existing(root, ["cov_scale_timeseries.csv", "artifacts/cov_scale_timeseries.csv"])
    p_sensors = find_first_existing(root, ["sensor_positions.csv", "artifacts/sensor_positions.csv"])
    ab = safe_read_csv(p_ab) if p_ab else None
    a3 = standardize_pred_df(safe_read_csv(p_a3), "A3") if p_a3 else None
    a4 = standardize_pred_df(safe_read_csv(p_a4), "A4") if p_a4 else None
    rel = merge_reliability_tables(
        gate_df=safe_read_csv(p_gate) if p_gate else None,
        weight_df=safe_read_csv(p_weight) if p_weight else None,
        cov_df=safe_read_csv(p_cov) if p_cov else None,
        reliability_df=safe_read_csv(p_rel) if p_rel else None,
    )
    sensors = safe_read_csv(p_sensors) if p_sensors else default_sensor_positions()
    return ab, a3, a4, rel, sensors


def main():
    parser = argparse.ArgumentParser(description="Generate RGCF paper figures")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--data_root", default=None, help="Root directory of experimental results")
    parser.add_argument("--output_dir", default="rgcf_figures_output")
    parser.add_argument("--demo", action="store_true", help="Generate demo data and figures")
    args = parser.parse_args()
    cfg = read_config(args.config)
    style_cfg = cfg.get("figure_style", {})
    apply_paper_style(style_cfg.get("font_family", "DejaVu Sans"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        data_root = generate_demo_data(out_dir / "demo_data")
    else:
        if args.data_root is None:
            raise SystemExit("Please provide --data_root or use --demo")
        data_root = Path(args.data_root)
    ab, a3, a4, rel, sensors = load_inputs(data_root, cfg)
    if a4 is not None:
        plot_scene(a4, sensors, out_dir, cfg)
    if ab is not None:
        plot_ablation_summary(ab, out_dir, cfg)
    if a3 is not None and a4 is not None:
        plot_trajectory_error(a3, a4, out_dir, cfg)
    if rel is not None:
        plot_reliability_chain(rel, out_dir, cfg)
        plot_fault_normal_distribution(rel, out_dir, cfg)
    if a4 is not None:
        plot_anomaly_heatmap(a4, rel, out_dir, cfg)
    print(f"[done] figures saved to: {out_dir}")

if __name__ == "__main__":
    main()
