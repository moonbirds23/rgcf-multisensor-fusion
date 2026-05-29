"""
Generate all 6 RGCF paper figures from real experimental data.

Usage:
    py -u scripts/run_rgcf_figures.py \
      --a1-dir "results/<A1_dir>" \
      --a3-dir "results/<A3_dir>" \
      --a4-dir "results/<A4_dir>" \
      --ablation-csv "results/rgcf_main_ablation_summary/ablation_summary.csv" \
      --out-dir "figures/rgcf_paper"

Or to find latest automatically:
    py -u scripts/run_rgcf_figures.py --auto --out-dir "figures/rgcf_paper"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rgcf_figures.style import apply_paper_style
from rgcf_figures.data_loader import (
    safe_read_csv, standardize_pred_df, standardize_ablation_df,
    merge_reliability_tables, default_sensor_positions,
)
from rgcf_figures.fig01_scene import plot_scene
from rgcf_figures.fig02_ablation import plot_ablation_summary
from rgcf_figures.fig03_trajectory_all_methods import plot_trajectory_all_methods
from rgcf_figures.fig04_reliability_chain import plot_reliability_chain
from rgcf_figures.fig05_fault_normal_distribution import plot_fault_normal_distribution
from rgcf_figures.fig06_anomaly_heatmap import plot_anomaly_heatmap
from rgcf_figures.fig07_temporal_error_heatmap import plot_temporal_error_heatmap


def find_latest_result(results_root: Path, substring: str) -> Path | None:
    if not results_root.exists():
        return None
    candidates = []
    for sub in sorted(results_root.iterdir()):
        if not sub.is_dir():
            continue
        if substring in sub.name and (sub / "meta.json").exists():
            candidates.append(sub)
    return candidates[-1] if candidates else None


def main():
    p = argparse.ArgumentParser(description="Generate 6 RGCF paper figures")
    p.add_argument("--a1-dir", type=str, default=None)
    p.add_argument("--a3-dir", type=str, default=None)
    p.add_argument("--a4-dir", type=str, default=None)
    p.add_argument("--ablation-csv", type=str, default=None)
    p.add_argument("--auto", action="store_true")
    p.add_argument("--out-dir", type=str, default="figures/rgcf_paper")
    args = p.parse_args()

    results_root = Path("results")

    if args.auto:
        a1_dir = find_latest_result(results_root, "hetero_robust_matrix_mixed_post_only_gnn")
        a3_dir = find_latest_result(results_root, "hetero_robust_matrix_mixed_dual_stream_plain_gnn")
        a4_dir = find_latest_result(results_root, "hetero_robust_matrix_mixed_v2_rgcf")
        ablation_csv = Path("results/rgcf_main_ablation_summary/ablation_summary.csv")
        print(f"[auto] A1: {a1_dir}")
        print(f"[auto] A3: {a3_dir}")
        print(f"[auto] A4: {a4_dir}")
    else:
        a1_dir = Path(args.a1_dir) if args.a1_dir else None
        a3_dir = Path(args.a3_dir) if args.a3_dir else None
        a4_dir = Path(args.a4_dir) if args.a4_dir else None
        ablation_csv = Path(args.ablation_csv) if args.ablation_csv else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_paper_style()
    cfg = {"figure_style": {"dpi": 300, "transparent": True, "save_svg": False}}

    # --- Load data ---
    # A1 pred timeseries (not used by most figures, but available)
    a1_df = None
    if a1_dir:
        pred_path = a1_dir / "artifacts" / "pred_timeseries.csv"
        a1_df = standardize_pred_df(safe_read_csv(pred_path), "A1")

    # A3 pred timeseries
    a3_df = None
    if a3_dir:
        pred_path = a3_dir / "artifacts" / "pred_timeseries.csv"
        a3_df = standardize_pred_df(safe_read_csv(pred_path), "A3")

    # A4 pred timeseries & reliability
    a4_df = None
    rel_df = None
    if a4_dir:
        pred_path = a4_dir / "artifacts" / "pred_timeseries.csv"
        a4_df = standardize_pred_df(safe_read_csv(pred_path), "A4")
        rel_df = merge_reliability_tables(
            gate_df=safe_read_csv(a4_dir / "artifacts" / "gate_timeseries.csv"),
            weight_df=safe_read_csv(a4_dir / "artifacts" / "weight_timeseries.csv"),
            cov_df=safe_read_csv(a4_dir / "artifacts" / "cov_scale_timeseries.csv"),
        )

    # Ablation summary
    ab_df = None
    if ablation_csv and ablation_csv.exists():
        ab_df = standardize_ablation_df(safe_read_csv(ablation_csv))

    sensors = default_sensor_positions()

    # --- Generate figures ---
    print("\n[1/6] Fig.1 Scene")
    if a4_df is not None:
        plot_scene(a4_df, sensors, out_dir, cfg)
    else:
        print("  SKIP: A4 data not found")

    print("\n[2/6] Fig.2 Ablation")
    if ab_df is not None:
        plot_ablation_summary(ab_df, out_dir, cfg)
    else:
        print("  SKIP: ablation summary not found")

    print("\n[3/7] Fig.3 Trajectory All Methods")
    base_ts = out_dir / "baseline_timeseries"
    if a1_df is not None and a3_df is not None and a4_df is not None:
        method_dfs = {}
        for name, fname in [("AVG","avg_timeseries.csv"), ("CI-multi","ci_multi_timeseries.csv"),
                            ("WAA-MM","waa_mm_timeseries.csv")]:
            p = base_ts / fname
            if p.exists():
                import pandas as pd
                method_dfs[name] = pd.read_csv(p)
        method_dfs["A1 Post-only"] = a1_df
        method_dfs["A3 Dual-stream"] = a3_df
        method_dfs["A4 Full RGCF"] = a4_df
        plot_trajectory_all_methods(
            method_dfs, output_dir=out_dir,
            method_order=["AVG","CI-multi","WAA-MM","A1 Post-only","A3 Dual-stream","A4 Full RGCF"],
            fault_windows=[(30.0,70.0)], transparent=True, save_svg=False)
    else:
        print("  SKIP: A1/A3/A4 data not found")

    print("\n[4/6] Fig.4 Reliability Chain")
    if rel_df is not None:
        plot_reliability_chain(rel_df, out_dir, cfg)
    else:
        print("  SKIP: reliability data not found (only A4 has gate/cov_scale)")

    print("\n[5/6] Fig.5 Fault/Normal Distribution")
    if rel_df is not None:
        plot_fault_normal_distribution(rel_df, out_dir, cfg)
    else:
        print("  SKIP: reliability data not found")

    print("\n[6/7] Fig.6 Anomaly Heatmap")
    if a4_df is not None:
        plot_anomaly_heatmap(a4_df, rel_df, out_dir, cfg)
    else:
        print("  SKIP: A4 data not found")

    # --- Fig.7 Temporal error heatmap ---
    print("\n[7/7] Fig.7 Temporal Error Heatmap")
    base_dir = out_dir / "baseline_timeseries"
    if a1_df is not None and a3_df is not None and a4_df is not None:
        method_dfs = {}
        # Rule-based baselines
        for name, fname in [("AVG", "avg_timeseries.csv"),
                            ("CI-multi", "ci_multi_timeseries.csv"),
                            ("WAA-MM", "waa_mm_timeseries.csv")]:
            p = base_dir / fname
            if p.exists():
                import pandas as pd
                method_dfs[name] = pd.read_csv(p)
        # Learned
        method_dfs["A1 Post-only"] = a1_df
        method_dfs["A3 Dual-stream"] = a3_df
        method_dfs["A4 Full RGCF"] = a4_df

        plot_temporal_error_heatmap(
            method_dfs,
            output_dir=out_dir,
            filename="fig07_temporal_error_heatmap",
            method_order=["AVG", "CI-multi", "WAA-MM",
                          "A1 Post-only", "A3 Dual-stream", "A4 Full RGCF"],
            fault_windows=[(30.0, 70.0)],
            color_scale_mode="global_percentile",
            vmax_percentile=97.0,
            strip_height=24,
            cmap="magma",
            vmin=0.0,
            show_title=False,
            transparent=True,
            save_svg=False,
        )
    else:
        print("  SKIP: A1/A3/A4 pred timeseries not all available")

    print(f"\n[done] figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
