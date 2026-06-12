"""
Collect ablation summary across AVG / CI-multi / WAA-MM / A1 / A3 / A4.

Usage (explicit dirs):
    py -u scripts/collect_rgcf_main_ablation.py \
      --post-only-result "results/<A1_dir>" \
      --dual-stream-result "results/<A3_dir>" \
      --rgcf-result "results/<A4_dir>" \
      --out-dir "results/rgcf_main_ablation_summary"

Usage (auto-latest):
    py -u scripts/collect_rgcf_main_ablation.py \
      --results-root "results" \
      --out-dir "results/rgcf_main_ablation_summary" \
      --auto-latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
#  Helpers
# ============================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: dict, key: str, default=None):
    v = d.get(key, default)
    if v is None:
        return None
    if isinstance(v, float) and (v != v):
        return None
    return v


def load_extra_metrics(result_dir: Path | None) -> dict:
    """Load quick_error_window_metrics.json + weight behavior from quick_gnn_metrics.json."""
    out = {}
    if result_dir is not None:
        err_path = result_dir / "metrics" / "quick_error_window_metrics.json"
        if err_path.exists():
            err = load_json(err_path)
            for k in ["overall_rmse_pos", "fault_window_rmse_pos",
                      "fault_window_p95_error_pos", "fault_window_max_error_pos",
                      "pre_fault_rmse_pos", "post_fault_rmse_pos",
                      "recovery_time_error"]:
                out[k] = safe_get(err, k)

        gnn_path = result_dir / "metrics" / "quick_gnn_metrics.json"
        if gnn_path.exists():
            gnn = load_json(gnn_path)
            for k in ["fault_top1_weight_rate", "fault_weight_below_equal_rate",
                      "fault_weight_below_010_rate"]:
                out[k] = safe_get(gnn, k)
    return out


# ============================================================
#  Find latest result dir by preset name
# ============================================================

def find_latest_result(results_root: Path, name_substring: str) -> Optional[Path]:
    if not results_root.exists():
        return None
    candidates = []
    for sub in sorted(results_root.iterdir()):
        if not sub.is_dir():
            continue
        if name_substring in sub.name:
            meta_path = sub / "meta.json"
            if meta_path.exists():
                candidates.append(sub)
    if not candidates:
        return None
    return candidates[-1]


# ============================================================
#  Collect
# ============================================================

def collect_ablation(
    post_only_dir: Optional[Path],
    dual_stream_dir: Optional[Path],
    rgcf_dir: Optional[Path],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # --- Rule-based baselines (from any result dir, prefer rgcf) ---
    baseline_source = rgcf_dir or post_only_dir or dual_stream_dir
    if baseline_source is not None:
        baseline_path = baseline_source / "metrics" / "quick_baseline_metrics.json"
        if baseline_path.exists():
            bl = load_json(baseline_path)

            # AVG
            rows.append({
                "variant": "AVG",
                "category": "rule",
                "result_dir": str(baseline_source) if baseline_source else "",
                "test_loss": None, "test_loss_pos": None, "test_loss_vel": None,
                "quick_rmse_pos": safe_get(bl, "rmse_avg"),
                "quick_rmse_full": None,
                "mean_weight_fault": safe_get(bl, "mean_weight_fault_avg"),
                "mean_weight_normal": safe_get(bl, "mean_weight_normal_avg"),
                "weight_sep": safe_get(bl, "weight_separation_normal_minus_fault_avg"),
                "mean_gate_fault": None, "mean_gate_normal": None, "gate_sep": None,
                "mean_cov_scale_fault": None, "mean_cov_scale_normal": None, "cov_ratio": None,
            })

            # CI-multi
            rows.append({
                "variant": "CI-multi",
                "category": "rule",
                "result_dir": str(baseline_source) if baseline_source else "",
                "test_loss": None, "test_loss_pos": None, "test_loss_vel": None,
                "quick_rmse_pos": safe_get(bl, "rmse_ci_multi"),
                "quick_rmse_full": None,
                "mean_weight_fault": safe_get(bl, "mean_weight_fault_ci_multi"),
                "mean_weight_normal": safe_get(bl, "mean_weight_normal_ci_multi"),
                "weight_sep": safe_get(bl, "weight_separation_normal_minus_fault_ci_multi"),
                "mean_gate_fault": None, "mean_gate_normal": None, "gate_sep": None,
                "mean_cov_scale_fault": None, "mean_cov_scale_normal": None, "cov_ratio": None,
            })

            # WAA-MM
            rows.append({
                "variant": "WAA-MM",
                "category": "rule",
                "result_dir": str(baseline_source) if baseline_source else "",
                "test_loss": None, "test_loss_pos": None, "test_loss_vel": None,
                "quick_rmse_pos": safe_get(bl, "rmse_waa_mm"),
                "quick_rmse_full": None,
                "mean_weight_fault": safe_get(bl, "mean_weight_fault_waa_mm"),
                "mean_weight_normal": safe_get(bl, "mean_weight_normal_waa_mm"),
                "weight_sep": safe_get(bl, "weight_separation_normal_minus_fault_waa_mm"),
                "mean_gate_fault": None, "mean_gate_normal": None, "gate_sep": None,
                "mean_cov_scale_fault": None, "mean_cov_scale_normal": None, "cov_ratio": None,
            })

    # --- Learned models ---
    learned_entries = [
        ("Posterior-only GNN", "learned", post_only_dir),
        ("Dual-stream Plain GNN", "learned", dual_stream_dir),
        ("Full RGCF", "learned+reliability", rgcf_dir),
    ]

    for variant, category, rdir in learned_entries:
        extra = load_extra_metrics(rdir)
        if rdir is None:
            rows.append({
                "variant": variant, "category": category, "result_dir": "",
                "test_loss": None, "test_loss_pos": None, "test_loss_vel": None,
                "quick_rmse_pos": None, "quick_rmse_full": None,
                "mean_weight_fault": None, "mean_weight_normal": None, "weight_sep": None,
                "mean_gate_fault": None, "mean_gate_normal": None, "gate_sep": None,
                "mean_cov_scale_fault": None, "mean_cov_scale_normal": None, "cov_ratio": None,
                **extra,
            })
            continue

        # Read test metrics from summary
        summary_path = rdir / "summary" / "training_summary.json"
        train_info = {}
        if summary_path.exists():
            train_info = load_json(summary_path)

        # Read quick GNN metrics
        gnn_path = rdir / "metrics" / "quick_gnn_metrics.json"
        gnn = {}
        if gnn_path.exists():
            gnn = load_json(gnn_path)

        rows.append({
            "variant": variant,
            "category": category,
            "result_dir": str(rdir),
            "test_loss": safe_get(train_info, "test_loss"),
            "test_loss_pos": safe_get(train_info, "test_loss_pos"),
            "test_loss_vel": safe_get(train_info, "test_loss_vel"),
            "quick_rmse_pos": safe_get(gnn, "rmse_gnn_pos"),
            "quick_rmse_full": safe_get(gnn, "rmse_gnn_full"),
            "mean_weight_fault": safe_get(gnn, "mean_weight_fault"),
            "mean_weight_normal": safe_get(gnn, "mean_weight_normal"),
            "weight_sep": safe_get(gnn, "weight_separation_normal_minus_fault"),
            "mean_gate_fault": safe_get(gnn, "mean_gate_fault"),
            "mean_gate_normal": safe_get(gnn, "mean_gate_normal"),
            "gate_sep": safe_get(gnn, "gate_separation_normal_minus_fault"),
            "mean_cov_scale_fault": safe_get(gnn, "mean_cov_scale_fault"),
            "mean_cov_scale_normal": safe_get(gnn, "mean_cov_scale_normal"),
            "cov_ratio": safe_get(gnn, "cov_scale_fault_to_normal_ratio"),
            **extra,
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = out_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[csv] {csv_path}")

    # Save JSON
    json_path = out_dir / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [dict(r) for _, r in df.iterrows()],
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"[json] {json_path}")

    # Print table
    print("\n" + "=" * 80)
    print("Ablation Summary")
    print("=" * 80)
    cols_show = ["variant", "category", "quick_rmse_pos", "mean_weight_fault",
                 "mean_weight_normal", "weight_sep",
                 "mean_gate_fault", "mean_gate_normal",
                 "mean_cov_scale_fault", "mean_cov_scale_normal", "cov_ratio"]
    print(df[cols_show].to_string(index=False))
    print("=" * 80)


# ============================================================
#  CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect RGCF main ablation summary across AVG/CI-multi/WAA-MM/A1/A3/A4."
    )
    p.add_argument("--post-only-result", type=str, default=None,
                   help="Result dir for A1 (Posterior-only GNN)")
    p.add_argument("--dual-stream-result", type=str, default=None,
                   help="Result dir for A3 (Dual-stream Plain GNN)")
    p.add_argument("--rgcf-result", type=str, default=None,
                   help="Result dir for A4 (Full RGCF)")
    p.add_argument("--results-root", type=str, default="results",
                   help="Results root for auto-latest mode")
    p.add_argument("--auto-latest", action="store_true",
                   help="Auto-find latest result dirs by preset name")
    p.add_argument("--out-dir", type=str,
                   default="results/rgcf_main_ablation_summary",
                   help="Output directory for summary files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    post_only_dir: Optional[Path] = None
    dual_stream_dir: Optional[Path] = None
    rgcf_dir: Optional[Path] = None

    if args.auto_latest:
        root = Path(args.results_root)
        post_only_dir = find_latest_result(root, "hetero_robust_matrix_mixed_post_only_gnn")
        dual_stream_dir = find_latest_result(root, "hetero_robust_matrix_mixed_dual_stream_plain_gnn")
        rgcf_dir = (
            find_latest_result(root, "hetero_robust_matrix_mixed_rgcf_v5")
            or find_latest_result(root, "RGCF-V5-Reliability-Gated Covariance-Calibrated Fusion")
            or find_latest_result(root, "hetero_robust_matrix_mixed_v2_rgcf")
        )
        print(f"[auto-latest] A1: {post_only_dir}")
        print(f"[auto-latest] A3: {dual_stream_dir}")
        print(f"[auto-latest] A4: {rgcf_dir}")
    else:
        if args.post_only_result:
            post_only_dir = Path(args.post_only_result)
        if args.dual_stream_result:
            dual_stream_dir = Path(args.dual_stream_result)
        if args.rgcf_result:
            rgcf_dir = Path(args.rgcf_result)

    out_dir = Path(args.out_dir)
    collect_ablation(post_only_dir, dual_stream_dir, rgcf_dir, out_dir)


if __name__ == "__main__":
    main()
