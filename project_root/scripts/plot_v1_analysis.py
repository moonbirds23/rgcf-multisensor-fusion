# -*- coding: utf-8 -*-
"""
plot_v1_analysis.py

用途：
    针对 NF-DKF V1 单次训练结果目录，自动生成 6 张正式分析图：
    1) fig01_v1_performance_bar.png
    2) fig02_v1_training_curves.png
    3) fig03_v1_mean_sensor_weights.png
    4) fig04_v1_weight_heatmap.png
    5) fig05_v1_weight_timeseries.png
    6) fig06_v1_weight_boxplot.png

可选：
    若传入 --compare_dir_v0，则额外生成
    7) fig07_v0_v1_mean_weight_compare.png

运行示例：
    python scripts/plot_v1_analysis.py --result_dir ".\\results\\20260320_150518__train__clean_v1_post_meas_direct__default_clean_4sensor_scene__post_meas_direct_fusion"

输出目录：
    默认输出到：
    <result_dir>\\artifacts\\plots_custom_v1

说明：
    - V1 是 direct fusion baseline，核心看性能、收敛、最终权重及其动态。
    - 虽然 artifacts 里可能有 gate_timeseries.csv，但 V1 正式分析主要使用其中的 w_s1~w_s4。
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Matplotlib 基础设置
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220


# =========================
# 基础工具
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    return {"_root": obj}


def maybe_load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return load_json(path)
    return {}


def maybe_load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        if isinstance(d, dict):
            out.update(d)
    return out


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def find_col(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    col_list = list(columns)
    lower_map = {c.lower(): c for c in col_list}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def fuzzy_find_col(columns: Iterable[str], keywords: Sequence[str]) -> Optional[str]:
    for c in columns:
        lc = c.lower()
        if all(k.lower() in lc for k in keywords):
            return c
    return None


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", fontsize: int = 9) -> None:
    for b in bars:
        h = b.get_height()
        if h is None or np.isnan(h):
            continue
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def safe_savefig(fig: plt.Figure, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# 数据读取
# =========================
def read_metrics_bundle(result_dir: Path) -> Dict[str, Any]:
    metrics_dir = result_dir / "metrics"
    summary_dir = result_dir / "summary"

    quick_baseline = maybe_load_json(metrics_dir / "quick_baseline_metrics.json")
    baseline = maybe_load_json(metrics_dir / "baseline_metrics.json")
    quick_gnn = maybe_load_json(metrics_dir / "quick_gnn_metrics.json")
    training_summary = maybe_load_json(summary_dir / "training_summary.json")
    sim_summary = maybe_load_json(summary_dir / "sim_summary.json")

    merged = merge_dicts(
        sim_summary,
        baseline,
        quick_baseline,
        training_summary,
        quick_gnn,
    )
    return merged


def get_history_df(result_dir: Path) -> Optional[pd.DataFrame]:
    return maybe_load_csv(result_dir / "history" / "train_history.csv")


def get_timeseries_df(result_dir: Path) -> Optional[pd.DataFrame]:
    # 统一接口名字虽然叫 gate_timeseries.csv，但 V1 主要读取其中的 w_s1~w_s4
    return maybe_load_csv(result_dir / "artifacts" / "gate_timeseries.csv")


def get_metric_value(metrics: Dict[str, Any], *candidates: str) -> Optional[float]:
    for c in candidates:
        if c in metrics:
            v = to_float(metrics[c])
            if v is not None:
                return v
    return None


def get_time_col(df: pd.DataFrame) -> str:
    t_col = find_col(df.columns, ["t", "time", "timestamp"])
    if t_col is not None:
        return t_col
    k_col = find_col(df.columns, ["k", "step", "idx", "index"])
    if k_col is not None:
        return k_col
    df["_auto_step"] = np.arange(len(df))
    return "_auto_step"


def extract_sensor_cols(df: pd.DataFrame, prefix: str, count: int = 4) -> List[str]:
    cols: List[str] = []
    for i in range(1, count + 1):
        exact = find_col(df.columns, [f"{prefix}{i}"])
        if exact is not None:
            cols.append(exact)
            continue
        fuzzy = fuzzy_find_col(df.columns, [prefix.rstrip("_"), f"s{i}"])
        if fuzzy is not None:
            cols.append(fuzzy)
    return cols


def compute_mean_weights_from_metrics_or_csv(
    metrics: Dict[str, Any],
    ts_df: Optional[pd.DataFrame],
) -> Tuple[List[str], List[float]]:
    labels = ["S1", "S2", "S3", "S4"]

    metric_keys = [f"mean_w_s{i}" for i in range(1, 5)]
    if all(k in metrics for k in metric_keys):
        vals = []
        for k in metric_keys:
            v = to_float(metrics.get(k))
            if v is None:
                raise ValueError(f"{k} 无法转换为 float")
            vals.append(v)
        return labels, vals

    if ts_df is not None:
        w_cols = extract_sensor_cols(ts_df, "w_s", count=4)
        if len(w_cols) == 4:
            vals = [float(pd.to_numeric(ts_df[c], errors="coerce").mean()) for c in w_cols]
            return labels, vals

    raise ValueError("未找到 mean_w_s1~4 或 w_s1~4")


def get_best_epoch(metrics: Dict[str, Any]) -> Optional[float]:
    return get_metric_value(
        metrics,
        "best_epoch",
        "best_val_epoch",
        "epoch_best",
        "best_epoch_idx",
    )


# =========================
# 图1：主性能对比柱状图
# =========================
def plot_01_performance_bar(
    out_dir: Path,
    metrics_v1: Dict[str, Any],
    metrics_v0: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    labels: List[str] = []
    values: List[float] = []

    baseline_pairs = [
        ("AVG", "rmse_avg"),
        ("CI-chain", "rmse_ci_chain"),
        ("CI-multi", "rmse_ci_multi"),
        ("Best-single", "rmse_best_single"),
    ]
    for label, key in baseline_pairs:
        v = get_metric_value(metrics_v1, key)
        if v is not None:
            labels.append(label)
            values.append(v)

    if metrics_v0 is not None:
        v0_pos = get_metric_value(metrics_v0, "rmse_gnn_pos")
        if v0_pos is not None:
            labels.append("V0-pos")
            values.append(v0_pos)

    v1_pos = get_metric_value(metrics_v1, "rmse_gnn_pos")
    if v1_pos is not None:
        labels.append("V1-pos")
        values.append(v1_pos)

    v1_full = get_metric_value(metrics_v1, "rmse_gnn_full")
    if v1_full is not None:
        labels.append("V1-full")
        values.append(v1_full)

    if len(labels) < 2:
        warnings.warn("图1跳过：性能字段不足。")
        return None

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)

    ax.set_title("V1 与经典基线的主性能对比（Clean 4-Sensor）", fontsize=13)
    ax.set_xlabel("Method")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.22)

    annotate_bars(ax, bars, fmt="{:.3f}")
    ax.text(
        0.99,
        0.97,
        "lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    out_path = out_dir / "fig01_v1_performance_bar.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图2：训练/验证损失曲线
# =========================
def plot_02_training_curves(
    out_dir: Path,
    history_df: Optional[pd.DataFrame],
    metrics: Dict[str, Any],
) -> Optional[Path]:
    if history_df is None or history_df.empty:
        warnings.warn("图2跳过：未找到 history/train_history.csv 或文件为空。")
        return None

    df = history_df.copy()

    epoch_col = find_col(df.columns, ["epoch", "ep"])
    if epoch_col is None:
        epoch_col = fuzzy_find_col(df.columns, ["epoch"])
    if epoch_col is None:
        df["_epoch_auto"] = np.arange(1, len(df) + 1)
        epoch_col = "_epoch_auto"

    train_col = find_col(df.columns, ["train_loss"])
    if train_col is None:
        train_col = fuzzy_find_col(df.columns, ["train", "loss"])

    val_col = find_col(df.columns, ["val_loss"])
    if val_col is None:
        val_col = fuzzy_find_col(df.columns, ["val", "loss"])

    if train_col is None and val_col is None:
        warnings.warn("图2跳过：未找到 train/val loss 列。")
        return None

    x = pd.to_numeric(df[epoch_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10.2, 5.6))

    if train_col is not None:
        y_train = pd.to_numeric(df[train_col], errors="coerce")
        ax.plot(x, y_train, label="Train loss", linewidth=1.8)

    if val_col is not None:
        y_val = pd.to_numeric(df[val_col], errors="coerce")
        ax.plot(x, y_val, label="Val loss", linewidth=1.8, linestyle="--")

    best_epoch = get_best_epoch(metrics)
    if best_epoch is not None:
        ax.axvline(best_epoch, linestyle=":", linewidth=1.5, label=f"Best epoch = {best_epoch:g}")

    ax.set_title("V1 训练与验证损失收敛曲线", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_path = out_dir / "fig02_v1_training_curves.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图3：平均最终融合权重柱状图
# =========================
def plot_03_mean_sensor_weights(
    out_dir: Path,
    metrics: Dict[str, Any],
    ts_df: Optional[pd.DataFrame],
) -> Optional[Path]:
    try:
        labels, values = compute_mean_weights_from_metrics_or_csv(metrics, ts_df)
    except Exception as e:
        warnings.warn(f"图3跳过：{e}")
        return None

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)

    ax.set_title("V1 各传感器平均最终融合权重", fontsize=13)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Average fusion weight")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.22)

    annotate_bars(ax, bars, fmt="{:.3f}")
    ax.text(
        0.99,
        0.97,
        "weights are normalized across sensors",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    out_path = out_dir / "fig03_v1_mean_sensor_weights.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图4：权重热力图
# =========================
def plot_04_weight_heatmap(
    out_dir: Path,
    ts_df: Optional[pd.DataFrame],
) -> Optional[Path]:
    if ts_df is None or ts_df.empty:
        warnings.warn("图4跳过：未找到 artifacts/gate_timeseries.csv 或文件为空。")
        return None

    df = ts_df.copy()
    time_col = get_time_col(df)
    w_cols = extract_sensor_cols(df, "w_s", count=4)

    if len(w_cols) != 4:
        warnings.warn("图4跳过：未找到完整的 w_s1~w_s4。")
        return None

    x = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    mat = df[w_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T

    if np.isnan(x).all():
        x = np.arange(mat.shape[1], dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    extent = [x[0], x[-1], 0.5, 4.5] if len(x) >= 2 else [0, mat.shape[1], 0.5, 4.5]
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("V1 最终融合权重热力图", fontsize=13)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sensor")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["S1", "S2", "S3", "S4"])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fusion weight")

    out_path = out_dir / "fig04_v1_weight_heatmap.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图5：权重时序曲线
# =========================
def plot_05_weight_timeseries(
    out_dir: Path,
    ts_df: Optional[pd.DataFrame],
) -> Optional[Path]:
    if ts_df is None or ts_df.empty:
        warnings.warn("图5跳过：未找到 artifacts/gate_timeseries.csv 或文件为空。")
        return None

    df = ts_df.copy()
    time_col = get_time_col(df)
    w_cols = extract_sensor_cols(df, "w_s", count=4)

    if len(w_cols) != 4:
        warnings.warn("图5跳过：未找到完整的 w_s1~w_s4。")
        return None

    x = pd.to_numeric(df[time_col], errors="coerce")
    if x.isna().all():
        x = pd.Series(np.arange(len(df)))

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    for i, c in enumerate(w_cols, start=1):
        y = pd.to_numeric(df[c], errors="coerce")
        ax.plot(x, y, label=f"S{i}", linewidth=1.4)

    ax.set_title("V1 各传感器最终融合权重时序曲线", fontsize=13)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Fusion weight")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.22)
    ax.legend(ncol=4)

    out_path = out_dir / "fig05_v1_weight_timeseries.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图6：权重分布箱线图
# =========================
def plot_06_weight_boxplot(
    out_dir: Path,
    ts_df: Optional[pd.DataFrame],
) -> Optional[Path]:
    if ts_df is None or ts_df.empty:
        warnings.warn("图6跳过：未找到 artifacts/gate_timeseries.csv 或文件为空。")
        return None

    df = ts_df.copy()
    w_cols = extract_sensor_cols(df, "w_s", count=4)

    if len(w_cols) != 4:
        warnings.warn("图6跳过：未找到完整的 w_s1~w_s4。")
        return None

    data = []
    labels = []
    for i, c in enumerate(w_cols, start=1):
        arr = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        data.append(arr)
        labels.append(f"S{i}")

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.boxplot(data, labels=labels, showfliers=False)

    ax.set_title("V1 各传感器最终融合权重分布", fontsize=13)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Fusion weight")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.22)

    out_path = out_dir / "fig06_v1_weight_boxplot.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图7：V0 vs V1 平均权重对比（可选）
# =========================
def plot_07_v0_v1_mean_weight_compare(
    out_dir: Path,
    metrics_v0: Dict[str, Any],
    ts_v0: Optional[pd.DataFrame],
    metrics_v1: Dict[str, Any],
    ts_v1: Optional[pd.DataFrame],
) -> Optional[Path]:
    try:
        labels0, vals0 = compute_mean_weights_from_metrics_or_csv(metrics_v0, ts_v0)
        labels1, vals1 = compute_mean_weights_from_metrics_or_csv(metrics_v1, ts_v1)
    except Exception as e:
        warnings.warn(f"图7跳过：{e}")
        return None

    if labels0 != labels1:
        warnings.warn("图7跳过：V0/V1 传感器标签不一致。")
        return None

    x = np.arange(len(labels0))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars0 = ax.bar(x - width / 2, vals0, width, label="V0")
    bars1 = ax.bar(x + width / 2, vals1, width, label="V1")

    ax.set_title("V0 与 V1 的平均节点权重分配差异", fontsize=13)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Average fusion weight")
    ax.set_xticks(x)
    ax.set_xticklabels(labels0)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend()

    annotate_bars(ax, bars0, fmt="{:.3f}", fontsize=8)
    annotate_bars(ax, bars1, fmt="{:.3f}", fontsize=8)

    out_path = out_dir / "fig07_v0_v1_mean_weight_compare.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 主流程
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot formal V1 analysis figures for NF-DKF result directory.")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="V1 结果目录，例如 .\\results\\20260320_150518__train__clean_v1_post_meas_direct__default_clean_4sensor_scene__post_meas_direct_fusion",
    )
    parser.add_argument(
        "--compare_dir_v0",
        type=str,
        default=None,
        help="可选：V0 结果目录。若提供，则生成 V0 vs V1 平均权重对比图。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录。默认 <result_dir>\\artifacts\\plots_custom_v1",
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir 不存在：{result_dir}")

    compare_dir_v0 = Path(args.compare_dir_v0) if args.compare_dir_v0 else None
    if compare_dir_v0 is not None and not compare_dir_v0.exists():
        raise FileNotFoundError(f"compare_dir_v0 不存在：{compare_dir_v0}")

    out_dir = Path(args.out_dir) if args.out_dir else (result_dir / "artifacts" / "plots_custom_v1")
    ensure_dir(out_dir)

    print("=" * 80)
    print("[INFO] 开始绘制 V1 分析图")
    print(f"[INFO] result_dir      = {result_dir}")
    print(f"[INFO] compare_dir_v0 = {compare_dir_v0}")
    print(f"[INFO] out_dir         = {out_dir}")
    print("=" * 80)

    metrics_v1 = read_metrics_bundle(result_dir)
    history_v1 = get_history_df(result_dir)
    ts_v1 = get_timeseries_df(result_dir)

    metrics_v0 = None
    ts_v0 = None
    if compare_dir_v0 is not None:
        metrics_v0 = read_metrics_bundle(compare_dir_v0)
        ts_v0 = get_timeseries_df(compare_dir_v0)

    saved_paths: List[Path] = []

    for fn_name, fn in [
        ("fig01", lambda: plot_01_performance_bar(out_dir, metrics_v1, metrics_v0)),
        ("fig02", lambda: plot_02_training_curves(out_dir, history_v1, metrics_v1)),
        ("fig03", lambda: plot_03_mean_sensor_weights(out_dir, metrics_v1, ts_v1)),
        ("fig04", lambda: plot_04_weight_heatmap(out_dir, ts_v1)),
        ("fig05", lambda: plot_05_weight_timeseries(out_dir, ts_v1)),
        ("fig06", lambda: plot_06_weight_boxplot(out_dir, ts_v1)),
    ]:
        try:
            p = fn()
            if p is not None:
                saved_paths.append(p)
                print(f"[SAVED] {fn_name}: {p}")
            else:
                print(f"[SKIP ] {fn_name}: required fields missing")
        except Exception as e:
            print(f"[ERROR] {fn_name}: {e}")

    if metrics_v0 is not None:
        try:
            p = plot_07_v0_v1_mean_weight_compare(out_dir, metrics_v0, ts_v0, metrics_v1, ts_v1)
            if p is not None:
                saved_paths.append(p)
                print(f"[SAVED] fig07: {p}")
            else:
                print("[SKIP ] fig07: required fields missing")
        except Exception as e:
            print(f"[ERROR] fig07: {e}")

    print("-" * 80)
    print(f"[DONE ] 共生成 {len(saved_paths)} 张图")
    for p in saved_paths:
        print(f"        {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()