# -*- coding: utf-8 -*-
"""
plot_v2_analysis.py

用途：
    针对 NF-DKF V2 单次训练结果目录，自动生成 6 张正式分析图：
    1) 主性能对比柱状图
    2) 训练/验证损失曲线
    3) 平均传感器融合权重柱状图
    4) 平均 Gate 激活柱状图
    5) Gate 热力图（时间 × 传感器）
    6) Gate–Weight 联动时序图

目录假设（与你当前结果目录设计一致）：
    result_dir/
    ├─ artifacts/
    │  ├─ gate_timeseries.csv
    │  └─ ...
    ├─ history/
    │  └─ train_history.csv
    ├─ metrics/
    │  ├─ quick_baseline_metrics.json
    │  ├─ baseline_metrics.json
    │  ├─ quick_gnn_metrics.json
    │  └─ ...
    ├─ summary/
    │  ├─ training_summary.json
    │  └─ sim_summary.json
    └─ ...

运行示例：
    python scripts/plot_v2_analysis.py --result_dir "results\\20260320_135750__train__clean_v2_post_meas_softgate__Dataset_V_1&2_seedCount60"

可选：
    python scripts/plot_v2_analysis.py --result_dir "..." --out_dir "...\artifacts\plots_custom"

说明：
    - 该脚本尽量做了“字段名容错”，适配你当前主线下可能略有差异的 csv/json 命名。
    - 若某个图所需字段缺失，会给出警告并跳过该图，不会导致整个脚本崩掉。
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
# 通用工具函数
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
        if not isinstance(d, dict):
            continue
        out.update(d)
    return out


def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


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
    col_list = list(columns)
    for c in col_list:
        lc = c.lower()
        if all(k.lower() in lc for k in keywords):
            return c
    return None


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", rotation: int = 0, fontsize: int = 9) -> None:
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
            rotation=rotation,
            fontsize=fontsize,
        )


def read_metrics_bundle(result_dir: Path) -> Dict[str, Any]:
    """
    尽可能把多个 json 的关键字段合到一个 dict 里。
    优先级上，后读到的字段会覆盖前面的同名字段。
    """
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


def get_gate_df(result_dir: Path) -> Optional[pd.DataFrame]:
    return maybe_load_csv(result_dir / "artifacts" / "gate_timeseries.csv")


def get_time_col(df: pd.DataFrame) -> str:
    """
    优先使用 t，其次 k，再否则用 index。
    """
    t_col = find_col(df.columns, ["t", "time", "timestamp"])
    if t_col is not None:
        return t_col
    k_col = find_col(df.columns, ["k", "step", "idx", "index"])
    if k_col is not None:
        return k_col
    # 若完全没有，就补一个
    df["_auto_step"] = np.arange(len(df))
    return "_auto_step"


def extract_sensor_cols(df: pd.DataFrame, prefix: str, count: int = 4) -> List[str]:
    """
    例如 prefix='g_s' -> ['g_s1', 'g_s2', 'g_s3', 'g_s4']
    """
    cols = []
    for i in range(1, count + 1):
        c = find_col(df.columns, [f"{prefix}{i}"])
        if c is None:
            # 再模糊匹配一次
            c = fuzzy_find_col(df.columns, [prefix.rstrip("_"), f"s{i}"])
        if c is not None:
            cols.append(c)
    return cols


def get_metric_value(metrics: Dict[str, Any], *candidates: str) -> Optional[float]:
    for c in candidates:
        if c in metrics:
            v = to_float(metrics[c])
            if v is not None:
                return v
    return None


def compute_or_fetch_mean_weights(metrics: Dict[str, Any], gate_df: Optional[pd.DataFrame]) -> Tuple[List[str], List[float]]:
    labels = ["S1", "S2", "S3", "S4"]
    values: List[float] = []

    metric_keys = [f"mean_w_s{i}" for i in range(1, 5)]
    all_in_metrics = all(k in metrics for k in metric_keys)

    if all_in_metrics:
        for k in metric_keys:
            v = to_float(metrics.get(k))
            if v is None:
                raise ValueError(f"{k} 存在但无法转换为 float。")
            values.append(v)
        return labels, values

    if gate_df is not None:
        weight_cols = extract_sensor_cols(gate_df, "w_s", count=4)
        if len(weight_cols) == 4:
            for c in weight_cols:
                values.append(float(pd.to_numeric(gate_df[c], errors="coerce").mean()))
            return labels, values

    raise ValueError("无法从 metrics 或 gate_timeseries.csv 中找到 mean_w_s1~4 / w_s1~4。")


def compute_or_fetch_mean_gates(metrics: Dict[str, Any], gate_df: Optional[pd.DataFrame]) -> Tuple[List[str], List[float], Optional[float], Optional[float]]:
    labels = ["S1", "S2", "S3", "S4"]
    values: List[float] = []

    metric_keys = [f"mean_g_s{i}" for i in range(1, 5)]
    all_in_metrics = all(k in metrics for k in metric_keys)

    mean_gate = get_metric_value(metrics, "mean_gate")
    std_gate = get_metric_value(metrics, "std_gate")

    if all_in_metrics:
        for k in metric_keys:
            v = to_float(metrics.get(k))
            if v is None:
                raise ValueError(f"{k} 存在但无法转换为 float。")
            values.append(v)
        return labels, values, mean_gate, std_gate

    if gate_df is not None:
        gate_cols = extract_sensor_cols(gate_df, "g_s", count=4)
        if len(gate_cols) == 4:
            for c in gate_cols:
                values.append(float(pd.to_numeric(gate_df[c], errors="coerce").mean()))
            arr = gate_df[gate_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            mean_gate = float(np.nanmean(arr))
            std_gate = float(np.nanstd(arr))
            return labels, values, mean_gate, std_gate

    raise ValueError("无法从 metrics 或 gate_timeseries.csv 中找到 mean_g_s1~4 / g_s1~4。")


def get_best_epoch(metrics: Dict[str, Any]) -> Optional[float]:
    return get_metric_value(
        metrics,
        "best_epoch",
        "best_val_epoch",
        "epoch_best",
        "best_epoch_idx",
    )


def safe_savefig(fig: plt.Figure, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# 图 1：主性能对比柱状图
# =========================
def plot_01_performance_bar(result_dir: Path, out_dir: Path, metrics: Dict[str, Any]) -> Optional[Path]:
    """
    主性能对比图：
    - rmse_avg
    - rmse_ci_chain
    - rmse_ci_multi
    - rmse_best_single
    - rmse_gnn_pos
    - 可选 rmse_gnn_full
    """
    label_key_pairs = [
        ("AVG", "rmse_avg"),
        ("CI-chain", "rmse_ci_chain"),
        ("CI-multi", "rmse_ci_multi"),
        ("Best-single", "rmse_best_single"),
        ("V2-pos", "rmse_gnn_pos"),
    ]

    # 若 full 有值，也加上
    if get_metric_value(metrics, "rmse_gnn_full") is not None:
        label_key_pairs.append(("V2-full", "rmse_gnn_full"))

    labels: List[str] = []
    values: List[float] = []

    for label, key in label_key_pairs:
        v = get_metric_value(metrics, key)
        if v is not None:
            labels.append(label)
            values.append(v)

    if len(labels) < 2:
        warnings.warn("图1跳过：性能字段不足，无法绘制主性能对比柱状图。")
        return None

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)

    ax.set_title("V2 与经典融合基线的主性能对比（Clean 4-Sensor）", fontsize=13)
    ax.set_xlabel("Method")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.2)

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

    out_path = out_dir / "fig01_performance_bar.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图 2：训练/验证损失曲线
# =========================
def plot_02_training_curves(result_dir: Path, out_dir: Path, history_df: Optional[pd.DataFrame], metrics: Dict[str, Any]) -> Optional[Path]:
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

    train_col = find_col(df.columns, ["train_loss", "train"])
    if train_col is None:
        train_col = fuzzy_find_col(df.columns, ["train", "loss"])

    val_col = find_col(df.columns, ["val_loss", "val"])
    if val_col is None:
        val_col = fuzzy_find_col(df.columns, ["val", "loss"])

    if train_col is None and val_col is None:
        warnings.warn("图2跳过：train_history.csv 中未找到 train/val loss 列。")
        return None

    x = pd.to_numeric(df[epoch_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    if train_col is not None:
        y_train = pd.to_numeric(df[train_col], errors="coerce")
        ax.plot(x, y_train, label="Train loss", linewidth=1.8)

    if val_col is not None:
        y_val = pd.to_numeric(df[val_col], errors="coerce")
        ax.plot(x, y_val, label="Val loss", linewidth=1.8, linestyle="--")

    best_epoch = get_best_epoch(metrics)
    if best_epoch is not None:
        ax.axvline(best_epoch, linestyle=":", linewidth=1.5, label=f"Best epoch = {best_epoch:g}")

    ax.set_title("V2 训练与验证损失收敛曲线", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.25)

    out_path = out_dir / "fig02_training_curves.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图 3：平均传感器融合权重柱状图
# =========================
def plot_03_mean_sensor_weights(result_dir: Path, out_dir: Path, metrics: Dict[str, Any], gate_df: Optional[pd.DataFrame]) -> Optional[Path]:
    try:
        labels, values = compute_or_fetch_mean_weights(metrics, gate_df)
    except Exception as e:
        warnings.warn(f"图3跳过：{e}")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)

    ax.set_title("各传感器平均最终融合权重", fontsize=13)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Average fusion weight")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)

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

    out_path = out_dir / "fig03_mean_sensor_weights.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图 4：平均 Gate 激活柱状图
# =========================
def plot_04_mean_gate_activation(result_dir: Path, out_dir: Path, metrics: Dict[str, Any], gate_df: Optional[pd.DataFrame]) -> Optional[Path]:
    try:
        labels, values, mean_gate, std_gate = compute_or_fetch_mean_gates(metrics, gate_df)
    except Exception as e:
        warnings.warn(f"图4跳过：{e}")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)

    ax.set_title("各传感器平均 Gate 激活水平", fontsize=13)
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Average gate value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)

    annotate_bars(ax, bars, fmt="{:.3f}")

    stat_lines = []
    if mean_gate is not None:
        stat_lines.append(f"global mean_gate = {mean_gate:.4f}")
    if std_gate is not None:
        stat_lines.append(f"global std_gate = {std_gate:.4f}")
    if stat_lines:
        ax.text(
            0.99,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    out_path = out_dir / "fig04_mean_gate_activation.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图 5：Gate 热力图
# =========================
def plot_05_gate_heatmap(result_dir: Path, out_dir: Path, gate_df: Optional[pd.DataFrame]) -> Optional[Path]:
    if gate_df is None or gate_df.empty:
        warnings.warn("图5跳过：未找到 artifacts/gate_timeseries.csv 或文件为空。")
        return None

    df = gate_df.copy()
    time_col = get_time_col(df)
    gate_cols = extract_sensor_cols(df, "g_s", count=4)

    if len(gate_cols) != 4:
        warnings.warn("图5跳过：gate_timeseries.csv 中未找到完整的 g_s1~g_s4 列。")
        return None

    x = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    mat = df[gate_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T  # [4, T]

    # 若 x 非严格有效，则退化为索引
    if np.isnan(x).all():
        x = np.arange(mat.shape[1], dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 3.8))

    # 使用 extent 让横轴是时间/步长
    extent = [x[0], x[-1], 0.5, 4.5] if len(x) >= 2 else [0, mat.shape[1], 0.5, 4.5]
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("Gate 随时间变化的热力图", fontsize=13)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sensor")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["S1", "S2", "S3", "S4"])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Gate value")

    out_path = out_dir / "fig05_gate_heatmap.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图 6：Gate–Weight 联动时序图
# =========================
def plot_06_gate_weight_joint_timeseries(result_dir: Path, out_dir: Path, gate_df: Optional[pd.DataFrame]) -> Optional[Path]:
    if gate_df is None or gate_df.empty:
        warnings.warn("图6跳过：未找到 artifacts/gate_timeseries.csv 或文件为空。")
        return None

    df = gate_df.copy()
    time_col = get_time_col(df)
    gate_cols = extract_sensor_cols(df, "g_s", count=4)
    weight_cols = extract_sensor_cols(df, "w_s", count=4)

    if len(gate_cols) != 4 or len(weight_cols) != 4:
        warnings.warn("图6跳过：gate_timeseries.csv 中未找到完整的 g_s1~g_s4 / w_s1~w_s4 列。")
        return None

    x = pd.to_numeric(df[time_col], errors="coerce")
    if x.isna().all():
        x = pd.Series(np.arange(len(df)))

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 9.5), sharex=True)

    for i in range(4):
        ax = axes[i]
        g = pd.to_numeric(df[gate_cols[i]], errors="coerce")
        w = pd.to_numeric(df[weight_cols[i]], errors="coerce")

        ax.plot(x, g, label="Gate", linewidth=1.3)
        ax.plot(x, w, label="Final weight", linewidth=1.3, linestyle="--")

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Value")
        ax.set_title(f"Sensor {i+1}", fontsize=10)
        ax.grid(True, alpha=0.22)

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time step")
    fig.suptitle("各传感器 Gate 与最终融合权重的联动时序", fontsize=13)

    out_path = out_dir / "fig06_gate_weight_joint_timeseries.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 主流程
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 6 formal analysis figures for NF-DKF V2 result directory.")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="V2 单次训练结果目录，例如 results/20260320_135750__train__clean_v2_post_meas_softgate__Dataset_V_1&2_seedCount60",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录。默认写到 <result_dir>/artifacts/plots_custom",
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir 不存在：{result_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (result_dir / "artifacts" / "plots_custom")
    ensure_dir(out_dir)

    metrics = read_metrics_bundle(result_dir)
    history_df = get_history_df(result_dir)
    gate_df = get_gate_df(result_dir)

    saved_paths: List[Path] = []

    p = plot_01_performance_bar(result_dir, out_dir, metrics)
    if p is not None:
        saved_paths.append(p)

    p = plot_02_training_curves(result_dir, out_dir, history_df, metrics)
    if p is not None:
        saved_paths.append(p)

    p = plot_03_mean_sensor_weights(result_dir, out_dir, metrics, gate_df)
    if p is not None:
        saved_paths.append(p)

    p = plot_04_mean_gate_activation(result_dir, out_dir, metrics, gate_df)
    if p is not None:
        saved_paths.append(p)

    p = plot_05_gate_heatmap(result_dir, out_dir, gate_df)
    if p is not None:
        saved_paths.append(p)

    p = plot_06_gate_weight_joint_timeseries(result_dir, out_dir, gate_df)
    if p is not None:
        saved_paths.append(p)

    print("=" * 80)
    print("绘图完成。")
    print(f"结果目录: {result_dir}")
    print(f"输出目录: {out_dir}")
    print("-" * 80)
    if saved_paths:
        for sp in saved_paths:
            print(f"[SAVED] {sp}")
    else:
        print("没有成功生成任何图，请检查结果目录中的 csv/json 字段命名。")
    print("=" * 80)


if __name__ == "__main__":
    main()