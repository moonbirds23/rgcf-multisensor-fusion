# -*- coding: utf-8 -*-
"""
plot_v0_analysis.py

用途：
    针对 NF-DKF V0 单次训练结果目录，自动生成 6 张正式分析图：
    1) fig01_v0_performance_bar.png
    2) fig02_v0_training_curves.png
    3) fig03_v0_mean_sensor_weights.png
    4) fig04_v0_weight_heatmap.png
    5) fig05_v0_trajectory_compare.png
    6) fig06_v0_perstep_error_curve.png

目录假设：
    result_dir/
    ├─ artifacts/
    │  ├─ gate_timeseries.csv     # 统一接口，V0 主要读取 w_s1~w_s4
    │  ├─ sim_arrays.npz          # 轨迹和逐步误差绘图来源
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
    python scripts/plot_v0_analysis.py --result_dir ".\\results\\20260322_190048__train__clean_baseline_v0__default_clean_4sensor_scene__original_gnn_fusion"

输出目录：
    默认输出到：
    <result_dir>\\artifacts\\plots_custom_v0

说明：
    - V0 是 post-only baseline，正式分析重点是性能、收敛、最终权重与轨迹误差。
    - 虽然 artifacts 里可能有 gate_timeseries.csv，但 V0 正式解读只使用其中的 w_s1~w_s4。
    - 若 sim_arrays.npz 的 key 名与你当前实现略有不同，脚本会尽量自动匹配；匹配不上则跳过对应图。
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
    # 统一接口文件；V0 主要读取其中的最终权重 w_s1~w_s4
    return maybe_load_csv(result_dir / "artifacts" / "gate_timeseries.csv")


def get_sim_npz_path(result_dir: Path) -> Path:
    return result_dir / "artifacts" / "sim_arrays.npz"


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
# sim_arrays.npz 解析
# =========================
def load_npz_dict(npz_path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        out = {k: data[k] for k in data.files}
        return out
    except Exception as e:
        warnings.warn(f"读取 sim_arrays.npz 失败: {e}")
        return None


def rank_npz_key(keys: Sequence[str], include_terms: Sequence[str], exclude_terms: Sequence[str] = ()) -> Optional[str]:
    """
    从 npz keys 中按模糊规则找最可能的 key。
    """
    best_key = None
    best_score = -1
    for k in keys:
        lk = k.lower()
        if any(ex in lk for ex in exclude_terms):
            continue
        score = sum(term in lk for term in include_terms)
        if score > best_score:
            best_score = score
            best_key = k
    if best_score <= 0:
        return None
    return best_key


def find_xy_like_array(npz_dict: Dict[str, np.ndarray], role: str) -> Optional[np.ndarray]:
    """
    role:
        - 'gt'
        - 'fused'
        - 'avg'
        - 'ci_chain'
        - 'ci_multi'
        - 'best_single'
    返回形状尽量为 [T, 2] 或 [T, >=2]
    """
    keys = list(npz_dict.keys())

    candidate_rules = {
        "gt": [
            (["gt"], ["err", "error"]),
            (["true"], ["err", "error"]),
            (["truth"], ["err", "error"]),
            (["x_true"], []),
            (["x_gt"], []),
        ],
        "fused": [
            (["fused"], ["err", "error"]),
            (["fusion"], ["err", "error"]),
            (["gnn"], ["err", "error"]),
            (["nn"], ["err", "error"]),
            (["x_fused"], []),
            (["x_gnn"], []),
        ],
        "avg": [
            (["avg"], ["err", "error"]),
            (["equal"], ["err", "error"]),
        ],
        "ci_chain": [
            (["ci", "chain"], ["err", "error"]),
        ],
        "ci_multi": [
            (["ci", "multi"], ["err", "error"]),
        ],
        "best_single": [
            (["best", "single"], ["err", "error"]),
            (["single", "best"], ["err", "error"]),
        ],
    }

    for include_terms, exclude_terms in candidate_rules.get(role, []):
        k = rank_npz_key(keys, include_terms, exclude_terms)
        if k is None:
            continue
        arr = np.asarray(npz_dict[k])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr
        if arr.ndim == 1 and arr.shape[0] >= 2:
            # 这种情况通常不是轨迹，跳过
            continue

    # 兜底：扫描所有二维数组，找 role 相关 key
    for k in keys:
        lk = k.lower()
        if role == "gt" and ("gt" in lk or "true" in lk or "truth" in lk):
            arr = np.asarray(npz_dict[k])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr
        if role == "fused" and ("fused" in lk or "gnn" in lk or "fusion" in lk or "nn" in lk):
            arr = np.asarray(npz_dict[k])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr

    return None


def find_error_like_array(npz_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    逐时刻误差数组，优先找 fused/gnn 对应的 error。
    允许返回：
        - [T]
        - [T, 2]
        - [T, d]
    """
    keys = list(npz_dict.keys())

    priority_rules = [
        (["gnn", "error"], []),
        (["fused", "error"], []),
        (["fusion", "error"], []),
        (["nn", "error"], []),
        (["pos", "error"], []),
        (["traj", "error"], []),
        (["error"], ["avg", "ci", "baseline"]),
        (["err"], ["avg", "ci", "baseline"]),
    ]

    for include_terms, exclude_terms in priority_rules:
        k = rank_npz_key(keys, include_terms, exclude_terms)
        if k is None:
            continue
        arr = np.asarray(npz_dict[k])
        if arr.ndim in (1, 2):
            return arr

    return None


def make_perstep_error_from_traj(gt: np.ndarray, fused: np.ndarray) -> Optional[np.ndarray]:
    if gt is None or fused is None:
        return None
    T = min(len(gt), len(fused))
    if T <= 0:
        return None
    gt2 = np.asarray(gt[:T, :2], dtype=float)
    fu2 = np.asarray(fused[:T, :2], dtype=float)
    err = np.linalg.norm(fu2 - gt2, axis=1)
    return err


# =========================
# 图1：主性能对比柱状图
# =========================
def plot_01_performance_bar(out_dir: Path, metrics: Dict[str, Any]) -> Optional[Path]:
    labels: List[str] = []
    values: List[float] = []

    pairs = [
        ("AVG", "rmse_avg"),
        ("CI-chain", "rmse_ci_chain"),
        ("CI-multi", "rmse_ci_multi"),
        ("Best-single", "rmse_best_single"),
        ("V0-pos", "rmse_gnn_pos"),
    ]

    for label, key in pairs:
        v = get_metric_value(metrics, key)
        if v is not None:
            labels.append(label)
            values.append(v)

    v_full = get_metric_value(metrics, "rmse_gnn_full")
    if v_full is not None:
        labels.append("V0-full")
        values.append(v_full)

    if len(labels) < 2:
        warnings.warn("图1跳过：性能字段不足。")
        return None

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)

    ax.set_title("V0 与经典融合基线的主性能对比（Clean 4-Sensor）", fontsize=13)
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

    out_path = out_dir / "fig01_v0_performance_bar.png"
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

    ax.set_title("V0 训练与验证损失收敛曲线", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_path = out_dir / "fig02_v0_training_curves.png"
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

    ax.set_title("V0 各传感器平均最终融合权重", fontsize=13)
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

    out_path = out_dir / "fig03_v0_mean_sensor_weights.png"
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

    ax.set_title("V0 最终融合权重热力图", fontsize=13)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sensor")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["S1", "S2", "S3", "S4"])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fusion weight")

    out_path = out_dir / "fig04_v0_weight_heatmap.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图5：轨迹对比图
# =========================
def plot_05_trajectory_compare(
    out_dir: Path,
    npz_dict: Optional[Dict[str, np.ndarray]],
) -> Optional[Path]:
    if npz_dict is None:
        warnings.warn("图5跳过：未找到 artifacts/sim_arrays.npz。")
        return None

    gt = find_xy_like_array(npz_dict, "gt")
    fused = find_xy_like_array(npz_dict, "fused")
    avg = find_xy_like_array(npz_dict, "avg")
    ci_chain = find_xy_like_array(npz_dict, "ci_chain")
    ci_multi = find_xy_like_array(npz_dict, "ci_multi")
    best_single = find_xy_like_array(npz_dict, "best_single")

    if gt is None or fused is None:
        warnings.warn("图5跳过：sim_arrays.npz 中未识别到 gt/fused 轨迹数组。")
        return None

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    gt2 = np.asarray(gt[:, :2], dtype=float)
    fu2 = np.asarray(fused[:, :2], dtype=float)
    ax.plot(gt2[:, 0], gt2[:, 1], linestyle="--", linewidth=2.0, label="GT")
    ax.plot(fu2[:, 0], fu2[:, 1], linewidth=1.8, label="V0 fused")

    if avg is not None:
        a2 = np.asarray(avg[:, :2], dtype=float)
        ax.plot(a2[:, 0], a2[:, 1], linewidth=1.2, label="AVG")
    if ci_chain is not None:
        c2 = np.asarray(ci_chain[:, :2], dtype=float)
        ax.plot(c2[:, 0], c2[:, 1], linewidth=1.2, label="CI-chain")
    if ci_multi is not None:
        cm2 = np.asarray(ci_multi[:, :2], dtype=float)
        ax.plot(cm2[:, 0], cm2[:, 1], linewidth=1.2, label="CI-multi")
    if best_single is not None:
        b2 = np.asarray(best_single[:, :2], dtype=float)
        ax.plot(b2[:, 0], b2[:, 1], linewidth=1.2, label="Best-single")

    ax.set_title("V0 融合轨迹与真值轨迹对比", fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.22)
    ax.axis("equal")
    ax.legend()

    out_path = out_dir / "fig05_v0_trajectory_compare.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 图6：逐时刻误差曲线
# =========================
def plot_06_perstep_error_curve(
    out_dir: Path,
    npz_dict: Optional[Dict[str, np.ndarray]],
) -> Optional[Path]:
    if npz_dict is None:
        warnings.warn("图6跳过：未找到 artifacts/sim_arrays.npz。")
        return None

    err = find_error_like_array(npz_dict)

    if err is None:
        gt = find_xy_like_array(npz_dict, "gt")
        fused = find_xy_like_array(npz_dict, "fused")
        err = make_perstep_error_from_traj(gt, fused)

    if err is None:
        warnings.warn("图6跳过：未从 sim_arrays.npz 中识别到逐时刻误差，也无法由轨迹重构。")
        return None

    err = np.asarray(err)

    if err.ndim == 2:
        if err.shape[1] == 2:
            y = np.linalg.norm(err, axis=1)
            ylabel = "Position error norm"
        else:
            y = np.linalg.norm(err, axis=1)
            ylabel = "Error norm"
    else:
        y = err.astype(float)
        ylabel = "Error"

    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(x, y, linewidth=1.5)

    ax.set_title("V0 单位时间跟踪误差曲线", fontsize=13)
    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)

    out_path = out_dir / "fig06_v0_perstep_error_curve.png"
    safe_savefig(fig, out_path)
    return out_path


# =========================
# 主流程
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot formal V0 analysis figures for NF-DKF result directory.")
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="V0 结果目录，例如 .\\results\\20260322_190048__train__clean_baseline_v0__default_clean_4sensor_scene__original_gnn_fusion",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录。默认 <result_dir>\\artifacts\\plots_custom_v0",
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir 不存在：{result_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (result_dir / "artifacts" / "plots_custom_v0")
    ensure_dir(out_dir)

    print("=" * 80)
    print("[INFO] 开始绘制 V0 分析图")
    print(f"[INFO] result_dir = {result_dir}")
    print(f"[INFO] out_dir    = {out_dir}")
    print("=" * 80)

    metrics = read_metrics_bundle(result_dir)
    history_df = get_history_df(result_dir)
    ts_df = get_timeseries_df(result_dir)
    npz_dict = load_npz_dict(get_sim_npz_path(result_dir))

    saved_paths: List[Path] = []

    jobs = [
        ("fig01", lambda: plot_01_performance_bar(out_dir, metrics)),
        ("fig02", lambda: plot_02_training_curves(out_dir, history_df, metrics)),
        ("fig03", lambda: plot_03_mean_sensor_weights(out_dir, metrics, ts_df)),
        ("fig04", lambda: plot_04_weight_heatmap(out_dir, ts_df)),
        ("fig05", lambda: plot_05_trajectory_compare(out_dir, npz_dict)),
        ("fig06", lambda: plot_06_perstep_error_curve(out_dir, npz_dict)),
    ]

    for name, fn in jobs:
        try:
            p = fn()
            if p is not None:
                saved_paths.append(p)
                print(f"[SAVED] {name}: {p}")
            else:
                print(f"[SKIP ] {name}: required fields missing")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("-" * 80)
    print(f"[DONE ] 共生成 {len(saved_paths)} 张图")
    for p in saved_paths:
        print(f"        {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()