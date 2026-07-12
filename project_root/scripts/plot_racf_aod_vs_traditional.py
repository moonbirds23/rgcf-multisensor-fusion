from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_DISPLAY = {
    "ME-RGCF-A0D": "RACF-AOD",
    "ME-RGCF-A0D-HS": "RACF-AOD",
    "CI-3T": "CI-3T",
    "AVG-3T": "Equal Weight",
    "WAA-MM-3T": "WAA-MM",
}

METHOD_ORDER = ["ME-RGCF-A0D", "ME-RGCF-A0D-HS", "CI-3T", "AVG-3T", "WAA-MM-3T"]
CORE_METHODS = ["ME-RGCF-A0D", "CI-3T", "AVG-3T", "WAA-MM-3T"]
FALLBACK_CORE_METHODS = ["ME-RGCF-A0D-HS", "CI-3T", "AVG-3T", "WAA-MM-3T"]

COLORS = {
    "ME-RGCF-A0D": "#2563eb",
    "ME-RGCF-A0D-HS": "#2563eb",
    "CI-3T": "#f97316",
    "AVG-3T": "#64748b",
    "WAA-MM-3T": "#16a34a",
}

SCENE_NAMES = {
    "S1R": "S1R basic",
    "S2R": "S2R maneuver",
}


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bundle(fig: plt.Figure, stem: Path) -> None:
    ensure_dir(stem.parent)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[saved] {stem.with_suffix('.png')}")
    print(f"[saved] {stem.with_suffix('.pdf')}")


def read_inputs(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_path = result_dir / "phase1r_aggregate_overall.csv"
    scene_path = result_dir / "phase1r_aggregate_by_scene.csv"
    missing = [str(p) for p in [overall_path, scene_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required aggregate files:\n" + "\n".join(missing))
    return pd.read_csv(overall_path), pd.read_csv(scene_path)


def pick_core_methods(overall: pd.DataFrame) -> list[str]:
    available = set(overall["method"].astype(str))
    if all(m in available for m in CORE_METHODS):
        return CORE_METHODS
    if all(m in available for m in FALLBACK_CORE_METHODS):
        return FALLBACK_CORE_METHODS
    wanted = [m for m in METHOD_ORDER if m in available]
    if not any(m.startswith("ME-RGCF-A0D") for m in wanted):
        raise ValueError("No AOD method found in aggregate file.")
    return wanted


def display_name(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def filtered(df: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    methods = list(methods)
    out = df[df["method"].isin(methods)].copy()
    out["method_display"] = out["method"].map(display_name)
    out["method_rank"] = out["method"].map({m: i for i, m in enumerate(methods)})
    out = out.sort_values(["method_rank", "scenario_id"] if "scenario_id" in out.columns else ["method_rank"])
    return out


def metric_value(df: pd.DataFrame, method: str, metric: str, scenario: str | None = None) -> float:
    view = df[df["method"].eq(method)]
    if scenario is not None:
        view = view[view["scenario_id"].eq(scenario)]
    if view.empty:
        return float("nan")
    return float(view.iloc[0][metric])


def lower_is_better_scores(values: dict[str, float]) -> dict[str, float]:
    finite = np.array([v for v in values.values() if math.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {k: 0.5 for k in values}
    best = float(np.nanmin(finite))
    worst = float(np.nanmax(finite))
    span = max(worst - best, 1e-9)
    return {k: 0.20 + 0.80 * (worst - v) / span if math.isfinite(v) else 0.20 for k, v in values.items()}


def improvement_pct(base: float, target: float) -> float:
    if not math.isfinite(base) or not math.isfinite(target) or abs(base) < 1e-12:
        return float("nan")
    return 100.0 * (base - target) / base


def normalized_visual_radii(gains: Iterable[float], min_radius: float = 0.42, max_radius: float = 0.98) -> list[float]:
    values = np.array([max(float(v), 0.0) if math.isfinite(float(v)) else 0.0 for v in gains], dtype=float)
    if values.size == 0:
        return []
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        return [0.70 for _ in values]
    return [float(min_radius + (v - lo) / (hi - lo) * (max_radius - min_radius)) for v in values]


def plot_six_dim_compass(overall: pd.DataFrame, by_scene: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    axes = [
        ("Overall\nRMSE", overall, "rmse_mean", None),
        ("Overall\nP99", overall, "p99_mean", None),
        ("Overall\nMax", overall, "max_mean", None),
        ("S1R\nRMSE", by_scene, "rmse_mean", "S1R"),
        ("S2R\nRMSE", by_scene, "rmse_mean", "S2R"),
        ("S2R\nP99", by_scene, "p99_mean", "S2R"),
    ]

    score_by_axis: list[dict[str, float]] = []
    raw_by_axis: list[dict[str, float]] = []
    for _, df, metric, scene in axes:
        raw = {m: metric_value(df, m, metric, scene) for m in methods}
        raw_by_axis.append(raw)
        score_by_axis.append(lower_is_better_scores(raw))

    theta = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    theta_closed = np.r_[theta, theta[0]]

    fig = plt.figure(figsize=(9.2, 8.5), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.06)
    ax.set_xticks(theta)
    ax.set_xticklabels([a[0] for a in axes], fontsize=11, color="#0f172a")
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["", "mid", "", "best"], fontsize=9, color="#64748b")
    ax.spines["polar"].set_color("#94a3b8")

    for method in methods:
        values = [score_by_axis[i][method] for i in range(len(axes))]
        closed = np.r_[values, values[0]]
        is_aod = method.startswith("ME-RGCF-A0D")
        color = COLORS.get(method, "#6b7280")
        ax.plot(theta_closed, closed, color=color, linewidth=3.3 if is_aod else 2.0, label=display_name(method), zorder=4)
        ax.fill(theta_closed, closed, color=color, alpha=0.16 if is_aod else 0.055, zorder=2)
        ax.scatter(theta, values, s=58 if is_aod else 36, color=color, edgecolor="white", linewidth=1.1, zorder=5)

    ax.set_title("Six-dimensional Performance Compass", y=1.10, fontsize=18, color="#0f172a")
    ax.text(
        0.5,
        -0.12,
        "Outward is better. Scores are normalized within RACF-AOD and traditional fusion baselines only.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=4, frameon=False, fontsize=10)
    save_bundle(fig, out_dir / "fig1_six_dim_performance_compass")


def _deep_indicators(overall: pd.DataFrame, by_scene: pd.DataFrame, method: str) -> dict[str, float]:
    overall_p95 = metric_value(overall, method, "p95_mean")
    overall_p99 = metric_value(overall, method, "p99_mean")
    overall_max = metric_value(overall, method, "max_mean")
    overall_rmse = metric_value(overall, method, "rmse_mean")

    s1_rmse = metric_value(by_scene, method, "rmse_mean", "S1R")
    s2_rmse = metric_value(by_scene, method, "rmse_mean", "S2R")
    s1_p95 = metric_value(by_scene, method, "p95_mean", "S1R")
    s2_p95 = metric_value(by_scene, method, "p95_mean", "S2R")
    s1_p99 = metric_value(by_scene, method, "p99_mean", "S1R")
    s2_p99 = metric_value(by_scene, method, "p99_mean", "S2R")

    return {
        "tail_onset": overall_p95 - overall_rmse,
        "tail_stretch": overall_p99 - overall_p95,
        "risk_envelope": overall_max - overall_rmse,
        "maneuver_rmse_penalty": s2_rmse - s1_rmse,
        "maneuver_p95_penalty": s2_p95 - s1_p95,
        "maneuver_p99_penalty": s2_p99 - s1_p99,
    }


def plot_improvement_bloom(overall: pd.DataFrame, by_scene: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    aod = next(m for m in methods if m.startswith("ME-RGCF-A0D"))
    baselines = [m for m in ["CI-3T", "AVG-3T", "WAA-MM-3T"] if m in methods]
    aod_ind = _deep_indicators(overall, by_scene, aod)
    baseline_ind = {m: _deep_indicators(overall, by_scene, m) for m in baselines}

    indicator_defs = [
        ("Tail onset\nP95-RMSE", "tail_onset", "#2563eb"),
        ("Extreme tail\nP99-P95", "tail_stretch", "#3b82f6"),
        ("Risk envelope\nMax-RMSE", "risk_envelope", "#60a5fa"),
        ("Maneuver gap\nS2R-S1R RMSE", "maneuver_rmse_penalty", "#16a34a"),
        ("Tail maneuver\nS2R-S1R P95", "maneuver_p95_penalty", "#22c55e"),
        ("Extreme maneuver\nS2R-S1R P99", "maneuver_p99_penalty", "#4ade80"),
    ]

    petals = []
    for label, key, color in indicator_defs:
        baseline_values = np.array([baseline_ind[m][key] for m in baselines], dtype=float)
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size == 0:
            continue
        traditional_median = float(np.median(baseline_values))
        traditional_best = float(np.min(baseline_values))
        aod_value = float(aod_ind[key])
        median_gain = improvement_pct(traditional_median, aod_value)
        best_gain = improvement_pct(traditional_best, aod_value)
        petals.append((label, median_gain, best_gain, aod_value, traditional_median, color))

    radii = normalized_visual_radii([p[1] for p in petals])
    theta = np.linspace(0, 2 * np.pi, len(petals), endpoint=False)
    width = 2 * np.pi / max(len(petals), 1) * 0.72

    fig = plt.figure(figsize=(9.3, 8.4), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.10)
    ax.set_xticks(theta)
    ax.set_xticklabels([p[0] for p in petals], fontsize=10.5, color="#0f172a")
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["", "norm.", "", "high"], fontsize=9, color="#64748b")
    ax.spines["polar"].set_color("#94a3b8")

    for i, (_, gain, best_gain, aod_value, traditional_median, color) in enumerate(petals):
        radius = radii[i]
        edge = color if gain >= 0 else "#ef4444"
        fill = color if gain >= 0 else "#fee2e2"
        ax.bar(theta[i], radius, width=width, bottom=0.0, color=fill, alpha=0.30, edgecolor=edge, linewidth=2.1)
        marker_color = color if best_gain >= 0 else "#f97316"
        ax.scatter([theta[i]], [radius], s=72, color=marker_color, edgecolor="white", linewidth=1.3, zorder=5)
        gain_text = f"{gain:+.1f}%"
        ax.text(theta[i], max(0.20, radius - 0.11), gain_text, ha="center", va="center", fontsize=9.3, color="#0f172a", fontweight="bold")
        ax.text(
            theta[i],
            min(1.04, radius + 0.12),
            f"{aod_value:.2f}/{traditional_median:.2f}",
            ha="center",
            va="center",
            fontsize=8.2,
            color="#475569",
        )

    ax.set_title("RACF-AOD Deep Robustness Bloom", y=1.10, fontsize=18, color="#0f172a")
    ax.text(
        0.5,
        -0.12,
        "Radius is normalized for visual balance. Labels keep true RACF-AOD / traditional-median values and lower-is-better reductions.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    save_bundle(fig, out_dir / "fig2_rmse_reduction_bloom")


def plot_overall_error_profile(overall: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    metrics = [("RMSE", "rmse_mean"), ("P95", "p95_mean"), ("P99", "p99_mean"), ("Max", "max_mean")]
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.2, 5.6), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    for i, method in enumerate(methods):
        vals = [metric_value(overall, method, metric) for _, metric in metrics]
        offset = (i - (len(methods) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=display_name(method),
            color=COLORS.get(method, "#6b7280"),
            alpha=0.88 if method.startswith("ME-RGCF-A0D") else 0.72,
            edgecolor="#ffffff",
            linewidth=1.0,
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=11)
    ax.set_ylabel("Position error (m)")
    ax.set_title("Overall Error Profile")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_ylim(0, max(overall[overall["method"].isin(methods)]["max_mean"]) * 1.22)
    save_bundle(fig, out_dir / "fig3_overall_error_profile")


def plot_scene_metric_bars(by_scene: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), facecolor="#f8fafc", constrained_layout=True)
    for ax, metric_label, metric in zip(axes, ["RMSE", "P99"], ["rmse_mean", "p99_mean"]):
        ax.set_facecolor("#f8fafc")
        scenes = ["S1R", "S2R"]
        x = np.arange(len(scenes))
        width = 0.18
        for i, method in enumerate(methods):
            vals = [metric_value(by_scene, method, metric, scene) for scene in scenes]
            offset = (i - (len(methods) - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                label=display_name(method),
                color=COLORS.get(method, "#6b7280"),
                alpha=0.90 if method.startswith("ME-RGCF-A0D") else 0.72,
                edgecolor="#ffffff",
                linewidth=1.0,
            )
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENE_NAMES.get(s, s) for s in scenes])
        ax.set_ylabel("Position error (m)")
        ax.set_title(f"By-scene {metric_label}")
    axes[0].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(1.06, -0.12))
    save_bundle(fig, out_dir / "fig4_scene_rmse_p99_comparison")


def build_improvement_table(overall: pd.DataFrame, by_scene: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    aod = next(m for m in methods if m.startswith("ME-RGCF-A0D"))
    rows = []
    baselines = [m for m in ["CI-3T", "AVG-3T", "WAA-MM-3T"] if m in methods]
    metrics = ["rmse_mean", "p95_mean", "p99_mean", "max_mean"]
    for scope, df, scene in [("Overall", overall, None), ("S1R", by_scene, "S1R"), ("S2R", by_scene, "S2R")]:
        for baseline in baselines:
            row = {"scope": scope, "baseline": display_name(baseline)}
            for metric in metrics:
                row[metric.replace("_mean", "_reduction_pct")] = improvement_pct(
                    metric_value(df, baseline, metric, scene),
                    metric_value(df, aod, metric, scene),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_deep_bloom_table(overall: pd.DataFrame, by_scene: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    aod = next(m for m in methods if m.startswith("ME-RGCF-A0D"))
    baselines = [m for m in ["CI-3T", "AVG-3T", "WAA-MM-3T"] if m in methods]
    indicator_labels = {
        "tail_onset": "P95-RMSE tail onset",
        "tail_stretch": "P99-P95 extreme-tail stretch",
        "risk_envelope": "Max-RMSE risk envelope",
        "maneuver_rmse_penalty": "S2R-S1R RMSE maneuver gap",
        "maneuver_p95_penalty": "S2R-S1R P95 tail maneuver",
        "maneuver_p99_penalty": "S2R-S1R P99 extreme maneuver",
    }
    aod_ind = _deep_indicators(overall, by_scene, aod)
    baseline_ind = {m: _deep_indicators(overall, by_scene, m) for m in baselines}
    rows = []
    for key, label in indicator_labels.items():
        baseline_values = [baseline_ind[m][key] for m in baselines]
        traditional_median = float(np.median(np.array(baseline_values, dtype=float)))
        traditional_best = float(np.min(np.array(baseline_values, dtype=float)))
        reduction_median = improvement_pct(traditional_median, float(aod_ind[key]))
        rows.append(
            {
                "indicator": key,
                "label": label,
                "racf_aod": float(aod_ind[key]),
                "traditional_median": traditional_median,
                "traditional_best": traditional_best,
                "reduction_vs_traditional_median_pct": reduction_median,
                "reduction_vs_traditional_best_pct": improvement_pct(traditional_best, float(aod_ind[key])),
            }
        )
    radii = normalized_visual_radii([r["reduction_vs_traditional_median_pct"] for r in rows])
    for row, radius in zip(rows, radii):
        row["visual_radius_normalized"] = radius
    return pd.DataFrame(rows)


def plot_deep_robustness_gap_dashboard(overall: pd.DataFrame, by_scene: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    table = build_deep_bloom_table(overall, by_scene, methods)
    labels = [
        "Tail onset\nP95-RMSE",
        "Extreme tail\nP99-P95",
        "Risk envelope\nMax-RMSE",
        "Maneuver gap\nS2R-S1R RMSE",
        "Tail maneuver\nS2R-S1R P95",
        "Extreme maneuver\nS2R-S1R P99",
    ]
    colors = ["#2563eb", "#3b82f6", "#60a5fa", "#16a34a", "#22c55e", "#4ade80"]
    ratio = (table["racf_aod"] / table["traditional_median"]).to_numpy(dtype=float)
    gains = table["reduction_vs_traditional_median_pct"].to_numpy(dtype=float)
    y = np.arange(len(table))[::-1]

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.5),
        facecolor="#f8fafc",
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.10},
    )
    for ax in (ax_l, ax_r):
        ax.set_facecolor("#f8fafc")

    # Left: normalized dumbbell. Traditional median is fixed at 1.0.
    ax_l.axvline(1.0, color="#64748b", linewidth=1.1, linestyle="--", alpha=0.80)
    for i, row in enumerate(table.itertuples(index=False)):
        yi = y[i]
        color = colors[i]
        ax_l.plot([ratio[i], 1.0], [yi, yi], color=color, linewidth=4.8, alpha=0.23, solid_capstyle="round")
        ax_l.scatter([1.0], [yi], s=96, color="#94a3b8", edgecolor="white", linewidth=1.4, zorder=4, label=None)
        ax_l.scatter([ratio[i]], [yi], s=120, color=color, edgecolor="white", linewidth=1.5, zorder=5)
        ax_l.text(
            min(ratio[i], 1.0) - 0.015,
            yi + 0.27,
            f"{row.racf_aod:.2f} / {row.traditional_median:.2f}",
            ha="left",
            va="center",
            fontsize=9,
            color="#475569",
        )

    x_min = max(0.0, float(np.nanmin(ratio)) - 0.08)
    ax_l.set_xlim(x_min, 1.07)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels, fontsize=10.4, color="#0f172a")
    ax_l.set_xlabel("Normalized indicator value (traditional median = 1.0)")
    ax_l.set_title("Deep Indicator Gap", fontsize=15, color="#0f172a", pad=14)
    ax_l.text(0.99, 1.02, "traditional median", transform=ax_l.get_xaxis_transform(), ha="right", fontsize=9, color="#64748b")
    ax_l.text(x_min, 1.02, "lower is better", transform=ax_l.get_xaxis_transform(), ha="left", fontsize=9, color="#2563eb")

    # Right: true reduction percentages.
    bars = ax_r.barh(y, gains, color=colors, alpha=0.82, edgecolor="white", linewidth=1.2, height=0.60)
    ax_r.axvline(0, color="#94a3b8", linewidth=0.9)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([""] * len(y))
    ax_r.set_xlabel("Reduction vs traditional median (%)")
    ax_r.set_title("True Reduction", fontsize=15, color="#0f172a", pad=14)
    x_max = max(32.0, float(np.nanmax(gains)) * 1.22)
    ax_r.set_xlim(0, x_max)
    for bar, gain in zip(bars, gains):
        ax_r.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{gain:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    fig.suptitle("RACF-AOD Deep Robustness Gap Dashboard", fontsize=19, fontweight="bold", color="#0f172a", y=0.99)
    fig.text(
        0.50,
        0.02,
        "Derived indicators avoid repeating the six-dimensional error axes. Values beside points are RACF-AOD / traditional-median in meters.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    save_bundle(fig, out_dir / "fig2_deep_robustness_gap_dashboard")


def plot_improvement_heatmap(table: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["rmse_reduction_pct", "p95_reduction_pct", "p99_reduction_pct", "max_reduction_pct"]
    labels = ["RMSE", "P95", "P99", "Max"]
    row_labels = [f"{r.scope} vs {r.baseline}" for r in table.itertuples(index=False)]
    values = table[metrics].to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(values))), 1.0)

    fig, ax = plt.subplots(figsize=(9.8, 6.1), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("RACF-AOD Error Reduction Against Traditional Baselines")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color="#0f172a", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.025)
    cbar.set_label("Error reduction (%)")
    save_bundle(fig, out_dir / "fig5_improvement_heatmap")


def write_summary(overall: pd.DataFrame, by_scene: pd.DataFrame, methods: list[str], improvement: pd.DataFrame, out_dir: Path) -> None:
    out_overall = filtered(overall, methods)
    out_scene = filtered(by_scene, methods)
    deep_bloom = build_deep_bloom_table(overall, by_scene, methods)
    out_overall.to_csv(out_dir / "filtered_overall.csv", index=False)
    out_scene.to_csv(out_dir / "filtered_by_scene.csv", index=False)
    improvement.to_csv(out_dir / "racf_aod_improvement_summary.csv", index=False)
    deep_bloom.to_csv(out_dir / "deep_robustness_bloom_source.csv", index=False)

    aod = next(m for m in methods if m.startswith("ME-RGCF-A0D"))
    ci_rmse = improvement[(improvement["scope"].eq("Overall")) & (improvement["baseline"].eq("CI-3T"))]["rmse_reduction_pct"]
    eq_rmse = improvement[(improvement["scope"].eq("Overall")) & (improvement["baseline"].eq("Equal Weight"))]["rmse_reduction_pct"]
    waa_rmse = improvement[(improvement["scope"].eq("Overall")) & (improvement["baseline"].eq("WAA-MM"))]["rmse_reduction_pct"]

    lines = [
        "# RACF-AOD vs traditional baselines figure source",
        "",
        f"Learned method used as RACF-AOD: `{aod}`.",
        "Historical learned versions are intentionally excluded.",
        "",
        "## Overall headline",
        "",
        f"- RMSE vs CI-3T: {float(ci_rmse.iloc[0]):.2f}% reduction" if not ci_rmse.empty else "- RMSE vs CI-3T: n/a",
        f"- RMSE vs Equal Weight: {float(eq_rmse.iloc[0]):.2f}% reduction" if not eq_rmse.empty else "- RMSE vs Equal Weight: n/a",
        f"- RMSE vs WAA-MM: {float(waa_rmse.iloc[0]):.2f}% reduction" if not waa_rmse.empty else "- RMSE vs WAA-MM: n/a",
        "",
        "## Filtered overall data",
        "",
        "```csv",
        out_overall.drop(columns=["method_rank"], errors="ignore").to_csv(index=False).strip(),
        "```",
        "",
        "## Filtered by-scene data",
        "",
        "```csv",
        out_scene.drop(columns=["method_rank"], errors="ignore").to_csv(index=False).strip(),
        "```",
        "",
        "## Improvement summary",
        "",
        "```csv",
        improvement.to_csv(index=False).strip(),
        "```",
        "",
        "## Deep robustness bloom source",
        "",
        "```csv",
        deep_bloom.to_csv(index=False).strip(),
        "```",
    ]
    (out_dir / "figure_source_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_dir / 'figure_source_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RACF-AOD vs traditional baseline report figures.")
    parser.add_argument(
        "--result-dir",
        default=r"results/实验数据/当前主线_Phase1R_ME/phase2_me_a0_dir_only",
        help="Directory containing phase1r_aggregate_overall.csv and phase1r_aggregate_by_scene.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"results/科研图像数据/racf_aod_vs_traditional",
        help="Output directory for PNG/PDF figures and source summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    result_dir = Path(args.result_dir)
    out_dir = ensure_dir(Path(args.out_dir))

    overall, by_scene = read_inputs(result_dir)
    methods = pick_core_methods(overall)
    overall_f = filtered(overall, methods)
    by_scene_f = filtered(by_scene, methods)

    plot_six_dim_compass(overall_f, by_scene_f, methods, out_dir)
    plot_deep_robustness_gap_dashboard(overall_f, by_scene_f, methods, out_dir)
    plot_overall_error_profile(overall_f, methods, out_dir)
    plot_scene_metric_bars(by_scene_f, methods, out_dir)
    improvement = build_improvement_table(overall_f, by_scene_f, methods)
    plot_improvement_heatmap(improvement, out_dir)
    write_summary(overall_f, by_scene_f, methods, improvement, out_dir)


if __name__ == "__main__":
    main()
