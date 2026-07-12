from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "A0": "#64748b",
    "A0D": "#2563eb",
    "A0D_ACCENT": "#7c3aed",
    "S1R": "#2563eb",
    "S2R": "#059669",
    "Evidence": "#f59e0b",
    "Track": "#8b5cf6",
    "Good": "#10b981",
    "Warn": "#f97316",
    "Bad": "#ef4444",
}


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 150,
            "savefig.dpi": 260,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _finite(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _score_lower(rows: dict[str, dict], metric: str) -> dict[str, float]:
    vals = np.array([_finite(v.get(metric)) for v in rows.values()], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {k: 0.5 for k in rows}
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-9)
    return {k: 0.18 + 0.82 * (hi - _finite(v.get(metric), hi)) / span for k, v in rows.items()}


def _score_higher(rows: dict[str, dict], metric: str, floor: float = 0.0) -> dict[str, float]:
    vals = np.array([max(_finite(v.get(metric), floor), floor) for v in rows.values()], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {k: 0.5 for k in rows}
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-9)
    return {k: 0.18 + 0.82 * (max(_finite(v.get(metric), floor), floor) - lo) / span for k, v in rows.items()}


def load_inputs(a0d_dir: Path, diag_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "overall": a0d_dir / "phase1r_aggregate_overall.csv",
        "by_scene": a0d_dir / "phase1r_aggregate_by_scene.csv",
        "run_summary": a0d_dir / "phase1r_run_summary.csv",
        "diag_summary": diag_dir / "a0d_directionality_summary.csv",
        "diag_scene": diag_dir / "a0d_directionality_by_scene.csv",
        "pair_matrix": diag_dir / "a0d_directionality_pair_matrix.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required A0D inputs:\n" + "\n".join(missing))
    return {name: pd.read_csv(path) for name, path in required.items()}


def plot_process_compass(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    diag = data["diag_summary"].set_index("method").to_dict("index")
    selected = {
        "A0": diag.get("A0", {}),
        "A0D": diag.get("A0D", {}),
    }
    selected["A0"].setdefault("raw_residual_corr_mean", 0.0)
    selected["A0"].setdefault("high_spread_corr_mean", 0.0)
    selected["A0"].setdefault("worst_track_hit_rate_mean", 0.0)
    selected["A0"].setdefault("worst_low_weight_rate_mean", 0.0)

    metrics = [
        ("Error\ncontrol", "error_pos_mean_mean", "lower"),
        ("Tail\ncontrol", "error_pos_max_mean", "lower"),
        ("Attention\nsplit", "row_std_mean_mean", "higher"),
        ("Evidence\ndirection", "high_spread_corr_mean", "higher"),
        ("Worst-track\nhit", "worst_track_hit_rate_mean", "higher"),
        ("Fusion\nsuppression", "worst_low_weight_rate_mean", "higher"),
    ]
    score_maps = {}
    for _, metric, direction in metrics:
        if direction == "lower":
            score_maps[metric] = _score_lower(selected, metric)
        else:
            score_maps[metric] = _score_higher(selected, metric)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(9.8, 9.2), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.05)
    ax.grid(color="#cbd5e1", linewidth=0.85, alpha=0.75)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_xticks(angles)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=11, color="#0f172a")
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["", "mid", "", "best"], fontsize=9, color="#64748b")

    for method, color, lw, alpha in [
        ("A0", COLORS["A0"], 2.2, 0.08),
        ("A0D", COLORS["A0D"], 3.6, 0.18),
    ]:
        vals = [score_maps[metric][method] for _, metric, _ in metrics]
        vals_closed = np.r_[vals, vals[0]]
        ax.plot(angles_closed, vals_closed, color=color, linewidth=lw, label=method, zorder=4)
        ax.fill(angles_closed, vals_closed, color=color, alpha=alpha, zorder=2)
        ax.scatter(angles, vals, s=62 if method == "A0D" else 42, color=color, edgecolor="white", linewidth=1.2, zorder=5)

    ax.set_title("A0D Process Compass", y=1.10, fontsize=20, fontweight="bold", color="#0f172a")
    ax.text(
        0.5,
        -0.13,
        "Outward is better. A0D turns A0's near-broadcast M->P attention into differentiated evidence-to-track interaction.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.2,
        color="#475569",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=11)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_bloom(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    by_scene = data["by_scene"]
    diag_scene = data["diag_scene"]
    a0d_scene = by_scene[by_scene["method"].eq("ME-RGCF-A0D")].set_index("scenario_id")
    ci_scene = by_scene[by_scene["method"].eq("CI-3T")].set_index("scenario_id")
    diag_a0d = diag_scene[diag_scene["method"].eq("A0D")].set_index("scenario")

    petals = [
        ("S1R\nRMSE", "S1R", "rmse_mean"),
        ("S1R\nP99", "S1R", "p99_mean"),
        ("S1R\ndir.", "S1R", "high_spread_corr_mean"),
        ("S2R\nRMSE", "S2R", "rmse_mean"),
        ("S2R\nP99", "S2R", "p99_mean"),
        ("S2R\ndir.", "S2R", "high_spread_corr_mean"),
    ]
    scores = []
    values = []
    colors = []
    for label, scene, metric in petals:
        if metric in {"rmse_mean", "p99_mean"}:
            val = _finite(a0d_scene.loc[scene, metric])
            base = _finite(ci_scene.loc[scene, metric])
            score = np.clip(base / max(val, 1e-9), 0.2, 1.18)
            score = min(score / 1.18, 1.0)
        else:
            val = _finite(diag_a0d.loc[scene, metric])
            score = np.clip(val / 0.18, 0.2, 1.0)
        scores.append(float(score))
        values.append(float(val))
        colors.append(COLORS[scene])

    theta = np.linspace(0, 2 * np.pi, len(petals), endpoint=False)
    width = 2 * np.pi / len(petals) * 0.70

    fig = plt.figure(figsize=(9.6, 8.8), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.14)
    ax.grid(color="#dbe3ee", linewidth=0.85, alpha=0.78)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_xticks(theta)
    ax.set_xticklabels([p[0] for p in petals], fontsize=11, color="#0f172a")
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["", "steady", "", "strong"], fontsize=9, color="#64748b")

    for i, (score, val, color) in enumerate(zip(scores, values, colors)):
        ax.bar(theta[i], score, width=width, bottom=0, color=color, alpha=0.25, edgecolor=color, linewidth=2.0)
        label = f"{val:.2f}" if i % 3 != 2 else f"{val:.3f}"
        ax.text(theta[i], max(0.34, score - 0.12), label, ha="center", va="center", fontsize=9.2, color="#0f172a")
        ax.scatter([theta[i]], [score], s=70, color=color, edgecolor="white", linewidth=1.3, zorder=4)

    ax.set_title("A0D Scenario Robustness Bloom", y=1.10, fontsize=20, fontweight="bold", color="#0f172a")
    ax.text(
        0.5,
        -0.13,
        "Petals summarize S1R/S2R accuracy, tail control, and residual-aware directionality. RMSE/P99 are scored against CI-3T.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_pair_attention_matrix(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    pair = data["pair_matrix"]
    a0d = pair[pair["method"].eq("A0D")]
    pivot = a0d.groupby(["p_node", "m_node"])["mean_mp_attn"].mean().unstack("m_node")
    pivot = pivot.reindex(index=["P1", "P2", "P3"], columns=["M1", "M2", "M3", "M4", "M5"])

    fig, ax = plt.subplots(figsize=(9.6, 5.9), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0.0, vmax=max(0.45, float(np.nanmax(pivot.values))))
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=12, color="#0f172a")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=12, color="#0f172a")
    ax.set_xlabel("")
    ax.set_ylabel("Posterior track nodes", fontsize=12, color="#0f172a", labelpad=10)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            color = "white" if value > 0.25 else "#0f172a"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    for x in [2.5]:
        ax.axvline(x, color="#f97316", linewidth=2.2, linestyle="--", alpha=0.85)
    ax.text(
        0.30,
        -0.135,
        "track measurements",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["Track"],
        fontweight="bold",
    )
    ax.text(
        0.70,
        -0.135,
        "evidence-only nodes",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["Evidence"],
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.025)
    cbar.set_label("mean M->P attention", fontsize=10)
    ax.set_title("A0D Pair-aware M->P Attention Matrix", fontsize=18, fontweight="bold", color="#0f172a", pad=24)
    ax.text(
        0.5,
        -0.285,
        "Rows are P1-P3. Columns are M1-M3 track-measurement nodes and M4-M5 evidence-only nodes. Non-uniform rows show that A0D is not broadcasting.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_directionality_relay(data: dict[str, pd.DataFrame], out_path: Path) -> None:
    row = data["diag_summary"].set_index("method").loc["A0D"]
    metrics = [
        ("Attention split", _finite(row["row_std_mean_mean"]) / 0.08, COLORS["A0D"]),
        ("Residual corr.", _finite(row["high_spread_corr_mean"]) / 0.18, COLORS["Evidence"]),
        ("Worst-track hit", _finite(row["worst_track_hit_rate_mean"]) / 0.45, COLORS["Warn"]),
        ("Low weight rate", _finite(row["worst_low_weight_rate_mean"]) / 0.80, COLORS["Good"]),
        ("High cov rate", _finite(row["worst_high_cov_rate_mean"]) / 0.60, COLORS["S2R"]),
    ]
    raw_values = [
        _finite(row["row_std_mean_mean"]),
        _finite(row["high_spread_corr_mean"]),
        _finite(row["worst_track_hit_rate_mean"]),
        _finite(row["worst_low_weight_rate_mean"]),
        _finite(row["worst_high_cov_rate_mean"]),
    ]
    labels = [m[0] for m in metrics]
    scores = [min(max(m[1], 0.0), 1.08) for m in metrics]
    colors = [m[2] for m in metrics]

    fig, ax = plt.subplots(figsize=(10.8, 6.8), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, scores, color=colors, alpha=0.78, edgecolor="white", linewidth=1.5)
    ax.axvline(1.0, color="#0f172a", linewidth=1.2, linestyle="--", alpha=0.65)
    ax.set_xlim(0, 1.16)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color="#0f172a")
    ax.set_xlabel("normalized target achievement", fontsize=11, color="#334155")
    ax.set_title("A0D Directionality Relay", fontsize=19, fontweight="bold", color="#0f172a", pad=18)

    for bar, raw in zip(bars, raw_values):
        ax.text(
            min(bar.get_width() + 0.025, 1.12),
            bar.get_y() + bar.get_height() / 2,
            f"{raw:.3f}",
            va="center",
            ha="left",
            fontsize=11,
            color="#0f172a",
            fontweight="bold",
        )

    ax.text(
        0.02,
        -0.28,
        f"Downstream response: Δw={_finite(row['delta_w_mean_mean']):+.3f}, "
        f"Δg={_finite(row['delta_g_mean_mean']):+.3f}, "
        f"Δcov={_finite(row['delta_cov_mean_mean']):+.3f}",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    ax.text(
        0.02,
        -0.38,
        "Interpretation: A0D already differentiates evidence-to-track attention; the remaining question is how strongly this signal propagates to weight/gate/covariance heads.",
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
    )
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def write_summary(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    overall = data["overall"]
    scene = data["by_scene"]
    diag = data["diag_summary"]
    diag_cols = [
        "method",
        "n_runs",
        "entropy_mean_mean",
        "row_std_mean_mean",
        "evidence_mass_mean_mean",
        "own_track_mass_mean_mean",
        "raw_residual_corr_mean",
        "high_spread_corr_mean",
        "worst_track_hit_rate_mean",
        "delta_w_mean_mean",
        "delta_cov_mean_mean",
        "worst_low_weight_rate_mean",
        "worst_high_cov_rate_mean",
    ]
    lines = [
        "# A0D report figure source summary",
        "",
        "## Overall",
        "```csv",
        overall.to_csv(index=False).strip(),
        "```",
        "",
        "## By scene",
        "```csv",
        scene.to_csv(index=False).strip(),
        "```",
        "",
        "## Directionality summary",
        "```csv",
        diag[diag_cols].to_csv(index=False).strip(),
        "```",
    ]
    (out_dir / "a0d_report_figure_source_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate A0D report figures from Phase2 A0D results and directionality diagnostics.")
    p.add_argument("--a0d-dir", default=r"E:\migration_packages\results\phase2_me_a0_dir_only")
    p.add_argument("--diag-dir", default="analysis_outputs/a0d_directionality_diagnostics")
    p.add_argument("--out-dir", default=r"results\科研图像数据\a0d_report_figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    a0d_dir = Path(args.a0d_dir)
    diag_dir = Path(args.diag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(a0d_dir, diag_dir)
    plot_process_compass(data, out_dir / "fig1_a0d_process_compass.png")
    plot_scenario_bloom(data, out_dir / "fig2_a0d_scenario_robustness_bloom.png")
    plot_pair_attention_matrix(data, out_dir / "fig3_a0d_pair_attention_matrix.png")
    plot_directionality_relay(data, out_dir / "fig4_a0d_directionality_relay.png")
    write_summary(data, out_dir)

    for name in [
        "fig1_a0d_process_compass.png",
        "fig2_a0d_scenario_robustness_bloom.png",
        "fig3_a0d_pair_attention_matrix.png",
        "fig4_a0d_directionality_relay.png",
        "a0d_report_figure_source_summary.md",
    ]:
        print(f"[saved] {out_dir / name}")


if __name__ == "__main__":
    main()
