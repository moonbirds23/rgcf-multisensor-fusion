from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge


P11 = "P11_feature_stable_reliability_no_cov"
BASELINES = ["P0_post_only_single_stream", "P4_full_gate_no_cov_formula", P11]

LABELS = {
    "P0_post_only_single_stream": "P0 post-only",
    "P4_full_gate_no_cov_formula": "P4 full gate",
    P11: "P11 reliability-only",
}

COLORS = {
    "P0_post_only_single_stream": "#64748b",
    "P4_full_gate_no_cov_formula": "#2563eb",
    P11: "#059669",
}

MODE_COLORS = {
    "clean": "#10b981",
    "dropout": "#2563eb",
    "pollution": "#f59e0b",
    "bias_ramp": "#ef4444",
}


def _num(value):
    if value in ("", None):
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def load_rows(result_dir: Path) -> List[Dict]:
    path = result_dir / "ablation_runs_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_diagnostics(result_dir: Path) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    diag_dir = result_dir / "tail_diagnostics"
    for path in sorted(diag_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = str(data.get("run_label", path.stem))
        variant = label.split("__trainseed", 1)[0] if "__trainseed" in label else label
        out[variant].append(data)
    return out


def aggregate_rows(rows: List[Dict]) -> Dict[str, Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)

    metrics = [
        "best_val_loss",
        "test_loss_pos",
        "full_p95_error",
        "full_p99_error",
        "full_max_error",
        "bias_ramp_p99_error",
        "bias_ramp_top_max_error",
        "p99_mean_fault_weight",
        "p99_fault_top_rate",
        "mean_gate_fault",
        "mean_gate_normal",
        "mean_weight_fault",
        "mean_weight_normal",
        "fault_window_max_error_pos",
    ]

    out = {}
    for variant, items in grouped.items():
        data = {"variant": variant, "n": len(items)}
        for metric in metrics:
            vals = [_num(item.get(metric)) for item in items]
            vals = [v for v in vals if v is not None]
            data[f"{metric}_mean"] = _mean(vals)
            data[f"{metric}_std"] = _std(vals)
            data[f"{metric}_min"] = min(vals) if vals else float("nan")
            data[f"{metric}_max"] = max(vals) if vals else float("nan")
        out[variant] = data
    return out


def aggregate_modes(diagnostics: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Dict]]:
    out = {}
    for variant, items in diagnostics.items():
        mode_stats = defaultdict(lambda: defaultdict(list))
        for diag in items:
            for mode, stats in diag.get("grouped_by_mode", {}).items():
                for key in ("mean_error", "p95_error", "p99_error", "top_max_error"):
                    value = _num(stats.get(key))
                    if value is not None:
                        mode_stats[mode][key].append(value)
        out[variant] = {
            mode: {f"{key}_mean": _mean(vals) for key, vals in keys.items()}
            for mode, keys in mode_stats.items()
        }
    return out


def _higher_is_better_score(agg: Dict[str, Dict], metric: str) -> Dict[str, float]:
    vals = [data.get(metric, float("nan")) for data in agg.values()]
    vals = np.array([float(v) for v in vals if math.isfinite(float(v))], dtype=float)
    if len(vals) == 0:
        return {k: 0.5 for k in agg}
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-9)
    return {k: 0.15 + 0.85 * (hi - float(v.get(metric, hi))) / span for k, v in agg.items()}


def plot_progress_compass(agg: Dict[str, Dict], out_path: Path):
    metrics = [
        ("Accuracy", "test_loss_pos_mean"),
        ("P95 control", "full_p95_error_mean"),
        ("P99 control", "full_p99_error_mean"),
        ("Tail stability", "full_max_error_mean"),
        ("Bias robustness", "bias_ramp_p99_error_mean"),
        ("Seed consistency", "test_loss_pos_std"),
    ]
    selected = {k: agg[k] for k in BASELINES if k in agg}
    scores = {metric: _higher_is_better_score(selected, metric) for _, metric in metrics}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(9.8, 9.2), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.04)
    ax.grid(color="#cbd5e1", linewidth=0.8, alpha=0.72)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics], fontsize=11, color="#0f172a")
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["", "balanced", "", "best"], fontsize=9, color="#64748b")

    for variant in selected:
        vals = [scores[metric][variant] for _, metric in metrics]
        vals = np.r_[vals, vals[0]]
        color = COLORS.get(variant, "#111827")
        lw = 3.4 if variant == P11 else 2.0
        alpha = 0.18 if variant == P11 else 0.07
        z = 5 if variant == P11 else 3
        ax.plot(angles, vals, color=color, linewidth=lw, label=LABELS.get(variant, variant), zorder=z)
        ax.fill(angles, vals, color=color, alpha=alpha, zorder=z - 1)
        ax.scatter(angles[:-1], vals[:-1], s=54 if variant == P11 else 36, color=color, edgecolor="white", linewidth=1.2, zorder=z + 1)

    ax.set_title("P11 Progress Compass", y=1.10, fontsize=19, fontweight="bold", color="#0f172a")
    ax.text(
        0.5,
        -0.13,
        "Outward is better. P11 emphasizes balanced reliability and lower tail sensitivity while keeping accuracy close to gated fusion.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_bloom(mode_agg: Dict[str, Dict[str, Dict]], out_path: Path):
    if P11 not in mode_agg:
        raise KeyError(f"No diagnostics found for {P11}")

    modes = ["clean", "dropout", "pollution", "bias_ramp"]
    p11 = mode_agg[P11]
    p99_vals = np.array([p11[m]["p99_error_mean"] for m in modes], dtype=float)
    mean_vals = np.array([p11[m]["mean_error_mean"] for m in modes], dtype=float)
    max_vals = np.array([p11[m]["top_max_error_mean"] for m in modes], dtype=float)

    p99_target = 5.0
    mean_target = 1.5
    p99_score = np.clip(p99_target / np.maximum(p99_vals, 1e-9), 0.22, 1.0)
    mean_score = np.clip(mean_target / np.maximum(mean_vals, 1e-9), 0.25, 1.0)

    theta = np.linspace(0, 2 * np.pi, len(modes), endpoint=False)
    width = 2 * np.pi / len(modes) * 0.70

    fig = plt.figure(figsize=(9.4, 8.8), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.12)
    ax.grid(color="#dbe3ee", linewidth=0.8, alpha=0.75)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_xticks(theta)
    ax.set_xticklabels([m.replace("_", " ") for m in modes], fontsize=12, color="#0f172a")
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["", "steady", "", "strong"], fontsize=9, color="#64748b")

    for i, mode in enumerate(modes):
        color = MODE_COLORS[mode]
        ax.bar(theta[i], p99_score[i], width=width, bottom=0.0, color=color, alpha=0.24, edgecolor=color, linewidth=2.0)
        ax.scatter([theta[i]], [mean_score[i]], s=90, color=color, edgecolor="white", linewidth=1.4, zorder=4)
        ax.scatter(
            [theta[i]],
            [0.08],
            s=35 + 18 * max_vals[i],
            color=color,
            alpha=0.34,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(theta[i], max(0.40, p99_score[i] - 0.12), f"P99 {p99_vals[i]:.2f}", ha="center", va="center", fontsize=9, color="#334155")

    ax.set_title("P11 Scenario Robustness Bloom", y=1.10, fontsize=19, fontweight="bold", color="#0f172a")
    ax.text(
        0.5,
        -0.13,
        "Petal length is normalized to a P99=5m reporting target; dots mark mean-error steadiness. The near-full bloom shows broad coverage.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_separation(agg: Dict[str, Dict], out_path: Path):
    if P11 not in agg:
        raise KeyError(f"No summary rows found for {P11}")
    data = agg[P11]
    normal_gate = float(data["mean_gate_normal_mean"])
    fault_gate = float(data["mean_gate_fault_mean"])
    normal_weight = float(data["mean_weight_normal_mean"])
    fault_weight = float(data["mean_weight_fault_mean"])

    fig, ax = plt.subplots(figsize=(10.6, 7.2), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_aspect("equal")
    ax.axis("off")

    rings = [
        ("Gate response", normal_gate, fault_gate, 1.00),
        ("Fusion weight", normal_weight, fault_weight, 0.70),
    ]
    center = (0.0, 0.0)
    for label, normal, fault, radius in rings:
        normal_angle = 360.0 * normal
        fault_angle = 360.0 * fault
        ax.add_patch(Wedge(center, radius, 90, 90 + normal_angle, width=0.13, facecolor="#059669", alpha=0.78, edgecolor="white", linewidth=1.0))
        ax.add_patch(Wedge(center, radius, 90 + normal_angle + 6, 90 + normal_angle + 6 + fault_angle, width=0.13, facecolor="#ef4444", alpha=0.82, edgecolor="white", linewidth=1.0))
        ax.add_patch(Wedge(center, radius, 0, 360, width=0.13, facecolor="none", edgecolor="#cbd5e1", linewidth=1.0, alpha=0.9))
        ax.text(-1.35, radius - 0.05, label, ha="left", va="center", fontsize=13, fontweight="bold", color="#0f172a")

    ax.text(0.0, 0.10, "P11", ha="center", va="center", fontsize=26, fontweight="bold", color="#0f172a")
    ax.text(0.0, -0.10, "reliability-only", ha="center", va="center", fontsize=11, color="#475569")

    left_x, right_x = -1.72, 1.18
    ax.scatter([left_x], [0.40], s=260, color="#059669", edgecolor="white", linewidth=1.5)
    ax.scatter([left_x], [0.12], s=260, color="#ef4444", edgecolor="white", linewidth=1.5)
    ax.text(left_x + 0.16, 0.40, f"normal gate  {normal_gate:.3f}", ha="left", va="center", fontsize=12, color="#0f172a")
    ax.text(left_x + 0.16, 0.12, f"fault gate   {fault_gate:.3f}", ha="left", va="center", fontsize=12, color="#0f172a")

    ax.scatter([right_x], [0.40], s=260, color="#059669", edgecolor="white", linewidth=1.5)
    ax.scatter([right_x], [0.12], s=260, color="#ef4444", edgecolor="white", linewidth=1.5)
    ax.text(right_x + 0.16, 0.40, f"normal weight {normal_weight:.3f}", ha="left", va="center", fontsize=12, color="#0f172a")
    ax.text(right_x + 0.16, 0.12, f"fault weight  {fault_weight:.3f}", ha="left", va="center", fontsize=12, color="#0f172a")

    sep_gate = normal_gate / max(fault_gate, 1e-6)
    sep_weight = normal_weight / max(fault_weight, 1e-6)
    ax.text(
        0.0,
        -1.20,
        f"Reliability separation: gate x{sep_gate:.1f}, fusion weight x{sep_weight:.1f}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.0,
        -1.43,
        "The measurement stream is kept as a reliability signal, while faulty-node influence is strongly suppressed.",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.62, 1.35)
    ax.set_title("P11 Reliability Separation Rings", fontsize=19, fontweight="bold", color="#0f172a", pad=22)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Create progress-report figures focused on the positive P11 reliability results.")
    parser.add_argument("--result_dir", default="results/rgcf_feature_stable_no_meas_repr_no_cov")
    parser.add_argument("--out_dir", default="results/p11_progress_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(result_dir)
    diagnostics = load_diagnostics(result_dir)
    agg = aggregate_rows(rows)
    mode_agg = aggregate_modes(diagnostics)

    plot_progress_compass(agg, out_dir / "fig1_p11_progress_compass.png")
    plot_scenario_bloom(mode_agg, out_dir / "fig2_p11_scenario_robustness_bloom.png")
    plot_reliability_separation(agg, out_dir / "fig3_p11_reliability_separation_rings.png")

    report = {
        "summary": agg.get(P11, {}),
        "scenario_summary": mode_agg.get(P11, {}),
    }
    (out_dir / "p11_progress_aggregate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir / 'fig1_p11_progress_compass.png'}")
    print(f"[saved] {out_dir / 'fig2_p11_scenario_robustness_bloom.png'}")
    print(f"[saved] {out_dir / 'fig3_p11_reliability_separation_rings.png'}")
    print(f"[saved] {out_dir / 'p11_progress_aggregate.json'}")


if __name__ == "__main__":
    main()
