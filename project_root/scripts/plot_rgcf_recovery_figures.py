from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge


VARIANT_LABELS = {
    "P0_post_only_single_stream": "P0 post-only",
    "P1_dual_stream_direct": "P1 direct dual-stream",
    "P2_gate_feature_path_only": "P2 gate-feature",
    "P3_gate_weight_path_only": "P3 gate-weight",
    "P4_full_gate_no_cov_formula": "P4 full gate",
    "P8_full_gate_aa_formula": "P8 AA full gate",
    "P11_feature_stable_reliability_no_cov": "P11 reliability-only",
    "P12_recovery_aware_full_gate_no_cov": "P12 recovery-aware",
    "M3_V5_current": "M3 current",
    "M4_V5_peer_consistency": "M4 peer-consistency",
}

VARIANT_COLORS = {
    "P0_post_only_single_stream": "#6b7280",
    "P1_dual_stream_direct": "#7c3aed",
    "P2_gate_feature_path_only": "#0891b2",
    "P3_gate_weight_path_only": "#16a34a",
    "P4_full_gate_no_cov_formula": "#2563eb",
    "P8_full_gate_aa_formula": "#9333ea",
    "P11_feature_stable_reliability_no_cov": "#059669",
    "P12_recovery_aware_full_gate_no_cov": "#dc2626",
    "M3_V5_current": "#ea580c",
    "M4_V5_peer_consistency": "#0f766e",
}

MODE_COLORS = {
    "bias_ramp": "#dc2626",
    "pollution": "#d97706",
    "dropout": "#2563eb",
    "clean": "#059669",
}


def _finite_float(value):
    if value in ("", None):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def _split_group_variant(name: str):
    if " | " in name:
        group, variant = name.split(" | ", 1)
        return group, variant
    return "", name


def _short_variant(name: str) -> str:
    group, variant = _split_group_variant(name)
    label = VARIANT_LABELS.get(variant, variant)
    return f"{group} {label}".strip()


def _color_variant(name: str) -> str:
    _, variant = _split_group_variant(name)
    return VARIANT_COLORS.get(variant, "#111827")


def _selected_variant(variant: str, selected: set[str] | None) -> bool:
    _, base_variant = _split_group_variant(variant)
    return selected is None or base_variant in selected or variant in selected


def load_result_dirs(result_dirs: List[Path], *, dir_labels: List[str] | None = None, selected: set[str] | None = None):
    rows = []
    diagnostics = defaultdict(list)
    use_dir_labels = bool(dir_labels)
    for idx, result_dir in enumerate(result_dirs):
        group = dir_labels[idx] if dir_labels and idx < len(dir_labels) else result_dir.name
        summary_path = result_dir / "ablation_runs_summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                for row in json.load(f):
                    variant = str(row["variant"])
                    if not _selected_variant(variant, selected):
                        continue
                    row = dict(row)
                    if use_dir_labels:
                        row["variant"] = f"{group} | {variant}"
                    rows.append(row)

        diag_dir = result_dir / "tail_diagnostics"
        if diag_dir.exists():
            for path in sorted(diag_dir.glob("*.json")):
                with path.open("r", encoding="utf-8") as f:
                    diag = json.load(f)
                label = str(diag.get("run_label", path.stem))
                if "__trainseed" in label:
                    variant = label.split("__trainseed", 1)[0]
                else:
                    variant = label
                if not _selected_variant(variant, selected):
                    continue
                if use_dir_labels:
                    variant = f"{group} | {variant}"
                diagnostics[variant].append(diag)
    return rows, diagnostics


def aggregate_rows(rows: List[Dict]) -> Dict[str, Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)

    out = {}
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
        "p99_active_fault_rate",
    ]
    for variant, items in grouped.items():
        agg = {"variant": variant, "n": len(items)}
        for metric in metrics:
            vals = [_finite_float(item.get(metric)) for item in items]
            vals = [v for v in vals if v is not None]
            agg[f"{metric}_mean"] = _mean(vals)
            agg[f"{metric}_std"] = _std(vals)
            agg[f"{metric}_max"] = max(vals) if vals else float("nan")
        out[variant] = agg
    return out


def _metric_badness(agg: Dict[str, Dict], key: str) -> Dict[str, float]:
    vals = [v.get(key, float("nan")) for v in agg.values()]
    vals = np.array([x for x in vals if math.isfinite(float(x))], dtype=float)
    lo = float(np.min(vals)) if len(vals) else 0.0
    hi = float(np.max(vals)) if len(vals) else 1.0
    span = max(hi - lo, 1e-9)
    return {name: (float(data.get(key, lo)) - lo) / span for name, data in agg.items()}


def plot_reliability_compass(agg: Dict[str, Dict], out_path: Path):
    metrics = [
        ("Mean error", "test_loss_pos_mean"),
        ("P99 tail", "full_p99_error_mean"),
        ("Worst spike", "full_max_error_mean"),
        ("Bias spike", "bias_ramp_top_max_error_mean"),
        ("Seed jitter", "test_loss_pos_std"),
    ]
    badness = {metric: _metric_badness(agg, metric) for _, metric in metrics}
    angles = np.linspace(0.0, 2.0 * np.pi, len(metrics), endpoint=False)
    angles = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(9.4, 8.8), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.grid(color="#cbd5e1", linewidth=0.8, alpha=0.75)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["low", "", "", "high"], fontsize=9, color="#64748b")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name for name, _ in metrics], fontsize=11, color="#0f172a")

    for variant, data in sorted(agg.items()):
        values = [badness[metric][variant] for _, metric in metrics]
        values = np.r_[values, values[0]]
        color = _color_variant(variant)
        ax.plot(angles, values, color=color, linewidth=2.4, label=_short_variant(variant))
        ax.fill(angles, values, color=color, alpha=0.10)
        ax.scatter(angles[:-1], values[:-1], s=42, color=color, edgecolor="white", linewidth=1.2, zorder=3)

    ax.set_title("Reliability Risk Compass", y=1.10, fontsize=18, fontweight="bold", color="#0f172a")
    ax.text(
        0.5,
        -0.13,
        "All axes are normalized risk: smaller enclosed area means better mean accuracy, tail behavior, and seed stability.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fault_mode_orbits(diagnostics: Dict[str, List[Dict]], out_path: Path):
    modes = ["clean", "dropout", "pollution", "bias_ramp"]
    theta = np.linspace(0, 2 * np.pi, len(modes), endpoint=False)
    offsets = np.linspace(-0.10, 0.10, max(len(diagnostics), 1))

    max_radius = 1.0
    mode_stats = defaultdict(dict)
    for variant, items in diagnostics.items():
        for mode in modes:
            p99_vals = []
            max_vals = []
            for diag in items:
                grouped = diag.get("grouped_by_mode", {})
                if mode in grouped:
                    p99_vals.append(_finite_float(grouped[mode].get("p99_error")))
                    max_vals.append(_finite_float(grouped[mode].get("top_max_error")))
            mode_stats[variant][mode] = {
                "p99": _mean([v for v in p99_vals if v is not None]),
                "max": _mean([v for v in max_vals if v is not None]),
            }
            if math.isfinite(mode_stats[variant][mode]["max"]):
                max_radius = max(max_radius, mode_stats[variant][mode]["max"])

    fig = plt.figure(figsize=(10.2, 9.0), facecolor="#f8fafc")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#f8fafc")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max_radius * 1.12)
    ax.grid(color="#cbd5e1", linewidth=0.8, alpha=0.8)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_xticks(theta)
    ax.set_xticklabels([m.replace("_", " ") for m in modes], fontsize=12, color="#0f172a")
    ax.tick_params(axis="y", labelsize=9, colors="#64748b")

    for j, (variant, stats) in enumerate(sorted(mode_stats.items())):
        color = _color_variant(variant)
        t = theta + offsets[j % len(offsets)]
        p99 = np.array([stats[m]["p99"] for m in modes], dtype=float)
        worst = np.array([stats[m]["max"] for m in modes], dtype=float)
        p99_closed = np.r_[p99, p99[0]]
        t_closed = np.r_[t, t[0]]
        ax.plot(t_closed, p99_closed, color=color, linewidth=2.2, alpha=0.9, label=_short_variant(variant))
        for ti, r1, r2, mode in zip(t, p99, worst, modes):
            if not math.isfinite(r1) or not math.isfinite(r2):
                continue
            ax.plot([ti, ti], [r1, r2], color=color, linewidth=1.8, alpha=0.55)
            ax.scatter([ti], [r1], s=42, color=color, edgecolor="white", linewidth=1.0, zorder=3)
            ax.scatter(
                [ti],
                [r2],
                s=max(60, min(280, 20 + r2 * 4)),
                color=MODE_COLORS.get(mode, color),
                alpha=0.65,
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
            )

    ax.set_title("Fault-Mode Tail Orbits", y=1.10, fontsize=18, fontweight="bold", color="#0f172a")
    ax.text(
        0.5,
        -0.13,
        "Each orbit is mean P99 error; outward bubbles mark worst spikes. A long bias-ramp spoke is the residual-risk signature.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residual_recovery_clock(diagnostics: Dict[str, List[Dict]], out_path: Path):
    selected = [v for v in diagnostics if v in VARIANT_LABELS]
    selected = sorted(selected, key=lambda x: list(VARIANT_LABELS).index(x) if x in VARIANT_LABELS else 99)
    selected = selected or sorted(diagnostics)
    n = len(selected)
    cols = min(2, max(1, n))
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(5.2 * cols, 4.9 * rows), facecolor="#f8fafc")

    max_err = 1.0
    for variant in selected:
        for diag in diagnostics[variant]:
            for row in diag.get("top20", []):
                if row.get("mode") == "bias_ramp":
                    max_err = max(max_err, float(row.get("error_pos", 0.0)))

    for idx, variant in enumerate(selected, 1):
        ax = fig.add_subplot(rows, cols, idx, polar=True)
        ax.set_facecolor("#f8fafc")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.05)
        ax.grid(color="#cbd5e1", linewidth=0.7, alpha=0.75)
        ax.spines["polar"].set_color("#94a3b8")
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="#64748b")
        ax.set_xticks(np.deg2rad([0, 60, 120, 180, 240, 300]))
        ax.set_xticklabels(["70s", "+2s", "+4s", "+6s", "-4s", "-2s"], fontsize=8, color="#64748b")

        for diag in diagnostics[variant]:
            for row in diag.get("top20", []):
                if row.get("mode") != "bias_ramp":
                    continue
                t = _finite_float(row.get("t"))
                w = _finite_float(row.get("fault_sensor_weight"))
                err = _finite_float(row.get("error_pos"))
                active = _finite_float(row.get("fault_active_any"))
                if t is None or w is None or err is None:
                    continue
                theta = ((t - 70.0) / 6.0) * np.pi
                size = 35.0 + 260.0 * (err / max_err) ** 1.2
                color = "#ef4444" if active and active > 0.5 else _color_variant(variant)
                alpha = 0.82 if active and active > 0.5 else 0.62
                ax.scatter(theta, min(max(w, 0.0), 1.0), s=size, color=color, alpha=alpha, edgecolor="white", linewidth=0.8)

        ax.add_patch(Wedge((0.0, 0.0), 1.02, -2, 2, transform=ax.transData._b, color="#0f172a", alpha=0.18))
        ax.add_patch(Circle((0.0, 0.0), 0.10, transform=ax.transData._b, color="#0f172a", alpha=0.08))
        ax.set_title(_short_variant(variant), fontsize=12, fontweight="bold", color="#0f172a", pad=18)

    fig.suptitle("Bias-Ramp Residual Recovery Clock", fontsize=18, fontweight="bold", color="#0f172a", y=1.02)
    fig.subplots_adjust(top=0.88, bottom=0.10, hspace=0.42, wspace=0.26)
    fig.text(
        0.5,
        0.01,
        "Angle is time around the 70s fault boundary; radius is the faulty sensor fusion weight; bubble size is position error.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw non-standard RGCF recovery/tail diagnostic figures.")
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        default=["results/rgcf_feature_stable_no_meas_repr_no_cov"],
        help="One or more result directories containing ablation_runs_summary.json and tail_diagnostics/.",
    )
    parser.add_argument("--out_dir", default="results/rgcf_recovery_figures")
    parser.add_argument(
        "--dir_labels",
        nargs="*",
        default=None,
        help="Optional labels for result_dirs. When set, variants are kept separate as '<label> | <variant>'.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant filter, useful when comparing two large result groups.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_dirs = [Path(p) for p in args.result_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {x.strip() for x in str(args.variants).split(",") if x.strip()} or None
    rows, diagnostics = load_result_dirs(result_dirs, dir_labels=args.dir_labels, selected=selected)
    if not rows:
        raise SystemExit("No ablation_runs_summary.json rows found.")
    agg = aggregate_rows(rows)

    plot_reliability_compass(agg, out_dir / "fig1_reliability_risk_compass.png")
    plot_fault_mode_orbits(diagnostics, out_dir / "fig2_fault_mode_tail_orbits.png")
    plot_residual_recovery_clock(diagnostics, out_dir / "fig3_bias_ramp_recovery_clock.png")

    aggregate_path = out_dir / "plot_aggregate_metrics.json"
    aggregate_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir / 'fig1_reliability_risk_compass.png'}")
    print(f"[saved] {out_dir / 'fig2_fault_mode_tail_orbits.png'}")
    print(f"[saved] {out_dir / 'fig3_bias_ramp_recovery_clock.png'}")
    print(f"[saved] {aggregate_path}")


if __name__ == "__main__":
    main()
