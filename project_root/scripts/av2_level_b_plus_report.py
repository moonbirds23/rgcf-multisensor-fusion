from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_level_b_plus_v11 import LEVEL_B_PLUS_PROTOCOL_VERSION


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_value(rows: list[dict[str, str]], metric: str, score: str) -> float:
    for row in rows:
        if row["metric"] == metric and row.get("score", "") == score:
            return float(row["median"])
    raise KeyError(f"Missing B3 summary metric={metric} score={score}")


def _combined_relative_spread_median(rows: list[dict[str, str]]) -> float:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        value = float(row["relative_spread_median"])
        if value == value:
            by_scene.setdefault(row["scenario_id"], []).append(value)
    return float(statistics.median(sum(values) / len(values) for values in by_scene.values()))


def run(root: Path) -> dict[str, Any]:
    metadata = root / "metadata" / "level_b_plus"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    motion = _read_csv(metadata / "av2_motion_statistics.csv")
    best = _read_csv(metadata / "posterior_best_source_counts.csv")
    switching = _read_csv(metadata / "posterior_switching.csv")
    b3 = _read_csv(metadata / "evidence_b3_across_scene_summary.csv")
    residual_rows = _read_csv(metadata / "evidence_residual_statistics.csv")
    faults = json.loads((metadata / "fault_directionality_summary.json").read_text(encoding="utf-8"))
    causality = json.loads((metadata / "causality_test_results.json").read_text(encoding="utf-8"))["summary"]
    locality = json.loads((metadata / "pair_feature_locality.json").read_text(encoding="utf-8"))

    heading_over_15 = sum(float(row["heading_change_total"]) > 15.0 * 3.141592653589793 / 180.0 for row in motion)
    speed_over_3 = sum(float(row["speed_range"]) > 3.0 for row in motion)
    best_counts = {row["source"]: int(row["best_scene_count"]) for row in best}
    time_best = {
        source: sum(float(row[f"{source}_best_step_fraction"]) for row in switching) / len(switching)
        for source in ("T1", "T2", "T3")
    }
    switching_scenes = sum(float(row["switch_count_mean"]) > 0.0 for row in switching)
    scene_best_fraction = {
        source: best_counts.get(source, 0) / len(switching)
        for source in ("T1", "T2", "T3")
    }
    # Frozen V1.1 GPU-entry gate.  Equality at 90% is allowed; exceeding it
    # for any posterior is not.  Time-step balance requires at least two
    # posterior sources to lead on at least 10% of evaluated time steps.
    scene_best_balance_pass = all(fraction <= 0.90 for fraction in scene_best_fraction.values())
    time_best_sources_at_least_10pct = [
        source for source, fraction in time_best.items() if fraction >= 0.10
    ]
    time_best_balance_pass = len(time_best_sources_at_least_10pct) >= 2
    b3_metrics = {
        "combined_pairwise_concordance_median": _summary_value(b3, "pairwise_concordance", "combined"),
        "combined_worst_source_accuracy_median": _summary_value(b3, "worst_source_accuracy", "combined"),
        "combined_spearman_median": _summary_value(b3, "spearman_position_error", "combined"),
        "median_relative_residual_spread": _combined_relative_spread_median(residual_rows),
    }
    b3_pass = (
        b3_metrics["combined_pairwise_concordance_median"] >= 0.55
        and b3_metrics["combined_worst_source_accuracy_median"] >= 0.40
        and b3_metrics["median_relative_residual_spread"] > 0.0
    )
    hard_integrity = all(
        (
            causality["future_perturbation_test"],
            causality["delay_direction_test"],
            causality["metadata_isolation_test"],
            causality["split_isolation_test"],
            locality["evidence_pair_column_locality"],
            locality["posterior_pair_row_locality"],
            locality["measurement_identity_alignment"],
            locality["feature_shape_test"],
            locality["finite_feature_rate"] == 1.0,
            faults["hard_interface_checks_pass"],
        )
    )
    gpu_smoke_training_authorized = bool(
        scene_best_balance_pass
        and time_best_balance_pass
        and b3_pass
        and faults["bias_successful_runs_gate"]["actual"] >= 12
        and hard_integrity
    )
    decision = "GO" if gpu_smoke_training_authorized else "NO-GO"
    summary = {
        "protocol": LEVEL_B_PLUS_PROTOCOL_VERSION,
        "scenes": len(motion),
        "measurement_seeds": [100, 101, 102],
        "motion": {"heading_change_over_15deg": heading_over_15, "speed_range_over_3mps": speed_over_3},
        "posterior": {
            "best_scene_counts": best_counts,
            "scene_best_fraction": scene_best_fraction,
            "mean_time_best_fraction": time_best,
            "switching_scenes": switching_scenes,
            "scene_best_balance_pass": scene_best_balance_pass,
            "time_best_sources_at_least_10pct": time_best_sources_at_least_10pct,
            "time_best_balance_pass": time_best_balance_pass,
        },
        "evidence": {**b3_metrics, "pass": b3_pass},
        "faults": faults,
        "causality": causality,
        "pair_locality": locality,
        "hard_integrity_pass": hard_integrity,
        "gpu_smoke_training_authorized": gpu_smoke_training_authorized,
        "decision": decision,
    }
    (metadata / "level_b_plus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Level B+ Report",
        "",
        "## 1. Protocol and frozen scenes",
        "",
        f"- Protocol: `{LEVEL_B_PLUS_PROTOCOL_VERSION}`",
        "- Data: 20 official-train focal VEHICLE scenes; measurement seeds 100, 101, 102.",
        "- Aggregation: per timestep → per scene/seed → seed average within scene → across-scene summary.",
        "",
        "## 2. AV2 motion statistics",
        "",
        f"- Scenes with cumulative heading change >15°: {heading_over_15}/20.",
        f"- Scenes with speed range >3 m/s: {speed_over_3}/20.",
        "",
        "## 3. Posterior complementarity",
        "",
        f"- Scene-best counts: T1={best_counts.get('T1', 0)}, T2={best_counts.get('T2', 0)}, T3={best_counts.get('T3', 0)}.",
        f"- Mean time-step best fractions: T1={time_best['T1']:.2%}, T2={time_best['T2']:.2%}, T3={time_best['T3']:.2%}.",
        f"- Scenes with at least one best-source switch: {switching_scenes}/20.",
        f"- Scene-best fractions: T1={scene_best_fraction['T1']:.2%}, T2={scene_best_fraction['T2']:.2%}, T3={scene_best_fraction['T3']:.2%}.",
        f"- Scene-best balance gate (each <=90%): {'PASS' if scene_best_balance_pass else 'FAIL'}.",
        f"- Time-best sources at >=10%: {', '.join(time_best_sources_at_least_10pct) or 'none'}; gate (at least two): {'PASS' if time_best_balance_pass else 'FAIL'}.",
        "",
        "## 4. Evidence discriminability",
        "",
        f"- Combined concordance median: {b3_metrics['combined_pairwise_concordance_median']:.4f}.",
        f"- Combined worst-source accuracy median: {b3_metrics['combined_worst_source_accuracy_median']:.4f}.",
        f"- Combined error-score Spearman median: {b3_metrics['combined_spearman_median']:.4f}.",
        f"- Combined relative residual-spread median: {b3_metrics['median_relative_residual_spread']:.4f}.",
        f"- B3 result: {'PASS' if b3_pass else 'FAIL'}.",
        "",
        "## 5. Single-fault directionality",
        "",
        f"- Dropout, delay, and both covariance-mismatch interface tests: {'PASS' if faults['hard_interface_checks_pass'] else 'FAIL'}.",
        f"- T2 bias successful runs: {faults['bias_successful_runs_gate']['actual']}/15; required 12/15.",
        f"- B4 result: {'PASS' if faults['b4_pass'] else 'FAIL'}.",
        "",
        "## 6. Causality tests",
        "",
        f"- Future perturbation: {'PASS' if causality['future_perturbation_test'] else 'FAIL'}; maximum prefix difference {causality['future_perturbation_max_abs_difference']:.1e}.",
        f"- Delay direction: {'PASS' if causality['delay_direction_test'] else 'FAIL'}; future accesses {causality['future_measurement_access_count']}.",
        f"- Metadata isolation: {'PASS' if causality['metadata_isolation_test'] else 'FAIL'}.",
        f"- Split isolation: {'PASS' if causality['split_isolation_test'] else 'FAIL'}.",
        "",
        "## 7. Pair-level feature locality",
        "",
        f"- Evidence pair-column locality: {'PASS' if locality['evidence_pair_column_locality'] else 'FAIL'}.",
        f"- Posterior pair-row locality: {'PASS' if locality['posterior_pair_row_locality'] else 'FAIL'}.",
        f"- Measurement identity alignment: {'PASS' if locality['measurement_identity_alignment'] else 'FAIL'}.",
        "",
        "## 8. Failures and anomalies",
        "",
        "- The report preserves all original B+ evidence and interface thresholds; the V1.1 GPU-entry balance and T2-bias gates are evaluated without tuning after results.",
        "",
        "## 9. Decision",
        "",
        f"- **{decision}**",
        f"- GPU smoke training authorized: **{'YES' if gpu_smoke_training_authorized else 'NO'}**.",
        "- Do not start Level C GPU training unless this report is `GO`. Any further protocol change requires an explicitly frozen successor configuration and a full B1–B6 rerun on the same scenes and seeds.",
        "",
    ]
    (reports / "LEVEL_B_PLUS_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the unified AV2 Level B+ report.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
