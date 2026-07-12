"""Write the GO/NO-GO record for the V3.1 20-scene nominal-only closure."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v1 import MEASUREMENT_SEEDS, NOMINAL_PROTOCOL_NAME


REFERENCE = {
    "time_best": {"T1": 0.6990, "T2": 0.1805, "T3": 0.1205},
    "concordance": 0.8788888888888888,
    "worst_source_accuracy": 0.8900000000000001,
    "spearman": 0.5808394447160614,
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_value(rows: list[dict[str, str]], metric: str, score: str) -> float:
    for row in rows:
        if row.get("metric") == metric and row.get("score") == score:
            return float(row["median"])
    raise RuntimeError(f"Missing evidence summary metric={metric}, score={score}")


def _relative_spread_median(rows: list[dict[str, str]]) -> float:
    per_scene: dict[str, list[float]] = {}
    for row in rows:
        value = float(row["relative_spread_median"])
        if value == value:
            per_scene.setdefault(row["scenario_id"], []).append(value)
    return float(statistics.median(sum(values) / len(values) for values in per_scene.values()))


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    metadata = root / "metadata"
    nominal = metadata / "nominal_v1"
    cache_audit = json.loads((metadata / "av2_nominal_cache_audit.json").read_text(encoding="utf-8"))
    config = json.loads((metadata / "av2_nominal_config_hash.json").read_text(encoding="utf-8"))
    integrity = json.loads((nominal / "av2_nominal_integrity_results.json").read_text(encoding="utf-8"))
    best = _rows(nominal / "posterior_best_source_counts.csv")
    switching = _rows(nominal / "posterior_switching.csv")
    evidence_summary = _rows(nominal / "evidence_b3_across_scene_summary.csv")
    evidence_residual = _rows(nominal / "evidence_residual_statistics.csv")

    if cache_audit["protocol_name"] != NOMINAL_PROTOCOL_NAME or config["protocol_name"] != NOMINAL_PROTOCOL_NAME:
        raise RuntimeError("Nominal cache/config protocol mismatch")
    if cache_audit["condition"] != "nominal" or cache_audit["fault_variants_generated"] != 0 or cache_audit["faults_enabled"]:
        raise RuntimeError("Non-nominal cache audit detected")
    if cache_audit["scenes"] != 20 or cache_audit["measurement_seeds"] != list(MEASUREMENT_SEEDS):
        raise RuntimeError("Frozen scene/seeds contract mismatch")

    best_counts = {row["source"]: int(row["best_scene_count"]) for row in best}
    time_best = {source: sum(float(row[f"{source}_best_step_fraction"]) for row in switching) / len(switching) for source in ("T1", "T2", "T3")}
    switching_ratio = sum(float(row["switch_count_mean"]) > 0.0 for row in switching) / len(switching)
    evidence = {
        "combined_concordance": _summary_value(evidence_summary, "pairwise_concordance", "combined"),
        "combined_worst_source_accuracy": _summary_value(evidence_summary, "worst_source_accuracy", "combined"),
        "combined_spearman": _summary_value(evidence_summary, "spearman_position_error", "combined"),
        "relative_residual_spread": _relative_spread_median(evidence_residual),
    }
    consistency = {
        "time_best_fraction_abs_difference": {source: abs(time_best[source] - REFERENCE["time_best"][source]) for source in time_best},
        "concordance_abs_difference": abs(evidence["combined_concordance"] - REFERENCE["concordance"]),
        "worst_source_accuracy_abs_difference": abs(evidence["combined_worst_source_accuracy"] - REFERENCE["worst_source_accuracy"]),
        "spearman_abs_difference": abs(evidence["combined_spearman"] - REFERENCE["spearman"]),
    }
    gates = {
        "protocol_is_nominal_only": True,
        "fault_variants_generated_zero": cache_audit["fault_variants_generated"] == 0,
        "scene_count_20": cache_audit["scenes"] == 20,
        "measurement_seeds_frozen": cache_audit["measurement_seeds"] == list(MEASUREMENT_SEEDS),
        "finite_ekf_output_rate": cache_audit["finite_feature_runs"] == 60,
        "positive_definite_covariance_rate": cache_audit["positive_definite_covariance_rate_generated_runs"] is not None and cache_audit["positive_definite_covariance_rate_generated_runs"] >= 0.999,
        "time_best_sources_at_least_10pct": sum(value >= 0.10 for value in time_best.values()) >= 2,
        "switching_scene_ratio": switching_ratio >= 0.25,
        "combined_concordance": evidence["combined_concordance"] >= 0.55,
        "combined_worst_source_accuracy": evidence["combined_worst_source_accuracy"] >= 0.40,
        "future_perturbation": integrity["future_perturbation_test"],
        "metadata_isolation": integrity["metadata_isolation_test"],
        "split_isolation": integrity["split_isolation_test"],
        "pair_locality": integrity["evidence_pair_column_locality"] and integrity["posterior_pair_row_locality"],
        "feature_shape": integrity["feature_shape_test"],
        "finite_feature_rate": integrity["finite_feature_rate"] == 1.0,
        "time_best_reproducibility": all(value <= 0.02 for value in consistency["time_best_fraction_abs_difference"].values()),
        "concordance_reproducibility": consistency["concordance_abs_difference"] <= 0.03,
        "worst_source_accuracy_reproducibility": consistency["worst_source_accuracy_abs_difference"] <= 0.03,
        "spearman_reproducibility": consistency["spearman_abs_difference"] <= 0.05,
    }
    go = all(gates.values())
    summary = {
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "condition": "nominal",
        "cache_audit": cache_audit,
        "sensor_config_hash": config["sensor_config_hash"],
        "posterior": {"best_scene_counts": best_counts, "time_best_fraction": time_best, "switching_scene_ratio": switching_ratio},
        "evidence": evidence,
        "integrity": {key: integrity[key] for key in ("future_perturbation_test", "metadata_isolation_test", "split_isolation_test", "evidence_pair_column_locality", "posterior_pair_row_locality", "measurement_identity_alignment", "feature_shape_test", "finite_feature_rate", "pass")},
        "reproducibility": consistency,
        "gates": gates,
        "small_nominal_test": "GO" if go else "NO-GO",
        "level_c_authorized": go,
    }
    (metadata / "av2_nominal_small_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# AV2 Nominal Small Test Report",
        "",
        "- Protocol: `AV2_NOMINAL_1000_V3.1`",
        "- Condition: nominal-only; fault variants generated: 0.",
        "- Frozen input: 20 official-train focal VEHICLE scenes; measurement seeds [100, 101, 102].",
        "",
        "## Results",
        "",
        f"- Time-step best fractions: T1={time_best['T1']:.2%}, T2={time_best['T2']:.2%}, T3={time_best['T3']:.2%}.",
        f"- Switching-scene ratio: {switching_ratio:.2%}.",
        f"- Combined evidence concordance / worst-source accuracy / Spearman: {evidence['combined_concordance']:.4f} / {evidence['combined_worst_source_accuracy']:.4f} / {evidence['combined_spearman']:.4f}.",
        f"- Relative residual-spread median: {evidence['relative_residual_spread']:.4f}.",
        f"- Engineering and causality/interface gates: {'PASS' if all(gates.values()) else 'FAIL'}.",
        "",
        "## Decision",
        "",
        "- OLD LEVEL B+ REPORT: ARCHIVED",
        f"- SMALL NOMINAL-ONLY TEST: {'GO' if go else 'NO-GO'}",
        f"- LEVEL C AUTHORIZED: {'YES' if go else 'NO'}",
    ]
    report_path = root / "reports" / "AV2_NOMINAL_SMALL_TEST_REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the AV2 V3.1 nominal-only small-test report.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
