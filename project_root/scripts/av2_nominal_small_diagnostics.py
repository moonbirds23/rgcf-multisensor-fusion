"""Run V3.1 B1--B3 and nominal-only integrity diagnostics on frozen inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v1 import MEASUREMENT_SEEDS, NOMINAL_PROTOCOL_NAME
from scripts.av2_level_b_plus_evidence import run as run_evidence
from scripts.av2_nominal_small_integrity import run as run_integrity
from scripts.av2_small_level_b_plus_b1_b2 import run as run_motion_posterior


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    metadata_dir = root / "metadata" / "nominal_v1"
    plot_dir = root / "plots" / "nominal_v1"
    log_dir = root / "logs" / "nominal_v1"
    run_motion_posterior(
        root,
        MEASUREMENT_SEEDS,
        output_metadata_dir=metadata_dir,
        output_plot_dir=plot_dir,
        protocol_name=NOMINAL_PROTOCOL_NAME,
    )
    evidence = run_evidence(
        root,
        count=20,
        measurement_seeds=MEASUREMENT_SEEDS,
        output_metadata_dir=metadata_dir,
        output_plot_dir=plot_dir,
        output_log_dir=log_dir,
        protocol_name=NOMINAL_PROTOCOL_NAME,
    )
    integrity = run_integrity(root, metadata_dir / "av2_nominal_integrity_results.json")
    result = {"protocol_name": NOMINAL_PROTOCOL_NAME, "evidence": evidence, "integrity_pass": integrity["pass"]}
    (log_dir / "nominal_diagnostics_completed.json").parent.mkdir(parents=True, exist_ok=True)
    (log_dir / "nominal_diagnostics_completed.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AV2 V3.1 nominal-only small-scene diagnostics.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
