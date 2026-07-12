"""Fail-fast GPU-host preflight for the AV2 V3.1 nominal Level C stage."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run(root: Path) -> dict[str, object]:
    import av2
    import matplotlib
    import numpy
    import pyarrow
    import scipy
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("AV2 V3.1 Level C requires CUDA; CPU fallback is forbidden")
    from configs.av2_nominal_1000_v1 import MEASUREMENT_SEEDS, NOMINAL_PROTOCOL_NAME
    from models.gnn_fusion import MeasurementEvaluatedRGCFA0Directional

    root = root.resolve()
    manifest_path = root / "manifests" / "level_b_20.csv"
    summary_path = root / "metadata" / "av2_nominal_small_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("V3.1 Level C requires the completed nominal small-test artifacts")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(rows) != 20 or len({row.get("scenario_id") for row in rows}) != 20:
        raise RuntimeError("Frozen level_b_20 manifest is invalid")
    if summary.get("small_nominal_test") != "GO" or not summary.get("level_c_authorized"):
        raise RuntimeError("V3.1 nominal-only GO is required before Level C")
    # Instantiate only; no data is scanned and no GPU work is launched.
    MeasurementEvaluatedRGCFA0Directional(
        post_in_dim=9, meas_in_dim=18, evidence_in_dim=16, pair_dim=8,
        hidden_dim=64, meas_hidden_dim=64, output_fusion_mode="info_diag",
    )
    result = {
        "status": "PASS",
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "root": str(root),
        "frozen_scenes": len(rows),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyarrow": pyarrow.__version__,
        "av2": getattr(av2, "__version__", "0.2.1"),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the GPU host for AV2 V3.1 Level C.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
