from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_TYPES = ["gps2d", "radar_rb", "aoa_only", "uwb_range_only"]
EXPECTED_POS = {
    "S1": [(-800.0, -200.0), (900.0, -100.0), (-300.0, 900.0), (700.0, 850.0)],
    "S2": [(-850.0, -350.0), (-350.0, -250.0), (-700.0, 450.0), (-150.0, 350.0)],
    "S3": [(-800.0, -200.0), (900.0, -100.0), (-300.0, 900.0), (700.0, 850.0)],
}
PRESETS = {
    "S1": "phase1_s1_balanced_hetero_nominal",
    "S2": "phase1_s2_clustered_hetero_nominal",
    "S3": "phase1_s3_maneuver_hetero_nominal",
}


def _check_imports() -> None:
    import numpy
    import pandas
    import scipy
    import sklearn
    import matplotlib
    import torch

    print("[python]", sys.version.replace("\n", " "))
    print("[numpy]", numpy.__version__)
    print("[pandas]", pandas.__version__)
    print("[scipy]", scipy.__version__)
    print("[sklearn]", sklearn.__version__)
    print("[matplotlib]", matplotlib.__version__)
    print("[torch]", torch.__version__)
    print("[torch.cuda]", torch.cuda.is_available())
    print("[torch.version.cuda]", torch.version.cuda)
    if torch.cuda.is_available():
        print("[gpu]", torch.cuda.get_device_name(0))


def _check_phase1_scenes() -> None:
    from core.config_loader import load_experiment_bundle
    from core.types import RunRequest
    from simulation.scenario_factory import build_scenario_from_bundle

    for scene_id, preset in PRESETS.items():
        bundle = load_experiment_bundle(
            RunRequest(mode="simulate", preset_name=preset, device="cpu")
        )
        artifacts = build_scenario_from_bundle(bundle).build()
        types = [node.sensor_type for node in artifacts.sensor_layout.sensors]
        pos = [tuple(map(float, node.position)) for node in artifacts.sensor_layout.sensors]
        truth_last = artifacts.truth.x4[-1, :2].round(3).tolist()

        print(f"[{scene_id}] preset={preset}")
        print(f"[{scene_id}] types={types}")
        print(f"[{scene_id}] pos={pos}")
        print(f"[{scene_id}] truth_last_xy={truth_last}")

        if types != EXPECTED_TYPES:
            raise RuntimeError(f"{scene_id} sensor types mismatch: {types}")
        if pos != EXPECTED_POS[scene_id]:
            raise RuntimeError(f"{scene_id} sensor positions mismatch: {pos}")

    if EXPECTED_POS["S1"] == EXPECTED_POS["S2"]:
        raise RuntimeError("S1 and S2 expected positions are identical; check the test script.")
    print("[phase1_scenes] ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU machine preflight check.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if torch.cuda.is_available() is false.",
    )
    args = parser.parse_args()

    _check_imports()

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false.")

    _check_phase1_scenes()
    print("[preflight] ok")


if __name__ == "__main__":
    main()
