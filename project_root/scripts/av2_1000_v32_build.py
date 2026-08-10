"""Build corrected AV2 V3.2 sim caches and feature shards in isolated paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v32 import (
    ARTIFACT_KEY,
    NOMINAL_PROTOCOL,
    NOMINAL_PROTOCOL_NAME,
    WARMUP_SECONDS,
    config_sha256,
    validate_native_noise_contract,
)
from tools.av2_1000.cache import build_feature_shards, build_sim
from tools.av2_1000.common import formal_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate unit-corrected, scenario-independent AV2 V3.2 artifacts."
    )
    parser.add_argument("stage", choices=("sim", "features", "all"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    args = parser.parse_args()

    validate_native_noise_contract()
    paths = formal_paths(args.root, artifact_key=ARTIFACT_KEY)
    digest = config_sha256()
    result: dict[str, object] = {
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "artifact_key": ARTIFACT_KEY,
        "config_sha256": digest,
    }
    if args.stage in ("sim", "all"):
        result["sim"] = build_sim(
            paths["manifests"],
            paths["truth"],
            paths["sim"],
            protocol=NOMINAL_PROTOCOL,
            protocol_name=NOMINAL_PROTOCOL_NAME,
            config_sha256=digest,
            scenario_scoped_rng=True,
        )
    if args.stage in ("features", "all"):
        result["features"] = build_feature_shards(
            paths["manifests"],
            paths["sim"],
            paths["features"],
            args.shard_size,
            protocol=NOMINAL_PROTOCOL,
            protocol_name=NOMINAL_PROTOCOL_NAME,
            config_sha256=digest,
            warmup_seconds=WARMUP_SECONDS,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
