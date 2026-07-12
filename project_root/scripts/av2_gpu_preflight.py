"""Strict CUDA-only preflight for the external AV2 experiment storage.

This command validates paths and runtime metadata only.  It never enumerates
raw AV2 files, builds a cache, or starts a training job.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.av2.paths import (  # noqa: E402
    Av2Paths,
    add_av2_data_root_argument,
    prepare_av2_output_dir,
    resolve_av2_paths,
)


ENVIRONMENT_RELATIVE_DIR = Path("environment")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate AV2 mounted storage and CUDA; no AV2 data is scanned."
    )
    add_av2_data_root_argument(parser)
    return parser


def _require_cuda(torch_module: Any) -> None:
    """Reject unavailable CUDA instead of silently using CPU."""

    if not bool(torch_module.cuda.is_available()):
        raise RuntimeError(
            "AV2 GPU preflight requires CUDA, but torch.cuda.is_available() is false. "
            "CPU fallback is intentionally disabled."
        )
    if int(torch_module.cuda.device_count()) < 1:
        raise RuntimeError("CUDA is reported available but no CUDA devices were found.")


def collect_preflight_metadata(torch_module: Any, paths: Av2Paths) -> dict[str, Any]:
    """Collect runtime and device facts without touching AV2 raw contents."""

    _require_cuda(torch_module)
    devices = []
    for index in range(int(torch_module.cuda.device_count())):
        properties = torch_module.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": str(torch_module.cuda.get_device_name(index)),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "root": str(paths.root),
            "raw_root": str(paths.raw_root),
            "environment_dir": str(paths.root / ENVIRONMENT_RELATIVE_DIR),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": str(torch_module.__version__),
            "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        },
        "cuda": {"available": True, "device_count": len(devices), "devices": devices},
    }


def write_preflight_metadata(paths: Av2Paths, metadata: dict[str, Any]) -> Path:
    """Write one JSON record to the sole output directory used by preflight."""

    environment_dir = prepare_av2_output_dir(paths, ENVIRONMENT_RELATIVE_DIR)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = environment_dir / f"av2_gpu_preflight_{timestamp}.json"
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    paths = resolve_av2_paths(args.data_root)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("AV2 GPU preflight requires a PyTorch CUDA installation.") from exc

    metadata = collect_preflight_metadata(torch, paths)
    output_path = write_preflight_metadata(paths, metadata)
    print(f"[av2-preflight] metadata written to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
