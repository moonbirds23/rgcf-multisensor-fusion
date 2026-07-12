from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from data.av2.paths import AV2_RAW_RELATIVE_DIR, resolve_av2_paths
from scripts.av2_gpu_preflight import (
    ENVIRONMENT_RELATIVE_DIR,
    _require_cuda,
    collect_preflight_metadata,
    write_preflight_metadata,
)


class _FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 2 if self.available else 0

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=24 * 1024**3, major=8, minor=6)


class _FakeTorch:
    __version__ = "test-cuda"
    version = SimpleNamespace(cuda="12.6")

    def __init__(self, available: bool = True) -> None:
        self.cuda = _FakeCuda(available)


def _paths(tmp_path: Path):
    root = tmp_path / "av2"
    (root / AV2_RAW_RELATIVE_DIR).mkdir(parents=True)
    return resolve_av2_paths(root, environ={})


class Av2GpuPreflightTests(unittest.TestCase):
    def test_cuda_is_strictly_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is intentionally disabled"):
            _require_cuda(_FakeTorch(available=False))

    def test_collects_device_metadata_without_raw_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = collect_preflight_metadata(_FakeTorch(), _paths(Path(temp_dir)))

            self.assertEqual(metadata["cuda"]["device_count"], 2)
            self.assertEqual(metadata["cuda"]["devices"][1]["name"], "Fake GPU 1")
            self.assertEqual(metadata["runtime"]["torch_cuda_version"], "12.6")

    def test_preflight_writes_only_to_environment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _paths(Path(temp_dir))
            output_path = write_preflight_metadata(paths, {"cuda": {"available": True}})

            self.assertEqual(output_path.parent, paths.root / ENVIRONMENT_RELATIVE_DIR)
            self.assertTrue(output_path.is_file())
            self.assertFalse((paths.root / "caches").exists())
            self.assertFalse((paths.root / "runs").exists())
