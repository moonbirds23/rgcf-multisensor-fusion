from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from data.av2.paths import (
    AV2_OUTPUT_RELATIVE_DIRS,
    AV2_RAW_RELATIVE_DIR,
    Av2PathError,
    add_av2_data_root_argument,
    prepare_av2_output_dir,
    prepare_av2_output_dirs,
    resolve_av2_paths,
)


def _make_root(tmp_path: Path, name: str = "av2") -> Path:
    root = tmp_path / name
    (root / AV2_RAW_RELATIVE_DIR).mkdir(parents=True)
    return root


class Av2PathTests(unittest.TestCase):
    def test_cli_data_root_overrides_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            cli_root = _make_root(tmp_path, "cli")
            env_root = _make_root(tmp_path, "env")

            paths = resolve_av2_paths(cli_root, environ={"PEFNET_AV2_ROOT": str(env_root)})

            self.assertEqual(paths.root, cli_root.resolve())
            self.assertEqual(paths.raw_root, cli_root / AV2_RAW_RELATIVE_DIR)

    def test_environment_root_is_used_when_cli_option_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _make_root(Path(temp_dir))

            paths = resolve_av2_paths(environ={"PEFNET_AV2_ROOT": str(root)})

            self.assertEqual(paths.root, root.resolve())

    def test_missing_configuration_has_no_default_path(self) -> None:
        with self.assertRaisesRegex(Av2PathError, "No default path"):
            resolve_av2_paths(environ={})

    def test_missing_raw_tree_is_rejected_without_creating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "av2"
            root.mkdir()

            with self.assertRaisesRegex(Av2PathError, "never created"):
                resolve_av2_paths(root, environ={})

            self.assertFalse((root / AV2_RAW_RELATIVE_DIR).exists())

    def test_prepare_creates_only_allowlisted_outputs_and_preserves_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _make_root(Path(temp_dir))
            raw_marker = root / AV2_RAW_RELATIVE_DIR / "must_not_change.txt"
            raw_marker.write_text("input-only", encoding="utf-8")
            paths = resolve_av2_paths(root, environ={})

            outputs = prepare_av2_output_dirs(paths)

            self.assertEqual(set(outputs), {path.as_posix() for path in AV2_OUTPUT_RELATIVE_DIRS})
            self.assertTrue(all(output.is_dir() for output in outputs.values()))
            self.assertEqual(raw_marker.read_text(encoding="utf-8"), "input-only")
            self.assertFalse((root / "unexpected").exists())

    def test_prepare_one_output_directory_does_not_create_unrelated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _make_root(Path(temp_dir))

            environment_dir = prepare_av2_output_dir(
                resolve_av2_paths(root, environ={}), Path("environment")
            )

            self.assertEqual(environment_dir, root / "environment")
            self.assertFalse((root / "caches").exists())
            self.assertFalse((root / "runs").exists())

    def test_output_symlink_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root = _make_root(tmp_path)
            outside = tmp_path / "outside"
            outside.mkdir()
            try:
                (root / "runs").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation is unavailable on this Windows host")

            with self.assertRaisesRegex(Av2PathError, "escapes AV2 root"):
                prepare_av2_output_dirs(resolve_av2_paths(root, environ={}))

    def test_parser_exposes_the_standard_data_root_option(self) -> None:
        parser = add_av2_data_root_argument(argparse.ArgumentParser())

        self.assertEqual(
            parser.parse_args(["--data-root", "D:/mounted/PEFNet_AV2"]).data_root,
            "D:/mounted/PEFNet_AV2",
        )
