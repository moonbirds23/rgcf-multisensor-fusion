"""Resolve and protect the external storage boundary for AV2 experiments.

This module deliberately has no fallback data directory.  AV2 raw data is an
input-only tree at ``data/raw/av2_motion_forecasting``; generated artifacts may
only be created in the enumerated output directories below.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Optional


AV2_ROOT_ENV_VAR = "PEFNET_AV2_ROOT"
AV2_RAW_RELATIVE_DIR = Path("data") / "raw" / "av2_motion_forecasting"

# Keep this allow-list small and explicit.  New AV2 artifact types must be
# added here before any writer is allowed to create their directories.
AV2_OUTPUT_RELATIVE_DIRS = (
    Path("data/manifests/pilot"),
    Path("data/manifests/formal"),
    Path("data/truth_cache/pilot"),
    Path("data/truth_cache/formal"),
    Path("data/feature_shards/pilot"),
    Path("data/feature_shards/formal"),
    Path("data/metadata"),
    Path("runs/pilot"),
    Path("runs/formal"),
    Path("checkpoints/pilot"),
    Path("checkpoints/formal"),
    Path("results/pilot"),
    Path("results/formal"),
    Path("caches/torch"),
    Path("caches/triton"),
    Path("caches/inductor"),
    Path("caches/pip"),
    Path("caches/temp"),
    Path("environment"),
    Path("backups/manifests"),
    Path("backups/configs"),
    Path("backups/final_results"),
)


class Av2PathError(ValueError):
    """Raised when the external AV2 storage contract is not satisfied."""


@dataclass(frozen=True)
class Av2Paths:
    """Validated AV2 locations.

    ``raw_root`` is never created or written by this module.  ``output_dirs``
    are returned only after :func:`prepare_av2_output_dirs` is called.
    """

    root: Path
    raw_root: Path


def add_av2_data_root_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard AV2 root option to an AV2-only command parser."""

    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Absolute AV2 artifact root. Overrides PEFNET_AV2_ROOT; no "
            "default location is used."
        ),
    )
    return parser


def _configured_root(data_root: Optional[str | Path], environ: Mapping[str, str]) -> str | Path:
    if data_root is not None and str(data_root).strip():
        return data_root
    env_root = environ.get(AV2_ROOT_ENV_VAR)
    if env_root and env_root.strip():
        return env_root
    raise Av2PathError(
        "AV2 data root is required: pass --data-root or set "
        f"{AV2_ROOT_ENV_VAR}. No default path is permitted."
    )


def _require_descendant(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Av2PathError(f"{label} escapes AV2 root: {path}") from exc


def resolve_av2_paths(
    data_root: Optional[str | Path] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Av2Paths:
    """Resolve CLI/env configuration and validate the existing raw input tree.

    This performs no directory enumeration and creates nothing.  In
    particular, a missing raw tree is an error rather than an invitation to
    create it, preserving the raw-data read-only contract.
    """

    env = os.environ if environ is None else environ
    configured = Path(_configured_root(data_root, env)).expanduser()
    if not configured.is_absolute():
        raise Av2PathError(
            f"AV2 data root must be absolute, got {configured!s}. "
            "Use an external mounted-drive path."
        )
    if not configured.exists() or not configured.is_dir():
        raise Av2PathError(f"AV2 data root must be an existing directory: {configured}")
    if not os.access(configured, os.R_OK):
        raise Av2PathError(f"AV2 data root is not readable: {configured}")

    root = configured.resolve(strict=True)
    raw_root = root / AV2_RAW_RELATIVE_DIR
    if not raw_root.exists() or not raw_root.is_dir():
        raise Av2PathError(
            "AV2 raw input directory is required and is never created by this "
            f"code: {raw_root}"
        )
    if not os.access(raw_root, os.R_OK):
        raise Av2PathError(f"AV2 raw input directory is not readable: {raw_root}")
    _require_descendant(raw_root.resolve(strict=True), root, label="AV2 raw input directory")
    return Av2Paths(root=root, raw_root=raw_root)


def _mkdir_allowed_descendant(root: Path, relative_dir: Path) -> Path:
    """Create one allow-listed directory without following an escaping symlink."""

    if relative_dir.is_absolute() or ".." in relative_dir.parts:
        raise Av2PathError(f"Invalid AV2 output relative path: {relative_dir}")

    current = root
    for part in relative_dir.parts:
        candidate = current / part
        if candidate.exists():
            resolved = candidate.resolve(strict=True)
            _require_descendant(resolved, root, label="AV2 output directory")
            if not resolved.is_dir():
                raise Av2PathError(f"AV2 output path component is not a directory: {candidate}")
            current = resolved
            continue
        candidate.mkdir()
        current = candidate
    resolved_target = current.resolve(strict=True)
    _require_descendant(resolved_target, root, label="AV2 output directory")
    return resolved_target


def prepare_av2_output_dirs(paths: Av2Paths) -> dict[str, Path]:
    """Create the fixed AV2 output tree and return absolute paths by key.

    The raw directory is not an output target and is never touched.  The
    explicit allow-list also prevents accidental writes to the repository,
    system temporary directories, or arbitrary folders under the mounted disk.
    """

    root = paths.root.resolve(strict=True)
    raw_root = paths.raw_root.resolve(strict=True)
    _require_descendant(raw_root, root, label="AV2 raw input directory")

    created: MutableMapping[str, Path] = {}
    for relative_dir in AV2_OUTPUT_RELATIVE_DIRS:
        output_dir = _prepare_one_output_dir(root, raw_root, relative_dir)
        created[relative_dir.as_posix()] = output_dir
    return dict(created)


def _prepare_one_output_dir(root: Path, raw_root: Path, relative_dir: Path) -> Path:
    if relative_dir not in AV2_OUTPUT_RELATIVE_DIRS:
        raise Av2PathError(f"AV2 output directory is not allow-listed: {relative_dir}")
    output_dir = _mkdir_allowed_descendant(root, relative_dir)
    if output_dir == raw_root or raw_root in output_dir.parents:
        raise Av2PathError(f"Raw AV2 input directory cannot be an output target: {output_dir}")
    return output_dir


def prepare_av2_output_dir(paths: Av2Paths, relative_dir: Path) -> Path:
    """Create one allow-listed AV2 output directory.

    Use this narrower API for commands such as preflight that must not create
    unrelated cache, checkpoint, or result directories.
    """

    root = paths.root.resolve(strict=True)
    raw_root = paths.raw_root.resolve(strict=True)
    _require_descendant(raw_root, root, label="AV2 raw input directory")
    return _prepare_one_output_dir(root, raw_root, relative_dir)
