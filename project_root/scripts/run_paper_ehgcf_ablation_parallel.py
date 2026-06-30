from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "run_phase1r_rgcf_benchmark.py"
DEFAULT_OUT_ROOT = Path(r"E:\migration_packages\results\paper_ehgcf_hypothesis_ablation")

ABLATION_METHODS = [
    ("posterior-only", "posterior_only", "Posterior-Only GNN"),
    ("posterior-only-calib", "posterior_only_calib", "Posterior-Only + Calib"),
    ("ehgcf-no-external", "ehgcf_no_external_evidence", "EHGCF w/o External Evidence"),
    ("ehgcf-no-hetero-gnn", "ehgcf_no_heterogeneous_gnn", "EHGCF w/o Heterogeneous GNN"),
    ("ehgcf-no-calib", "ehgcf_no_calibrated_fusion", "EHGCF w/o Calibrated Fusion"),
    ("ehgcf", "ehgcf", "EHGCF"),
]


def _parse_seed_range(text: str) -> tuple[int, int]:
    clean = str(text).strip()
    if "-" in clean:
        lo, hi = clean.split("-", 1)
        return int(lo), int(hi)
    value = int(clean)
    return value, value


def _method_filter(text: str | None) -> list[tuple[str, str, str]]:
    if not text:
        return list(ABLATION_METHODS)
    aliases = {m[0]: m for m in ABLATION_METHODS}
    aliases.update({m[1]: m for m in ABLATION_METHODS})
    out = []
    for part in str(text).split(","):
        key = part.strip()
        if not key:
            continue
        if key not in aliases:
            allowed = ", ".join(m[0] for m in ABLATION_METHODS)
            raise ValueError(f"Unknown method '{key}'. Allowed: {allowed}")
        method = aliases[key]
        if method not in out:
            out.append(method)
    return out


def _read_dataset_dir(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset_dir = str(row.get("dataset_dir") or "").strip()
            if dataset_dir and dataset_dir != "phase1r_mixed_train_store":
                return dataset_dir
    return None


def _quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)


def _build_benchmark_cmd(
    *,
    python_exe: str,
    method: str,
    out_dir: Path,
    dataset_store_root: Path,
    mixed_dataset_dir: str,
    device: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_exe,
        str(args.benchmark_script),
        "--methods",
        method,
        "--out-dir",
        str(out_dir),
        "--dataset-store-root",
        str(dataset_store_root),
        "--mixed-dataset-dir",
        str(mixed_dataset_dir),
        "--device",
        device,
        "--model-seeds",
        args.model_seeds,
        "--train-seed-range",
        args.train_seed_range,
        "--val-seed-range",
        args.val_seed_range,
        "--test-seed-range",
        args.test_seed_range,
        "--epochs",
        str(args.effective_epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--skip-rule-baselines",
    ]
    if args.resume:
        cmd.append("--resume")
    if args.smoke:
        cmd.append("--smoke")
    if args.no_sim_cache:
        cmd.append("--no-sim-cache")
    if args.force_regenerate_sims:
        cmd.append("--force-regenerate-sims")
    return cmd


def _prepare_dataset(args: argparse.Namespace, out_root: Path) -> str:
    if args.mixed_dataset_dir:
        return str(args.mixed_dataset_dir)

    from run_phase1r_rgcf_benchmark import (
        build_scenes,
        prepare_mixed_dataset,
        SeedRanges,
    )

    train = _parse_seed_range(args.train_seed_range)
    val = _parse_seed_range(args.val_seed_range)
    test = _parse_seed_range(args.test_seed_range)
    scenes = build_scenes(args.scenario_presets)
    smoke_duration = float(args.smoke_duration) if args.smoke else None
    dataset_dir, _ = prepare_mixed_dataset(
        scenes=scenes,
        seed_ranges=SeedRanges(train=train, val=val, test=test),
        epochs=args.effective_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.dataset_device,
        dataset_store_root=str(args.dataset_store_root),
        mixed_dataset_dir=None,
        use_sim_cache=not args.no_sim_cache,
        force_regenerate_sims=args.force_regenerate_sims,
        smoke_duration=smoke_duration,
    )
    (out_root / "fixed_mixed_dataset_dir.txt").write_text(str(dataset_dir), encoding="utf-8")
    return str(dataset_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch EHGCF hypothesis-driven ablation groups in parallel.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without generating data or launching training.")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--dataset-store-root", default=str(PROJECT_ROOT / "dataset_store"))
    p.add_argument("--mixed-dataset-dir", default=None, help="Fixed mixed dataset dir shared by every group. If omitted, this script creates one before launching groups.")
    p.add_argument("--python", dest="python_exe", default=sys.executable)
    p.add_argument("--benchmark-script", default=str(BENCHMARK_SCRIPT))
    p.add_argument("--methods", default=None, help="Optional comma-separated subset. Default runs all six ablation groups.")
    p.add_argument("--devices", default="cuda", help="Comma-separated devices used round-robin, e.g. cuda:0,cuda:1.")
    p.add_argument("--dataset-device", default="cpu", help="Device passed while preparing the fixed dataset.")
    p.add_argument("--max-parallel", type=int, default=0, help="Maximum concurrent processes. Default = number of selected methods.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-sim-cache", action="store_true")
    p.add_argument("--force-regenerate-sims", action="store_true")
    p.add_argument("--scenario-presets", default=None)
    p.add_argument("--model-seeds", default="0,1,2,3,4")
    p.add_argument("--train-seed-range", default="10-69")
    p.add_argument("--val-seed-range", default="70-89")
    p.add_argument("--test-seed-range", default="90-109")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--smoke-duration", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.benchmark_script = Path(args.benchmark_script)
    args.dataset_store_root = Path(args.dataset_store_root)
    args.effective_epochs = int(args.epochs if args.epochs is not None else (2 if args.smoke else 80))
    out_root = Path(args.out_root)
    if args.smoke and out_root == DEFAULT_OUT_ROOT:
        out_root = DEFAULT_OUT_ROOT.with_name("paper_ehgcf_hypothesis_ablation_smoke")
    out_root.mkdir(parents=True, exist_ok=True)

    methods = _method_filter(args.methods)
    devices = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    if not devices:
        devices = ["cuda"]

    print("=" * 88)
    print("[EHGCF hypothesis ablation]")
    print(f"[out_root] {out_root}")
    print(f"[methods] {', '.join(m[2] for m in methods)}")
    print(f"[devices] {', '.join(devices)}")

    if args.dry_run:
        mixed_dataset_dir = args.mixed_dataset_dir or "<FIXED_MIXED_DATASET_DIR_CREATED_BEFORE_PARALLEL_RUN>"
        print(f"[dataset] {mixed_dataset_dir}")
    else:
        mixed_dataset_dir = _prepare_dataset(args, out_root)
        print(f"[dataset] shared fixed mixed dataset: {mixed_dataset_dir}")

    commands = []
    for idx, (method_cli, slug, display_name) in enumerate(methods):
        device = devices[idx % len(devices)]
        method_out = out_root / slug
        cmd = _build_benchmark_cmd(
            python_exe=args.python_exe,
            method=method_cli,
            out_dir=method_out,
            dataset_store_root=args.dataset_store_root,
            mixed_dataset_dir=mixed_dataset_dir,
            device=device,
            args=args,
        )
        commands.append((display_name, cmd, method_out))

    print("=" * 88)
    print("[commands]")
    for display_name, cmd, method_out in commands:
        print(f"\n# {display_name}")
        print(_quote_cmd(cmd))
        print(f"# output: {method_out}")

    if args.dry_run:
        print("=" * 88)
        print("[dry-run] no training launched.")
        return

    max_parallel = int(args.max_parallel) if int(args.max_parallel) > 0 else len(commands)
    running: list[tuple[str, subprocess.Popen]] = []
    pending = list(commands)
    failures: list[str] = []

    while pending or running:
        while pending and len(running) < max_parallel:
            display_name, cmd, method_out = pending.pop(0)
            method_out.mkdir(parents=True, exist_ok=True)
            log_path = method_out / "parallel_stdout.log"
            print("=" * 88)
            print(f"[launch] {display_name}")
            print(f"[log] {log_path}")
            log_file = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            proc._rgcf_log_file = log_file  # type: ignore[attr-defined]
            running.append((display_name, proc))

        still_running: list[tuple[str, subprocess.Popen]] = []
        for display_name, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((display_name, proc))
                continue
            log_file = getattr(proc, "_rgcf_log_file", None)
            if log_file is not None:
                log_file.close()
            if code != 0:
                failures.append(f"{display_name} exited with code {code}")
                print(f"[failed] {display_name} code={code}")
            else:
                print(f"[done] {display_name}")
        running = still_running

        if running:
            import time

            time.sleep(5.0)

    print("=" * 88)
    print(f"[done] outputs root: {out_root}")
    print(f"[dataset] shared fixed mixed dataset: {mixed_dataset_dir}")
    if failures:
        for item in failures:
            print(f"[failure] {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
