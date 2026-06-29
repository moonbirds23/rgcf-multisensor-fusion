from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "run_phase1r_rgcf_benchmark.py"
DEFAULT_OUT_ROOT = Path(r"E:\migration_packages\results\paper_ehgcf_final_compare")


def _bool_flag(enabled: bool, flag: str) -> List[str]:
    return [flag] if enabled else []


def _append_if_value(cmd: List[str], flag: str, value: str | None) -> None:
    if value not in (None, ""):
        cmd.extend([flag, str(value)])


def _read_dataset_dir(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        dataset_dir = str(row.get("dataset_dir") or "").strip()
        if dataset_dir and dataset_dir != "phase1r_mixed_train_store":
            return dataset_dir
    return None


def build_benchmark_command(
    *,
    python_exe: str,
    methods: str,
    out_dir: Path,
    args: argparse.Namespace,
    mixed_dataset_dir: str | None,
) -> List[str]:
    cmd = [
        python_exe,
        str(args.benchmark_script),
        "--methods",
        methods,
        "--out-dir",
        str(out_dir),
        "--dataset-store-root",
        str(args.dataset_store_root),
        "--device",
        args.device,
        "--model-seeds",
        args.model_seeds,
        "--train-seed-range",
        args.train_seed_range,
        "--val-seed-range",
        args.val_seed_range,
        "--test-seed-range",
        args.test_seed_range,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
    ]
    _append_if_value(cmd, "--mixed-dataset-dir", mixed_dataset_dir)
    cmd.extend(_bool_flag(args.resume, "--resume"))
    cmd.extend(_bool_flag(args.smoke, "--smoke"))
    cmd.extend(_bool_flag(args.no_sim_cache, "--no-sim-cache"))
    cmd.extend(_bool_flag(args.force_regenerate_sims, "--force-regenerate-sims"))
    return cmd


def print_command(label: str, cmd: Iterable[str]) -> None:
    print("=" * 88)
    print(f"[{label}]")
    print(" ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd))


def run_command(label: str, cmd: List[str], *, dry_run: bool) -> None:
    print_command(label, cmd)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT.parent), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the compact EHGCF paper experiment suite on a GPU host.")
    p.add_argument("--mode", choices=["main", "ablation", "all"], default="all")
    p.add_argument("--dry-run", action="store_true", help="Print the benchmark commands without launching training.")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--dataset-store-root", default=str(PROJECT_ROOT / "dataset_store"))
    p.add_argument("--mixed-dataset-dir", default=None, help="Optional fixed mixed dataset dir. If omitted, the main run builds one from fixed seeds.")
    p.add_argument("--python", dest="python_exe", default=sys.executable)
    p.add_argument("--benchmark-script", default=str(BENCHMARK_SCRIPT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-sim-cache", action="store_true")
    p.add_argument("--force-regenerate-sims", action="store_true")
    p.add_argument("--model-seeds", default="0,1,2,3,4")
    p.add_argument("--train-seed-range", default="10-69")
    p.add_argument("--val-seed-range", default="70-89")
    p.add_argument("--test-seed-range", default="90-109")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.benchmark_script = Path(args.benchmark_script)
    args.dataset_store_root = Path(args.dataset_store_root)
    out_root = Path(args.out_root)
    if args.smoke and out_root == DEFAULT_OUT_ROOT:
        out_root = DEFAULT_OUT_ROOT.with_name("paper_ehgcf_smoke_check")
    out_root.mkdir(parents=True, exist_ok=True)

    main_dir = out_root / "main"
    ablation_dir = out_root / "ablation"
    mixed_dataset_dir = args.mixed_dataset_dir

    if args.mode in {"main", "all"}:
        cmd = build_benchmark_command(
            python_exe=args.python_exe,
            methods="ehgcf",
            out_dir=main_dir,
            args=args,
            mixed_dataset_dir=mixed_dataset_dir,
        )
        run_command("main: EHGCF vs rule baselines", cmd, dry_run=args.dry_run)
        if not mixed_dataset_dir and not args.dry_run:
            mixed_dataset_dir = _read_dataset_dir(main_dir / "phase1r_run_summary.csv")
            if mixed_dataset_dir:
                print(f"[dataset] Reusing mixed dataset for later runs: {mixed_dataset_dir}")

    if args.mode in {"ablation", "all"}:
        cmd = build_benchmark_command(
            python_exe=args.python_exe,
            methods="posterior-only,rgcf,me-a0,ehgcf-no-calib,ehgcf",
            out_dir=ablation_dir,
            args=args,
            mixed_dataset_dir=mixed_dataset_dir,
        )
        run_command("ablation: compact EHGCF paper ablations", cmd, dry_run=args.dry_run)

    print("=" * 88)
    print(f"[done] outputs root: {out_root}")
    if args.dry_run and not args.mixed_dataset_dir and args.mode == "all":
        print("[note] Dry-run cannot know the generated dataset dir. In a real --mode all run, the ablation stage reuses the main-stage dataset.")


if __name__ == "__main__":
    main()
