from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_RESULTS_ROOT = Path(r"E:\migration_packages\results")
DEFAULT_PAPER_ROOT = DEFAULT_RESULTS_ROOT / "paper_ehgcf_final_compare"

METHOD_RENAME = {
    "ME-RGCF-A0D": "EHGCF",
    "ME_RGCF_A0D": "EHGCF",
    "EHGCF": "EHGCF",
    "EHGCF w/o calibrated fusion": "EHGCF w/o calibrated fusion",
    "ME-RGCF-A0": "ME-RGCF-A0",
    "RGCF": "RGCF",
    "Posterior-only": "Posterior-only",
}

MAIN_METHOD_ORDER = ["EHGCF", "CI-3T", "WAA-MM-3T", "AVG-3T", "single-T1", "single-T2", "single-T3"]
ABLATION_METHOD_ORDER = ["Posterior-only", "RGCF", "ME-RGCF-A0", "EHGCF w/o calibrated fusion", "EHGCF"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def rename_method(value: str) -> str:
    value = str(value or "")
    return METHOD_RENAME.get(value, value)


def add_source(rows: Iterable[Dict[str, str]], *, source_file: Path, source_role: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        item = dict(row)
        if "method" in item:
            item["method"] = rename_method(item["method"])
        item["source_role"] = source_role
        item["source_file"] = str(source_file)
        out.append(item)
    return out


def method_rank(method: str, order: Sequence[str]) -> int:
    try:
        return order.index(method)
    except ValueError:
        return len(order) + 100


def existing_result_dirs(results_root: Path) -> Dict[str, Path]:
    return {
        "existing_ehgcf_final": results_root / "phase2_me_a0_dir_only",
        "existing_rgcf_process": results_root / "phase1r_rgcf_compare",
        "existing_me_a0_process": results_root / "phase2_me_a0_only",
        "existing_hs_diagnostic_only": results_root / "phase2_me_a0_dir_hs_only",
    }


def select_main_dir(paper_root: Path, results_root: Path) -> Path:
    candidate = paper_root / "main"
    if (candidate / "phase1r_aggregate_overall.csv").exists():
        return candidate
    return existing_result_dirs(results_root)["existing_ehgcf_final"]


def select_ablation_dirs(paper_root: Path, results_root: Path) -> List[tuple[str, Path]]:
    candidate = paper_root / "ablation"
    if (candidate / "phase1r_aggregate_overall.csv").exists():
        return [("paper_ablation", candidate)]
    dirs = existing_result_dirs(results_root)
    return [
        ("existing_rgcf_process", dirs["existing_rgcf_process"]),
        ("existing_me_a0_process", dirs["existing_me_a0_process"]),
        ("existing_ehgcf_final", dirs["existing_ehgcf_final"]),
    ]


def build_manifest(paths: Iterable[tuple[str, Path]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role, path in paths:
        rows.append({
            "source_role": role,
            "path": str(path),
            "exists": path.exists(),
            "rows": count_rows(path) if path.is_file() else "",
        })
    return rows


def detail_method_from_path(path: Path) -> str:
    text = path.name
    if "EHGCF_no_calibrated_fusion" in text:
        return "EHGCF w/o calibrated fusion"
    if "Posterior_only" in text:
        return "Posterior-only"
    if "ME_RGCF_A0D" in text or "EHGCF" in text:
        return "EHGCF"
    if "ME_RGCF_A0" in text:
        return "ME-RGCF-A0"
    if "RGCF" in text:
        return "RGCF"
    return ""


def collect_mechanism_pool(source_dirs: Sequence[tuple[str, Path]], out_path: Path, *, max_rows: int) -> int:
    selected_prefixes = (
        "run_label",
        "method",
        "scenario_id",
        "scenario_label",
        "seed",
        "k",
        "t",
        "error_pos",
        "g_s",
        "w_s",
        "cov_scale_s",
        "mp_attn_p",
        "mm_attn_m",
        "mp_pair_res_p",
        "mp_pair_rank_p",
    )
    detail_files: List[tuple[str, Path]] = []
    keep_cols: List[str] = []
    for source_role, source_dir in source_dirs:
        detail_dir = source_dir / "eval_details"
        if not detail_dir.exists():
            continue
        for detail_file in sorted(detail_dir.glob("*_errors.csv")):
            detail_files.append((source_role, detail_file))
            with detail_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for col in reader.fieldnames or []:
                    if col.startswith(selected_prefixes) and col not in keep_cols:
                        keep_cols.append(col)

    fieldnames = ["source_role", "source_file", "method", *[c for c in keep_cols if c != "method"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8", newline="")
    written = 0
    try:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for source_role, detail_file in detail_files:
            method = detail_method_from_path(detail_file)
            with detail_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if max_rows and written >= max_rows:
                        return written
                    out_row = {k: row.get(k, "") for k in fieldnames}
                    out_row["source_role"] = source_role
                    out_row["source_file"] = str(detail_file)
                    out_row["method"] = method or rename_method(row.get("method", ""))
                    writer.writerow(out_row)
                    written += 1
    finally:
        out_f.close()
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect EHGCF paper tables and mechanism plotting data.")
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--paper-root", default=str(DEFAULT_PAPER_ROOT))
    p.add_argument("--out-dir", default=None, help="Default: <paper-root>/paper_tables")
    p.add_argument("--mechanism-max-rows", type=int, default=0, help="0 means keep all selected mechanism rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    paper_root = Path(args.paper_root)
    out_dir = Path(args.out_dir) if args.out_dir else paper_root / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_dir = select_main_dir(paper_root, results_root)
    main_overall_path = main_dir / "phase1r_aggregate_overall.csv"
    main_scene_path = main_dir / "phase1r_aggregate_by_scene.csv"
    main_rows = add_source(read_csv(main_overall_path), source_file=main_overall_path, source_role="main_overall")
    scene_rows = add_source(read_csv(main_scene_path), source_file=main_scene_path, source_role="main_by_scene")
    main_rows.sort(key=lambda r: method_rank(str(r.get("method", "")), MAIN_METHOD_ORDER))
    scene_rows.sort(key=lambda r: (str(r.get("scenario_id", "")), method_rank(str(r.get("method", "")), MAIN_METHOD_ORDER)))
    write_csv(out_dir / "table_main_overall.csv", main_rows)
    write_csv(out_dir / "table_by_scene.csv", scene_rows)

    ablation_source_dirs = select_ablation_dirs(paper_root, results_root)
    ablation_rows: List[Dict[str, str]] = []
    for role, source_dir in ablation_source_dirs:
        path = source_dir / "phase1r_aggregate_overall.csv"
        rows = add_source(read_csv(path), source_file=path, source_role=role)
        ablation_rows.extend([r for r in rows if r.get("method") in ABLATION_METHOD_ORDER])
    ablation_rows.sort(key=lambda r: method_rank(str(r.get("method", "")), ABLATION_METHOD_ORDER))
    write_csv(out_dir / "table_ablation.csv", ablation_rows)

    mechanism_sources = list(ablation_source_dirs)
    if all(source_dir.resolve() != main_dir.resolve() for _, source_dir in mechanism_sources):
        mechanism_sources = [("paper_or_existing_main", main_dir), *mechanism_sources]
    n_mech = collect_mechanism_pool(
        mechanism_sources,
        out_dir / "mechanism_plot_pool.csv",
        max_rows=args.mechanism_max_rows,
    )

    manifest_paths: List[tuple[str, Path]] = [
        ("main_overall", main_overall_path),
        ("main_by_scene", main_scene_path),
        ("main_run_summary", main_dir / "phase1r_run_summary.csv"),
    ]
    for role, source_dir in ablation_source_dirs:
        manifest_paths.extend([
            (f"{role}_overall", source_dir / "phase1r_aggregate_overall.csv"),
            (f"{role}_by_scene", source_dir / "phase1r_aggregate_by_scene.csv"),
            (f"{role}_run_summary", source_dir / "phase1r_run_summary.csv"),
            (f"{role}_eval_details", source_dir / "eval_details"),
        ])
    manifest = build_manifest(manifest_paths)
    manifest.append({
        "source_role": "mechanism_plot_pool",
        "path": str(out_dir / "mechanism_plot_pool.csv"),
        "exists": True,
        "rows": n_mech,
    })
    write_csv(out_dir / "result_sources_manifest.csv", manifest)

    print(f"[tables] {out_dir}")
    print(f"[main_source] {main_dir}")
    print(f"[mechanism_rows] {n_mech}")


if __name__ == "__main__":
    main()
