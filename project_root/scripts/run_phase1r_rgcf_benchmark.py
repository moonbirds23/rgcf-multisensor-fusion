from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from core.config_loader import load_experiment_bundle
from core.types import RunRequest


DEFAULT_SCENES = {
    "S1R": "phase1r_basic_3track_2evidence_nominal",
    "S2R": "phase1r_maneuver_3track_2evidence_nominal",
}
SCENE_LABELS = {
    "S1R": "basic-3track-2evidence",
    "S2R": "maneuver-3track-2evidence",
}
RULE_METHODS = ["single-T1", "single-T2", "single-T3", "AVG-3T", "WAA-MM-3T", "CI-3T"]


@dataclass(frozen=True)
class LearnedMethodSpec:
    cli_name: str
    display_name: str
    run_slug: str
    preset_suffix: str
    model_name: str


LEARNED_METHOD_SPECS = {
    "rgcf": LearnedMethodSpec("rgcf", "RGCF", "RGCF", "rgcf", "phase1r_rgcf"),
    "me-a0": LearnedMethodSpec("me-a0", "ME-RGCF-A0", "ME_RGCF_A0", "me_rgcf_a0", "me_rgcf_a0"),
}
LEARNED_METHODS = [LEARNED_METHOD_SPECS["rgcf"].display_name]


@dataclass(frozen=True)
class Phase1RScene:
    scene_id: str
    label: str
    preset_name: str


@dataclass(frozen=True)
class SeedRanges:
    train: Tuple[int, int]
    val: Tuple[int, int]
    test: Tuple[int, int]


def parse_seed_range(text: str) -> Tuple[int, int]:
    clean = str(text).strip()
    if "-" in clean:
        lo, hi = clean.split("-", 1)
        out = (int(lo), int(hi))
    else:
        v = int(clean)
        out = (v, v)
    if out[1] < out[0]:
        raise ValueError(f"Invalid seed range: {text}")
    return out


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = parse_seed_range(part)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def parse_methods(text: str | None) -> List[LearnedMethodSpec]:
    raw = str(text or "rgcf").strip().lower()
    if raw in {"", "default"}:
        raw = "rgcf"
    specs: List[LearnedMethodSpec] = []
    seen = set()
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key == "me_a0":
            key = "me-a0"
        if key not in LEARNED_METHOD_SPECS:
            allowed = ", ".join(sorted(LEARNED_METHOD_SPECS))
            raise ValueError(f"Unknown --methods entry '{part}'. Allowed: {allowed}")
        if key not in seen:
            specs.append(LEARNED_METHOD_SPECS[key])
            seen.add(key)
    if not specs:
        specs.append(LEARNED_METHOD_SPECS["rgcf"])
    return specs


def parse_mapping(text: str | None) -> Dict[str, str]:
    if not text:
        return {}
    out: Dict[str, str] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build_scenes(spec: str | None) -> List[Phase1RScene]:
    presets = dict(DEFAULT_SCENES)
    presets.update(parse_mapping(spec))
    return [
        Phase1RScene(scene_id=sid, label=SCENE_LABELS.get(sid, sid), preset_name=presets[sid])
        for sid in ("S1R", "S2R")
    ]


def configure_seed_ranges(bundle, ranges: SeedRanges) -> None:
    bundle.train.train_seed_start, bundle.train.train_seed_end = ranges.train
    bundle.train.val_seed_start, bundle.train.val_seed_end = ranges.val
    bundle.train.test_seed_start, bundle.train.test_seed_end = ranges.test


def build_learned_bundle(
    scene: Phase1RScene,
    method: LearnedMethodSpec,
    *,
    model_seed: int,
    seed_ranges: SeedRanges,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    device: str,
    smoke_duration: float | None = None,
):
    bundle = load_experiment_bundle(
        RunRequest(
            mode="train",
            preset_name=f"{scene.preset_name}_{method.preset_suffix}",
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            repeat_runs=1,
            experiment_name=None,
        )
    )
    if str(bundle.fault.mode) != "clean":
        raise RuntimeError(f"Phase1R requires clean/no-pollution presets, got fault_mode={bundle.fault.mode}")
    configure_seed_ranges(bundle, seed_ranges)
    bundle.train.model_seed = int(model_seed)
    bundle.model.model_name = method.model_name
    bundle.model.use_post_stream = True
    bundle.model.use_meas_stream = True
    bundle.model.use_gate = True
    bundle.model.output_fusion_mode = "info_diag"
    bundle.identity.model_name = bundle.model.model_name
    if smoke_duration is not None:
        bundle.scenario.motion.T = float(smoke_duration)
    return bundle


def build_rgcf_bundle(scene: Phase1RScene, **kwargs):
    return build_learned_bundle(scene, LEARNED_METHOD_SPECS["rgcf"], **kwargs)


def build_dataset_source_bundle(
    scene: Phase1RScene,
    *,
    seed_ranges: SeedRanges,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    device: str,
    smoke_duration: float | None = None,
):
    bundle = build_rgcf_bundle(
        scene,
        model_seed=0,
        seed_ranges=seed_ranges,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        device=device,
        smoke_duration=smoke_duration,
    )
    bundle.identity.preset_name = f"phase1r_{scene.scene_id}_dataset_source"
    bundle.identity.experiment_name = f"phase1r_{scene.scene_id}_dataset_source"
    return bundle


def annotate_sims(sims: Sequence[Dict], *, scene: Phase1RScene, split_name: str) -> List[Dict]:
    out = []
    for sim in sims:
        sim["phase1r_scene_id"] = scene.scene_id
        sim["phase1r_scene_label"] = scene.label
        sim["phase1r_scene_preset"] = scene.preset_name
        sim["split_name"] = split_name
        out.append(sim)
    return out


def split_mixed_by_scene(sims: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for sim in sims:
        sid = sim.get("phase1r_scene_id")
        if not sid:
            raise ValueError("Mixed Phase1R dataset is missing phase1r_scene_id; regenerate it with this script.")
        out.setdefault(str(sid), []).append(sim)
    return out


def prepare_mixed_dataset(
    *,
    scenes: Sequence[Phase1RScene],
    seed_ranges: SeedRanges,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    device: str,
    dataset_store_root: str,
    mixed_dataset_dir: str | None,
    use_sim_cache: bool,
    force_regenerate_sims: bool,
    smoke_duration: float | None,
) -> Tuple[str, Dict[str, Dict[str, List[Dict]]]]:
    from data.dataset_store import load_raw_dataset_store, save_raw_dataset_store
    from training.trainer import build_sim_list_from_seed_range

    if mixed_dataset_dir:
        ds = load_raw_dataset_store(mixed_dataset_dir)
        by_scene = {
            scene.scene_id: {
                "train": split_mixed_by_scene(ds["train_sims"]).get(scene.scene_id, []),
                "val": split_mixed_by_scene(ds["val_sims"]).get(scene.scene_id, []),
                "test": split_mixed_by_scene(ds["test_sims"]).get(scene.scene_id, []),
            }
            for scene in scenes
        }
        return str(mixed_dataset_dir), by_scene

    scene_splits: Dict[str, Dict[str, List[Dict]]] = {}
    mixed_train: List[Dict] = []
    mixed_val: List[Dict] = []
    mixed_test: List[Dict] = []
    dataset_bundle = None

    for scene in scenes:
        print("=" * 88)
        print(f"[dataset] building Phase1R sims for {scene.scene_id}: {scene.preset_name}")
        bundle = build_dataset_source_bundle(
            scene,
            seed_ranges=seed_ranges,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            device=device,
            smoke_duration=smoke_duration,
        )
        if dataset_bundle is None:
            dataset_bundle = bundle
        splits = {
            "train": annotate_sims(
                build_sim_list_from_seed_range(
                    bundle,
                    seed_ranges.train[0],
                    seed_ranges.train[1],
                    "train",
                    use_sim_cache=use_sim_cache,
                    force_regenerate_sims=force_regenerate_sims,
                ),
                scene=scene,
                split_name="train",
            ),
            "val": annotate_sims(
                build_sim_list_from_seed_range(
                    bundle,
                    seed_ranges.val[0],
                    seed_ranges.val[1],
                    "val",
                    use_sim_cache=use_sim_cache,
                    force_regenerate_sims=force_regenerate_sims,
                ),
                scene=scene,
                split_name="val",
            ),
            "test": annotate_sims(
                build_sim_list_from_seed_range(
                    bundle,
                    seed_ranges.test[0],
                    seed_ranges.test[1],
                    "test",
                    use_sim_cache=use_sim_cache,
                    force_regenerate_sims=force_regenerate_sims,
                ),
                scene=scene,
                split_name="test",
            ),
        }
        scene_splits[scene.scene_id] = splits
        mixed_train.extend(splits["train"])
        mixed_val.extend(splits["val"])
        mixed_test.extend(splits["test"])

    if dataset_bundle is None:
        raise RuntimeError("No Phase1R scenes configured.")
    dataset_bundle.identity.preset_name = "phase1r_s1r_s2r_mixed_nominal"
    dataset_bundle.identity.experiment_name = "phase1r_s1r_s2r_mixed_nominal"
    dataset_bundle.identity.scene_name = "S1R_S2R_mixed"
    dataset_bundle.identity.fault_mode = "clean"
    ds_dir = save_raw_dataset_store(
        dataset_bundle,
        mixed_train,
        mixed_val,
        mixed_test,
        root_dir=dataset_store_root,
        note="phase1r_mixed_nominal_s1r_s2r",
    )
    print(f"[dataset] mixed training store: {ds_dir}")
    return str(ds_dir), scene_splits


def save_rows(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metric_summary(errors: Sequence[float]) -> Dict[str, float | int | str]:
    vals = [float(v) for v in errors if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"num_points": 0, "rmse": "", "p95": "", "p99": "", "max": ""}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "num_points": int(arr.size),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _info_fuse(xs: Sequence[np.ndarray], ps: Sequence[np.ndarray], ws: np.ndarray) -> np.ndarray:
    y = np.zeros((4, 4), dtype=np.float64)
    eta = np.zeros((4,), dtype=np.float64)
    for x, p, w in zip(xs, ps, ws):
        invp = np.linalg.pinv(p + 1e-6 * np.eye(4))
        y += float(w) * invp
        eta += float(w) * (invp @ x)
    return np.linalg.pinv(y + 1e-9 * np.eye(4)) @ eta


def _ci3_weights(xs: Sequence[np.ndarray], ps: Sequence[np.ndarray], step: float = 0.05) -> np.ndarray:
    best_w = np.array([1.0 / 3.0] * 3, dtype=np.float64)
    best_score = None
    grid = np.arange(0.0, 1.0 + 1e-12, step)
    invps = [np.linalg.pinv(p + 1e-6 * np.eye(4)) for p in ps]
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-12:
                continue
            ws = np.array([w1, w2, max(w3, 0.0)], dtype=np.float64)
            y = ws[0] * invps[0] + ws[1] * invps[1] + ws[2] * invps[2]
            p = np.linalg.pinv(y + 1e-9 * np.eye(4))
            score = float(np.linalg.slogdet(p)[1])
            if best_score is None or score < best_score:
                best_score = score
                best_w = ws
    return best_w / max(float(best_w.sum()), 1e-9)


def baseline_errors_for_sim(sim: Dict) -> Dict[str, List[float]]:
    from simulation.fusion_baselines import fuse_waa_mm_sequence

    truth = np.asarray(sim["x_truth_4d"], dtype=np.float64)
    truth_xy = truth[:, :2]
    xhat = np.asarray(sim.get("track_xhat", sim["xhat"]), dtype=np.float64)
    phat = np.asarray(sim.get("track_Phat", sim["Phat"]), dtype=np.float64)
    valid = np.asarray(sim.get("track_valid_mask", sim["valid_mask"]), dtype=np.float64)
    k_count, n_track, _ = xhat.shape
    out: Dict[str, List[float]] = {}

    for i in range(min(3, n_track)):
        err = np.sqrt(np.sum((xhat[:, i, :2] - truth_xy) ** 2, axis=1))
        out[f"single-T{i + 1}"] = list(err)

    avg_xy = np.zeros_like(truth_xy)
    for k in range(k_count):
        idx = np.flatnonzero(valid[k] > 0.5)
        if idx.size == 0:
            idx = np.arange(n_track)
        avg_xy[k] = np.mean(xhat[k, idx, :2], axis=0)
    out["AVG-3T"] = list(np.sqrt(np.sum((avg_xy - truth_xy) ** 2, axis=1)))

    x_waa, _, _ = fuse_waa_mm_sequence(xhat, phat, valid)
    out["WAA-MM-3T"] = list(np.sqrt(np.sum((x_waa[:, :2] - truth_xy) ** 2, axis=1)))

    ci_xy = np.zeros_like(truth_xy)
    for k in range(k_count):
        idx = np.flatnonzero(valid[k] > 0.5)
        if idx.size < 3:
            idx = np.arange(n_track)
        idx = idx[:3]
        xs = [xhat[k, i] for i in idx]
        ps = [phat[k, i] for i in idx]
        ws = _ci3_weights(xs, ps)
        ci_xy[k] = _info_fuse(xs, ps, ws)[:2]
    out["CI-3T"] = list(np.sqrt(np.sum((ci_xy - truth_xy) ** 2, axis=1)))
    return out


def evaluate_rule_baselines(*, scene: Phase1RScene, test_sims: Sequence[Dict], out_dir: Path, seed_ranges: SeedRanges) -> List[Dict]:
    buckets = {name: [] for name in RULE_METHODS}
    for sim in test_sims:
        for name, vals in baseline_errors_for_sim(sim).items():
            buckets.setdefault(name, []).extend(vals)
    rows: List[Dict] = []
    for name in RULE_METHODS:
        rows.append({
            "status": "ok",
            "scenario_id": scene.scene_id,
            "scenario_label": scene.label,
            "scenario_preset": scene.preset_name,
            "method": name,
            "method_category": "rule_baseline",
            "model_seed": "",
            "run_label": f"phase1r_{scene.scene_id}_{name}",
            "run_dir": "",
            "dataset_dir": "phase1r_mixed_train_store",
            "train_seed_range": f"{seed_ranges.train[0]}-{seed_ranges.train[1]}",
            "val_seed_range": f"{seed_ranges.val[0]}-{seed_ranges.val[1]}",
            "test_seed_range": f"{seed_ranges.test[0]}-{seed_ranges.test[1]}",
            **metric_summary(buckets.get(name, [])),
        })
    save_json(out_dir / "eval_details" / f"phase1r_{scene.scene_id}_rule_baselines.json", {"scene": asdict(scene), "rows": rows})
    return rows


def sensor_health_rows(scenes: Sequence[Phase1RScene], scene_splits: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
    rows: List[Dict] = []
    for scene in scenes:
        sims = scene_splits[scene.scene_id]["test"]
        per_sensor = {0: [], 1: [], 2: []}
        for sim in sims:
            truth_xy = np.asarray(sim["x_truth_4d"], dtype=np.float64)[:, :2]
            xhat = np.asarray(sim.get("track_xhat", sim["xhat"]), dtype=np.float64)
            for i in range(min(3, xhat.shape[1])):
                per_sensor[i].extend(list(np.sqrt(np.sum((xhat[:, i, :2] - truth_xy) ** 2, axis=1))))
        for i in range(3):
            rows.append({
                "scenario_id": scene.scene_id,
                "scenario_label": scene.label,
                "sensor": f"T{i + 1}",
                **metric_summary(per_sensor[i]),
            })
    return rows


def evidence_report_rows(scenes: Sequence[Phase1RScene], scene_splits: Dict[str, Dict[str, List[Dict]]]) -> List[Dict]:
    rows: List[Dict] = []
    for scene in scenes:
        residuals = {0: [], 1: []}
        for sim in scene_splits[scene.scene_id]["test"]:
            arr = np.asarray(sim.get("evidence_residual_to_prior", np.empty((0, 0))), dtype=np.float64)
            for i in range(min(2, arr.shape[1] if arr.ndim == 2 else 0)):
                residuals[i].extend([float(v) for v in arr[:, i] if np.isfinite(float(v))])
        for i in range(2):
            rows.append({
                "scenario_id": scene.scene_id,
                "scenario_label": scene.label,
                "sensor": f"E{i + 1}",
                "residual_metric": "normalized_residual_to_track_prior",
                **metric_summary(residuals[i]),
            })
    return rows


def evaluate_learned(
    *,
    model,
    bundle,
    scene: Phase1RScene,
    test_sims: Sequence[Dict],
    out_dir: Path,
    run_label: str,
) -> Dict:
    import torch
    from training.evaluator import evaluate_single_sim_fusion_with_timeseries

    device = torch.device(bundle.base.runtime.device)
    errors: List[float] = []
    detail_rows: List[Dict] = []
    for sim in test_sims:
        ev = evaluate_single_sim_fusion_with_timeseries(sim, model, bundle, device)
        seed = sim.get("seed")
        for row in ev["timeseries"]:
            err = float(row["error_pos"])
            errors.append(err)
            out_row = {
                "run_label": run_label,
                "scenario_id": scene.scene_id,
                "scenario_label": scene.label,
                "seed": seed,
                "k": row.get("k"),
                "t": row.get("t"),
                "error_pos": err,
            }
            for key, value in row.items():
                if (
                    key.startswith("w_s")
                    or key.startswith("cov_scale_s")
                    or key.startswith("g_s")
                    or key.startswith("mp_attn_p")
                    or key.startswith("mm_attn_m")
                ):
                    out_row[key] = value
            detail_rows.append(out_row)
    safe_label = run_label.replace("\\", "_").replace("/", "_").replace(":", "_")
    save_rows(detail_rows, out_dir / "eval_details" / f"{safe_label}_errors.csv")
    out = metric_summary(errors)
    out["num_sims"] = len(test_sims)
    return out


def _mean(values: Iterable[float]) -> float | str:
    vals = [float(v) for v in values if v not in ("", None) and math.isfinite(float(v))]
    return "" if not vals else sum(vals) / len(vals)


def _std(values: Iterable[float]) -> float | str:
    vals = [float(v) for v in values if v not in ("", None) and math.isfinite(float(v))]
    if len(vals) <= 1:
        return ""
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def aggregate_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    def grouped(keys: Sequence[str]) -> List[Dict]:
        buckets: Dict[Tuple, List[Dict]] = {}
        for row in ok_rows:
            buckets.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
        out: List[Dict] = []
        for key, items in sorted(buckets.items()):
            entry = {k: v for k, v in zip(keys, key)}
            entry["n_runs"] = len(items)
            entry["num_points_total"] = sum(int(i.get("num_points") or 0) for i in items)
            for metric in ("rmse", "p95", "p99", "max"):
                vals = [i.get(metric) for i in items]
                entry[f"{metric}_mean"] = _mean(vals)
                entry[f"{metric}_std"] = _std(vals)
            out.append(entry)
        return out

    return {
        "by_scene": grouped(["scenario_id", "scenario_label", "method", "method_category"]),
        "overall": grouped(["method", "method_category"]),
    }


def save_run_outputs(rows: List[Dict], out_dir: Path) -> None:
    save_json(out_dir / "phase1r_run_summary.json", rows)
    save_rows(rows, out_dir / "phase1r_run_summary.csv")
    aggregates = aggregate_rows(rows)
    save_json(out_dir / "phase1r_aggregate_by_scene.json", aggregates["by_scene"])
    save_rows(aggregates["by_scene"], out_dir / "phase1r_aggregate_by_scene.csv")
    save_json(out_dir / "phase1r_aggregate_overall.json", aggregates["overall"])
    save_rows(aggregates["overall"], out_dir / "phase1r_aggregate_overall.csv")


def save_plan(
    *,
    out_dir: Path,
    scenes: Sequence[Phase1RScene],
    learned_methods: Sequence[LearnedMethodSpec],
    model_seeds: Sequence[int],
    seed_ranges: SeedRanges,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    smoke: bool,
) -> List[Dict]:
    plan_rows: List[Dict] = []
    for scene in scenes:
        for method in RULE_METHODS:
            plan_rows.append({
                "action": "evaluate_rule_baseline",
                "scenario_id": scene.scene_id,
                "scenario_label": scene.label,
                "scenario_preset": scene.preset_name,
                "method": method,
                "model_seed": "",
                "train_seed_range": f"{seed_ranges.train[0]}-{seed_ranges.train[1]}",
                "val_seed_range": f"{seed_ranges.val[0]}-{seed_ranges.val[1]}",
                "test_seed_range": f"{seed_ranges.test[0]}-{seed_ranges.test[1]}",
                "epochs": "",
                "lr": "",
                "batch_size": "",
                "hidden_dim": "",
                "profile": "smoke" if smoke else "formal",
            })
    for method in learned_methods:
        for model_seed in model_seeds:
            plan_rows.append({
                "action": f"train_{method.cli_name}_on_mixed_eval_s1r_s2r",
                "scenario_id": "S1R+S2R",
                "scenario_label": "mixed-train",
                "scenario_preset": ",".join(s.preset_name for s in scenes),
                "method": method.display_name,
                "model_seed": model_seed,
                "train_seed_range": f"{seed_ranges.train[0]}-{seed_ranges.train[1]} per scene",
                "val_seed_range": f"{seed_ranges.val[0]}-{seed_ranges.val[1]} per scene",
                "test_seed_range": f"{seed_ranges.test[0]}-{seed_ranges.test[1]}",
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "profile": "smoke" if smoke else "formal",
            })
    plan = {
        "benchmark": "Phase1R RGCF Corrected 3-Track 2-Evidence Benchmark",
        "scenes": [asdict(s) for s in scenes],
        "methods": RULE_METHODS + [m.display_name for m in learned_methods],
        "main_method": "RGCF",
        "sensor_protocol": "T1/T2/T3 are posterior track sensors; E1/E2 are measurement evidence only.",
        "training_protocol": "single S1R/S2R mixed train/val store; evaluate separately on S1R and S2R",
        "rule_baselines": RULE_METHODS,
        "model_init_seeds": list(model_seeds),
        "seed_ranges": {"train": list(seed_ranges.train), "val": list(seed_ranges.val), "test": list(seed_ranges.test)},
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "profile": "smoke" if smoke else "formal",
        "runs": plan_rows,
    }
    save_json(out_dir / "phase1r_plan.json", plan)
    save_rows(plan_rows, out_dir / "phase1r_plan.csv")
    return plan_rows


def load_existing_rows(out_dir: Path) -> List[Dict]:
    path = out_dir / "phase1r_run_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase1R RGCF corrected GPU benchmark.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--methods", default="rgcf", help="Learned methods to run: rgcf, me-a0, or rgcf,me-a0. Rule baselines always run.")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "phase1r_rgcf_compare"))
    p.add_argument("--dataset-store-root", default=str(PROJECT_ROOT / "dataset_store"))
    p.add_argument("--scenario-presets", default=None)
    p.add_argument("--mixed-dataset-dir", default=None)
    p.add_argument("--model-seeds", default=None)
    p.add_argument("--train-seed-range", "--train-seeds", dest="train_seed_range", default=None)
    p.add_argument("--val-seed-range", "--val-seeds", dest="val_seed_range", default=None)
    p.add_argument("--test-seed-range", "--test-seeds", dest="test_seed_range", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-sim-cache", action="store_true")
    p.add_argument("--force-regenerate-sims", action="store_true")
    p.add_argument("--smoke-duration", type=float, default=20.0, help="Trajectory duration used only with --smoke.")
    return p.parse_args()


def resolved_runtime(args: argparse.Namespace) -> Tuple[List[int], SeedRanges, int]:
    if args.smoke:
        model_seeds = parse_int_list(args.model_seeds or "0")
        train = parse_seed_range(args.train_seed_range or "10-11")
        val = parse_seed_range(args.val_seed_range or "70")
        test = parse_seed_range(args.test_seed_range or "90")
        epochs = int(args.epochs if args.epochs is not None else 2)
    else:
        model_seeds = parse_int_list(args.model_seeds or "0,1,2,3,4")
        train = parse_seed_range(args.train_seed_range or "10-69")
        val = parse_seed_range(args.val_seed_range or "70-89")
        test = parse_seed_range(args.test_seed_range or "90-109")
        epochs = int(args.epochs if args.epochs is not None else 80)
    return model_seeds, SeedRanges(train=train, val=val, test=test), epochs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    scenes = build_scenes(args.scenario_presets)
    learned_methods = parse_methods(args.methods)
    model_seeds, seed_ranges, epochs = resolved_runtime(args)
    smoke_duration = float(args.smoke_duration) if args.smoke else None
    plan_rows = save_plan(
        out_dir=out_dir,
        scenes=scenes,
        learned_methods=learned_methods,
        model_seeds=model_seeds,
        seed_ranges=seed_ranges,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        smoke=args.smoke,
    )
    print("=" * 88)
    print("Phase1R RGCF benchmark plan")
    print("[train_scene_set] S1R+S2R")
    print(f"[methods] {', '.join(RULE_METHODS + [m.display_name for m in learned_methods])}")
    print(f"[model_init_seeds] {', '.join(str(s) for s in model_seeds)}")
    print(f"[runs] {len(plan_rows)}")
    print(f"[plan_json] {out_dir / 'phase1r_plan.json'}")
    print("=" * 88)
    if args.dry_run:
        return

    from core.result_manager import ResultManager
    from experiments.train import run_train_experiment

    mixed_dataset_dir, scene_splits = prepare_mixed_dataset(
        scenes=scenes,
        seed_ranges=seed_ranges,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
        dataset_store_root=args.dataset_store_root,
        mixed_dataset_dir=args.mixed_dataset_dir,
        use_sim_cache=not args.no_sim_cache,
        force_regenerate_sims=args.force_regenerate_sims,
        smoke_duration=smoke_duration,
    )

    health = sensor_health_rows(scenes, scene_splits)
    save_json(out_dir / "phase1r_sensor_health_by_scene.json", health)
    save_rows(health, out_dir / "phase1r_sensor_health_by_scene.csv")
    evidence = evidence_report_rows(scenes, scene_splits)
    save_json(out_dir / "phase1r_evidence_residual_report.json", evidence)
    save_rows(evidence, out_dir / "phase1r_evidence_residual_report.csv")

    rows = load_existing_rows(out_dir) if args.resume else []
    completed = {str(r.get("run_label")) for r in rows if r.get("status") == "ok"}
    baseline_done = {str(r.get("scenario_id")) for r in rows if r.get("method_category") == "rule_baseline" and r.get("status") == "ok"}

    for scene in scenes:
        if scene.scene_id in baseline_done:
            continue
        print(f"[rule_baselines] evaluating {scene.scene_id}")
        baseline_rows = evaluate_rule_baselines(
            scene=scene,
            test_sims=scene_splits[scene.scene_id]["test"],
            out_dir=out_dir,
            seed_ranges=seed_ranges,
        )
        for row in baseline_rows:
            row["dataset_dir"] = mixed_dataset_dir
        rows.extend(baseline_rows)
    save_run_outputs(rows, out_dir)

    train_scene = scenes[0]
    total_learned_runs = len(learned_methods) * len(model_seeds)
    run_idx = 0
    for method in learned_methods:
        for model_seed in model_seeds:
            run_idx += 1
            train_run_label = f"phase1r_mixed_{method.run_slug}__modelseed{model_seed}"
            eval_labels = [f"{train_run_label}__test_{scene.scene_id}" for scene in scenes]
            print("=" * 88)
            print(f"[run {run_idx}/{total_learned_runs}] {train_run_label}")
            if args.resume and all(label in completed for label in eval_labels):
                print(f"[skip completed] {train_run_label}")
                continue

            bundle = build_learned_bundle(
                train_scene,
                method,
                model_seed=model_seed,
                seed_ranges=seed_ranges,
                epochs=epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                device=args.device,
                smoke_duration=smoke_duration,
            )
            bundle.identity.preset_name = train_run_label
            bundle.identity.experiment_name = train_run_label
            bundle.identity.scene_name = "S1R_S2R_mixed"

            res = run_train_experiment(
                bundle,
                epochs=epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                dataset_store_root=args.dataset_store_root,
                dataset_dir=mixed_dataset_dir,
            )
            rm = ResultManager(bundle, mode="train", experiment_name_override=train_run_label)
            rm.save_train_result(
                train_info=res.train_info,
                history=res.history,
                quick_baseline_metrics=res.quick_baseline_metrics,
                quick_gnn_metrics=res.quick_gnn_metrics,
                quick_sim=res.quick_sim,
                model=res.model,
                quick_gnn_timeseries=res.quick_gnn_timeseries,
            )

            for scene in scenes:
                eval_run_label = f"{train_run_label}__test_{scene.scene_id}"
                if args.resume and eval_run_label in completed:
                    continue
                metrics = evaluate_learned(
                    model=res.model,
                    bundle=bundle,
                    scene=scene,
                    test_sims=scene_splits[scene.scene_id]["test"],
                    out_dir=out_dir,
                    run_label=eval_run_label,
                )
                rows.append({
                    "status": "ok",
                    "scenario_id": scene.scene_id,
                    "scenario_label": scene.label,
                    "scenario_preset": scene.preset_name,
                    "method": method.display_name,
                    "method_category": "learned",
                    "model_seed": model_seed,
                    "run_label": eval_run_label,
                    "train_run_label": train_run_label,
                    "run_dir": str(rm.run_dir),
                    "dataset_dir": mixed_dataset_dir,
                    "train_dataset_protocol": "S1R/S2R mixed",
                    "train_seed_range": f"{seed_ranges.train[0]}-{seed_ranges.train[1]} per scene",
                    "val_seed_range": f"{seed_ranges.val[0]}-{seed_ranges.val[1]} per scene",
                    "test_seed_range": f"{seed_ranges.test[0]}-{seed_ranges.test[1]}",
                    "epochs": epochs,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "hidden_dim": args.hidden_dim,
                    "best_epoch": res.train_info.get("best_epoch", ""),
                    "best_val_loss": res.train_info.get("best_val_loss", ""),
                    "test_loss": res.train_info.get("test_loss", ""),
                    "test_loss_pos": res.train_info.get("test_loss_pos", ""),
                    **metrics,
                })
                completed.add(eval_run_label)
            save_run_outputs(rows, out_dir)

    print("=" * 88)
    print("[done] Phase1R RGCF benchmark")
    save_run_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
