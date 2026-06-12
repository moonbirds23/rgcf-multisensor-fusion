from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import load_experiment_bundle
from core.types import RunRequest


DEFAULT_SCENES = {
    "S1": "phase1_s1_balanced_hetero_nominal",
    "S2": "phase1_s2_clustered_hetero_nominal",
    "S3": "phase1_s3_maneuver_hetero_nominal",
}

SCENE_LABELS = {
    "S1": "S1-balanced",
    "S2": "S2-clustered",
    "S3": "S3-maneuver",
}

DEFAULT_METHODS = [
    "P0_post_only_single_stream",
    "P1_dual_stream_direct",
    "P11_feature_stable_reliability_no_cov",
]

EXTENDED_METHODS = [
    "P0_post_only_single_stream",
    "P1_dual_stream_direct",
    "P4_full_gate_no_cov_formula",
    "P11_feature_stable_reliability_no_cov",
    "P12_recovery_aware_full_gate_no_cov",
]

RULE_BASELINES = ["AVG", "CI-multi", "WAA-MM", "best-single"]


@dataclass(frozen=True)
class Phase1Scene:
    scene_id: str
    label: str
    preset_name: str
    dataset_dir: str = ""


@dataclass(frozen=True)
class SeedRanges:
    train: Tuple[int, int]
    val: Tuple[int, int]
    test: Tuple[int, int]


def _disable_gate_loss(model) -> None:
    model.gate_supervision_weight = 0.0
    model.gate_prior_weight = 0.0


def _disable_cov(model) -> None:
    model.use_cov_calibration = False
    model.use_cov_in_fusion = False
    model.cov_weight_beta = 0.0
    model.cov_prior_weight = 0.0
    model.cov_sep_weight = 0.0


def apply_method_variant(bundle, method: str):
    """Apply Phase-1 method variants on top of a scene preset.

    The scene preset owns scenario/fault settings. This function only changes
    model-side choices so Agent 1 can evolve S1/S2/S3 independently.
    """
    model = bundle.model

    if method == "P0_post_only_single_stream":
        model.model_name = "original_gnn_fusion"
        model.use_post_stream = True
        model.use_meas_stream = False
        model.use_gate = False
        model.fusion_variant = "post_only"
        return bundle

    if method == "P1_dual_stream_direct":
        model.model_name = "post_meas_direct_fusion"
        model.use_post_stream = True
        model.use_meas_stream = True
        model.use_gate = False
        model.fusion_variant = "post_meas_direct"
        return bundle

    if method == "P4_full_gate_no_cov_formula":
        model.model_name = "post_meas_soft_gate_fusion"
        model.use_post_stream = True
        model.use_meas_stream = True
        model.use_gate = True
        model.gate_init_bias = 0.0
        model.use_gate_supervision = True
        model.use_balanced_gate_loss = True
        model.gate_supervision_weight = 0.05
        model.gate_prior_weight = 0.005
        model.gate_prior_mean = 0.75
        model.normal_gate_target = 0.8
        model.fault_gate_target = 0.2
        _disable_cov(model)
        model.use_gate_on_meas_feature = True
        model.gate_weight_alpha = 1.0
        model.fusion_variant = "Full Gate without Covariance Path"
        return bundle

    if method == "P11_feature_stable_reliability_no_cov":
        model.model_name = "post_meas_soft_gate_fusion"
        model.use_post_stream = True
        model.use_meas_stream = True
        model.use_gate = True
        model.gate_init_bias = 0.0
        model.use_gate_supervision = True
        model.use_balanced_gate_loss = True
        model.gate_supervision_weight = 0.05
        model.gate_prior_weight = 0.005
        model.gate_prior_mean = 0.75
        model.normal_gate_target = 0.8
        model.fault_gate_target = 0.2
        _disable_cov(model)
        model.use_meas_in_representation = False
        model.use_gate_on_meas_feature = False
        model.gate_weight_alpha = 1.0
        model.base_logit_temperature = 2.0
        model.weight_uniform_mix = 0.05
        model.fusion_variant = (
            "Feature-Stable Reliability-Weighted Fusion without Measurement "
            "Representation or Covariance Path"
        )
        return bundle

    if method == "P12_recovery_aware_full_gate_no_cov":
        model.model_name = "post_meas_soft_gate_fusion"
        model.use_post_stream = True
        model.use_meas_stream = True
        model.use_gate = True
        model.gate_init_bias = 0.0
        model.use_gate_supervision = True
        model.use_balanced_gate_loss = True
        model.gate_supervision_weight = 0.05
        model.gate_prior_weight = 0.005
        model.gate_prior_mean = 0.75
        model.normal_gate_target = 0.8
        model.fault_gate_target = 0.2
        _disable_cov(model)
        model.use_gate_on_meas_feature = True
        model.gate_weight_alpha = 1.0
        model.use_error_aware_gate_target = True
        model.gate_error_tau = 5.0
        model.gate_target_min = 0.1
        model.fault_weight_loss_weight = 0.05
        model.fault_weight_margin = 0.1
        model.fusion_variant = "Recovery-Aware Full Gate without Covariance Path"
        return bundle

    if method == "P5_cov_fusion_formula_only":
        model.model_name = "post_meas_soft_gate_fusion"
        model.use_post_stream = True
        model.use_meas_stream = True
        model.use_gate = True
        model.gate_init_bias = 0.0
        model.use_gate_on_meas_feature = False
        model.gate_weight_alpha = 0.0
        _disable_gate_loss(model)
        model.use_cov_calibration = True
        model.use_cov_in_fusion = True
        model.cov_weight_beta = 0.0
        model.fusion_variant = "Covariance Fusion Formula Only"
        return bundle

    raise ValueError(f"Unknown Phase-1 method variant: {method}")


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


def parse_mapping(text: str | None) -> Dict[str, str]:
    if not text:
        return {}
    out: Dict[str, str] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected KEY=value mapping item, got: {part}")
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build_scenes(preset_overrides: str | None, dataset_dirs: str | None) -> List[Phase1Scene]:
    presets = dict(DEFAULT_SCENES)
    presets.update(parse_mapping(preset_overrides))
    datasets = parse_mapping(dataset_dirs)
    scenes: List[Phase1Scene] = []
    for scene_id in ("S1", "S2", "S3"):
        scenes.append(
            Phase1Scene(
                scene_id=scene_id,
                label=SCENE_LABELS.get(scene_id, scene_id),
                preset_name=presets[scene_id],
                dataset_dir=datasets.get(scene_id, ""),
            )
        )
    return scenes


def annotate_phase1_sims(
    sims: Sequence[Dict],
    *,
    scene: Phase1Scene,
    split_name: str,
) -> List[Dict]:
    out: List[Dict] = []
    for sim in sims:
        sim["phase1_scene_id"] = scene.scene_id
        sim["phase1_scene_label"] = scene.label
        sim["phase1_scene_preset"] = scene.preset_name
        sim["split_name"] = split_name
        out.append(sim)
    return out


def configure_seed_ranges(bundle, ranges: SeedRanges) -> None:
    bundle.train.train_seed_start, bundle.train.train_seed_end = ranges.train
    bundle.train.val_seed_start, bundle.train.val_seed_end = ranges.val
    bundle.train.test_seed_start, bundle.train.test_seed_end = ranges.test


def build_bundle(
    scene: Phase1Scene,
    method: str,
    *,
    model_seed: int,
    seed_ranges: SeedRanges,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    device: str,
    require_clean: bool,
):
    try:
        bundle = load_experiment_bundle(
            RunRequest(
                mode="train",
                preset_name=scene.preset_name,
                device=device,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                repeat_runs=1,
                experiment_name=None,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load Phase-1 preset '{scene.preset_name}' for {scene.scene_id}. "
            "Agent 1 may still need to add this preset, or pass --scenario-presets."
        ) from exc

    fault_mode = str(getattr(bundle.fault, "mode", ""))
    if require_clean and fault_mode != "clean":
        raise RuntimeError(
            f"Phase 1 requires nominal/clean presets. {scene.scene_id} preset "
            f"'{scene.preset_name}' resolved to fault_mode='{fault_mode}'."
        )

    configure_seed_ranges(bundle, seed_ranges)
    apply_method_variant(bundle, method)
    bundle.train.model_seed = int(model_seed)

    run_label = f"phase1_mixed_{method}__modelseed{model_seed}"
    bundle.identity.preset_name = run_label
    bundle.identity.experiment_name = run_label
    return bundle


def build_phase1_sims_for_scene(
    *,
    scene: Phase1Scene,
    seed_ranges: SeedRanges,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    device: str,
    require_clean: bool,
    use_sim_cache: bool,
    force_regenerate_sims: bool,
) -> Tuple[object, Dict[str, List[Dict]]]:
    from training.trainer import build_sim_list_from_seed_range

    bundle = build_bundle(
        scene,
        "P0_post_only_single_stream",
        model_seed=0,
        seed_ranges=seed_ranges,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        device=device,
        require_clean=require_clean,
    )
    bundle.identity.preset_name = f"phase1_{scene.scene_id}_dataset_source"
    bundle.identity.experiment_name = f"phase1_{scene.scene_id}_dataset_source"

    train_sims = build_sim_list_from_seed_range(
        bundle,
        seed_ranges.train[0],
        seed_ranges.train[1],
        "train",
        use_sim_cache=use_sim_cache,
        force_regenerate_sims=force_regenerate_sims,
    )
    val_sims = build_sim_list_from_seed_range(
        bundle,
        seed_ranges.val[0],
        seed_ranges.val[1],
        "val",
        use_sim_cache=use_sim_cache,
        force_regenerate_sims=force_regenerate_sims,
    )
    test_sims = build_sim_list_from_seed_range(
        bundle,
        seed_ranges.test[0],
        seed_ranges.test[1],
        "test",
        use_sim_cache=use_sim_cache,
        force_regenerate_sims=force_regenerate_sims,
    )
    splits = {
        "train": annotate_phase1_sims(train_sims, scene=scene, split_name="train"),
        "val": annotate_phase1_sims(val_sims, scene=scene, split_name="val"),
        "test": annotate_phase1_sims(test_sims, scene=scene, split_name="test"),
    }
    return bundle, splits


def _split_mixed_sims_by_scene(sims: Sequence[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for sim in sims:
        scene_id = sim.get("phase1_scene_id")
        if not scene_id:
            raise ValueError(
                "Mixed Phase-1 dataset is missing 'phase1_scene_id'. "
                "Regenerate it with this script instead of using an old per-scene store."
            )
        out.setdefault(str(scene_id), []).append(sim)
    return out


def prepare_mixed_phase1_dataset(
    *,
    scenes: Sequence[Phase1Scene],
    seed_ranges: SeedRanges,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_dim: int,
    device: str,
    require_clean: bool,
    dataset_store_root: str,
    mixed_dataset_dir: str | None,
    use_sim_cache: bool,
    force_regenerate_sims: bool,
) -> Tuple[str, Dict[str, Dict[str, List[Dict]]]]:
    from data.dataset_store import load_raw_dataset_store, save_raw_dataset_store

    if mixed_dataset_dir:
        ds = load_raw_dataset_store(mixed_dataset_dir)
        by_scene = {
            scene.scene_id: {
                "train": _split_mixed_sims_by_scene(ds["train_sims"]).get(scene.scene_id, []),
                "val": _split_mixed_sims_by_scene(ds["val_sims"]).get(scene.scene_id, []),
                "test": _split_mixed_sims_by_scene(ds["test_sims"]).get(scene.scene_id, []),
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
        print(f"[dataset] building Phase-1 sims for {scene.scene_id}: {scene.preset_name}")
        bundle, splits = build_phase1_sims_for_scene(
            scene=scene,
            seed_ranges=seed_ranges,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            device=device,
            require_clean=require_clean,
            use_sim_cache=use_sim_cache,
            force_regenerate_sims=force_regenerate_sims,
        )
        if dataset_bundle is None:
            dataset_bundle = bundle
        scene_splits[scene.scene_id] = splits
        mixed_train.extend(splits["train"])
        mixed_val.extend(splits["val"])
        mixed_test.extend(splits["test"])

    if dataset_bundle is None:
        raise RuntimeError("No Phase-1 scenes were provided.")

    dataset_bundle.identity.preset_name = "phase1_s1_s2_s3_mixed_nominal"
    dataset_bundle.identity.experiment_name = "phase1_s1_s2_s3_mixed_nominal"
    dataset_bundle.identity.scene_name = "S1_S2_S3_mixed"
    dataset_bundle.identity.fault_mode = "clean"

    ds_dir = save_raw_dataset_store(
        dataset_bundle,
        mixed_train,
        mixed_val,
        mixed_test,
        root_dir=dataset_store_root,
        note="phase1_mixed_nominal_s1_s2_s3",
    )
    print(f"[dataset] mixed training store: {ds_dir}")
    return str(ds_dir), scene_splits


def metric_summary(errors: Sequence[float]) -> Dict[str, float | int | str]:
    vals = [float(v) for v in errors if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"num_points": 0, "rmse": "", "p95": "", "p99": "", "max": ""}
    import numpy as np

    arr = np.asarray(vals, dtype=np.float64)
    return {
        "num_points": int(arr.size),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


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


def save_plan(
    *,
    out_dir: Path,
    scenes: Sequence[Phase1Scene],
    methods: Sequence[str],
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
        for baseline in RULE_BASELINES:
            plan_rows.append(
                {
                    "action": "evaluate_rule_baseline",
                    "scenario_id": scene.scene_id,
                    "scenario_label": scene.label,
                    "scenario_preset": scene.preset_name,
                    "train_dataset": "S1/S2/S3 mixed",
                    "test_dataset": scene.scene_id,
                    "method": baseline,
                    "model_seed": "",
                    "train_seed_range": f"{seed_ranges.train[0]}-{seed_ranges.train[1]}",
                    "val_seed_range": f"{seed_ranges.val[0]}-{seed_ranges.val[1]}",
                    "test_seed_range": f"{seed_ranges.test[0]}-{seed_ranges.test[1]}",
                    "epochs": "",
                    "lr": "",
                    "batch_size": "",
                    "hidden_dim": "",
                    "profile": "smoke" if smoke else "formal",
                }
            )

    for method in methods:
        for model_seed in model_seeds:
            plan_rows.append(
                {
                    "action": "train_on_mixed_eval_s1_s2_s3",
                    "scenario_id": "S1+S2+S3",
                    "scenario_label": "mixed-train",
                    "scenario_preset": ",".join(s.preset_name for s in scenes),
                    "train_dataset": "S1/S2/S3 mixed",
                    "test_dataset": "S1,S2,S3 separately",
                    "method": method,
                    "model_seed": model_seed,
                    "train_seed_range": f"{seed_ranges.train[0]}-{seed_ranges.train[1]} per scene",
                    "val_seed_range": f"{seed_ranges.val[0]}-{seed_ranges.val[1]} per scene",
                    "test_seed_range": f"{seed_ranges.test[0]}-{seed_ranges.test[1]} per scene",
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "profile": "smoke" if smoke else "formal",
                }
            )

    save_json(
        out_dir / "phase1_plan.json",
        {
            "benchmark": "Phase 1 Nominal Fusion Benchmark",
            "scenes": [asdict(s) for s in scenes],
            "methods": list(methods),
            "main_method": "P11_feature_stable_reliability_no_cov",
            "training_protocol": "single S1/S2/S3 mixed training set; evaluate separately on S1, S2, S3",
            "rule_baselines": list(RULE_BASELINES),
            "model_init_seeds": list(model_seeds),
            "seed_ranges": {
                "train": list(seed_ranges.train),
                "val": list(seed_ranges.val),
                "test": list(seed_ranges.test),
            },
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "profile": "smoke" if smoke else "formal",
            "runs": plan_rows,
            "agent1_required_presets": [
                s.preset_name for s in scenes if s.preset_name in DEFAULT_SCENES.values()
            ],
        },
    )
    save_rows(plan_rows, out_dir / "phase1_plan.csv")
    return plan_rows


def load_existing_rows(out_dir: Path) -> List[Dict]:
    path = out_dir / "phase1_run_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def save_run_outputs(rows: List[Dict], out_dir: Path) -> None:
    save_json(out_dir / "phase1_run_summary.json", rows)
    save_rows(rows, out_dir / "phase1_run_summary.csv")
    aggregates = aggregate_rows(rows)
    save_json(out_dir / "phase1_aggregate_by_scene.json", aggregates["by_scene"])
    save_rows(aggregates["by_scene"], out_dir / "phase1_aggregate_by_scene.csv")
    save_json(out_dir / "phase1_aggregate_overall.json", aggregates["overall"])
    save_rows(aggregates["overall"], out_dir / "phase1_aggregate_overall.csv")
    print(f"[summary_json] {out_dir / 'phase1_run_summary.json'}")
    print(f"[summary_csv] {out_dir / 'phase1_run_summary.csv'}")
    print(f"[aggregate_csv] {out_dir / 'phase1_aggregate_by_scene.csv'}")


def _mean(values: Iterable[float]) -> float | str:
    vals = [float(v) for v in values if v not in ("", None) and math.isfinite(float(v))]
    if not vals:
        return ""
    return sum(vals) / len(vals)


def _std(values: Iterable[float]) -> float | str:
    vals = [float(v) for v in values if v not in ("", None) and math.isfinite(float(v))]
    if len(vals) <= 1:
        return ""
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def aggregate_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    metrics = ["rmse", "p95", "p99", "max"]

    def grouped(keys: Sequence[str]) -> List[Dict]:
        buckets: Dict[Tuple, List[Dict]] = {}
        for row in ok_rows:
            key = tuple(row.get(k, "") for k in keys)
            buckets.setdefault(key, []).append(row)
        out: List[Dict] = []
        for key, items in sorted(buckets.items()):
            entry = {k: v for k, v in zip(keys, key)}
            entry["n_runs"] = len(items)
            entry["num_points_total"] = sum(int(i.get("num_points") or 0) for i in items)
            for metric in metrics:
                vals = [i.get(metric) for i in items]
                entry[f"{metric}_mean"] = _mean(vals)
                entry[f"{metric}_std"] = _std(vals)
            out.append(entry)
        return out

    return {
        "by_scene": grouped(["scenario_id", "scenario_label", "method", "method_category"]),
        "overall": grouped(["method", "method_category"]),
    }


def evaluate_learned_testset(
    *,
    model,
    bundle,
    scene: Phase1Scene,
    test_sims: Sequence[Dict],
    out_dir: Path,
    run_label: str,
) -> Dict:
    import torch

    from training.evaluator import evaluate_single_sim_fusion_with_timeseries

    device = torch.device(bundle.base.runtime.device)
    detail_rows: List[Dict] = []
    errors: List[float] = []

    for sim in test_sims:
        ev = evaluate_single_sim_fusion_with_timeseries(sim, model, bundle, device)
        seed = sim.get("seed")
        for row in ev["timeseries"]:
            err = float(row["error_pos"])
            errors.append(err)
            detail_rows.append(
                {
                    "run_label": run_label,
                    "scenario_id": scene.scene_id,
                    "scenario_label": scene.label,
                    "seed": seed,
                    "k": row.get("k"),
                    "t": row.get("t"),
                    "error_pos": err,
                }
            )

    detail_dir = out_dir / "phase1_eval_details"
    safe_label = run_label.replace("\\", "_").replace("/", "_").replace(":", "_")
    save_rows(detail_rows, detail_dir / f"{safe_label}_errors.csv")
    out = metric_summary(errors)
    out["num_sims"] = len(test_sims)
    return out


def _baseline_error_tracks(sim) -> Dict[str, List[float]]:
    import numpy as np

    from simulation.fusion_baselines import extract_baseline_trajectories, fuse_waa_mm_sequence

    truth_xy = np.asarray(sim["x_truth_4d"], dtype=np.float64)[:, 0:2]
    tracks = extract_baseline_trajectories(sim)
    out: Dict[str, List[float]] = {}

    mapping = {
        "AVG": "avg_xy",
        "CI-multi": "ci_multi_xy",
        "best-single": "best_single_xy",
    }
    for name, key in mapping.items():
        if key not in tracks:
            continue
        xy = np.asarray(tracks[key], dtype=np.float64)
        out[name] = list(np.sqrt(np.sum((xy - truth_xy) ** 2, axis=1)))

    valid_mask = sim.get("valid_mask", None)
    if valid_mask is None:
        valid_mask = np.ones(sim["xhat"].shape[:2], dtype=np.float64)
    x_waa, _, _ = fuse_waa_mm_sequence(sim["xhat"], sim["Phat"], valid_mask)
    waa_xy = np.asarray(x_waa, dtype=np.float64)[:, 0:2]
    out["WAA-MM"] = list(np.sqrt(np.sum((waa_xy - truth_xy) ** 2, axis=1)))
    return out


def evaluate_rule_baselines(
    *,
    scene: Phase1Scene,
    test_sims: Sequence[Dict],
    out_dir: Path,
    seed_ranges: SeedRanges,
) -> List[Dict]:
    buckets = {name: [] for name in RULE_BASELINES}
    for sim in test_sims:
        for name, vals in _baseline_error_tracks(sim).items():
            buckets.setdefault(name, []).extend(vals)

    rows: List[Dict] = []
    for name in RULE_BASELINES:
        metrics = metric_summary(buckets.get(name, []))
        row = {
            "status": "ok",
            "scenario_id": scene.scene_id,
            "scenario_label": scene.label,
            "scenario_preset": scene.preset_name,
            "method": name,
            "method_category": "rule_baseline",
            "model_seed": "",
            "run_label": f"phase1_{scene.scene_id}_{name}",
            "run_dir": "",
            "dataset_dir": "phase1_mixed_train_store",
            "train_seed_range": f"{seed_ranges.train[0]}-{seed_ranges.train[1]}",
            "val_seed_range": f"{seed_ranges.val[0]}-{seed_ranges.val[1]}",
            "test_seed_range": f"{seed_ranges.test[0]}-{seed_ranges.test[1]}",
            **metrics,
        }
        rows.append(row)

    detail_dir = out_dir / "phase1_eval_details"
    save_json(
        detail_dir / f"phase1_{scene.scene_id}_rule_baselines.json",
        {"scenario": asdict(scene), "num_test_sims": len(test_sims), "rows": rows},
    )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Phase 1 Nominal Fusion Benchmark over S1/S2/S3."
    )
    p.add_argument("--dry-run", action="store_true", help="Print and save the run plan only.")
    p.add_argument("--smoke", action="store_true", help="Use a tiny seed grid and short training.")
    p.add_argument("--out-dir", default="results/phase1_nominal_benchmark")
    p.add_argument("--dataset-store-root", default="dataset_store")
    p.add_argument(
        "--scenario-presets",
        default=None,
        help=(
            "Comma mapping for Agent-1 presets, for example "
            "S1=phase1_s1_balanced_hetero_nominal,"
            "S2=phase1_s2_clustered_hetero_nominal,"
            "S3=phase1_s3_maneuver_hetero_nominal."
        ),
    )
    p.add_argument(
        "--dataset-dirs",
        default=None,
        help="Deprecated. Phase 1 now uses one mixed S1/S2/S3 training store.",
    )
    p.add_argument(
        "--mixed-dataset-dir",
        default=None,
        help="Optional existing mixed Phase-1 dataset store generated by this script.",
    )
    p.add_argument(
        "--methods",
        default="default",
        help="default/compare (P0,P1,P11), main (P11), extended, or comma-separated variants.",
    )
    p.add_argument("--model-seeds", default=None, help="Model init seeds, e.g. 0,1,2,3,4.")
    p.add_argument("--train-seed-range", default=None, help="Rollout train seed range, e.g. 10-69.")
    p.add_argument("--val-seed-range", default=None, help="Rollout val seed range, e.g. 70-89.")
    p.add_argument("--test-seed-range", default=None, help="Rollout test seed range, e.g. 90-109.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true", help="Skip run labels already in phase1_run_summary.json.")
    p.add_argument(
        "--allow-nonclean-presets",
        action="store_true",
        help="Disable the nominal fault_mode='clean' guard. Intended only for compatibility debugging.",
    )
    p.add_argument(
        "--no-sim-cache",
        action="store_true",
        help="Disable simulation cache when building the mixed dataset.",
    )
    p.add_argument(
        "--force-regenerate-sims",
        action="store_true",
        help="Ignore existing cached simulations when building the mixed dataset.",
    )
    return p.parse_args()


def select_methods(spec: str) -> List[str]:
    spec = str(spec).strip()
    if spec == "default":
        return list(DEFAULT_METHODS)
    if spec == "main":
        return ["P11_feature_stable_reliability_no_cov"]
    if spec == "compare":
        return [
            "P0_post_only_single_stream",
            "P1_dual_stream_direct",
            "P11_feature_stable_reliability_no_cov",
        ]
    if spec == "extended":
        return list(EXTENDED_METHODS)
    return [x.strip() for x in spec.split(",") if x.strip()]


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


def print_plan_summary(plan_rows: List[Dict], out_dir: Path) -> None:
    train_rows = [r for r in plan_rows if r["action"] == "train_on_mixed_eval_s1_s2_s3"]
    baseline_rows = [r for r in plan_rows if r["action"] == "evaluate_rule_baseline"]
    scenes = sorted({r["scenario_id"] for r in baseline_rows})
    methods = sorted({r["method"] for r in train_rows})
    model_seeds = sorted({str(r["model_seed"]) for r in train_rows})
    print("=" * 88)
    print("Phase 1 Nominal Fusion Benchmark plan")
    print("[train_scene_set] S1+S2+S3")
    print(f"[test_scenes] {', '.join(scenes)}")
    print("[training_protocol] one S1/S2/S3 mixed train/val set, separate S1/S2/S3 tests")
    print(f"[methods] {', '.join(methods)}")
    print(f"[model_init_seeds] {', '.join(model_seeds)}")
    print(f"[train_eval_runs] {len(train_rows)}")
    print(f"[rule_baseline_rows] {len(baseline_rows)}")
    print(f"[out_dir] {out_dir}")
    print(f"[plan_json] {out_dir / 'phase1_plan.json'}")
    print(f"[plan_csv] {out_dir / 'phase1_plan.csv'}")
    print("=" * 88)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.dataset_dirs:
        raise ValueError(
            "--dataset-dirs belongs to the old per-scene training protocol. "
            "Use --mixed-dataset-dir for the current S1/S2/S3 mixed protocol."
        )

    scenes = build_scenes(args.scenario_presets, args.dataset_dirs)
    methods = select_methods(args.methods)
    model_seeds, seed_ranges, epochs = resolved_runtime(args)

    plan_rows = save_plan(
        out_dir=out_dir,
        scenes=scenes,
        methods=methods,
        model_seeds=model_seeds,
        seed_ranges=seed_ranges,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        smoke=args.smoke,
    )
    print_plan_summary(plan_rows, out_dir)

    if args.dry_run:
        return

    from core.result_manager import ResultManager
    from experiments.train import run_train_experiment

    mixed_dataset_dir, scene_splits = prepare_mixed_phase1_dataset(
        scenes=scenes,
        seed_ranges=seed_ranges,
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
        require_clean=not args.allow_nonclean_presets,
        dataset_store_root=args.dataset_store_root,
        mixed_dataset_dir=args.mixed_dataset_dir,
        use_sim_cache=not args.no_sim_cache,
        force_regenerate_sims=args.force_regenerate_sims,
    )

    rows = load_existing_rows(out_dir) if args.resume else []
    completed = {str(r.get("run_label")) for r in rows if r.get("status") == "ok"}
    baseline_done = {
        str(r.get("scenario_id"))
        for r in rows
        if r.get("method_category") == "rule_baseline" and r.get("status") == "ok"
    }

    for scene in scenes:
        if scene.scene_id not in baseline_done:
            print(f"[rule_baselines] evaluating {scene.scene_id} on mixed-store test split")
            baseline_rows = evaluate_rule_baselines(
                scene=scene,
                test_sims=scene_splits[scene.scene_id]["test"],
                out_dir=out_dir,
                seed_ranges=seed_ranges,
            )
            for row in baseline_rows:
                row["dataset_dir"] = mixed_dataset_dir
            rows.extend(baseline_rows)
            baseline_done.add(scene.scene_id)
    save_run_outputs(rows, out_dir)

    total_runs = len(methods) * len(model_seeds)
    run_idx = 0
    train_scene = scenes[0]
    for method in methods:
        for model_seed in model_seeds:
            run_idx += 1
            train_run_label = f"phase1_mixed_{method}__modelseed{model_seed}"
            eval_labels = [f"{train_run_label}__test_{scene.scene_id}" for scene in scenes]
            print("=" * 88)
            print(f"[run {run_idx}/{total_runs}] {train_run_label}")
            if args.resume and all(label in completed for label in eval_labels):
                print(f"[skip completed] {train_run_label}")
                continue

            bundle = build_bundle(
                train_scene,
                method,
                model_seed=model_seed,
                seed_ranges=seed_ranges,
                epochs=epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                device=args.device,
                require_clean=not args.allow_nonclean_presets,
            )
            bundle.identity.preset_name = train_run_label
            bundle.identity.experiment_name = train_run_label
            bundle.identity.scene_name = "S1_S2_S3_mixed"

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
                    print(f"[skip completed eval] {eval_run_label}")
                    continue

                metrics = evaluate_learned_testset(
                    model=res.model,
                    bundle=bundle,
                    scene=scene,
                    test_sims=scene_splits[scene.scene_id]["test"],
                    out_dir=out_dir,
                    run_label=eval_run_label,
                )
                row = {
                    "status": "ok",
                    "scenario_id": scene.scene_id,
                    "scenario_label": scene.label,
                    "scenario_preset": scene.preset_name,
                    "method": method,
                    "method_category": "learned",
                    "model_seed": model_seed,
                    "run_label": eval_run_label,
                    "train_run_label": train_run_label,
                    "run_dir": str(rm.run_dir),
                    "dataset_dir": mixed_dataset_dir,
                    "train_dataset_protocol": "S1/S2/S3 mixed",
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
                }
                rows.append(row)
                completed.add(eval_run_label)

            save_run_outputs(rows, out_dir)

    print("=" * 88)
    print("[done] Phase 1 nominal benchmark")
    save_run_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
