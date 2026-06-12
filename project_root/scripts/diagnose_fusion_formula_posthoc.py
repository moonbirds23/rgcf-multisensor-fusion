from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_store import load_raw_dataset_store
from features.dataset import build_dataset_from_sim
from models.model_factory import build_model_from_bundle
from scripts.run_rgcf_v5_ablation import build_variant


DEFAULT_DATASET_DIR = (
    "dataset_store/20260601_144533__hetero_robust_matrix_mixed_v2_rgcf__"
    "hetero_robust_matr__70722427"
)
DEFAULT_M4_CHECKPOINT = (
    "results/20260601_224351__train__M4_V5_peer_consistency__hetero_4sensor_scene__"
    "post_meas_soft_gate_fusion/checkpoints/best_model.pt"
)
DEFAULT_M3_CHECKPOINT = (
    "results/20260601_170351__train__M3_V5_current__hetero_4sensor_scene__"
    "post_meas_soft_gate_fusion/checkpoints/best_model.pt"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return _project_root() / path


def _weighted_aa(xhat: np.ndarray, weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    active = weights * valid
    denom = np.sum(active)
    if denom <= 1e-12:
        active = valid
        denom = max(float(np.sum(active)), 1e-12)
    return np.sum(xhat * active[:, None], axis=0) / denom


def _equal_aa(xhat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    denom = max(float(np.sum(valid)), 1e-12)
    return np.sum(xhat * valid[:, None], axis=0) / denom


def _info_fusion(xhat: np.ndarray, pdiag: np.ndarray, weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    active = weights * valid
    y = 1.0 / np.maximum(pdiag, 1e-6)
    denom = np.sum(active[:, None] * y, axis=0)
    numer = np.sum(active[:, None] * y * xhat, axis=0)
    return numer / np.maximum(denom, 1e-12)


def _summary(errors: List[float]) -> Dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "n": int(arr.size),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def evaluate_formula_variants(*, variant: str, checkpoint: Path, dataset_dir: Path) -> Dict:
    bundle = build_variant(variant, epochs=80, lr=1e-3, batch_size=64, hidden_dim=64)
    device = torch.device("cpu")
    model = build_model_from_bundle(bundle).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds_obj = load_raw_dataset_store(dataset_dir)
    test_sims = ds_obj["test_sims"]

    errors = {
        "model_default": [],
        "info_no_covscale": [],
        "info_with_covscale": [],
        "aa_model_weights": [],
        "aa_equal_valid": [],
    }
    mode_errors = {k: {} for k in errors}
    top_rows = {k: [] for k in errors}

    for sim in test_sims:
        ds = build_dataset_from_sim(sim, bundle)
        mode = str(sim.get("effective_fault_mode") or sim.get("fault_mode") or "unknown")
        for k in errors:
            mode_errors[k].setdefault(mode, [])
        for i in range(len(ds)):
            item = ds[i]
            post_feat = item["post_feat"].to(device)
            meas_feat = item.get("meas_feat")
            if meas_feat is not None:
                meas_feat = meas_feat.to(device)
            mask = item["mask"].to(device)
            target = item["target"].detach().cpu().numpy()
            with torch.no_grad():
                out = model(post_feat=post_feat, mask=mask, meas_feat=meas_feat, return_weights=True)

            pred_default = out.pred.detach().cpu().numpy()
            weights = out.weights.detach().cpu().numpy()
            if weights.ndim > 1:
                weights = weights[0]
            valid = item["mask"].detach().cpu().numpy()

            pf = item["post_feat"].detach().cpu().numpy()
            xhat = pf[:, 0:4].copy()
            xhat[:, 0] *= float(bundle.scenario.pos_scale)
            xhat[:, 1] *= float(bundle.scenario.pos_scale)
            xhat[:, 2] *= float(bundle.scenario.vel_scale)
            xhat[:, 3] *= float(bundle.scenario.vel_scale)
            pdiag = np.expm1(pf[:, 4:8]).clip(min=1e-6)
            cov_scale = out.aux.get("cov_scale", None)
            if cov_scale is None:
                cov_np = np.ones(pdiag.shape[0], dtype=np.float64)
            else:
                cov_np = cov_scale.detach().cpu().numpy()
                if cov_np.ndim > 1:
                    cov_np = cov_np[0]

            preds = {
                "model_default": pred_default,
                "info_no_covscale": _info_fusion(xhat, pdiag, weights, valid),
                "info_with_covscale": _info_fusion(xhat, pdiag * cov_np[:, None], weights, valid),
                "aa_model_weights": _weighted_aa(xhat, weights, valid),
                "aa_equal_valid": _equal_aa(xhat, valid),
            }
            for name, pred in preds.items():
                err = float(np.sqrt(np.sum((pred[0:2] - target[0:2]) ** 2)))
                errors[name].append(err)
                mode_errors[name][mode].append(err)
                row = {
                    "seed": sim.get("seed"),
                    "mode": mode,
                    "sid": sim.get("effective_fault_sensor_id"),
                    "k": int(i),
                    "t": float(ds.t[i]),
                    "error_pos": err,
                }
                top_rows[name].append(row)

    result = {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "summary": {name: _summary(vals) for name, vals in errors.items()},
        "by_mode": {
            name: {mode: _summary(vals) for mode, vals in per_mode.items() if vals}
            for name, per_mode in mode_errors.items()
        },
        "top10": {
            name: sorted(rows, key=lambda r: r["error_pos"], reverse=True)[:10]
            for name, rows in top_rows.items()
        },
    }
    return result


def _rank_rows(result: Dict) -> List[Dict]:
    rows = []
    for name, metrics in result["summary"].items():
        row = {"formula": name, **metrics}
        bias = result["by_mode"].get(name, {}).get("bias_ramp", {})
        row["bias_ramp_p99"] = bias.get("p99", "")
        row["bias_ramp_max"] = bias.get("max", "")
        rows.append(row)
    return sorted(rows, key=lambda r: (r["p99"], r["max"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Posthoc compare information-style fusion and AA fusion formulas.")
    parser.add_argument("--variant", default="M4_V5_peer_consistency")
    parser.add_argument("--checkpoint", default=DEFAULT_M4_CHECKPOINT)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out_dir", default="results/fusion_formula_posthoc")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_formula_variants(
        variant=args.variant,
        checkpoint=_resolve(args.checkpoint),
        dataset_dir=_resolve(args.dataset_dir),
    )
    stem = args.variant.replace("\\", "_").replace("/", "_")
    json_path = out_dir / f"{stem}_formula_diagnosis.json"
    csv_path = out_dir / f"{stem}_formula_rank.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = _rank_rows(result)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[json] {json_path}")
    print(f"[csv] {csv_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
