from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import load_experiment_bundle
from core.types import RunRequest


DEFAULT_DATASET_DIR = (
    "dataset_store\\20260601_144533__hetero_robust_matrix_mixed_v2_rgcf__"
    "hetero_robust_matr__70722427"
)


def _set_identity(bundle, name: str):
    bundle.identity.preset_name = name
    bundle.identity.experiment_name = name
    return bundle


def _disable_gate_loss(model):
    model.gate_supervision_weight = 0.0
    model.gate_prior_weight = 0.0


def _disable_cov(model):
    model.use_cov_calibration = False
    model.use_cov_in_fusion = False
    model.cov_weight_beta = 0.0
    model.cov_prior_weight = 0.0
    model.cov_sep_weight = 0.0


def _base_bundle(preset: str, *, epochs: int, lr: float, batch_size: int, hidden_dim: int):
    return load_experiment_bundle(
        RunRequest(
            mode="train",
            preset_name=preset,
            device="cpu",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            repeat_runs=1,
            experiment_name=None,
        )
    )


def build_variant(name: str, *, epochs: int, lr: float, batch_size: int, hidden_dim: int):
    if name == "P0_post_only_single_stream":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_post_only_gnn",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        return _set_identity(bundle, name)

    if name == "P1_dual_stream_direct":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_dual_stream_plain_gnn",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        return _set_identity(bundle, name)

    if name == "P2_gate_feature_path_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = True
        bundle.model.gate_weight_alpha = 0.0
        return _set_identity(bundle, name)

    if name == "P3_gate_weight_path_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = False
        bundle.model.gate_weight_alpha = 1.0
        return _set_identity(bundle, name)

    if name == "P4_full_gate_no_cov_formula":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = True
        bundle.model.gate_weight_alpha = 1.0
        return _set_identity(bundle, name)

    if name == "P12_recovery_aware_full_gate_no_cov":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = True
        bundle.model.gate_weight_alpha = 1.0
        bundle.model.use_error_aware_gate_target = True
        bundle.model.gate_error_tau = 5.0
        bundle.model.gate_target_min = 0.1
        bundle.model.fault_weight_loss_weight = 0.05
        bundle.model.fault_weight_margin = 0.1
        bundle.model.fusion_variant = "Recovery-Aware Full Gate without Covariance Path"
        return _set_identity(bundle, name)

    if name == "P11_feature_stable_reliability_no_cov":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_meas_in_representation = False
        bundle.model.use_gate_on_meas_feature = False
        bundle.model.gate_weight_alpha = 1.0
        bundle.model.base_logit_temperature = 2.0
        bundle.model.weight_uniform_mix = 0.05
        bundle.model.fusion_variant = "Feature-Stable Reliability-Weighted Fusion without Measurement Representation or Covariance Path"
        return _set_identity(bundle, name)

    if name == "P5_cov_fusion_formula_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_gate_on_meas_feature = False
        bundle.model.gate_weight_alpha = 0.0
        _disable_gate_loss(bundle.model)
        bundle.model.use_cov_calibration = True
        bundle.model.use_cov_in_fusion = True
        bundle.model.cov_weight_beta = 0.0
        return _set_identity(bundle, name)

    if name == "P6_cov_weight_formula_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_gate_on_meas_feature = False
        bundle.model.gate_weight_alpha = 0.0
        _disable_gate_loss(bundle.model)
        bundle.model.use_cov_calibration = True
        bundle.model.use_cov_in_fusion = False
        bundle.model.cov_weight_beta = 1.0
        return _set_identity(bundle, name)

    if name == "P7_full_gate_cov_formula":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        return _set_identity(bundle, name)

    if name == "P8_full_gate_aa_formula":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = True
        bundle.model.gate_weight_alpha = 1.0
        bundle.model.output_fusion_mode = "aa"
        bundle.model.fusion_variant = "Dual-Stream Gated Adaptive-Average Fusion"
        return _set_identity(bundle, name)

    if name == "P9_gate_aa_cov_aux_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_gate_on_meas_feature = True
        bundle.model.gate_weight_alpha = 1.0
        bundle.model.output_fusion_mode = "aa"
        bundle.model.use_cov_calibration = True
        bundle.model.use_cov_in_fusion = False
        bundle.model.cov_weight_beta = 0.0
        bundle.model.fusion_variant = "Dual-Stream Gated AA Fusion with Cov Auxiliary Head"
        return _set_identity(bundle, name)

    if name == "P10_full_formula_peer_optional":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5_peer",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        return _set_identity(bundle, name)

    if name == "M0_A3_plain":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_dual_stream_plain_gnn",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        return _set_identity(bundle, name)

    if name == "M1_A4_gate_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = True
        bundle.model.gate_weight_alpha = 1.0
        return _set_identity(bundle, name)

    if name == "M2_V5a_gate_cov_fusion_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_cov_in_fusion = True
        bundle.model.cov_weight_beta = 0.0
        return _set_identity(bundle, name)

    if name == "M3_V5_current":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        return _set_identity(bundle, name)

    if name == "M4_V5_peer_consistency":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5_peer",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        return _set_identity(bundle, name)

    if name == "M4A_no_peer_features":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_peer_consistency_features = False
        bundle.model.post_in_dim = 9
        return _set_identity(bundle, name)

    if name == "M4A_zero_peer_features":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_peer_consistency_features = True
        bundle.model.zero_peer_consistency_features = True
        bundle.model.post_in_dim = 15
        return _set_identity(bundle, name)

    if name == "M4A_no_fault_weight_loss":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.fault_weight_loss_weight = 0.0
        return _set_identity(bundle, name)

    if name == "M4A_no_uniform_mix":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.weight_uniform_mix = 0.0
        return _set_identity(bundle, name)

    if name == "M4A_no_logit_temperature":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.base_logit_temperature = 1.0
        return _set_identity(bundle, name)

    if name == "M4A_no_cov_in_fusion":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_cov_in_fusion = False
        return _set_identity(bundle, name)

    if name == "M4A_no_cov_module":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        return _set_identity(bundle, name)

    if name == "M4A_no_gate_feature_path":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_gate_on_meas_feature = False
        return _set_identity(bundle, name)

    if name == "M4A_no_gate_weight_bias":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.gate_weight_alpha = 0.0
        return _set_identity(bundle, name)

    if name == "M4A_peer_loo_median":
        bundle = build_variant(
            "M4_V5_peer_consistency",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.peer_consistency_mode = "loo_median"
        return _set_identity(bundle, name)

    if name == "E1_gate_feature_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = True
        bundle.model.gate_weight_alpha = 0.0
        return _set_identity(bundle, name)

    if name == "E2_gate_weight_only":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_v2_rgcf",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        _disable_cov(bundle.model)
        bundle.model.use_gate_on_meas_feature = False
        bundle.model.gate_weight_alpha = 1.0
        return _set_identity(bundle, name)

    if name == "E4_cov_fusion_only_no_gate":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_gate_on_meas_feature = False
        bundle.model.gate_weight_alpha = 0.0
        _disable_gate_loss(bundle.model)
        bundle.model.use_cov_calibration = True
        bundle.model.use_cov_in_fusion = True
        bundle.model.cov_weight_beta = 0.0
        return _set_identity(bundle, name)

    if name == "E5_cov_weight_only_no_gate":
        bundle = _base_bundle(
            "hetero_robust_matrix_mixed_rgcf_v5",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.use_gate_on_meas_feature = False
        bundle.model.gate_weight_alpha = 0.0
        _disable_gate_loss(bundle.model)
        bundle.model.use_cov_calibration = True
        bundle.model.use_cov_in_fusion = False
        bundle.model.cov_weight_beta = 1.0
        return _set_identity(bundle, name)

    if name == "E7_cov_fusion_no_cov_loss":
        bundle = build_variant(
            "E4_cov_fusion_only_no_gate",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        bundle.model.cov_prior_weight = 0.0
        bundle.model.cov_sep_weight = 0.0
        return _set_identity(bundle, name)

    raise ValueError(f"Unknown variant: {name}")


DEFAULT_VARIANTS = [
    "M0_A3_plain",
    "M1_A4_gate_only",
    "M2_V5a_gate_cov_fusion_only",
    "M3_V5_current",
    "M4_V5_peer_consistency",
]

ALL_VARIANTS = [
    *DEFAULT_VARIANTS,
    "P11_feature_stable_reliability_no_cov",
    "P12_recovery_aware_full_gate_no_cov",
    "E1_gate_feature_only",
    "E2_gate_weight_only",
    "E4_cov_fusion_only_no_gate",
    "E5_cov_weight_only_no_gate",
    "E7_cov_fusion_no_cov_loss",
]

PAPER_M4_VARIANTS = [
    "M3_V5_current",
    "M4_V5_peer_consistency",
    "M4A_no_peer_features",
    "M4A_zero_peer_features",
    "M4A_no_fault_weight_loss",
    "M4A_no_uniform_mix",
    "M4A_no_logit_temperature",
    "M4A_no_cov_in_fusion",
    "M4A_no_cov_module",
    "M4A_no_gate_feature_path",
    "M4A_no_gate_weight_bias",
    "M4A_peer_loo_median",
]

PAPER_MAIN_VARIANTS = [
    "P0_post_only_single_stream",
    "P1_dual_stream_direct",
    "P2_gate_feature_path_only",
    "P3_gate_weight_path_only",
    "P4_full_gate_no_cov_formula",
    "P8_full_gate_aa_formula",
]

PAPER_COV_DIAG_VARIANTS = [
    "P4_full_gate_no_cov_formula",
    "P5_cov_fusion_formula_only",
    "P6_cov_weight_formula_only",
    "P7_full_gate_cov_formula",
    "P8_full_gate_aa_formula",
    "P9_gate_aa_cov_aux_only",
]

PAPER_MAIN_WITH_PEER_VARIANTS = [
    *PAPER_MAIN_VARIANTS,
    "P10_full_formula_peer_optional",
]

STABLE_MAIN_VARIANTS = [
    "P0_post_only_single_stream",
    "P4_full_gate_no_cov_formula",
    "P11_feature_stable_reliability_no_cov",
]

RECOVERY_MAIN_VARIANTS = [
    "P4_full_gate_no_cov_formula",
    "P11_feature_stable_reliability_no_cov",
    "P12_recovery_aware_full_gate_no_cov",
]


def _metric(d: Dict, key: str):
    value = d.get(key)
    return "" if value is None else value


def save_summary(rows: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_runs_summary.json"
    csv_path = out_dir / "ablation_runs_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"[summary_json] {json_path}")
    print(f"[summary_csv] {csv_path}")


def load_existing_summary(out_dir: Path) -> List[Dict]:
    json_path = out_dir / "ablation_runs_summary.json"
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in existing summary: {json_path}")
    return data


def parse_args():
    p = argparse.ArgumentParser(description="Run RGCF-V5 gate/cov ablation on one fixed dataset.")
    p.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--out_dir", default="results/rgcf_v5_ablation_batch")
    p.add_argument(
        "--train_seeds",
        default="0",
        help="Comma-separated model initialization seeds. Use 0,1,2,3,4 for paper statistics.",
    )
    p.add_argument(
        "--variants",
        default="default",
        help="default, all, paper_main, paper_cov_diag, paper_main_with_peer, paper_m4, stable_main, recovery_main, or comma-separated variant names.",
    )
    p.add_argument("--dry_run", action="store_true", help="Only print selected variants and their key config values.")
    p.add_argument("--tail_diagnostics", action="store_true", help="Evaluate full test-set tail metrics after each run.")
    p.add_argument(
        "--paper_config",
        action="store_true",
        help="Use the paper peer-consistency setup: paper_m4 variants, train seeds 0..4, tail diagnostics, paper out dir.",
    )
    p.add_argument(
        "--main_paper_config",
        action="store_true",
        help="Use the main paper ablation setup around dual-stream, gate, and fusion-formula contributions.",
    )
    p.add_argument(
        "--cov_diag_config",
        action="store_true",
        help="Use the covariance/formula diagnosis setup comparing AA and covariance-aware fusion.",
    )
    return p.parse_args()


def select_variants(spec: str) -> List[str]:
    spec = str(spec).strip()
    if spec == "default":
        return list(DEFAULT_VARIANTS)
    if spec == "all":
        return list(ALL_VARIANTS)
    if spec == "paper_m4":
        return list(PAPER_M4_VARIANTS)
    if spec == "paper_main":
        return list(PAPER_MAIN_VARIANTS)
    if spec == "paper_cov_diag":
        return list(PAPER_COV_DIAG_VARIANTS)
    if spec == "paper_main_with_peer":
        return list(PAPER_MAIN_WITH_PEER_VARIANTS)
    if spec == "stable_main":
        return list(STABLE_MAIN_VARIANTS)
    if spec == "recovery_main":
        return list(RECOVERY_MAIN_VARIANTS)
    return [x.strip() for x in spec.split(",") if x.strip()]


def _variant_config_row(bundle) -> Dict:
    model = bundle.model
    return {
        "variant": bundle.identity.preset_name,
        "model_name": model.model_name,
        "fusion_variant": model.fusion_variant,
        "use_post_stream": model.use_post_stream,
        "use_meas_stream": model.use_meas_stream,
        "use_gate": model.use_gate,
        "post_in_dim": model.post_in_dim,
        "use_peer_consistency_features": model.use_peer_consistency_features,
        "zero_peer_consistency_features": getattr(model, "zero_peer_consistency_features", False),
        "peer_consistency_mode": getattr(model, "peer_consistency_mode", "median"),
        "base_logit_temperature": model.base_logit_temperature,
        "weight_uniform_mix": model.weight_uniform_mix,
        "output_fusion_mode": getattr(model, "output_fusion_mode", "info_diag"),
        "fault_weight_loss_weight": model.fault_weight_loss_weight,
        "use_meas_in_representation": getattr(model, "use_meas_in_representation", True),
        "use_gate_on_meas_feature": model.use_gate_on_meas_feature,
        "gate_weight_alpha": model.gate_weight_alpha,
        "use_error_aware_gate_target": getattr(model, "use_error_aware_gate_target", False),
        "gate_error_tau": getattr(model, "gate_error_tau", 5.0),
        "gate_target_min": getattr(model, "gate_target_min", 0.1),
        "use_cov_calibration": model.use_cov_calibration,
        "use_cov_in_fusion": model.use_cov_in_fusion,
        "cov_weight_beta": model.cov_weight_beta,
        "cov_prior_weight": model.cov_prior_weight,
        "cov_sep_weight": model.cov_sep_weight,
    }


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _finite(values: List[float]) -> List[float]:
    return [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _mean(values: List[float]):
    values = _finite(values)
    if not values:
        return ""
    import numpy as np

    return float(np.mean(values))


def _percentile(values: List[float], q: float):
    values = _finite(values)
    if not values:
        return ""
    import numpy as np

    return float(np.percentile(values, q))


def _fault_weight_at_row(row: Dict, sid):
    if sid is None:
        return None
    key = f"w_s{int(sid)}"
    if key not in row:
        return None
    return float(row[key])


def _fault_is_top_at_row(row: Dict, sid) -> bool | None:
    if sid is None:
        return None
    weights = [(int(k[3:]), float(v)) for k, v in row.items() if k.startswith("w_s")]
    if not weights:
        return None
    top_sid = max(weights, key=lambda x: x[1])[0]
    return top_sid == int(sid)


def _run_tail_diagnostics(*, model, bundle, dataset_dir: str, out_dir: Path, run_label: str) -> Dict:
    import numpy as np
    import torch

    from data.dataset_store import load_raw_dataset_store
    from training.evaluator import evaluate_single_sim_fusion_with_timeseries

    ds = load_raw_dataset_store(dataset_dir)
    test_sims = ds["test_sims"]
    device = torch.device(bundle.base.runtime.device)

    point_rows: List[Dict] = []
    sim_rows: List[Dict] = []

    for sim in test_sims:
        ev = evaluate_single_sim_fusion_with_timeseries(sim, model, bundle, device)
        summary = ev["summary"]
        mode = sim.get("effective_fault_mode") or sim.get("fault_mode") or "unknown"
        sid = sim.get("effective_fault_sensor_id")
        seed = sim.get("seed")

        sim_rows.append(
            {
                "seed": seed,
                "mode": mode,
                "sid": sid,
                "overall_rmse_pos": summary.get("overall_rmse_pos"),
                "fault_window_rmse_pos": summary.get("fault_window_rmse_pos"),
                "fault_window_p95_error_pos": summary.get("fault_window_p95_error_pos"),
                "fault_window_max_error_pos": summary.get("fault_window_max_error_pos"),
                "fault_top1_weight_rate": summary.get("fault_top1_weight_rate"),
                "fault_weight_below_010_rate": summary.get("fault_weight_below_010_rate"),
            }
        )

        for row in ev["timeseries"]:
            fault_weight = _fault_weight_at_row(row, sid)
            fault_is_top = _fault_is_top_at_row(row, sid)
            point_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "sid": sid,
                    "k": row.get("k"),
                    "t": row.get("t"),
                    "error_pos": float(row["error_pos"]),
                    "fault_active_any": float(row.get("fault_active_any", 0.0)),
                    "fault_sensor_weight": fault_weight,
                    "fault_is_top": fault_is_top,
                }
            )

    errors = [float(r["error_pos"]) for r in point_rows]
    p95 = _percentile(errors, 95)
    p99 = _percentile(errors, 99)
    max_error = max(errors) if errors else ""
    p99_rows = [r for r in point_rows if p99 != "" and float(r["error_pos"]) >= float(p99)]

    grouped = {}
    by_mode = defaultdict(list)
    for row in point_rows:
        by_mode[str(row.get("mode", "unknown"))].append(row)
    for mode, rows in by_mode.items():
        mode_errors = [float(r["error_pos"]) for r in rows]
        grouped[mode] = {
            "n": len(rows),
            "mean_error": _mean(mode_errors),
            "p95_error": _percentile(mode_errors, 95),
            "p99_error": _percentile(mode_errors, 99),
            "top_max_error": max(mode_errors) if mode_errors else "",
        }

    p99_fault_weights = [
        float(r["fault_sensor_weight"])
        for r in p99_rows
        if r.get("fault_sensor_weight") is not None
    ]
    p99_fault_top = [
        1.0 if bool(r["fault_is_top"]) else 0.0
        for r in p99_rows
        if r.get("fault_is_top") is not None
    ]
    p99_active = [float(r.get("fault_active_any", 0.0)) for r in p99_rows]
    p99_bias = [r for r in p99_rows if str(r.get("mode")) == "bias_ramp"]

    top_rows = sorted(point_rows, key=lambda r: float(r["error_pos"]), reverse=True)[:20]
    top_csv_rows = [
        {
            "rank": i + 1,
            "seed": r.get("seed"),
            "mode": r.get("mode"),
            "sid": r.get("sid"),
            "k": r.get("k"),
            "t": r.get("t"),
            "error_pos": r.get("error_pos"),
            "fault_active_any": r.get("fault_active_any"),
            "fault_sensor_weight": r.get("fault_sensor_weight"),
            "fault_is_top": r.get("fault_is_top"),
        }
        for i, r in enumerate(top_rows)
    ]

    diag = {
        "run_label": run_label,
        "num_points": len(point_rows),
        "p95_error": p95,
        "p99_error": p99,
        "max_error": max_error,
        "p99_count": len(p99_rows),
        "p99_mean_fault_weight": _mean(p99_fault_weights),
        "p99_fault_top_rate": _mean(p99_fault_top),
        "p99_active_fault_rate": _mean(p99_active),
        "p99_bias_ramp_count": len(p99_bias),
        "p99_bias_ramp_rate": len(p99_bias) / max(len(p99_rows), 1),
        "bias_ramp_top_max_error": grouped.get("bias_ramp", {}).get("top_max_error", ""),
        "bias_ramp_p99_error": grouped.get("bias_ramp", {}).get("p99_error", ""),
        "grouped_by_mode": grouped,
        "top20": top_csv_rows,
        "sim_summary": sim_rows,
    }

    diag_dir = out_dir / "tail_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    safe_label = run_label.replace("\\", "_").replace("/", "_").replace(":", "_")
    (diag_dir / f"{safe_label}.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    with (diag_dir / f"{safe_label}_top20.csv").open("w", encoding="utf-8", newline="") as f:
        if top_csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(top_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(top_csv_rows)

    return {
        "full_num_points": diag["num_points"],
        "full_p95_error": diag["p95_error"],
        "full_p99_error": diag["p99_error"],
        "full_max_error": diag["max_error"],
        "p99_mean_fault_weight": diag["p99_mean_fault_weight"],
        "p99_fault_top_rate": diag["p99_fault_top_rate"],
        "p99_active_fault_rate": diag["p99_active_fault_rate"],
        "p99_bias_ramp_count": diag["p99_bias_ramp_count"],
        "p99_bias_ramp_rate": diag["p99_bias_ramp_rate"],
        "bias_ramp_top_max_error": diag["bias_ramp_top_max_error"],
        "bias_ramp_p99_error": diag["bias_ramp_p99_error"],
    }


def main():
    args = parse_args()
    if args.paper_config:
        args.variants = "paper_m4"
        args.train_seeds = "0,1,2,3,4"
        args.tail_diagnostics = True
        if args.out_dir == "results/rgcf_v5_ablation_batch":
            args.out_dir = "results/rgcf_v5_m4_paper_ablation"
    if args.main_paper_config:
        args.variants = "paper_main"
        args.train_seeds = "0,1,2,3,4"
        args.tail_diagnostics = True
        if args.out_dir == "results/rgcf_v5_ablation_batch":
            args.out_dir = "results/rgcf_main_paper_ablation"
    if args.cov_diag_config:
        args.variants = "paper_cov_diag"
        args.train_seeds = "0,1,2,3,4"
        args.tail_diagnostics = True
        if args.out_dir == "results/rgcf_v5_ablation_batch":
            args.out_dir = "results/rgcf_cov_formula_diagnosis"

    variants = select_variants(args.variants)
    train_seeds = _parse_int_list(args.train_seeds)
    out_dir = Path(args.out_dir)
    rows: List[Dict] = load_existing_summary(out_dir)
    completed_run_labels = {str(row.get("run_label")) for row in rows if row.get("run_label")}

    print(f"[dataset_dir] {args.dataset_dir}")
    print(f"[variants] {', '.join(variants)}")
    print(f"[train_seeds] {', '.join(map(str, train_seeds))}")
    print(f"[tail_diagnostics] {args.tail_diagnostics}")

    if args.dry_run:
        for variant in variants:
            bundle = build_variant(
                variant,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
            )
            bundle.train.model_seed = train_seeds[0] if train_seeds else None
            print(json.dumps(_variant_config_row(bundle), ensure_ascii=False))
        return

    from core.result_manager import ResultManager
    from experiments.train import run_train_experiment

    total_runs = len(variants) * max(len(train_seeds), 1)
    run_idx = 0
    for variant in variants:
        for train_seed in train_seeds:
            run_idx += 1
            print("=" * 88)
            print(f"[run {run_idx}/{total_runs}] {variant} | train_seed={train_seed}")
            bundle = build_variant(
                variant,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
            )
            bundle.train.model_seed = int(train_seed)
            run_label = f"{variant}__trainseed{train_seed}"
            if run_label in completed_run_labels:
                print(f"[skip completed] {run_label}", flush=True)
                continue
            bundle.identity.preset_name = run_label
            bundle.identity.experiment_name = run_label

            res = run_train_experiment(
                bundle,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                dataset_dir=args.dataset_dir,
            )
            rm = ResultManager(bundle, mode="train", experiment_name_override=run_label)
            rm.save_train_result(
                train_info=res.train_info,
                history=res.history,
                quick_baseline_metrics=res.quick_baseline_metrics,
                quick_gnn_metrics=res.quick_gnn_metrics,
                quick_sim=res.quick_sim,
                model=res.model,
                quick_gnn_timeseries=res.quick_gnn_timeseries,
            )

            row = {
                "variant": variant,
                "run_label": run_label,
                "train_seed": train_seed,
                "run_dir": str(rm.run_dir),
                "dataset_dir": args.dataset_dir,
                "best_epoch": _metric(res.train_info, "best_epoch"),
                "best_val_loss": _metric(res.train_info, "best_val_loss"),
                "test_loss": _metric(res.train_info, "test_loss"),
                "test_loss_pos": _metric(res.train_info, "test_loss_pos"),
                "quick_rmse_pos": _metric(res.quick_gnn_metrics, "rmse_gnn_pos"),
                "fault_window_rmse_pos": _metric(res.quick_gnn_metrics, "fault_window_rmse_pos"),
                "fault_window_p95_error_pos": _metric(res.quick_gnn_metrics, "fault_window_p95_error_pos"),
                "fault_window_max_error_pos": _metric(res.quick_gnn_metrics, "fault_window_max_error_pos"),
                "mean_gate_fault": _metric(res.quick_gnn_metrics, "mean_gate_fault"),
                "mean_gate_normal": _metric(res.quick_gnn_metrics, "mean_gate_normal"),
                "mean_weight_fault": _metric(res.quick_gnn_metrics, "mean_weight_fault"),
                "mean_weight_normal": _metric(res.quick_gnn_metrics, "mean_weight_normal"),
                "mean_cov_scale_fault": _metric(res.quick_gnn_metrics, "mean_cov_scale_fault"),
                "mean_cov_scale_normal": _metric(res.quick_gnn_metrics, "mean_cov_scale_normal"),
                "cov_scale_fault_to_normal_ratio": _metric(res.quick_gnn_metrics, "cov_scale_fault_to_normal_ratio"),
            }
            if args.tail_diagnostics:
                print(f"[tail_diagnostics] start {run_label}", flush=True)
                tail_metrics = _run_tail_diagnostics(
                    model=res.model,
                    bundle=bundle,
                    dataset_dir=args.dataset_dir,
                    out_dir=out_dir,
                    run_label=run_label,
                )
                print(f"[tail_diagnostics] done {run_label}", flush=True)
                row.update(tail_metrics)
            rows.append(row)
            completed_run_labels.add(run_label)
            save_summary(rows, out_dir)

    print("=" * 88)
    print("[done]")
    save_summary(rows, out_dir)


if __name__ == "__main__":
    main()
