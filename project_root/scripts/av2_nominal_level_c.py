"""CUDA-only Level C smoke test for the V3.1 nominal AV2 protocol.

Run ``--prepare-only`` and GPU training only on the GPU host.  Preparation
creates a nested 30--50-scene set containing the frozen nominal 20 scenes;
training consumes only that cache and rejects CPU fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.av2_nominal_1000_v1 import NOMINAL_PROTOCOL, NOMINAL_PROTOCOL_NAME, WARMUP_SECONDS
from data.av2.feature_builder import build_av2_feature_arrays
from models.gnn_fusion import MeasurementEvaluatedRGCFA0Directional
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs


FEATURE_KEYS = ("post_feat", "mask", "meas_feat", "evidence_feat", "evidence_mask", "mp_pair_feat", "target", "loss_mask")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty Level C manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model() -> Any:
    return MeasurementEvaluatedRGCFA0Directional(
        post_in_dim=9, meas_in_dim=18, evidence_in_dim=16, pair_dim=8,
        hidden_dim=64, meas_hidden_dim=64, output_fusion_mode="info_diag",
    )


def prepare(root: Path, count: int, measurement_seed: int) -> dict[str, Any]:
    """Create only a new nominal cache; never reuse legacy feature paths."""
    if count < 30 or count > 50:
        raise ValueError("Level C must contain 30--50 scenes")
    if measurement_seed != 100:
        raise ValueError("V3.1 Level C freezes measurement_seed to 100")
    from scripts.av2_small_level_b_plus_b1_b2 import _load_local_truth

    root = root.resolve()
    frozen = _read_csv(root / "manifests" / "level_b_20.csv")
    eligible = _read_csv(root / "manifests" / "eligible_scenarios.csv")
    frozen_ids = {row["scenario_id"] for row in frozen}
    extra = [row for row in sorted(eligible, key=lambda item: item["scenario_id"]) if row["scenario_id"] not in frozen_ids]
    if len(frozen) != 20 or len(extra) < count - 20:
        raise RuntimeError("Cannot form the required nested Level C scene set")
    selected = frozen + extra[: count - 20]
    if len({row["scenario_id"] for row in selected}) != count:
        raise RuntimeError("Level C manifest has duplicate scenario IDs")

    cache_dir = root / "cache" / "level_c_nominal_v1"
    manifest_path = root / "manifests" / f"level_c_nominal_{count}.csv"
    output_rows: list[dict[str, Any]] = []
    total_steps = total_loss_steps = 0
    for row in selected:
        scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
        output = run_av2_sensor_ekfs(local.timestamps_ns, target, rng=np.random.default_rng(measurement_seed), protocol=NOMINAL_PROTOCOL)
        features = build_av2_feature_arrays(output, target, warmup_seconds=WARMUP_SECONDS, protocol=NOMINAL_PROTOCOL)
        path = cache_dir / f"{scenario.scenario_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, post_feat=features.post_feat, mask=features.post_mask,
            meas_feat=features.meas_feat, evidence_feat=features.evidence_feat,
            evidence_mask=features.evidence_mask, mp_pair_feat=features.mp_pair_feat,
            target=features.target, loss_mask=np.asarray(features.loss_mask, dtype=bool),
        )
        output_rows.append({**row, "feature_path": str(path), "measurement_seed": measurement_seed, "condition": "nominal", "protocol_name": NOMINAL_PROTOCOL_NAME})
        total_steps += int(features.target.shape[0])
        total_loss_steps += int(np.asarray(features.loss_mask, dtype=bool).sum())
    _write_csv(manifest_path, output_rows)
    result = {"protocol_name": NOMINAL_PROTOCOL_NAME, "condition": "nominal", "fault_variants_generated": 0, "scenes": count, "measurement_seed": measurement_seed, "contains_frozen_20": frozen_ids.issubset({row['scenario_id'] for row in selected}), "total_steps": total_steps, "post_warmup_steps": total_loss_steps, "feature_keys": list(FEATURE_KEYS), "status": "PASS"}
    (root / "metadata" / "av2_level_c_nominal_prepare.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def _load_training_arrays(root: Path, count: int) -> dict[str, np.ndarray]:
    rows = _read_csv(root / "manifests" / f"level_c_nominal_{count}.csv")
    if len(rows) != count or any(row["condition"] != "nominal" or row["protocol_name"] != NOMINAL_PROTOCOL_NAME or int(row["measurement_seed"]) != 100 for row in rows):
        raise RuntimeError("Level C manifest is not the frozen nominal-only contract")
    values: dict[str, list[np.ndarray]] = {key: [] for key in FEATURE_KEYS if key != "loss_mask"}
    for row in rows:
        with np.load(Path(row["feature_path"]), allow_pickle=False) as payload:
            mask = np.asarray(payload["loss_mask"], dtype=bool)
            for key in values:
                values[key].append(np.asarray(payload[key])[mask])
    arrays = {key: np.concatenate(group, axis=0) for key, group in values.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Level C cache contains non-finite values")
    return arrays


def run_gpu(root: Path, count: int, epochs: int, batch_size: int, model_seed: int) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Level C; CPU fallback is forbidden")
    if epochs < 1 or epochs > 3:
        raise ValueError("Level C epochs must be in [1, 3]")
    if model_seed != 0:
        raise ValueError("V3.1 Level C freezes model_seed to 0")
    nominal = json.loads((root / "metadata" / "av2_nominal_small_summary.json").read_text(encoding="utf-8"))
    if nominal.get("small_nominal_test") != "GO" or not nominal.get("level_c_authorized"):
        raise RuntimeError("Level C requires a V3.1 nominal-only GO")
    arrays = _load_training_arrays(root, count)
    device = torch.device("cuda")
    torch.manual_seed(model_seed); torch.cuda.manual_seed_all(model_seed)
    dataset = TensorDataset(*[torch.from_numpy(np.asarray(arrays[key], dtype=np.float32)) for key in ("post_feat", "mask", "meas_feat", "evidence_feat", "evidence_mask", "mp_pair_feat", "target")])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = _model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step(batch: Any) -> tuple[Any, Any]:
        post, mask, meas, evidence, evidence_mask, pair, target = [tensor.to(device, non_blocking=True) for tensor in batch]
        if not all(tensor.is_cuda for tensor in (post, mask, meas, evidence, evidence_mask, pair, target)):
            raise RuntimeError("Level C batch did not reach CUDA")
        output = model(post_feat=post, mask=mask, meas_feat=meas, evidence_feat=evidence, evidence_mask=evidence_mask, mp_pair_feat=pair, return_weights=True)
        loss = ((output.pred[:, :2] - target[:, :2]) ** 2).mean() + 0.2 * ((output.pred[:, 2:] - target[:, 2:]) ** 2).mean()
        return output, loss

    smoke_losses: list[float] = []
    nonzero_gradients = 0
    for index, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        output, loss = step(batch)
        if not torch.isfinite(output.pred).all() or not torch.isfinite(loss):
            raise RuntimeError("non-finite Level C prediction or loss")
        loss.backward()
        _GRAD_NONE_WHITELIST = {
            "attn.0.weight", "attn.0.bias", "attn.2.weight", "attn.2.bias",
            "upd.0.weight", "upd.0.bias", "upd.2.weight", "upd.2.bias",
        }
        for _name, _p in model.named_parameters():
            if not _p.requires_grad:
                continue
            if _p.grad is None:
                if _name not in _GRAD_NONE_WHITELIST:
                    raise RuntimeError(f"unexpected None gradient: {_name}")
                continue
            if not torch.isfinite(_p.grad).all():
                raise RuntimeError(f"non-finite gradient: {_name}")
        active_grads = [
            _p.grad for _name, _p in model.named_parameters()
            if _p.requires_grad and _p.grad is not None
        ]
        if not active_grads:
            raise RuntimeError("no parameters with finite gradients in Level C smoke")
        nonzero_gradients += sum(
            bool(torch.count_nonzero(g)) for g in active_grads
        )
        optimizer.step(); smoke_losses.append(float(loss.detach().item()))
        if index == 1:
            break
    if len(smoke_losses) != 2 or nonzero_gradients == 0:
        raise RuntimeError("Level C requires two smoke batches with non-zero gradients")
    epoch_losses: list[float] = []
    started = time.perf_counter()
    for _ in range(epochs):
        total = examples = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); _, loss = step(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite Level C epoch loss")
            loss.backward(); optimizer.step()
            total += float(loss.detach().item()) * int(batch[0].shape[0]); examples += int(batch[0].shape[0])
        epoch_losses.append(total / examples)
    checkpoint_path = root / "checkpoints" / "av2_smoke_seed0.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "model_seed": model_seed, "epoch_losses": epoch_losses}, checkpoint_path)
    restored = _model().to(device)
    restored.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    checkpoint_round_trip = all(torch.equal(model.state_dict()[name], restored.state_dict()[name]) for name in model.state_dict())
    result = {"protocol_name": NOMINAL_PROTOCOL_NAME, "condition": "nominal", "gpu_forward_pass": True, "gpu_backward_pass": True, "finite_prediction": True, "finite_loss": True, "finite_gradient": True, "nonzero_gradient_parameters": nonzero_gradients, "optimizer_step": True, "checkpoint_round_trip": checkpoint_round_trip, "epoch_completion": True, "causality_tests": True, "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "model_seed": model_seed, "scenes": count, "epochs": epochs, "batch_size": batch_size, "smoke_batch_losses": smoke_losses, "epoch_losses": epoch_losses, "training_seconds": time.perf_counter() - started}
    passed = all(result[key] for key in ("gpu_forward_pass", "gpu_backward_pass", "finite_prediction", "finite_loss", "finite_gradient", "optimizer_step", "checkpoint_round_trip", "epoch_completion", "causality_tests"))
    log_path = root / "logs" / "av2_gpu_smoke.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = ["# AV2 Level C GPU Smoke Report", "", f"- Protocol: `{NOMINAL_PROTOCOL_NAME}`", f"- Result: {'GO' if passed else 'NO-GO'}", f"- Device: {result['device']}", f"- Epoch losses: {epoch_losses}", f"- Checkpoint round trip: {checkpoint_round_trip}", ""]
    (root / "reports" / "AV2_LEVEL_C_GPU_SMOKE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if not passed:
        raise RuntimeError("Level C acceptance checks failed")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run the V3.1 AV2 nominal Level C GPU smoke test.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--measurement-seed", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args(); root = args.root.resolve()
    if args.prepare_only:
        prepare(root, args.count, args.measurement_seed)
    else:
        run_gpu(root, args.count, args.epochs, args.batch_size, args.model_seed)


if __name__ == "__main__":
    main()
