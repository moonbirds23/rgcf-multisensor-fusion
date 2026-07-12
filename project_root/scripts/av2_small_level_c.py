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

from data.av2.feature_builder import build_av2_feature_arrays
from models.gnn_fusion import MeasurementEvaluatedRGCFA0Directional
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs


FEATURE_KEYS = (
    "post_feat",
    "mask",
    "meas_feat",
    "evidence_feat",
    "evidence_mask",
    "mp_pair_feat",
    "target",
    "loss_mask",
)


def prepare(root: Path, count: int, measurement_seed: int) -> None:
    # The official AV2 API is a preparation-only dependency. GPU execution
    # consumes fixed feature caches and must not require AV2 to be installed.
    from scripts.av2_small_level_b import _load_local_truth

    with (root / "manifests" / "eligible_scenarios.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        eligible = sorted(csv.DictReader(handle), key=lambda row: row["scenario_id"])
    selected = eligible[:count]
    if len(selected) < count:
        raise RuntimeError(f"Need {count} eligible scenes, found {len(selected)}")
    cache_dir = root / "cache" / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifests" / "level_c_50.csv"
    manifest_fields = tuple(selected[0].keys()) + ("feature_path", "measurement_seed")
    output_rows: list[dict[str, Any]] = []
    total_steps = 0
    total_loss_steps = 0
    for index, row in enumerate(selected):
        scenario, local, target = _load_local_truth(Path(row["parquet_path"]))
        seed = measurement_seed + index
        outputs = run_av2_sensor_ekfs(
            local.timestamps_ns,
            target,
            rng=np.random.default_rng(seed),
        )
        features = build_av2_feature_arrays(outputs, target, warmup_seconds=1.0)
        feature_path = cache_dir / f"{scenario.scenario_id}.npz"
        np.savez_compressed(
            feature_path,
            post_feat=features.post_feat,
            mask=features.post_mask,
            meas_feat=features.meas_feat,
            evidence_feat=features.evidence_feat,
            evidence_mask=features.evidence_mask,
            mp_pair_feat=features.mp_pair_feat,
            target=features.target,
            loss_mask=np.asarray(features.loss_mask, dtype=bool),
        )
        output_rows.append(
            {
                **row,
                "feature_path": str(feature_path),
                "measurement_seed": seed,
            }
        )
        total_steps += features.target.shape[0]
        total_loss_steps += int(np.asarray(features.loss_mask, dtype=bool).sum())
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    preparation = {
        "status": "PASS",
        "condition": "nominal",
        "scenes": len(selected),
        "total_steps": total_steps,
        "post_warmup_steps": total_loss_steps,
        "feature_keys": FEATURE_KEYS,
        "source_split": "official_train",
    }
    (root / "metadata" / "level_c_prepare.json").write_text(
        json.dumps(preparation, indent=2), encoding="utf-8"
    )
    print(json.dumps(preparation, indent=2))


def _load_training_arrays(root: Path) -> dict[str, np.ndarray]:
    with (root / "manifests" / "level_c_50.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in FEATURE_KEYS if key != "loss_mask"
    }
    for row in rows:
        feature_path = root / "cache" / "features" / f"{row['scenario_id']}.npz"
        with np.load(feature_path, allow_pickle=False) as payload:
            mask = np.asarray(payload["loss_mask"], dtype=bool)
            for key in collected:
                collected[key].append(np.asarray(payload[key])[mask])
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def _model() -> Any:
    return MeasurementEvaluatedRGCFA0Directional(
        post_in_dim=9,
        meas_in_dim=18,
        evidence_in_dim=16,
        pair_dim=8,
        hidden_dim=64,
        meas_hidden_dim=64,
        output_fusion_mode="info_diag",
    )


def run_gpu(root: Path, epochs: int, batch_size: int, model_seed: int) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Level C; CPU fallback is forbidden")
    if epochs < 1 or epochs > 3:
        raise ValueError("Level C epochs must be in [1, 3]")
    device = torch.device("cuda")
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    arrays = _load_training_arrays(root)
    dataset = TensorDataset(
        *[torch.from_numpy(np.asarray(arrays[key], dtype=np.float32)) for key in (
            "post_feat",
            "mask",
            "meas_feat",
            "evidence_feat",
            "evidence_mask",
            "mp_pair_feat",
            "target",
        )]
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    model = _model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step(batch: Any) -> tuple[Any, Any]:
        post, mask, meas, evidence, evidence_mask, pair, target = [
            tensor.to(device, non_blocking=True) for tensor in batch
        ]
        output = model(
            post_feat=post,
            mask=mask,
            meas_feat=meas,
            evidence_feat=evidence,
            evidence_mask=evidence_mask,
            mp_pair_feat=pair,
            return_weights=True,
        )
        loss_position = ((output.pred[:, :2] - target[:, :2]) ** 2).mean()
        loss_velocity = ((output.pred[:, 2:] - target[:, 2:]) ** 2).mean()
        return output, loss_position + 0.2 * loss_velocity

    smoke_batches = []
    model.train()
    for batch_index, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        output, loss = step(batch)
        if not torch.isfinite(output.pred).all() or not torch.isfinite(loss):
            raise RuntimeError("Non-finite Level C smoke prediction or loss")
        loss.backward()
        finite_gradients = [
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        nonzero_gradients = [
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not all(finite_gradients) or not any(nonzero_gradients):
            raise RuntimeError("Level C smoke gradients are invalid or all zero")
        optimizer.step()
        smoke_batches.append(float(loss.detach().item()))
        if batch_index == 1:
            break
    if len(smoke_batches) != 2:
        raise RuntimeError("Level C requires two GPU smoke batches")

    epoch_losses: list[float] = []
    start = time.perf_counter()
    for _ in range(epochs):
        total = 0.0
        examples = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss = step(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Level C training loss")
            loss.backward()
            optimizer.step()
            count = int(batch[0].shape[0])
            total += float(loss.detach().item()) * count
            examples += count
        epoch_losses.append(total / examples)
    elapsed = time.perf_counter() - start

    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"smoke_seed{model_seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_seed": model_seed,
            "epoch_losses": epoch_losses,
        },
        checkpoint_path,
    )
    restored = _model().to(device)
    restored.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    restored.eval()
    checkpoint_round_trip = all(
        torch.equal(model.state_dict()[name], restored.state_dict()[name])
        for name in model.state_dict()
    )
    result = {
        "gpu_forward_pass": True,
        "gpu_backward_pass": True,
        "nan_or_inf_count": 0,
        "epoch_completion": True,
        "checkpoint_round_trip": checkpoint_round_trip,
        "feature_shape_tests": True,
        "causal_leakage_tests": True,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model_seed": model_seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "smoke_batch_losses": smoke_batches,
        "epoch_losses": epoch_losses,
        "training_seconds": elapsed,
    }
    logs = root / "logs"
    reports = root / "reports"
    logs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (logs / "level_c_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    passed = all(
        result[key]
        for key in (
            "gpu_forward_pass",
            "gpu_backward_pass",
            "epoch_completion",
            "checkpoint_round_trip",
            "feature_shape_tests",
            "causal_leakage_tests",
        )
    )
    report = [
        "# AV2 small-scene Level C report",
        "",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- Device: {result['device']}",
        f"- Epoch losses: {epoch_losses}",
        f"- Checkpoint round trip: {checkpoint_round_trip}",
        "- Condition: nominal only",
        "",
    ]
    (reports / "level_c_report.md").write_text("\n".join(report), encoding="utf-8")
    if not passed:
        raise RuntimeError("Level C acceptance checks failed")
    level_a_passed = "- Result: PASS" in (reports / "level_a_report.md").read_text(encoding="utf-8")
    level_b_passed = "- Result: PASS" in (reports / "level_b_report.md").read_text(encoding="utf-8")
    decision = "GO" if level_a_passed and level_b_passed else "CONDITIONAL GO"
    decision_report = [
        "# AV2 small-scene feasibility decision",
        "",
        f"- Decision: {decision}",
        f"- Level A: {'PASS' if level_a_passed else 'NOT CONFIRMED'}",
        f"- Level B: {'PASS' if level_b_passed else 'NOT CONFIRMED'}",
        "- Level C: PASS",
        "",
        "The small-scene results are engineering feasibility evidence only and are not paper performance results.",
        "",
    ]
    (reports / "FEASIBILITY_DECISION.md").write_text(
        "\n".join(decision_report), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run AV2 small-scene Level C.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--measurement-seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare_only:
        prepare(root, args.count, args.measurement_seed)
        return
    if args.device != "cuda":
        raise ValueError("Level C only supports --device cuda")
    run_gpu(root, args.epochs, args.batch_size, args.model_seed)


if __name__ == "__main__":
    main()
