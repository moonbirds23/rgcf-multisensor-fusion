from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stop_patience: int = 12
    train_seed_start: int = 10
    train_seed_end: int = 69
    val_seed_start: int = 70
    val_seed_end: int = 79
    test_seed_start: int = 80
    test_seed_end: int = 89
    repeat_runs: int = 10
    repeat_seed_stride: int = 100
