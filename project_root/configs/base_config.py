from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    seed: int = 7
    device: str = "cpu"


@dataclass
class ResultConfig:
    result_root: str = "results"
    save_model_ckpt: bool = True


@dataclass
class BaseConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    results: ResultConfig = field(default_factory=ResultConfig)
