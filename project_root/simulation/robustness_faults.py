from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Tuple

from configs.fault_config import (
    DropoutFaultConfig,
    DropoutWindow,
    FaultConfig,
    PollutionFaultConfig,
)
from core.types import ExperimentBundle

DEFAULT_WINDOW: Tuple[float, float] = (30.0, 70.0)

# A 10-slot deterministic mixture: clean 20%, pollution 30%,
# bias_ramp 30%, dropout 20%.
DEFAULT_MATRIX_CASES = [
    "clean",
    "pollution",
    "bias_ramp",
    "dropout",
    "pollution",
    "bias_ramp",
    "clean",
    "dropout",
    "pollution",
    "bias_ramp",
]

DEFAULT_BALANCED_MATRIX_CASES = [
    {"mode": "clean", "sensor_id": None},
    {"mode": "clean", "sensor_id": None},
    {"mode": "clean", "sensor_id": None},
    {"mode": "clean", "sensor_id": None},

    {"mode": "pollution", "sensor_id": 1},
    {"mode": "pollution", "sensor_id": 2},
    {"mode": "pollution", "sensor_id": 3},
    {"mode": "pollution", "sensor_id": 4},
    {"mode": "pollution", "sensor_id": 1},
    {"mode": "pollution", "sensor_id": 3},

    {"mode": "bias_ramp", "sensor_id": 1},
    {"mode": "bias_ramp", "sensor_id": 2},
    {"mode": "bias_ramp", "sensor_id": 3},
    {"mode": "bias_ramp", "sensor_id": 4},
    {"mode": "bias_ramp", "sensor_id": 2},
    {"mode": "bias_ramp", "sensor_id": 4},

    {"mode": "dropout", "sensor_id": 1},
    {"mode": "dropout", "sensor_id": 2},
    {"mode": "dropout", "sensor_id": 3},
    {"mode": "dropout", "sensor_id": 4},
]


def _stable_int_hash(*items: Any) -> int:
    text = "::".join(str(x) for x in items)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _select_sensor_id(fault: FaultConfig, seed: int) -> int:
    sensor_ids = list(getattr(fault, "candidate_sensor_ids", None) or [1, 2, 3, 4])
    h = _stable_int_hash("sensor", int(seed))
    idx = h % len(sensor_ids)
    return int(sensor_ids[idx])


def _select_matrix_case(fault: FaultConfig, seed: int) -> str:
    cases = list(getattr(fault, "matrix_cases", None) or DEFAULT_MATRIX_CASES)
    return str(cases[int(seed) % len(cases)])


def _select_matrix_entry(fault: FaultConfig, seed: int):
    cases = list(getattr(fault, "matrix_cases", None) or DEFAULT_BALANCED_MATRIX_CASES)
    entry = cases[int(seed) % len(cases)]

    if isinstance(entry, dict):
        mode = str(entry.get("mode", "clean"))
        sensor_id = entry.get("sensor_id", None)
        return mode, sensor_id

    return str(entry), None


def _window(fault: FaultConfig) -> Tuple[float, float]:
    win = getattr(fault, "window", None)
    if win is None:
        return DEFAULT_WINDOW
    return float(win[0]), float(win[1])


def _clean_fault() -> FaultConfig:
    return FaultConfig(
        mode="clean",
        dropout=DropoutFaultConfig(enabled=False, windows=[]),
        pollution=PollutionFaultConfig(enabled=False, target_sensor_ids=[]),
    )


def _pollution_fault(sensor_id: int, window: Tuple[float, float]) -> FaultConfig:
    return FaultConfig(
        mode="pollution",
        dropout=DropoutFaultConfig(enabled=False, windows=[]),
        pollution=PollutionFaultConfig(
            enabled=True,
            target_sensor_ids=[int(sensor_id)],
            bias_rw_sigma=0.0,
            jump_prob=1.0,
            jump_sigma=12.0,
            active_time_window=window,
            fault_shape="jump",
        ),
    )


def _bias_ramp_fault(sensor_id: int, window: Tuple[float, float]) -> FaultConfig:
    return FaultConfig(
        mode="bias_ramp",
        dropout=DropoutFaultConfig(enabled=False, windows=[]),
        pollution=PollutionFaultConfig(
            enabled=True,
            target_sensor_ids=[int(sensor_id)],
            bias_rw_sigma=0.0,
            jump_prob=0.0,
            jump_sigma=18.0,
            active_time_window=window,
            fault_shape="bias_ramp",
        ),
    )


def _dropout_fault(sensor_id: int, window: Tuple[float, float]) -> FaultConfig:
    return FaultConfig(
        mode="dropout",
        dropout=DropoutFaultConfig(
            enabled=True,
            windows=[DropoutWindow(sensor_ids=[int(sensor_id)], t0=window[0], t1=window[1])],
        ),
        pollution=PollutionFaultConfig(enabled=False, target_sensor_ids=[]),
    )


def materialize_robustness_fault_bundle(bundle: ExperimentBundle) -> Tuple[ExperimentBundle, Dict]:
    """Turn robustness-matrix semantic faults into one concrete trajectory fault."""
    mode = str(getattr(bundle.fault, "mode", "clean"))
    robust_modes = {
        "robust_matrix_mixed",
        "random_window_pollution",
        "bias_ramp",
        "dropout_window",
    }
    if mode not in robust_modes:
        return bundle, {
            "effective_fault_mode": mode,
            "effective_fault_sensor_id": None,
            "effective_fault_window": getattr(bundle.fault.pollution, "active_time_window", None),
            "source_fault_mode": mode,
        }

    seed = int(bundle.base.runtime.seed)
    window = _window(bundle.fault)

    if mode == "robust_matrix_mixed":
        case, case_sensor_id = _select_matrix_entry(bundle.fault, seed)
    else:
        case, case_sensor_id = mode, None

    if case_sensor_id is not None:
        sensor_id = int(case_sensor_id)
    else:
        sensor_id = _select_sensor_id(bundle.fault, seed)

    if case == "clean":
        materialized_fault = _clean_fault()
        effective_sensor_id = None
    elif case in ("pollution", "random_window_pollution"):
        materialized_fault = _pollution_fault(sensor_id, window)
        effective_sensor_id = sensor_id
        case = "pollution"
    elif case == "bias_ramp":
        materialized_fault = _bias_ramp_fault(sensor_id, window)
        effective_sensor_id = sensor_id
    elif case in ("dropout", "dropout_window"):
        materialized_fault = _dropout_fault(sensor_id, window)
        effective_sensor_id = sensor_id
        case = "dropout"
    else:
        raise ValueError(f"Unsupported robustness matrix case: {case}")

    out = copy.deepcopy(bundle)
    out.fault = materialized_fault

    meta = {
        "effective_fault_mode": case,
        "effective_fault_sensor_id": effective_sensor_id,
        "effective_fault_window": window,
        "source_fault_mode": mode,
    }
    return out, meta
