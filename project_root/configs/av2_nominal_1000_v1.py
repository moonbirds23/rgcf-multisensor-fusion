"""Frozen nominal-only AV2 protocol for the V3.1 execution plan.

This configuration deliberately reuses the approved V1.1 sensor geometry and
noise, while declaring a separate nominal-only experiment boundary.  It does
not alter EKF dynamics, initialization, feature schema, scenes, or measurement
seeds, and it contains no fault sampling controls.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .av2_level_b_plus_v11 import AGGRESSIVE_PROTOCOL, MEASUREMENT_SEEDS


NOMINAL_PROTOCOL_NAME = "AV2_NOMINAL_1000_V3.1"
NOMINAL_CONDITION = "nominal"
NOMINAL_CACHE_SCHEMA_VERSION = "AV2_NOMINAL_CACHE_V1"
WARMUP_SECONDS = 1.0
NOMINAL_PROTOCOL = AGGRESSIVE_PROTOCOL

# These values are intentionally literal, so a cache audit can reject any
# accidental reintroduction of a fault/mixed-quality data path.
NOMINAL_CONDITIONS = {
    "nominal_only": True,
    "enable_bias": False,
    "enable_drift": False,
    "enable_noise_burst": False,
    "enable_dropout": False,
    "enable_delay": False,
    "enable_covariance_mismatch": False,
    "enable_fault_sampling": False,
    "active_conditions": ["nominal"],
    "fault_probability": 0.0,
}


def canonical_config() -> dict[str, Any]:
    """Return the complete hashable nominal-only contract."""
    return {
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "condition": NOMINAL_CONDITION,
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "warmup_seconds": WARMUP_SECONDS,
        "conditions": NOMINAL_CONDITIONS,
        "sensors": [
            {
                "name": sensor.name,
                "role": sensor.role,
                "measurement_model": sensor.measurement_model,
                "local_position_xy_m": list(sensor.local_position_xy_m)
                if sensor.local_position_xy_m is not None
                else None,
                "rate_hz": sensor.rate_hz,
                "noise": list(sensor.noise_standard_deviations),
            }
            for sensor in NOMINAL_PROTOCOL.sensors
        ],
        "ekf": {
            "state": "[px, py, vx, vy]",
            "process_acceleration_sigma_mps2": 2.0,
            "initial_velocity_mps": [0.0, 0.0],
            "initial_velocity_variance": 36.0,
        },
        "feature_schema_version": "av2_pilot_feature_v1",
    }


def config_sha256() -> str:
    payload = json.dumps(canonical_config(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
