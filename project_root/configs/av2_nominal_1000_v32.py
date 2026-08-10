"""Corrected nominal-only AV2 protocol for a full V3.2 regeneration.

V3.1 is intentionally left untouched because its frozen caches interpret the
configured angular standard deviations as radians.  V3.2 stores every noise
standard deviation in the measurement model's native unit: metres for range
and position, radians for bearing.  It also declares scenario-scoped random
streams so a shared measurement-seed label does not replay the same noise
sequence in every scene.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from math import radians
from typing import Any

from .av2_config import SensorSpec
from .av2_level_b_plus_v11 import AGGRESSIVE_PROTOCOL, MEASUREMENT_SEEDS


NOMINAL_PROTOCOL_NAME = "AV2_NOMINAL_1000_V3.2"
NOMINAL_CONDITION = "nominal"
NOMINAL_CACHE_SCHEMA_VERSION = "AV2_NOMINAL_CACHE_V2"
ARTIFACT_KEY = "av2_1000_v32"
WARMUP_SECONDS = 1.0
RNG_DERIVATION = "sha256(protocol_name|scenario_id|measurement_seed|measurement)[:4]-little"


def _sensor(
    name: str,
    role: str,
    measurement_model: str,
    position: tuple[float, float] | None,
    rate_hz: float,
    noise: tuple[tuple[str, float], ...],
) -> SensorSpec:
    return SensorSpec(name, role, measurement_model, position, rate_hz, noise)


# The numerical angular values below are radians.  The field names deliberately
# say ``bearing_rad`` so the serialized contract cannot again claim degrees
# while passing unconverted values to a radian measurement model.
NOMINAL_PROTOCOL = replace(
    AGGRESSIVE_PROTOCOL,
    sensors=(
        _sensor("T1", "posterior", "gps_2d", None, 5.0, (("x_m", 4.5), ("y_m", 4.5))),
        _sensor(
            "T2", "posterior", "range_bearing", (80.0, -50.0), 5.0,
            (("range_m", 2.0), ("bearing_rad", radians(0.40))),
        ),
        _sensor(
            "T3", "posterior", "range_bearing", (-60.0, 70.0), 5.0,
            (("range_m", 2.4), ("bearing_rad", radians(0.45))),
        ),
        _sensor(
            "E1", "evidence", "aoa_bearing", (90.0, 65.0), 5.0,
            (("bearing_rad", radians(0.80)),),
        ),
        _sensor("E2", "evidence", "uwb_range", (-85.0, -50.0), 10.0, (("range_m", 1.2),)),
    ),
)
NOMINAL_PROTOCOL.validate()


_EXPECTED_NATIVE_FIELDS = {
    "gps_2d": ("x_m", "y_m"),
    "range_bearing": ("range_m", "bearing_rad"),
    "aoa_bearing": ("bearing_rad",),
    "uwb_range": ("range_m",),
}


def validate_native_noise_contract() -> None:
    """Reject unit labels that do not match the generator's native units."""
    for sensor in NOMINAL_PROTOCOL.sensors:
        fields = tuple(name for name, _ in sensor.noise_standard_deviations)
        expected = _EXPECTED_NATIVE_FIELDS[sensor.measurement_model]
        if fields != expected:
            raise ValueError(
                f"{sensor.name}: native noise fields must be {expected}, got {fields}"
            )


validate_native_noise_contract()


def canonical_config() -> dict[str, Any]:
    return {
        "protocol_name": NOMINAL_PROTOCOL_NAME,
        "cache_schema_version": NOMINAL_CACHE_SCHEMA_VERSION,
        "artifact_key": ARTIFACT_KEY,
        "condition": NOMINAL_CONDITION,
        "measurement_seeds": list(MEASUREMENT_SEEDS),
        "rng_derivation": RNG_DERIVATION,
        "warmup_seconds": WARMUP_SECONDS,
        "nominal_only": True,
        "sensors": [
            {
                "name": sensor.name,
                "role": sensor.role,
                "measurement_model": sensor.measurement_model,
                "local_position_xy_m": list(sensor.local_position_xy_m)
                if sensor.local_position_xy_m is not None
                else None,
                "rate_hz": sensor.rate_hz,
                "noise_native_units": [list(value) for value in sensor.noise_standard_deviations],
            }
            for sensor in NOMINAL_PROTOCOL.sensors
        ],
        "ekf": {
            "state": "[px, py, vx, vy]",
            "process_acceleration_sigma_mps2": 2.0,
            "initial_velocity_mps": [0.0, 0.0],
            "initial_velocity_variance": 36.0,
            "initialization": "unchanged_from_V3.1",
        },
        "feature_schema_version": "av2_pilot_feature_v1",
    }


def config_sha256() -> str:
    payload = json.dumps(canonical_config(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
