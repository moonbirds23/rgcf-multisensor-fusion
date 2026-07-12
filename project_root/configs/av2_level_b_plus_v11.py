"""Pre-registered aggressive AV2 Level B+ v1.1 configuration.

This module leaves the frozen AV2 Pilot v1.0 configuration untouched.  It is
only the explicitly authorised Level B+ rebalancing protocol; all scene IDs,
measurement seeds, EKF dynamics, initialization, and decision thresholds are
inherited unchanged.
"""
from __future__ import annotations

from dataclasses import replace

from .av2_config import AV2_PILOT_V1, Av2PilotProtocol, SensorSpec


LEVEL_B_PLUS_PROTOCOL_VERSION = "AV2_LEVEL_B_PLUS_V1.1_AGGRESSIVE"
MEASUREMENT_SEEDS = (100, 101, 102)
T2_BIAS_ONSET_SECONDS = 2.5
T2_BIAS_RAMP_DURATION_SECONDS = 1.5
T2_BIAS_TERMINAL_RANGE_M = 10.0


def _sensor(
    name: str,
    role: str,
    measurement_model: str,
    position: tuple[float, float] | None,
    rate_hz: float,
    noise: tuple[tuple[str, float], ...],
) -> SensorSpec:
    return SensorSpec(name, role, measurement_model, position, rate_hz, noise)


AGGRESSIVE_PROTOCOL: Av2PilotProtocol = replace(
    AV2_PILOT_V1,
    sensors=(
        _sensor("T1", "posterior", "gps_2d", None, 5.0, (("x_m", 4.5), ("y_m", 4.5))),
        _sensor("T2", "posterior", "range_bearing", (80.0, -50.0), 5.0, (("range_m", 2.0), ("bearing_deg", 0.40))),
        _sensor("T3", "posterior", "range_bearing", (-60.0, 70.0), 5.0, (("range_m", 2.4), ("bearing_deg", 0.45))),
        _sensor("E1", "evidence", "aoa_bearing", (90.0, 65.0), 5.0, (("bearing_deg", 0.8),)),
        _sensor("E2", "evidence", "uwb_range", (-85.0, -50.0), 10.0, (("range_m", 1.2),)),
    ),
)

# Reuse the frozen structural checks for roles, state model, causal rules, and
# fault-interface constraints.  Numerical values above are the documented B+
# v1.1 exception and not a mutation of AV2_PILOT_V1.
AGGRESSIVE_PROTOCOL.validate()
