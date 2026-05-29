from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.types import ExperimentBundle
from simulation.scenario_factory import build_scenario_from_bundle
from simulation.sensor_factory import build_sensors_from_bundle
from simulation.faults import build_fault_manager_from_bundle
from simulation.robustness_faults import materialize_robustness_fault_bundle
from .ekf import CVEKF, EKFState
from .measurement_models import (
    h_gps2d,
    H_gps2d,
    h_radar_rb,
    H_radar_rb,
    h_aoa_only,
    H_aoa_only,
    h_uwb_range_only,
    H_uwb_range_only,
)
from .fusion_baselines import evaluate_baselines_from_sim


@dataclass
class RunnerOutputs:
    """
    单次 rollout 结果容器
    """
    sim: Dict[str, np.ndarray]
    baseline_metrics: Dict[str, float]
    fault_logs: List[Dict]


def _sigma_a_for_time(t: float, T: float, base_sigma: float, turn_sigma: float) -> float:
    if (40.0 <= t < 70.0) or (85.0 <= t < 100.0):
        return turn_sigma
    return base_sigma


def _build_initial_ekf_state(first_truth_x4: np.ndarray) -> EKFState:
    x0 = np.array(first_truth_x4, dtype=np.float64).copy()
    P0 = np.diag([50.0**2, 50.0**2, 10.0**2, 10.0**2]).astype(np.float64)
    return EKFState(x=x0, P=P0)


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _sensor_type_code(sensor_type: str) -> int:
    mapping = {
        "gps2d": 0,
        "radar_rb": 1,
        "aoa_only": 2,
        "uwb_range_only": 3,
    }
    if sensor_type not in mapping:
        raise ValueError(f"Unsupported sensor_type: {sensor_type}")
    return mapping[sensor_type]


def _is_fault_active_from_config(bundle: ExperimentBundle, sensor_id: int, t: float) -> bool:
    fault = bundle.fault

    if bool(getattr(fault.dropout, "enabled", False)):
        for window in getattr(fault.dropout, "windows", []):
            if sensor_id in set(getattr(window, "sensor_ids", [])):
                if float(window.t0) <= t <= float(window.t1):
                    return True

    pollution = getattr(fault, "pollution", None)
    if pollution is not None and bool(getattr(pollution, "enabled", False)):
        if sensor_id in set(getattr(pollution, "target_sensor_ids", [])):
            active_time_window = getattr(pollution, "active_time_window", None)
            if active_time_window is None:
                return True
            t0, t1 = active_time_window
            if float(t0) <= t <= float(t1):
                return True

    return False


def _predict_measurement_and_jacobian(
    sensor,
    st_pred: EKFState,
) -> Tuple[np.ndarray, np.ndarray]:
    x = st_pred.x

    if sensor.sensor_type == "gps2d":
        z_pred = np.asarray(h_gps2d(x), dtype=np.float64)
        H = np.asarray(H_gps2d(x), dtype=np.float64)
        return z_pred, H

    if sensor.sensor_type == "radar_rb":
        z_pred = np.asarray(h_radar_rb(x, sensor.sx, sensor.sy), dtype=np.float64)
        H = np.asarray(H_radar_rb(x, sensor.sx, sensor.sy), dtype=np.float64)
        return z_pred, H

    if sensor.sensor_type == "aoa_only":
        z_pred = np.asarray(h_aoa_only(x, sensor.sx, sensor.sy), dtype=np.float64)
        H = np.asarray(H_aoa_only(x, sensor.sx, sensor.sy), dtype=np.float64)
        return z_pred, H

    if sensor.sensor_type == "uwb_range_only":
        z_pred = np.asarray(h_uwb_range_only(x, sensor.sx, sensor.sy), dtype=np.float64)
        H = np.asarray(H_uwb_range_only(x, sensor.sx, sensor.sy), dtype=np.float64)
        return z_pred, H

    raise ValueError(f"Unsupported sensor_type: {sensor.sensor_type}")


def _normalize_innovation_for_sensor(sensor_type: str, innov: np.ndarray) -> np.ndarray:
    out = np.asarray(innov, dtype=np.float64).copy()

    if sensor_type == "radar_rb":
        # [range, bearing]
        if out.shape[0] >= 2:
            out[1] = _wrap_angle(out[1])

    elif sensor_type == "aoa_only":
        # [bearing]
        if out.shape[0] >= 1:
            out[0] = _wrap_angle(out[0])

    return out


def _safe_nis(innov: np.ndarray, S: np.ndarray) -> float:
    eps = 1e-9
    try:
        Sinv = np.linalg.pinv(S + eps * np.eye(S.shape[0], dtype=np.float64))
        nis = float(innov.T @ Sinv @ innov)
    except Exception:
        nis = 0.0
    return max(nis, 0.0)


def _ekf_update_dispatch(
    ekf: CVEKF,
    ekf_state: EKFState,
    sensor,
    z: np.ndarray,
    R: np.ndarray,
) -> EKFState:
    if sensor.sensor_type == "gps2d":
        return ekf.update(
            ekf_state,
            z,
            R,
            h_gps2d,
            H_gps2d,
            None,
        )

    if sensor.sensor_type == "radar_rb":
        return ekf.update(
            ekf_state,
            z,
            R,
            h_radar_rb,
            H_radar_rb,
            1,
            sensor.sx,
            sensor.sy,
        )

    if sensor.sensor_type == "aoa_only":
        return ekf.update(
            ekf_state,
            z,
            R,
            h_aoa_only,
            H_aoa_only,
            0,
            sensor.sx,
            sensor.sy,
        )

    if sensor.sensor_type == "uwb_range_only":
        return ekf.update(
            ekf_state,
            z,
            R,
            h_uwb_range_only,
            H_uwb_range_only,
            None,
            sensor.sx,
            sensor.sy,
        )

    raise ValueError(f"Unsupported sensor_type: {sensor.sensor_type}")


def run_single_simulation(bundle: ExperimentBundle) -> RunnerOutputs:
    """
    Step 5 rollout 主入口。

    当前在原有输出基础上，额外记录：
    - xpred / Ppred
    - zpred_store
    - innovation_store
    - S_store / Sdiag_store
    - nis_store
    - sensor_type_codes / sensor_type_names / sensor_pos

    目的：
    为 V1 meas-stream 提供标准 innovation / NIS / geometry 所需统计量。
    """
    bundle, fault_meta = materialize_robustness_fault_bundle(bundle)

    scenario = build_scenario_from_bundle(bundle)
    artifacts = scenario.build()

    sensors = build_sensors_from_bundle(bundle)
    fault_manager = build_fault_manager_from_bundle(bundle)

    truth_t = artifacts.truth.t
    truth_x4 = artifacts.truth.x4
    K = truth_x4.shape[0]
    N = len(sensors)

    rng = np.random.default_rng(bundle.base.runtime.seed)

    xhat = np.zeros((K, N, 4), dtype=np.float64)
    Phat = np.zeros((K, N, 4, 4), dtype=np.float64)
    xpred = np.zeros((K, N, 4), dtype=np.float64)
    Ppred = np.zeros((K, N, 4, 4), dtype=np.float64)
    valid_mask = np.zeros((K, N), dtype=np.float64)
    fault_active_mask = np.zeros((K, N), dtype=np.float32)

    z_store = np.empty((K, N), dtype=object)
    R_store = np.empty((K, N), dtype=object)

    zpred_store = np.empty((K, N), dtype=object)
    innovation_store = np.empty((K, N), dtype=object)
    S_store = np.empty((K, N), dtype=object)
    Sdiag_store = np.empty((K, N), dtype=object)
    nis_store = np.full((K, N), np.nan, dtype=np.float64)

    fault_logs: List[Dict] = []

    ekf_states: List[EKFState] = []
    for _ in range(N):
        ekf_states.append(_build_initial_ekf_state(truth_x4[0]))

    sensor_type_codes = np.array(
        [_sensor_type_code(sensor.sensor_type) for sensor in sensors],
        dtype=np.int64,
    )
    sensor_type_names = np.array([sensor.sensor_type for sensor in sensors], dtype=object)
    sensor_pos = np.array(
        [[float(getattr(sensor, "sx", 0.0)), float(getattr(sensor, "sy", 0.0))] for sensor in sensors],
        dtype=np.float64,
    )

    for k in range(K):
        tk = float(truth_t[k])
        x_true_k = truth_x4[k]

        scene_spec = bundle.scene_spec
        if scene_spec is None:
            raise ValueError("bundle.scene_spec is None，说明 ExperimentBundle 装配阶段没有正确构建 SceneRuntimeSpec")

        motion_cfg = scene_spec.config.motion
        scene_T = float(scene_spec.T)
        scene_dt = float(scene_spec.dt)

        sigma_a = _sigma_a_for_time(
            t=tk,
            T=scene_T,
            base_sigma=float(motion_cfg.sigma_a_base),
            turn_sigma=float(motion_cfg.sigma_a_turn),
        )
        ekf = CVEKF(dt=scene_dt, sigma_a=sigma_a)

        for i, sensor in enumerate(sensors):
            if _is_fault_active_from_config(bundle, int(sensor.sid), tk):
                fault_active_mask[k, i] = 1.0

            st_pred = ekf.predict(ekf_states[i])

            xpred[k, i] = st_pred.x
            Ppred[k, i] = st_pred.P

            z_nom, R_nom, valid_nom = sensor.measure(x_true_k, tk, rng)

            z_fin, R_fin, valid_fin, traces = fault_manager.apply(
                sensor_id=sensor.sid,
                sensor_type=sensor.sensor_type,
                t=tk,
                truth_state=x_true_k,
                z=z_nom,
                R=R_nom,
                valid=valid_nom,
                rng=rng,
                metadata={"sensor_name": sensor.name},
            )

            for tr in traces:
                if bool(tr.triggered):
                    fault_active_mask[k, i] = 1.0
                fault_logs.append({
                    "k": k,
                    "t": float(tk),
                    "sensor_id": int(sensor.sid),
                    "sensor_name": sensor.name,
                    "sensor_type": sensor.sensor_type,
                    "fault_name": tr.fault_name,
                    "triggered": bool(tr.triggered),
                    "info": tr.info,
                })

            st_upd = st_pred

            zpred_store[k, i] = None
            innovation_store[k, i] = None
            S_store[k, i] = None
            Sdiag_store[k, i] = None
            nis_store[k, i] = np.nan

            if valid_fin and (z_fin is not None) and (R_fin is not None):
                z_pred_i, H_i = _predict_measurement_and_jacobian(sensor, st_pred)

                z_fin_arr = np.asarray(z_fin, dtype=np.float64).reshape(-1)
                R_fin_arr = np.asarray(R_fin, dtype=np.float64)
                innov_i = z_fin_arr - z_pred_i
                innov_i = _normalize_innovation_for_sensor(sensor.sensor_type, innov_i)

                S_i = H_i @ st_pred.P @ H_i.T + R_fin_arr
                Sdiag_i = np.diag(S_i).astype(np.float64)
                nis_i = _safe_nis(innov_i, S_i)

                zpred_store[k, i] = z_pred_i
                innovation_store[k, i] = innov_i
                S_store[k, i] = S_i
                Sdiag_store[k, i] = Sdiag_i
                nis_store[k, i] = nis_i

                st_upd = _ekf_update_dispatch(ekf, st_pred, sensor, z_fin_arr, R_fin_arr)

            ekf_states[i] = st_upd

            xhat[k, i] = st_upd.x
            Phat[k, i] = st_upd.P
            valid_mask[k, i] = 1.0 if valid_fin else 0.0
            z_store[k, i] = z_fin
            R_store[k, i] = R_fin

    sim = {
        "t": truth_t.copy(),
        "x_truth_4d": truth_x4.copy(),
        "x_truth_5d": artifacts.truth.x5.copy(),
        "xhat": xhat,
        "Phat": Phat,
        "xpred": xpred,
        "Ppred": Ppred,
        "valid_mask": valid_mask,
        "fault_active_mask": fault_active_mask,
        "z_store": z_store,
        "R_store": R_store,
        "zpred_store": zpred_store,
        "innovation_store": innovation_store,
        "S_store": S_store,
        "Sdiag_store": Sdiag_store,
        "nis_store": nis_store,
        "sensor_type_codes": sensor_type_codes,
        "sensor_type_names": sensor_type_names,
        "sensor_pos": sensor_pos,
        "scene_name": bundle.identity.scene_name,
        "fault_mode": bundle.identity.fault_mode,
        "effective_fault_mode": fault_meta.get("effective_fault_mode"),
        "effective_fault_sensor_id": fault_meta.get("effective_fault_sensor_id"),
        "effective_fault_window": fault_meta.get("effective_fault_window"),
        "source_fault_mode": fault_meta.get("source_fault_mode"),
    }

    baseline_metrics = evaluate_baselines_from_sim(sim)

    return RunnerOutputs(
        sim=sim,
        baseline_metrics=baseline_metrics,
        fault_logs=fault_logs,
    )
