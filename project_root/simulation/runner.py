from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.types import ExperimentBundle
from simulation.scenario_factory import build_scenario_from_bundle
from simulation.sensor_factory import build_sensors_from_layout
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


def _safe_measurement_residual_norm(sensor, z, R, x_ref: np.ndarray) -> float:
    if z is None or R is None:
        return float("nan")
    x_ref = np.asarray(x_ref, dtype=np.float64)
    if sensor.sensor_type == "gps2d":
        z_pred = np.asarray(h_gps2d(x_ref), dtype=np.float64)
    elif sensor.sensor_type == "radar_rb":
        z_pred = np.asarray(h_radar_rb(x_ref, sensor.sx, sensor.sy), dtype=np.float64)
    elif sensor.sensor_type == "aoa_only":
        z_pred = np.asarray(h_aoa_only(x_ref, sensor.sx, sensor.sy), dtype=np.float64)
    elif sensor.sensor_type == "uwb_range_only":
        z_pred = np.asarray(h_uwb_range_only(x_ref, sensor.sx, sensor.sy), dtype=np.float64)
    else:
        return float("nan")
    innov = np.asarray(z, dtype=np.float64).reshape(-1) - z_pred
    innov = _normalize_innovation_for_sensor(sensor.sensor_type, innov)
    R_arr = np.asarray(R, dtype=np.float64)
    try:
        rinv = np.linalg.pinv(R_arr + 1e-9 * np.eye(R_arr.shape[0], dtype=np.float64))
        return float(np.sqrt(max(float(innov.T @ rinv @ innov), 0.0)))
    except Exception:
        return float(np.linalg.norm(innov))


def _attach_phase1r_role_views(sim: Dict[str, np.ndarray], sensors, sensor_roles: List[str]) -> None:
    roles = np.array([str(r or "track") for r in sensor_roles], dtype=object)
    track_idx = np.flatnonzero(roles == "track")
    evidence_idx = np.flatnonzero(roles == "evidence")
    sim["sensor_roles"] = roles
    sim["track_indices"] = track_idx.astype(np.int64)
    sim["evidence_indices"] = evidence_idx.astype(np.int64)

    if track_idx.size > 0:
        for src, dst in (
            ("xhat", "track_xhat"),
            ("Phat", "track_Phat"),
            ("xpred", "track_xpred"),
            ("Ppred", "track_Ppred"),
            ("valid_mask", "track_valid_mask"),
            ("z_store", "track_z"),
            ("R_store", "track_R"),
            ("innovation_store", "track_innovation"),
            ("nis_store", "track_nis"),
        ):
            if src in sim:
                sim[dst] = sim[src][:, track_idx].copy()
        sim["track_sensor_type_codes"] = sim["sensor_type_codes"][track_idx].copy()
        sim["track_sensor_type_names"] = sim["sensor_type_names"][track_idx].copy()
        sim["track_sensor_pos"] = sim["sensor_pos"][track_idx].copy()

    if evidence_idx.size > 0:
        for src, dst in (
            ("z_store", "evidence_z"),
            ("R_store", "evidence_R"),
            ("valid_mask", "evidence_valid_mask"),
            ("zpred_store", "evidence_zpred"),
            ("innovation_store", "evidence_innovation"),
            ("nis_store", "evidence_nis"),
        ):
            if src in sim:
                sim[dst] = sim[src][:, evidence_idx].copy()
        sim["evidence_sensor_type_codes"] = sim["sensor_type_codes"][evidence_idx].copy()
        sim["evidence_sensor_type_names"] = sim["sensor_type_names"][evidence_idx].copy()
        sim["evidence_sensor_pos"] = sim["sensor_pos"][evidence_idx].copy()

    if track_idx.size > 0 and evidence_idx.size > 0:
        track_xhat = np.asarray(sim["track_xhat"], dtype=np.float64)
        track_valid = np.asarray(sim["track_valid_mask"], dtype=np.float64)
        k_count, evidence_count = sim["evidence_z"].shape
        track_count = track_idx.size
        residual_prior = np.full((k_count, evidence_count), np.nan, dtype=np.float64)
        residual_track = np.full((k_count, evidence_count, track_count), np.nan, dtype=np.float64)
        for k in range(k_count):
            valid_tracks = np.flatnonzero(track_valid[k] > 0.5)
            if valid_tracks.size == 0:
                prior = np.mean(track_xhat[k], axis=0)
            else:
                prior = np.mean(track_xhat[k, valid_tracks], axis=0)
            for local_e, global_e in enumerate(evidence_idx):
                sensor = sensors[int(global_e)]
                z = sim["z_store"][k, global_e]
                R = sim["R_store"][k, global_e]
                residual_prior[k, local_e] = _safe_measurement_residual_norm(sensor, z, R, prior)
                for local_t in range(track_count):
                    residual_track[k, local_e, local_t] = _safe_measurement_residual_norm(
                        sensor,
                        z,
                        R,
                        track_xhat[k, local_t],
                    )
        sim["evidence_residual_to_prior"] = residual_prior
        sim["evidence_residual_to_track"] = residual_track


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

    sensors = build_sensors_from_layout(artifacts.sensor_layout)
    sensor_roles = [
        str(getattr(node, "sensor_role", "track") or "track")
        for node in artifacts.sensor_layout.sensors
    ]
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

        motion_cfg = artifacts.motion
        scene_T = float(motion_cfg.T)
        scene_dt = float(motion_cfg.dt)

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
    _attach_phase1r_role_views(sim, sensors, sensor_roles)

    baseline_metrics = evaluate_baselines_from_sim(sim)

    return RunnerOutputs(
        sim=sim,
        baseline_metrics=baseline_metrics,
        fault_logs=fault_logs,
    )
