from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SENSORS = ["S1", "S2", "S3", "S4"]

def generate_demo_data(output_dir, seed=7):
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 120, 1201)
    x = 500 + 300 * np.sin(2*np.pi*t/120)
    y = 500 + 300 * np.sin(4*np.pi*t/120)
    fault = ((t >= 30) & (t <= 70)).astype(int)

    # A3/A4 predictions with different noise/fault behavior
    base_noise = rng.normal(0, 0.9, size=(len(t), 2))
    fault_bump = fault[:, None] * np.column_stack([2.2*np.sin(0.45*t), 2.0*np.cos(0.35*t)])
    a3_pred = np.column_stack([x, y]) + base_noise + 0.55*fault_bump
    a4_pred = np.column_stack([x, y]) + rng.normal(0, 0.75, size=(len(t), 2)) + 0.40*fault_bump
    # add one max spike to mimic realistic Max behavior
    a4_pred[np.argmin(np.abs(t-54.2))] += np.array([6.0, -5.8])
    for name, pred in [("A3", a3_pred), ("A4", a4_pred)]:
        df = pd.DataFrame({
            "t": t, "truth_x": x, "truth_y": y,
            "pred_x": pred[:,0], "pred_y": pred[:,1],
            "fault_active_any": fault,
        })
        df["error_pos"] = np.sqrt((df.pred_x-df.truth_x)**2 + (df.pred_y-df.truth_y)**2)
        df.to_csv(out/f"{name}_pred_timeseries.csv", index=False)

    # reliability long table
    rows = []
    for si, s in enumerate(SENSORS):
        is_fault_sensor = (s == "S2")
        for tt, ff in zip(t, fault):
            active = int(ff and is_fault_sensor)
            normal_wave = 0.03*np.sin(0.06*tt + si)
            gate = 0.68 + normal_wave + rng.normal(0, 0.03)
            weight = 0.26 + 0.04*np.sin(0.08*tt + si) + rng.normal(0, 0.015)
            cov = 4.5 + 0.2*np.sin(0.05*tt+si) + rng.normal(0, 0.05)
            if active:
                gate = 0.20 + rng.normal(0, 0.035)
                weight = 0.105 + rng.normal(0, 0.025)
                cov = 5.0 + rng.normal(0, 0.08)
            rows.append({"t": tt, "sensor_id": s, "gate": np.clip(gate, 0, 1),
                         "weight": np.clip(weight, 0, 1), "cov_scale": max(cov, 1.0),
                         "fault_active": active, "valid": 1})
    rel = pd.DataFrame(rows)
    rel["effective_info"] = rel["weight"] / rel["cov_scale"]
    rel.to_csv(out/"reliability_timeseries.csv", index=False)
    rel.pivot(index="t", columns="sensor_id", values="gate").rename(columns=lambda c: "g_"+c.lower()).reset_index().to_csv(out/"gate_timeseries.csv", index=False)
    rel.pivot(index="t", columns="sensor_id", values="weight").rename(columns=lambda c: "w_"+c.lower()).reset_index().to_csv(out/"weight_timeseries.csv", index=False)
    rel.pivot(index="t", columns="sensor_id", values="cov_scale").rename(columns=lambda c: "cov_scale_"+c.lower()).reset_index().to_csv(out/"cov_scale_timeseries.csv", index=False)

    ab = pd.DataFrame([
        {"method":"AVG", "type":"Rule", "overall_rmse":2.751, "fault_rmse":np.nan, "fault_p95":np.nan, "gate_sep":np.nan, "info_ratio":np.nan},
        {"method":"CI-multi", "type":"Rule", "overall_rmse":3.220, "fault_rmse":np.nan, "fault_p95":np.nan, "gate_sep":np.nan, "info_ratio":np.nan},
        {"method":"WAA-MM", "type":"Rule", "overall_rmse":2.452, "fault_rmse":np.nan, "fault_p95":np.nan, "gate_sep":np.nan, "info_ratio":np.nan},
        {"method":"A1 Posterior-only", "type":"Learned", "overall_rmse":2.082, "fault_rmse":2.824, "fault_p95":4.982, "gate_sep":np.nan, "info_ratio":np.nan},
        {"method":"A3 Dual-stream", "type":"Learned", "overall_rmse":1.989, "fault_rmse":2.596, "fault_p95":3.960, "gate_sep":np.nan, "info_ratio":np.nan},
        {"method":"A4 Full RGCF", "type":"Learned+Rel.", "overall_rmse":1.941, "fault_rmse":2.545, "fault_p95":3.901, "gate_sep":0.475, "info_ratio":0.30},
    ])
    ab.to_csv(out/"ablation_summary.csv", index=False)
    pd.DataFrame({
        "sensor_id": SENSORS,
        "x": [100, 900, 100, 900],
        "y": [100, 120, 900, 900],
        "sensor_type": ["gps2d", "radar_rb", "aoa_only", "uwb_range_only"]
    }).to_csv(out/"sensor_positions.csv", index=False)
    return out
