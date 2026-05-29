from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

SENSOR_IDS = ["S1", "S2", "S3", "S4"]

def read_config(config_path=None):
    """Read config as dict from JSON/YAML (JSON-only if no yaml)."""
    if config_path is None:
        return {}
    p = Path(config_path)
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_path(data_root, rel_or_abs):
    if rel_or_abs is None:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return Path(data_root) / p

def safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)

def safe_read_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def find_first_existing(data_root, candidates):
    root = Path(data_root)
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    return None

def standardize_pred_df(df, method="A4"):
    if df is None:
        return None
    df = df.copy()
    aliases = [
        ("time", "t"), ("k_time", "t"),
        ("truth_px", "truth_x"), ("truth_x", "truth_x"),
        ("gt_x", "truth_x"), ("gt_px", "truth_x"), ("x_true", "truth_x"),
        ("truth_py", "truth_y"), ("truth_y", "truth_y"),
        ("gt_y", "truth_y"), ("gt_py", "truth_y"), ("y_true", "truth_y"),
        ("pred_px", "pred_x"), ("pred_x", "pred_x"),
        ("px_pred", "pred_x"), ("x_pred", "pred_x"),
        ("pred_py", "pred_y"), ("pred_y", "pred_y"),
        ("py_pred", "pred_y"), ("y_pred", "pred_y"),
    ]
    for old, new in aliases:
        if old in df.columns:
            df[new] = df[old]
    if "t" not in df.columns:
        df["t"] = np.arange(len(df), dtype=float) * 0.1
    if "error_pos" not in df.columns and {"truth_x", "truth_y", "pred_x", "pred_y"}.issubset(df.columns):
        df["error_pos"] = np.sqrt((df["pred_x"] - df["truth_x"])**2 + (df["pred_y"] - df["truth_y"])**2)
    if "method" not in df.columns:
        df["method"] = method
    if "fault_active_any" not in df.columns:
        # fallback from per-sensor fault columns
        fault_cols = [c for c in df.columns if c.startswith("fault_active")]
        if fault_cols:
            df["fault_active_any"] = df[fault_cols].max(axis=1)
        else:
            df["fault_active_any"] = 0
    return df

def wide_to_long_metric(df, prefix, value_name):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None
    id_vars = [c for c in ["t", "k"] if c in df.columns]
    out = df.melt(id_vars=id_vars, value_vars=cols, var_name="sensor_raw", value_name=value_name)
    out["sensor_id"] = out["sensor_raw"].str.extract(r"([sS]\d+)")[0].str.upper()
    out = out.drop(columns=["sensor_raw"])
    return out

def merge_reliability_tables(gate_df=None, weight_df=None, cov_df=None, reliability_df=None):
    if reliability_df is not None and {"t", "sensor_id"}.issubset(reliability_df.columns):
        out = reliability_df.copy()
        if "effective_info" not in out.columns and {"weight", "cov_scale"}.issubset(out.columns):
            out["effective_info"] = out["weight"] / out["cov_scale"].replace(0, np.nan)
        return out

    parts = []
    if gate_df is not None:
        if {"t", "sensor_id", "gate"}.issubset(gate_df.columns):
            parts.append(gate_df[["t", "sensor_id", "gate"]].copy())
        else:
            parts.append(wide_to_long_metric(gate_df, "g_", "gate"))
    if weight_df is not None:
        if {"t", "sensor_id", "weight"}.issubset(weight_df.columns):
            parts.append(weight_df[["t", "sensor_id", "weight"]].copy())
        else:
            parts.append(wide_to_long_metric(weight_df, "w_", "weight"))
    if cov_df is not None:
        if {"t", "sensor_id", "cov_scale"}.issubset(cov_df.columns):
            parts.append(cov_df[["t", "sensor_id", "cov_scale"]].copy())
        else:
            # support cov_s1, cov_scale_s1, c_s1
            cov_long = wide_to_long_metric(cov_df, "cov_scale_", "cov_scale")
            if cov_long is None:
                cov_long = wide_to_long_metric(cov_df, "cov_", "cov_scale")
            if cov_long is None:
                cov_long = wide_to_long_metric(cov_df, "c_", "cov_scale")
            parts.append(cov_long)
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    for p in parts:
        if "sensor_id" in p.columns:
            p["sensor_id"] = p["sensor_id"].astype(str).str.upper().str.strip()
    out = parts[0]
    for p in parts[1:]:
        out = pd.merge(out, p, on=["t", "sensor_id"], how="outer")
    if "cov_scale" not in out.columns:
        out["cov_scale"] = 1.0
    if "effective_info" not in out.columns and {"weight", "cov_scale"}.issubset(out.columns):
        out["effective_info"] = out["weight"] / out["cov_scale"].replace(0, np.nan)
    if "fault_active" not in out.columns:
        # try wide fault columns from any source
        out["fault_active"] = 0
    return out

def standardize_ablation_df(df):
    """Map real ablation CSV columns to expected names."""
    if df is None:
        return None
    df = df.copy()
    aliases = {
        "variant": "method",
        "quick_rmse_pos": "overall_rmse",
        "fault_window_rmse_pos": "fault_rmse",
        "fault_window_p95_error_pos": "fault_p95",
        "fault_window_max_error_pos": "fault_max",
        "gate_separation_normal_minus_fault": "gate_sep",
        "mean_gate_fault": "mean_gate_fault",
        "mean_gate_normal": "mean_gate_normal",
        "cov_scale_fault_to_normal_ratio": "info_ratio",
        "mean_cov_scale_fault": "mean_cov_fault",
        "mean_cov_scale_normal": "mean_cov_normal",
        "weight_separation_normal_minus_fault": "weight_sep",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df


def default_sensor_positions():
    return pd.DataFrame({
        "sensor_id": ["S1", "S2", "S3", "S4"],
        "x": [100.0, 900.0, 100.0, 900.0],
        "y": [100.0, 120.0, 900.0, 900.0],
        "sensor_type": ["gps2d", "radar_rb", "aoa_only", "uwb_range_only"],
    })
