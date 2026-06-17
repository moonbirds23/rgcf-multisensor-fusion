from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


P_COUNT = 3
M_COUNT = 5
EVIDENCE_M = (3, 4)
TRACK_M = (0, 1, 2)
EPS = 1e-12


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve(path_like: str | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return _project_root() / path


def _f(value, default=np.nan) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_name(value: str) -> str:
    return str(value).replace("\\", "_").replace("/", "_").replace(":", "_")


def _mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _corr(x, y) -> float:
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    if xa.size <= 1 or float(np.std(xa)) <= EPS or float(np.std(ya)) <= EPS:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _spearman3(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    return _corr(_rankdata(a), _rankdata(b))


def _kendall3(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    total = 0
    score = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            da = np.sign(a[i] - a[j])
            db = np.sign(b[i] - b[j])
            if da == 0 or db == 0:
                continue
            total += 1
            score += float(da * db)
    if total == 0:
        return float("nan")
    return score / float(total)


def _metric_summary(prefix: str, values: Sequence[float]) -> Dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class RunData:
    method: str
    scenario: str
    model_seed: str
    file: Path
    attn: np.ndarray
    error: np.ndarray
    weights: np.ndarray | None
    gates: np.ndarray | None
    cov: np.ndarray | None
    pair_res: np.ndarray | None
    pair_rank: np.ndarray | None
    skipped: Dict[str, str]


def _method_from_name(path: Path) -> str:
    name = path.name
    if "ME_RGCF_A0D" in name:
        return "A0D"
    if "ME_RGCF_A0" in name:
        return "A0"
    if "RGCF" in name:
        return "RGCF"
    return "unknown"


def _scenario_from_name(path: Path) -> str:
    match = re.search(r"test_(S\dR)", path.name)
    return match.group(1) if match else "unknown"


def _seed_from_name(path: Path) -> str:
    match = re.search(r"modelseed(\d+)", path.name)
    return match.group(1) if match else ""


def _read_detail_file(path: Path, method: str | None = None) -> RunData:
    method = method or _method_from_name(path)
    scenario = _scenario_from_name(path)
    seed = _seed_from_name(path)
    attn_rows: List[List[List[float]]] = []
    err_rows: List[float] = []
    weight_rows: List[List[float]] = []
    gate_rows: List[List[float]] = []
    cov_rows: List[List[float]] = []
    res_rows: List[List[List[float]]] = []
    rank_rows: List[List[List[float]]] = []
    skipped: Dict[str, str] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        has_attn = all(f"mp_attn_p{i}_m{j}" in fields for i in range(1, 4) for j in range(1, 6))
        has_w = all(f"w_s{i}" in fields for i in range(1, 4))
        has_g = all(f"g_s{i}" in fields for i in range(1, 4))
        has_cov = all(f"cov_scale_s{i}" in fields for i in range(1, 4))
        has_pair = all(
            f"mp_pair_res_p{i}_m{j}" in fields and f"mp_pair_rank_p{i}_m{j}" in fields
            for i in range(1, 4)
            for j in range(4, 6)
        )
        if not has_attn:
            raise ValueError(f"{path} does not contain complete mp_attn_p*_m* columns.")
        if not has_w:
            skipped["weights"] = "missing w_s1..w_s3"
        if not has_g:
            skipped["gates"] = "missing g_s1..g_s3"
        if not has_cov:
            skipped["cov_scale"] = "missing cov_scale_s1..cov_scale_s3"
        if not has_pair:
            skipped["pair"] = "missing mp_pair_res/rank evidence columns"

        for row in reader:
            attn_rows.append([
                [_f(row.get(f"mp_attn_p{i}_m{j}")) for j in range(1, 6)]
                for i in range(1, 4)
            ])
            err_rows.append(_f(row.get("error_pos")))
            if has_w:
                weight_rows.append([_f(row.get(f"w_s{i}")) for i in range(1, 4)])
            if has_g:
                gate_rows.append([_f(row.get(f"g_s{i}")) for i in range(1, 4)])
            if has_cov:
                cov_rows.append([_f(row.get(f"cov_scale_s{i}")) for i in range(1, 4)])
            if has_pair:
                res_rows.append([
                    [_f(row.get(f"mp_pair_res_p{i}_m{j}")) for j in range(4, 6)]
                    for i in range(1, 4)
                ])
                rank_rows.append([
                    [_f(row.get(f"mp_pair_rank_p{i}_m{j}")) for j in range(4, 6)]
                    for i in range(1, 4)
                ])

    attn = np.asarray(attn_rows, dtype=np.float64)
    error = np.asarray(err_rows, dtype=np.float64)
    weights = np.asarray(weight_rows, dtype=np.float64) if weight_rows else None
    gates = np.asarray(gate_rows, dtype=np.float64) if gate_rows else None
    cov = np.asarray(cov_rows, dtype=np.float64) if cov_rows else None
    pair_res = np.asarray(res_rows, dtype=np.float64) if res_rows else None
    pair_rank = np.asarray(rank_rows, dtype=np.float64) if rank_rows else None
    return RunData(method, scenario, seed, path, attn, error, weights, gates, cov, pair_res, pair_rank, skipped)


def _find_detail_files(result_dir: Path | None, method: str) -> List[Path]:
    if result_dir is None:
        return []
    detail = result_dir / "eval_details"
    if not detail.exists():
        raise FileNotFoundError(f"Missing eval_details directory: {detail}")
    if method == "A0D":
        pattern = "*ME_RGCF_A0D*modelseed*errors.csv"
    elif method == "A0":
        pattern = "*ME_RGCF_A0__modelseed*errors.csv"
    else:
        pattern = "*errors.csv"
    return sorted(detail.glob(pattern))


def _attention_metrics(attn: np.ndarray) -> Dict[str, float | int]:
    entropy = -np.sum(attn * np.log(np.clip(attn, EPS, 1.0)), axis=2)
    row_std = np.std(attn, axis=1)
    evidence_mass = np.sum(attn[:, :, list(EVIDENCE_M)], axis=2)
    track_mass = np.sum(attn[:, :, list(TRACK_M)], axis=2)
    own_track = np.stack([attn[:, i, i] for i in range(P_COUNT)], axis=1)
    other_track = track_mass - own_track
    out: Dict[str, float | int] = {
        "num_rows": int(attn.shape[0]),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy, ddof=1)) if entropy.size > 1 else float("nan"),
        "entropy_p95": float(np.percentile(entropy, 95)),
        "row_std_mean": float(np.mean(row_std)),
        "row_std_p95": float(np.percentile(row_std, 95)),
        "evidence_mass_mean": float(np.mean(evidence_mass)),
        "track_mass_mean": float(np.mean(track_mass)),
        "own_track_mass_mean": float(np.mean(own_track)),
        "other_track_mass_mean": float(np.mean(other_track)),
    }
    out["evidence_to_track_mass_ratio"] = float(out["evidence_mass_mean"]) / max(float(out["track_mass_mean"]), EPS)
    return out


def _directionality_metrics(
    run: RunData,
    *,
    mask: np.ndarray | None = None,
    spread_low_q: float = 30.0,
    spread_mid_lo_q: float = 40.0,
    spread_mid_hi_q: float = 70.0,
    spread_high_q: float = 70.0,
) -> Dict[str, float | int]:
    if run.pair_res is None or run.pair_rank is None:
        return {}
    n = run.attn.shape[0]
    base_mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if base_mask.size != n or not np.any(base_mask):
        return {}

    ev_attn = run.attn[:, :, list(EVIDENCE_M)]
    res = run.pair_res
    rank = run.pair_rank
    centered = res - np.nanmean(res, axis=1, keepdims=True)

    selected = base_mask[:, None, None] & np.isfinite(rank) & np.isfinite(ev_attn)
    out: Dict[str, float | int] = {
        "raw_residual_corr": _corr(rank[selected], ev_attn[selected]),
        "residual_log_corr": _corr(res[selected], ev_attn[selected]),
        "residual_centered_corr": _corr(centered[selected], ev_attn[selected]),
    }

    ev_sum = np.sum(ev_attn, axis=2, keepdims=True)
    ev_norm = ev_attn / np.maximum(ev_sum, EPS)
    out["evidence_normalized_corr"] = _corr(rank[selected], ev_norm[selected])

    # Within-evidence gap: does the M4-vs-M5 residual gap map to M4-vs-M5 attention gap?
    gap_mask = base_mask[:, None] & np.all(np.isfinite(rank), axis=2) & np.all(np.isfinite(ev_attn), axis=2)
    res_gap = rank[:, :, 0] - rank[:, :, 1]
    attn_gap = ev_attn[:, :, 0] - ev_attn[:, :, 1]
    out["within_evidence_gap_corr"] = _corr(res_gap[gap_mask], attn_gap[gap_mask])

    spread = np.nanmax(res, axis=1) - np.nanmin(res, axis=1)  # [K,2]
    spread_vals = spread[base_mask]
    spread_vals = spread_vals[np.isfinite(spread_vals)]
    if spread_vals.size > 0:
        low_thr = float(np.percentile(spread_vals, spread_low_q))
        mid_lo = float(np.percentile(spread_vals, spread_mid_lo_q))
        mid_hi = float(np.percentile(spread_vals, spread_mid_hi_q))
        high_thr = float(np.percentile(spread_vals, spread_high_q))
        for name, ev_mask in (
            ("low_spread", spread <= low_thr),
            ("mid_spread", (spread >= mid_lo) & (spread <= mid_hi)),
            ("high_spread", spread >= high_thr),
        ):
            full_mask = base_mask[:, None, None] & ev_mask[:, None, :] & np.isfinite(rank) & np.isfinite(ev_attn)
            out[f"{name}_corr"] = _corr(rank[full_mask], ev_attn[full_mask])
            out[f"{name}_n_pairs"] = int(np.sum(full_mask))
        out["spread_low_threshold"] = low_thr
        out["spread_high_threshold"] = high_thr
    return out


def _ranking_metrics(run: RunData, *, mask: np.ndarray | None = None, high_spread_only: bool = False, spread_top_frac: float = 0.30) -> Dict[str, float | int]:
    if run.pair_res is None or run.pair_rank is None:
        return {}
    n = run.attn.shape[0]
    base_mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    ev_attn = run.attn[:, :, list(EVIDENCE_M)]
    res = run.pair_res

    spread = np.nanmax(res, axis=1) - np.nanmin(res, axis=1)  # [K,2]
    if high_spread_only:
        vals = spread[base_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {}
        spread_mask = spread >= float(np.percentile(vals, 100.0 * (1.0 - spread_top_frac)))
    else:
        spread_mask = np.ones_like(spread, dtype=bool)

    hits = []
    top2_hits = []
    spearman = []
    kendall = []
    worst_minus_best = []
    worst_minus_mean = []
    event_count = 0
    for k in range(n):
        if not base_mask[k]:
            continue
        for e in range(2):
            if not spread_mask[k, e]:
                continue
            r = res[k, :, e]
            a = ev_attn[k, :, e]
            if not np.all(np.isfinite(r)) or not np.all(np.isfinite(a)):
                continue
            worst = int(np.argmax(r))
            best = int(np.argmin(r))
            attn_order = np.argsort(a)[::-1]
            hits.append(float(attn_order[0] == worst))
            top2_hits.append(float(worst in attn_order[:2]))
            spearman.append(_spearman3(r, a))
            kendall.append(_kendall3(r, a))
            others = [i for i in range(P_COUNT) if i != worst]
            worst_minus_best.append(float(a[worst] - a[best]))
            worst_minus_mean.append(float(a[worst] - np.mean(a[others])))
            event_count += 1
    return {
        "ranking_events": event_count,
        "worst_track_hit_rate": _mean(hits),
        "top2_hit_rate": _mean(top2_hits),
        "spearman_rank_corr": _mean(spearman),
        "kendall_tau": _mean(kendall),
        "worst_minus_best_attn": _mean(worst_minus_best),
        "worst_minus_mean_attn": _mean(worst_minus_mean),
    }


def _downstream_metrics(run: RunData, *, spread_top_frac: float) -> Dict[str, float | int]:
    if run.pair_res is None or run.weights is None:
        return {}
    n = run.attn.shape[0]
    res = run.pair_res
    spread = np.nanmax(res, axis=1) - np.nanmin(res, axis=1)
    vals = spread[np.isfinite(spread)]
    if vals.size == 0:
        return {}
    high_mask = spread >= float(np.percentile(vals, 100.0 * (1.0 - spread_top_frac)))
    deltas_w = []
    deltas_g = []
    deltas_cov = []
    low_weight = []
    low_gate = []
    high_cov = []
    events = 0
    for k in range(n):
        for e in range(2):
            if not high_mask[k, e]:
                continue
            r = res[k, :, e]
            if not np.all(np.isfinite(r)):
                continue
            worst = int(np.argmax(r))
            others = [i for i in range(P_COUNT) if i != worst]
            w_delta = float(run.weights[k, worst] - np.mean(run.weights[k, others]))
            deltas_w.append(w_delta)
            low_weight.append(float(w_delta < 0.0))
            if run.gates is not None:
                g_delta = float(run.gates[k, worst] - np.mean(run.gates[k, others]))
                deltas_g.append(g_delta)
                low_gate.append(float(g_delta < 0.0))
            if run.cov is not None:
                c_delta = float(run.cov[k, worst] - np.mean(run.cov[k, others]))
                deltas_cov.append(c_delta)
                high_cov.append(float(c_delta > 0.0))
            events += 1
    return {
        "downstream_events": events,
        "delta_w_mean": _mean(deltas_w),
        "delta_g_mean": _mean(deltas_g),
        "delta_cov_mean": _mean(deltas_cov),
        "worst_low_weight_rate": _mean(low_weight),
        "worst_low_gate_rate": _mean(low_gate),
        "worst_high_cov_rate": _mean(high_cov),
        "delta_error_sensor_skipped": "not available in eval_details",
    }


def _error_bin_masks(error: np.ndarray, tail_frac: float) -> Dict[str, np.ndarray]:
    finite = np.isfinite(error)
    vals = error[finite]
    if vals.size == 0:
        return {}
    q30 = float(np.percentile(vals, 30))
    q70 = float(np.percentile(vals, 70))
    qtail = float(np.percentile(vals, 100.0 * (1.0 - tail_frac)))
    return {
        "low_error": finite & (error <= q30),
        "mid_error": finite & (error > q30) & (error <= q70),
        "high_error": finite & (error > q70),
        "tail_error": finite & (error >= qtail),
    }


def _run_metrics(run: RunData, spread_top_frac: float) -> Dict:
    row: Dict[str, float | int | str] = {
        "method": run.method,
        "scenario": run.scenario,
        "model_seed": run.model_seed,
        "file": str(run.file),
    }
    row.update(_attention_metrics(run.attn))
    row.update(_metric_summary("error_pos", run.error))
    row.update(_directionality_metrics(run))
    row.update(_ranking_metrics(run))
    high_rank = _ranking_metrics(run, high_spread_only=True, spread_top_frac=spread_top_frac)
    for key, value in high_rank.items():
        row[f"high_spread_{key}"] = value
    row.update(_downstream_metrics(run, spread_top_frac=spread_top_frac))
    if run.skipped:
        row["skipped_metrics"] = "; ".join(f"{k}: {v}" for k, v in sorted(run.skipped.items()))
    else:
        row["skipped_metrics"] = ""
    return row


def _aggregate_rows(rows: List[Dict], keys: Sequence[str], metric_names: Sequence[str]) -> List[Dict]:
    buckets: Dict[tuple, List[Dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    out: List[Dict] = []
    for key, items in sorted(buckets.items()):
        entry = {k: v for k, v in zip(keys, key)}
        entry["n_runs"] = len(items)
        for metric in metric_names:
            vals = [_f(item.get(metric)) for item in items]
            entry[f"{metric}_mean"] = _mean(vals)
            entry[f"{metric}_std"] = _std(vals)
            entry[f"{metric}_min"] = _percentile(vals, 0)
            entry[f"{metric}_max"] = _percentile(vals, 100)
        out.append(entry)
    return out


def _error_bin_rows(runs: List[RunData], *, spread_top_frac: float, tail_frac: float) -> List[Dict]:
    out: List[Dict] = []
    for run in runs:
        if run.method != "A0D":
            continue
        for bin_name, mask in _error_bin_masks(run.error, tail_frac).items():
            row: Dict[str, float | int | str] = {
                "method": run.method,
                "scenario": run.scenario,
                "model_seed": run.model_seed,
                "error_bin": bin_name,
                "num_rows": int(np.sum(mask)),
            }
            if not np.any(mask):
                out.append(row)
                continue
            attention = _attention_metrics(run.attn[mask])
            row.update({f"bin_{k}": v for k, v in attention.items()})
            for key in (
                "entropy_mean",
                "row_std_mean",
                "evidence_mass_mean",
                "track_mass_mean",
                "own_track_mass_mean",
                "other_track_mass_mean",
                "evidence_to_track_mass_ratio",
            ):
                row[key] = attention.get(key, "")
            row.update(_directionality_metrics(run, mask=mask))
            row.update(_ranking_metrics(run, mask=mask))
            row.update({f"error_{k}": v for k, v in _metric_summary("pos", run.error[mask]).items()})
            # High-spread share inside this error bin.
            if run.pair_res is not None:
                spread = np.nanmax(run.pair_res, axis=1) - np.nanmin(run.pair_res, axis=1)
                vals = spread[np.isfinite(spread)]
                if vals.size > 0:
                    high_thr = float(np.percentile(vals, 100.0 * (1.0 - spread_top_frac)))
                    ev_high = spread >= high_thr
                    row["high_spread_event_share"] = float(np.sum(ev_high[mask]) / max(2 * int(np.sum(mask)), 1))
            if run.weights is not None:
                row["w_s1_mean"] = float(np.mean(run.weights[mask, 0]))
                row["w_s2_mean"] = float(np.mean(run.weights[mask, 1]))
                row["w_s3_mean"] = float(np.mean(run.weights[mask, 2]))
            if run.gates is not None:
                row["g_s1_mean"] = float(np.mean(run.gates[mask, 0]))
                row["g_s2_mean"] = float(np.mean(run.gates[mask, 1]))
                row["g_s3_mean"] = float(np.mean(run.gates[mask, 2]))
            if run.cov is not None:
                row["cov_scale_s1_mean"] = float(np.mean(run.cov[mask, 0]))
                row["cov_scale_s2_mean"] = float(np.mean(run.cov[mask, 1]))
                row["cov_scale_s3_mean"] = float(np.mean(run.cov[mask, 2]))
            out.append(row)
    return out


def _pair_matrix_rows(runs: List[RunData]) -> List[Dict]:
    rows: List[Dict] = []
    for run in runs:
        mean_attn = np.nanmean(run.attn, axis=0)
        for p in range(P_COUNT):
            for m in range(M_COUNT):
                rows.append({
                    "method": run.method,
                    "scenario": run.scenario,
                    "model_seed": run.model_seed,
                    "p_node": f"P{p + 1}",
                    "m_node": f"M{m + 1}",
                    "mean_mp_attn": float(mean_attn[p, m]),
                })
    return rows


def _attention_compare_rows(seed_rows: List[Dict]) -> List[Dict]:
    metrics = [
        "entropy_mean",
        "row_std_mean",
        "evidence_mass_mean",
        "track_mass_mean",
        "own_track_mass_mean",
        "other_track_mass_mean",
        "evidence_to_track_mass_ratio",
    ]
    by_key = {(r["scenario"], r["model_seed"], r["method"]): r for r in seed_rows}
    out: List[Dict] = []
    keys = sorted({(r["scenario"], r["model_seed"]) for r in seed_rows})
    for scenario, seed in keys:
        a0 = by_key.get((scenario, seed, "A0"))
        a0d = by_key.get((scenario, seed, "A0D"))
        if not a0 or not a0d:
            continue
        row = {"scenario": scenario, "model_seed": seed}
        for metric in metrics:
            row[f"A0_{metric}"] = a0.get(metric, "")
            row[f"A0D_{metric}"] = a0d.get(metric, "")
            row[f"delta_{metric}"] = _f(a0d.get(metric)) - _f(a0.get(metric))
        out.append(row)
    return out


def _metric_correlation_rows(seed_rows: List[Dict]) -> List[Dict]:
    metrics = [
        "raw_residual_corr",
        "high_spread_corr",
        "evidence_normalized_corr",
        "worst_track_hit_rate",
        "high_spread_worst_track_hit_rate",
        "row_std_mean",
        "evidence_mass_mean",
        "own_track_mass_mean",
        "error_pos_mean",
        "error_pos_max",
    ]
    rows = [r for r in seed_rows if r.get("method") == "A0D"]
    out: List[Dict] = []
    for a in metrics:
        for b in metrics:
            out.append({
                "metric_x": a,
                "metric_y": b,
                "corr": _corr([_f(r.get(a)) for r in rows], [_f(r.get(b)) for r in rows]),
            })
    return out


def _seed_worst_rows(seed_rows: List[Dict]) -> List[Dict]:
    rows = [r for r in seed_rows if r.get("method") == "A0D"]
    specs = [
        ("lowest_raw_residual_corr", "raw_residual_corr", True),
        ("lowest_high_spread_corr", "high_spread_corr", True),
        ("lowest_hit_rate", "worst_track_hit_rate", True),
        ("highest_tail_error", "error_pos_max", False),
        ("highest_rmse_proxy", "error_pos_mean", False),
    ]
    out = []
    for label, metric, ascending in specs:
        valid = [r for r in rows if math.isfinite(_f(r.get(metric)))]
        valid = sorted(valid, key=lambda r: _f(r.get(metric)), reverse=not ascending)
        if not valid:
            continue
        r = valid[0]
        out.append({
            "worst_case": label,
            "metric": metric,
            "scenario": r.get("scenario"),
            "model_seed": r.get("model_seed"),
            "value": r.get(metric),
            "file": r.get("file"),
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline A0D M->P directionality diagnostics from eval_details CSV files.")
    p.add_argument("--a0d-dir", default=r"E:\migration_packages\results\phase2_me_a0_dir_only")
    p.add_argument("--a0-dir", default=r"E:\migration_packages\results\phase2_me_a0_only")
    p.add_argument("--out-dir", default=str(_project_root() / "results" / "a0d_directionality_diagnostics"))
    p.add_argument("--spread-top-frac", type=float, default=0.30)
    p.add_argument("--tail-frac", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    a0d_dir = _resolve(args.a0d_dir)
    a0_dir = _resolve(args.a0_dir)
    out_dir = _resolve(args.out_dir)
    if out_dir is None:
        raise ValueError("--out-dir is required.")
    out_dir.mkdir(parents=True, exist_ok=True)

    spread_top_frac = min(max(float(args.spread_top_frac), 0.01), 0.99)
    tail_frac = min(max(float(args.tail_frac), 0.001), 0.50)

    files: List[tuple[str, Path]] = []
    files.extend(("A0D", p) for p in _find_detail_files(a0d_dir, "A0D"))
    files.extend(("A0", p) for p in _find_detail_files(a0_dir, "A0"))
    if not files:
        raise RuntimeError("No eval_details CSV files found.")

    runs: List[RunData] = []
    for method, path in files:
        print(f"[read] {method}: {path}")
        runs.append(_read_detail_file(path, method=method))

    seed_rows = [_run_metrics(run, spread_top_frac=spread_top_frac) for run in runs]
    metric_names = [
        "entropy_mean",
        "row_std_mean",
        "evidence_mass_mean",
        "track_mass_mean",
        "own_track_mass_mean",
        "other_track_mass_mean",
        "evidence_to_track_mass_ratio",
        "raw_residual_corr",
        "residual_log_corr",
        "residual_centered_corr",
        "evidence_normalized_corr",
        "within_evidence_gap_corr",
        "high_spread_corr",
        "mid_spread_corr",
        "low_spread_corr",
        "worst_track_hit_rate",
        "top2_hit_rate",
        "spearman_rank_corr",
        "kendall_tau",
        "high_spread_worst_track_hit_rate",
        "delta_w_mean",
        "delta_g_mean",
        "delta_cov_mean",
        "worst_low_weight_rate",
        "worst_low_gate_rate",
        "worst_high_cov_rate",
        "error_pos_mean",
        "error_pos_max",
    ]
    by_scene_rows = _aggregate_rows(seed_rows, ["method", "scenario"], metric_names)
    summary_rows = _aggregate_rows(seed_rows, ["method"], metric_names)
    error_bin_rows = _error_bin_rows(runs, spread_top_frac=spread_top_frac, tail_frac=tail_frac)
    pair_rows = _pair_matrix_rows(runs)
    compare_rows = _attention_compare_rows(seed_rows)
    corr_rows = _metric_correlation_rows(seed_rows)
    worst_rows = _seed_worst_rows(seed_rows)

    skipped = [
        {
            "method": run.method,
            "scenario": run.scenario,
            "model_seed": run.model_seed,
            "file": str(run.file),
            "skipped_metrics": run.skipped,
        }
        for run in runs
        if run.skipped
    ]

    _write_csv(out_dir / "a0d_directionality_by_seed.csv", seed_rows)
    _write_csv(out_dir / "a0d_directionality_by_scene.csv", by_scene_rows)
    _write_csv(out_dir / "a0d_directionality_summary.csv", summary_rows)
    _write_csv(out_dir / "a0d_directionality_by_error_bin.csv", error_bin_rows)
    _write_csv(out_dir / "a0d_directionality_pair_matrix.csv", pair_rows)
    _write_csv(out_dir / "a0_vs_a0d_attention_compare.csv", compare_rows)
    _write_csv(out_dir / "a0d_directionality_metric_correlations.csv", corr_rows)
    _write_csv(out_dir / "a0d_directionality_seed_worst_table.csv", worst_rows)
    _write_csv(out_dir / "a0d_directionality_skipped_metrics.csv", skipped)

    report = {
        "args": {
            "a0d_dir": str(a0d_dir),
            "a0_dir": str(a0_dir),
            "out_dir": str(out_dir),
            "spread_top_frac": spread_top_frac,
            "tail_frac": tail_frac,
        },
        "summary": summary_rows,
        "by_scene": by_scene_rows,
        "seed_worst_table": worst_rows,
        "skipped_metrics": skipped,
        "decision_hints": [
            "If high_spread_corr and evidence_normalized_corr are much higher than raw_residual_corr, the raw residual_corr metric is too coarse.",
            "If high-spread ranking is weak while row_std is high, P rows differ but not in residual direction; inspect directional loss.",
            "If ranking aligns but delta_w/delta_cov/delta_g do not respond, inspect P-P/head/fusion outputs before changing pair features.",
            "If tail_error bins lose directionality, consider tail-aware directionality or pollution-window diagnostics.",
        ],
    }
    _write_json(out_dir / "a0d_directionality_summary.json", report)

    print(f"[done] wrote diagnostics to {out_dir}")
    for row in summary_rows:
        print({
            "method": row.get("method"),
            "row_std": row.get("row_std_mean_mean"),
            "raw_corr": row.get("raw_residual_corr_mean"),
            "high_corr": row.get("high_spread_corr_mean"),
            "ev_norm_corr": row.get("evidence_normalized_corr_mean"),
            "hit": row.get("worst_track_hit_rate_mean"),
        })


if __name__ == "__main__":
    main()
