from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# =========================================================
# 基本工具
# =========================================================

def rmse_pos(track_xy: np.ndarray, truth_xy: np.ndarray) -> float:
    e = track_xy - truth_xy
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))


def rmse_full(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    e = pred - truth
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def mae_pos(track_xy: np.ndarray, truth_xy: np.ndarray) -> float:
    e = np.abs(track_xy - truth_xy)
    return float(np.mean(np.sum(e, axis=1)))


# =========================================================
# CI 工具
# =========================================================

def ci_fuse_two(
    x1: np.ndarray,
    P1: np.ndarray,
    x2: np.ndarray,
    P2: np.ndarray,
    objective: str = "logdet",
    n_grid: int = 101,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    两节点 CI 融合
    """
    P1_inv = np.linalg.inv(P1)
    P2_inv = np.linalg.inv(P2)

    best_w = None
    best_x = None
    best_P = None
    best_score = None

    ws = np.linspace(0.0, 1.0, n_grid)

    for w in ws:
        Y = w * P1_inv + (1.0 - w) * P2_inv
        P = np.linalg.inv(Y)
        x = P @ (w * P1_inv @ x1 + (1.0 - w) * P2_inv @ x2)

        if objective == "trace":
            score = np.trace(P)
        else:
            score = np.linalg.slogdet(P)[1]

        if best_score is None or score < best_score:
            best_score = score
            best_w = float(w)
            best_x = x
            best_P = P

    return best_x, best_P, best_w


def ci_fuse_multi(
    xs: List[np.ndarray],
    Ps: List[np.ndarray],
    objective: str = "logdet",
    step: float = 0.05,
    min_w: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    多节点 CI 融合（简单网格搜索版）
    当前先针对 4 节点最常用场景。
    """
    n = len(xs)
    assert n == len(Ps), "xs and Ps length mismatch"

    # 当前先做 4 节点常用情形，便于与你现有 clean 版本对齐
    if n != 4:
        raise NotImplementedError("Current ci_fuse_multi supports n=4 only.")

    grid = np.arange(min_w, 1.0 + 1e-12, step)

    invPs = [np.linalg.inv(P) for P in Ps]

    best_score = None
    best_x = None
    best_P = None
    best_w = None

    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                w4 = 1.0 - (w1 + w2 + w3)
                if w4 < min_w or w4 > 1.0:
                    continue

                ws = np.array([w1, w2, w3, w4], dtype=np.float64)
                if np.any(ws < min_w):
                    continue

                Y = np.zeros_like(Ps[0], dtype=np.float64)
                y = np.zeros_like(xs[0], dtype=np.float64)

                for i in range(4):
                    Y += ws[i] * invPs[i]
                    y += ws[i] * (invPs[i] @ xs[i])

                P = np.linalg.inv(Y)
                x = P @ y

                if objective == "trace":
                    score = np.trace(P)
                else:
                    score = np.linalg.slogdet(P)[1]

                if best_score is None or score < best_score:
                    best_score = score
                    best_x = x
                    best_P = P
                    best_w = ws

    return best_x, best_P, best_w


# =========================================================
# WAA-MM
# =========================================================

def fuse_waa_mm_single(
    xhats: np.ndarray,
    Phats: np.ndarray,
    valid: np.ndarray,
    eps: float = 1e-6,
):
    """Weighted Arithmetic Average with Moment Matching for one time step."""
    xhats = np.asarray(xhats, dtype=np.float64)
    Phats = np.asarray(Phats, dtype=np.float64)
    valid = np.asarray(valid).astype(bool)

    N, D = xhats.shape
    weights = np.zeros((N,), dtype=np.float64)
    idx = np.where(valid)[0]

    if idx.size == 0:
        return (
            np.zeros((D,), dtype=np.float64),
            np.eye(D, dtype=np.float64) * 1e6,
            weights,
        )

    if idx.size == 1:
        j = int(idx[0])
        weights[j] = 1.0
        return xhats[j].copy(), Phats[j].copy(), weights

    traces = np.array(
        [np.trace(Phats[j]) for j in idx],
        dtype=np.float64,
    )
    scores = 1.0 / np.maximum(traces, eps)
    beta = scores / np.maximum(np.sum(scores), eps)

    for local_pos, j in enumerate(idx):
        weights[j] = beta[local_pos]

    x_fused = np.sum(
        beta[:, None] * xhats[idx],
        axis=0,
    )

    P_fused = np.zeros((D, D), dtype=np.float64)
    for local_pos, j in enumerate(idx):
        dx = (xhats[j] - x_fused).reshape(D, 1)
        P_fused += beta[local_pos] * (Phats[j] + dx @ dx.T)

    P_fused = 0.5 * (P_fused + P_fused.T)
    return x_fused, P_fused, weights


def fuse_waa_mm_sequence(
    xhat: np.ndarray,
    Phat: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
):
    """WAA-MM fusion for a full simulation sequence."""
    xhat = np.asarray(xhat)
    Phat = np.asarray(Phat)
    valid_mask = np.asarray(valid_mask)

    K, N, D = xhat.shape
    x_fused = np.zeros((K, D), dtype=np.float64)
    P_fused = np.zeros((K, D, D), dtype=np.float64)
    weights = np.zeros((K, N), dtype=np.float64)

    for k in range(K):
        xk, Pk, wk = fuse_waa_mm_single(
            xhats=xhat[k],
            Phats=Phat[k],
            valid=valid_mask[k],
            eps=eps,
        )
        x_fused[k] = xk
        P_fused[k] = Pk
        weights[k] = wk

    return x_fused, P_fused, weights


# =========================================================
# 权重行为统计
# =========================================================

def compute_fault_normal_weight_stats(
    weights: np.ndarray,
    fault_active_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
):
    """Compute mean weight for fault vs normal sensors."""
    weights = np.asarray(weights, dtype=np.float64)
    fault_active_mask = np.asarray(fault_active_mask) > 0.5

    if valid_mask is None:
        valid = np.ones_like(weights, dtype=bool)
    else:
        valid = np.asarray(valid_mask) > 0.5

    fault_sel = fault_active_mask & valid
    normal_sel = (~fault_active_mask) & valid

    out = {}

    if fault_sel.sum() > 0:
        out["mean_weight_fault"] = float(np.mean(weights[fault_sel]))
    else:
        out["mean_weight_fault"] = float("nan")

    if normal_sel.sum() > 0:
        out["mean_weight_normal"] = float(np.mean(weights[normal_sel]))
    else:
        out["mean_weight_normal"] = float("nan")

    if np.isfinite(out.get("mean_weight_fault", np.nan)) and np.isfinite(
        out.get("mean_weight_normal", np.nan)
    ):
        out["weight_separation_normal_minus_fault"] = (
            out["mean_weight_normal"] - out["mean_weight_fault"]
        )
    else:
        out["weight_separation_normal_minus_fault"] = float("nan")

    return out


# =========================================================
# baseline 轨迹提取
# =========================================================

def extract_baseline_trajectories(sim: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    从标准 sim 结果中提取：
    - truth
    - s1~sN 单节点轨迹
    - AVG
    - CI-chain
    - CI-multi
    - best-single
    """
    x_truth = sim["x_truth_4d"]        # [K,4]
    xhat = sim["xhat"]                 # [K,N,4]
    Phat = sim["Phat"]                 # [K,N,4,4]

    K, N, _ = xhat.shape
    truth_xy = x_truth[:, 0:2].copy()

    out = {"truth_xy": truth_xy}

    for i in range(N):
        out[f"s{i+1}_xy"] = xhat[:, i, 0:2].copy()

    # AVG
    avg_xy = np.mean(xhat[:, :, 0:2], axis=1)
    out["avg_xy"] = avg_xy

    # CI-chain
    ci_chain_xy = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        x_cur = xhat[k, 0].copy()
        P_cur = Phat[k, 0].copy()
        for i in range(1, N):
            x_cur, P_cur, _ = ci_fuse_two(
                x_cur, P_cur,
                xhat[k, i], Phat[k, i],
                objective="logdet",
                n_grid=101
            )
        ci_chain_xy[k] = x_cur[0:2]
    out["ci_chain_xy"] = ci_chain_xy

    # CI-multi（当前默认 N=4）
    if N == 4:
        ci_multi_xy = np.zeros((K, 2), dtype=np.float64)
        for k in range(K):
            xs = [xhat[k, i].copy() for i in range(4)]
            Ps = [Phat[k, i].copy() for i in range(4)]
            x_f, _, _ = ci_fuse_multi(xs, Ps, objective="logdet", step=0.05, min_w=0.0)
            ci_multi_xy[k] = x_f[0:2]
        out["ci_multi_xy"] = ci_multi_xy

    # best-single（按全局 position RMSE 选）
    single_tracks = [xhat[:, i, 0:2].copy() for i in range(N)]
    single_rmses = [rmse_pos(tr, truth_xy) for tr in single_tracks]
    best_idx = int(np.argmin(single_rmses))
    out["best_single_xy"] = single_tracks[best_idx]
    out["best_single_name"] = f"s{best_idx + 1}"

    return out


def evaluate_baselines_from_sim(sim: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    对单次仿真结果计算 baseline metrics.
    包含 AVG / CI-multi / WAA-MM / best-single.
    """
    x_truth = sim["x_truth_4d"]
    xhat = sim["xhat"]
    Phat = sim["Phat"]
    valid_mask = sim.get("valid_mask", None)

    K, N, _ = xhat.shape
    truth_xy = x_truth[:, 0:2]

    if valid_mask is None:
        valid_mask = np.ones((K, N), dtype=float)

    # single sensor rmse
    rmses = []
    for i in range(N):
        rmses.append(rmse_pos(xhat[:, i, 0:2], truth_xy))

    # avg
    avg_xy = np.mean(xhat[:, :, 0:2], axis=1)
    rmse_avg = rmse_pos(avg_xy, truth_xy)

    # ci chain (kept for backward compat, not used in ablation summary)
    ci_chain_xy = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        x_cur = xhat[k, 0].copy()
        P_cur = Phat[k, 0].copy()
        for i in range(1, N):
            x_cur, P_cur, _ = ci_fuse_two(
                x_cur, P_cur,
                xhat[k, i], Phat[k, i],
                objective="logdet",
                n_grid=101
            )
        ci_chain_xy[k] = x_cur[0:2]
    rmse_ci_chain = rmse_pos(ci_chain_xy, truth_xy)

    # ci multi
    rmse_ci_multi = float("nan")
    if N == 4:
        ci_multi_xy = np.zeros((K, 2), dtype=np.float64)
        for k in range(K):
            xs = [xhat[k, i].copy() for i in range(4)]
            Ps = [Phat[k, i].copy() for i in range(4)]
            x_f, _, _ = ci_fuse_multi(xs, Ps, objective="logdet", step=0.05, min_w=0.0)
            ci_multi_xy[k] = x_f[0:2]
        rmse_ci_multi = rmse_pos(ci_multi_xy, truth_xy)

    # WAA-MM
    x_waa_mm, _, weights_waa_mm = fuse_waa_mm_sequence(xhat, Phat, valid_mask)
    rmse_waa_mm = rmse_pos(x_waa_mm[:, 0:2], truth_xy)

    out = {}
    for i, r in enumerate(rmses):
        out[f"rmse_s{i+1}"] = float(r)
    out["rmse_avg"] = float(rmse_avg)
    out["rmse_ci_chain"] = float(rmse_ci_chain)
    out["rmse_ci_multi"] = float(rmse_ci_multi)
    out["rmse_waa_mm"] = float(rmse_waa_mm)
    out["rmse_best_single"] = float(min(rmses))

    # weight stats for non-learned baselines
    fault_active_mask = sim.get("fault_active_mask", None)
    if fault_active_mask is not None:
        # AVG weights
        weights_avg = np.zeros((K, N), dtype=np.float64)
        for k in range(K):
            v_count = int(np.sum(valid_mask[k] > 0.5))
            if v_count > 0:
                weights_avg[k, valid_mask[k] > 0.5] = 1.0 / v_count
        stats_avg = compute_fault_normal_weight_stats(weights_avg, fault_active_mask, valid_mask)
        for k, v in stats_avg.items():
            out[f"{k}_avg"] = v

        # WAA-MM weights
        stats_waa = compute_fault_normal_weight_stats(weights_waa_mm, fault_active_mask, valid_mask)
        for k, v in stats_waa.items():
            out[f"{k}_waa_mm"] = v

    return out