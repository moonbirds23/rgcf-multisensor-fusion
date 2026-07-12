"""Offline metrics for diagonal Gaussian fusion uncertainty.

These utilities evaluate already-produced predictions and reported diagonal
covariances.  They make no assumptions about model architecture or temporal
causality; callers are responsible for supplying only causally valid outputs.
All calculations use NumPy float64, including when given torch tensors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2

try:  # Keep the module usable in lightweight NumPy-only evaluation tools.
    import torch
except ImportError:  # pragma: no cover - torch is a project dependency.
    torch = None


def _as_float64_array(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _validated_rows(
    prediction: Any,
    target: Any,
    covariance_diag: Any,
    valid_mask: Any | None,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Flatten inputs and retain finite, positive-variance state samples."""
    if min_variance <= 0.0:
        raise ValueError("min_variance must be positive")

    prediction = _as_float64_array(prediction)
    target = _as_float64_array(target)
    covariance_diag = _as_float64_array(covariance_diag)
    if prediction.shape != target.shape or prediction.shape != covariance_diag.shape:
        raise ValueError("prediction, target, and covariance_diag must have identical shapes")
    if prediction.ndim < 1 or prediction.shape[-1] == 0:
        raise ValueError("inputs must have a non-empty state dimension")

    sample_shape = prediction.shape[:-1]
    state_dim = prediction.shape[-1]
    prediction = prediction.reshape(-1, state_dim)
    target = target.reshape(-1, state_dim)
    covariance_diag = covariance_diag.reshape(-1, state_dim)

    if valid_mask is None:
        requested = np.ones(prediction.shape[0], dtype=bool)
    else:
        mask = _as_float64_array(valid_mask)
        try:
            requested = np.broadcast_to(mask, sample_shape).reshape(-1).astype(bool, copy=False)
        except ValueError as exc:
            raise ValueError("valid_mask must broadcast to prediction.shape[:-1]") from exc

    finite_and_positive = (
        np.isfinite(prediction).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(covariance_diag).all(axis=1)
        & (covariance_diag >= min_variance).all(axis=1)
    )
    selected = requested & finite_and_positive
    error = prediction[selected] - target[selected]
    variance = covariance_diag[selected]
    return error, variance, int(prediction.shape[0]), int(selected.sum())


def diagonal_nees(
    prediction: Any,
    target: Any,
    covariance_diag: Any,
    *,
    valid_mask: Any | None = None,
    min_variance: float = 1e-8,
) -> np.ndarray:
    """Return one normalized-estimation-error-squared (NEES) value per sample."""
    error, variance, _, _ = _validated_rows(prediction, target, covariance_diag, valid_mask, min_variance)
    return np.sum((error * error) / variance, axis=1)


def diagonal_gaussian_nll(
    prediction: Any,
    target: Any,
    covariance_diag: Any,
    *,
    valid_mask: Any | None = None,
    min_variance: float = 1e-8,
) -> np.ndarray:
    """Return full multivariate diagonal-Gaussian NLL values per sample."""
    error, variance, _, _ = _validated_rows(prediction, target, covariance_diag, valid_mask, min_variance)
    state_dim = variance.shape[1]
    return 0.5 * (state_dim * np.log(2.0 * np.pi) + np.sum(np.log(variance) + (error * error) / variance, axis=1))


def diagonal_uncertainty_metrics(
    prediction: Any,
    target: Any,
    covariance_diag: Any,
    *,
    valid_mask: Any | None = None,
    coverage_level: float = 0.95,
    min_variance: float = 1e-8,
) -> dict[str, float | int]:
    """Summarize ANEES, chi-square coverage, and Gaussian NLL.

    Invalid rows (masked rows, non-finite values, or variance below
    ``min_variance``) are excluded and reported through ``invalid_count``.
    If no rows remain, the three aggregate metrics are ``nan`` rather than a
    misleading zero.  Coverage is the fraction whose full-state NEES is below
    the chi-square quantile for the state dimension.
    """
    if not 0.0 < coverage_level < 1.0:
        raise ValueError("coverage_level must lie strictly between 0 and 1")
    error, variance, sample_count, valid_count = _validated_rows(
        prediction, target, covariance_diag, valid_mask, min_variance
    )
    state_dim = _as_float64_array(prediction).shape[-1]
    threshold = float(chi2.ppf(coverage_level, df=state_dim))
    if valid_count == 0:
        return {
            "sample_count": sample_count,
            "valid_count": 0,
            "invalid_count": sample_count,
            "state_dim": state_dim,
            "coverage_level": float(coverage_level),
            "coverage_threshold": threshold,
            "anees": float("nan"),
            "coverage": float("nan"),
            "nll": float("nan"),
        }
    nees = np.sum((error * error) / variance, axis=1)
    nll = 0.5 * (
        state_dim * np.log(2.0 * np.pi)
        + np.sum(np.log(variance) + (error * error) / variance, axis=1)
    )
    return {
        "sample_count": sample_count,
        "valid_count": valid_count,
        "invalid_count": sample_count - valid_count,
        "state_dim": state_dim,
        "coverage_level": float(coverage_level),
        "coverage_threshold": threshold,
        "anees": float(np.mean(nees)),
        "coverage": float(np.mean(nees <= threshold)),
        "nll": float(np.mean(nll)),
    }
