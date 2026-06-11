"""Signal filtering utilities for F0 smoothing (model-agnostic, NaN-aware)."""

from __future__ import annotations

import numpy as np


def nanmean(values: np.ndarray) -> float:
    """Compute mean ignoring NaN/inf values."""
    return np.nanmean(values).item() if np.any(np.isfinite(values)) else 0.0


def nanmedian(values: np.ndarray) -> float:
    """Compute median ignoring NaN/inf values."""
    return np.nanmedian(values).item() if np.any(np.isfinite(values)) else 0.0


def mean_filter(signal: np.ndarray, win_length: int = 9) -> np.ndarray:
    """Mean filter ignoring zeros (UV frames).

    Args:
        signal: (T,) float32
        win_length: odd window size

    Returns:
        filtered: (T,) float32
    """
    T = len(signal)
    half = win_length // 2
    result = signal.copy().astype(np.float64)

    for t in range(T):
        left = max(0, t - half)
        right = min(T, t + half + 1)
        window = signal[left:right]
        valid = window > 0
        if valid.any():
            result[t] = window[valid].mean()
        else:
            result[t] = 0.0

    return result.astype(np.float32)


def median_filter(signal: np.ndarray, win_length: int = 9) -> np.ndarray:
    """Median filter ignoring zeros (UV frames).

    Args:
        signal: (T,) float32
        win_length: odd window size

    Returns:
        filtered: (T,) float32
    """
    T = len(signal)
    half = win_length // 2
    result = signal.copy().astype(np.float64)

    for t in range(T):
        left = max(0, t - half)
        right = min(T, t + half + 1)
        window = signal[left:right]
        valid = window[window > 0]
        if len(valid) > 0:
            result[t] = np.median(valid)
        else:
            result[t] = 0.0

    return result.astype(np.float32)


def interpolate_uv(f0: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Linear interpolation of unvoiced frames in log domain.

    Args:
        f0: (T,) float32, 0=unvoiced
        uv: (T,) bool, True=unvoiced

    Returns:
        f0_interp: (T,) float32
    """
    f0 = f0.copy().astype(np.float64)
    f0[f0 <= 0] = np.nan
    # Find voiced segments
    voiced_idx = np.where(~uv)[0]
    if len(voiced_idx) < 2:
        f0[~uv] = 0.0
        return f0.astype(np.float32)

    # Interpolate in log domain
    log_f0 = np.log(np.maximum(f0, 1e-10))
    log_f0[uv] = np.interp(
        np.where(uv)[0],
        voiced_idx,
        log_f0[voiced_idx],
    )
    result = np.exp(log_f0)
    result[~uv] = f0[~uv]
    return result.astype(np.float32)
