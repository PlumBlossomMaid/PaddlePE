"""Ensemble / TTA methods for F0 refinement (model-agnostic)."""

from __future__ import annotations

import numpy as np


def ensemble_f0(
    f0s: np.ndarray,
    key_shifts: list[float],
    uv_penalty: float = 12.0,
) -> np.ndarray:
    """DP-based ensemble of multiple F0 predictions
    with key shifts (FCPE style).

    Given multiple F0 predictions from different key shifts (transpositions),
    find the optimal path through all candidates using dynamic programming.

    Args:
        f0s: (T, K) float32, K predictions from different key shifts
        key_shifts: list of K key shifts in semitones
        uv_penalty: penalty for UV (unvoiced) frames

    Returns:
        f0_result: (T,) float32, ensembled F0
    """
    T, K = f0s.shape
    uv_penalty_sq = uv_penalty * uv_penalty

    # Convert all F0s to the same note space
    shift_factors = 2.0 ** (
        np.array(key_shifts, dtype=np.float64) / 12.0
    )  # (K,)
    f0s_aligned = f0s / shift_factors[None, :]  # (T, K)

    # Convert to note (MIDI-like)
    notes = 12.0 * np.log2(np.maximum(f0s_aligned, 1e-10) / 440.0) + 69.0
    notes[notes < 0] = 0

    # DP forward
    dp = np.zeros((T, K), dtype=np.float64)
    backtrack = np.zeros((T, K), dtype=np.int64)

    # Init: UV penalty for first frame
    dp[0] = (notes[0] <= 0) * uv_penalty_sq

    for t in range(1, T):
        # Penalty matrix: (K, K) where prev=row, curr=col
        t_uv = notes[t] <= 0
        t1_uv = notes[t - 1] <= 0

        # UV penalty for current frame
        penalty = np.zeros((K, K), dtype=np.float64)
        penalty += uv_penalty_sq * t_uv[None, :]

        # L2 distance penalty for V->V transitions
        l2 = (
            (notes[t - 1][:, None] - notes[t][None, :])
            * (~t1_uv)[:, None]
            * (~t_uv)[None, :]
        )
        l2 = l2 * l2 - 0.5
        l2 = np.maximum(l2, 0)
        penalty += l2

        # UV->V transition penalty
        penalty += t1_uv[:, None] * (~t_uv)[None, :] * uv_penalty_sq * 2

        # Choose min
        scores = dp[t - 1][:, None] + penalty
        backtrack[t] = np.argmin(scores, axis=0)
        dp[t] = np.min(scores, axis=0)

    # Backtrack
    f0_result = np.zeros(T, dtype=np.float32)
    best = np.argmin(dp[-1])
    for t in range(T - 1, -1, -1):
        f0_result[t] = f0s[t, best]
        best = backtrack[t, best]

    return f0_result
