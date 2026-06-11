"""F0 decoding algorithms (model-agnostic)."""

from __future__ import annotations

import numpy as np


def argmax(logits: np.ndarray, bins_to_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take argmax over pitch bins.

    Args:
        logits: (T, num_bins) float32
        bins_to_freq: (num_bins,) float32, mapping from bin index to Hz

    Returns:
        (bins, f0_hz) each (T,)
    """
    bins = np.argmax(logits, axis=-1)
    f0 = bins_to_freq[bins]
    return bins, f0


def weighted_argmax(
    logits: np.ndarray,
    bins_to_freq: np.ndarray,
    window: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted average around argmax for sub-bin precision.

    Args:
        logits: (T, num_bins)
        bins_to_freq: (num_bins,)
        window: number of bins in local window (must be odd)

    Returns:
        (bins, f0_hz), f0 is sub-bin interpolated
    """
    T, num_bins = logits.shape
    hard_bins = np.argmax(logits, axis=-1)  # (T,)
    half_w = window // 2

    # Build local windows
    start = np.clip(hard_bins - half_w, 0, None)
    end = np.clip(hard_bins + half_w + 1, None, num_bins)
    f0_float = np.zeros(T, dtype=np.float64)

    for t in range(T):
        s, e = start[t], end[t]
        if s >= e:
            f0_float[t] = bins_to_freq[hard_bins[t]]
            continue
        probs = np.exp(logits[t, s:e] - np.max(logits[t, s:e]))
        probs /= probs.sum() + 1e-10
        freqs = bins_to_freq[s:e]
        f0_float[t] = np.dot(probs, freqs)

    return hard_bins, f0_float.astype(np.float32)


def viterbi(
    logits: np.ndarray,
    bins_to_freq: np.ndarray,
    transition_width: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi decoding with triangular transition matrix.

    Args:
        logits: (T, num_bins)
        bins_to_freq: (num_bins,)
        transition_width: max bin transition per frame

    Returns:
        (bins, f0_hz) smoothed by Viterbi
    """
    T, num_bins = logits.shape
    # Build triangular transition matrix
    trans = np.maximum(transition_width - np.abs(np.arange(num_bins)[:, None] - np.arange(num_bins)[None, :]), 0)
    trans = trans / trans.sum(axis=-1, keepdims=True)

    # Log probabilities
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True) + 1e-10)
    log_trans = np.log(trans + 1e-10)

    # Forward pass
    delta = np.zeros((T, num_bins), dtype=np.float64)
    psi = np.zeros((T, num_bins), dtype=np.int64)
    delta[0] = log_probs[0]

    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_trans + log_probs[t][None, :]
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0)

    # Backtrack
    bins = np.zeros(T, dtype=np.int64)
    bins[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        bins[t] = psi[t + 1, bins[t + 1]]

    f0 = bins_to_freq[bins]
    return bins, f0


def local_argmax(
    logits: np.ndarray,
    cent_table: np.ndarray,
    threshold: float = 0.05,
    window: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Local argmax weighted averaging in cent space (FCPE style).

    Args:
        logits: (T, num_bins) sigmoid outputs in [0,1]
        cent_table: (num_bins,) cent values per bin
        threshold: confidence threshold for UV
        window: local window size

    Returns:
        (cents, f0_hz), each (T,)
    """
    T, num_bins = logits.shape
    hard_bins = np.argmax(logits, axis=-1)
    half_w = window // 2

    cents = np.zeros(T, dtype=np.float64)
    for t in range(T):
        center = hard_bins[t]
        s = max(0, center - half_w)
        e = min(num_bins, center + half_w + 1)
        if s >= e:
            cents[t] = cent_table[center]
            continue
        y_local = logits[t, s:e] + 1e-10
        c_local = cent_table[s:e]
        cents[t] = np.dot(y_local, c_local) / y_local.sum()

    f0 = cent_to_f0(cents)
    # Apply UV mask
    confidence = np.max(logits, axis=-1)
    cents[confidence <= threshold] = -float("inf")
    f0[confidence <= threshold] = 0.0
    return cents.astype(np.float32), f0


def cent_to_f0(cent: np.ndarray) -> np.ndarray:
    """Convert cent to Hz. f0 = 10 * 2^(cent / 1200)"""
    return 10.0 * (2.0 ** (cent / 1200.0))


def f0_to_cent(f0: np.ndarray) -> np.ndarray:
    """Convert Hz to cent. cent = 1200 * log2(f0 / 10)"""
    return 1200.0 * np.log2(np.maximum(f0, 1e-10) / 10.0)
