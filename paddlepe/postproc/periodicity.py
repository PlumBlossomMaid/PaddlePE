"""Periodicity / confidence estimation from pitch distributions
(model-agnostic)."""

from __future__ import annotations

import numpy as np


def entropy(logits: np.ndarray) -> np.ndarray:
    """Entropy-based periodicity.

    Lower entropy = more periodic (voiced).
    Result normalized to [0, 1] where 1 = fully periodic.

    Args:
        logits: (T, num_bins)

    Returns:
        periodicity: (T,) float32
    """
    # Softmax
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True) + 1e-10
    # Entropy
    ent = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
    max_ent = np.log(logits.shape[-1])
    periodicity = 1.0 - ent / max_ent
    return periodicity.astype(np.float32)


def periodicity_max(logits: np.ndarray) -> np.ndarray:
    """Max-probability periodicity.

    Args:
        logits: (T, num_bins)

    Returns:
        periodicity: (T,) float32
    """
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True) + 1e-10
    return np.max(probs, axis=-1).astype(np.float32)


def periodicity_sum(logits: np.ndarray) -> np.ndarray:
    """Sum-of-activations periodicity.

    Args:
        logits: (T, num_bins) sigmoid outputs

    Returns:
        periodicity: (T,) float32, clipped to [0, 1]
    """
    return np.clip(np.sum(np.exp(logits), axis=-1), 0, 1).astype(np.float32)
