"""UV (unvoiced) detection and thresholding algorithms (model-agnostic)."""

from __future__ import annotations

import numpy as np


def threshold_at(f0: np.ndarray, confidence: np.ndarray, value: float) -> np.ndarray:
    """Fixed threshold: frames with confidence < value are set to 0.

    Args:
        f0: (T,) float32
        confidence: (T,) float32
        value: threshold

    Returns:
        f0_masked: (T,) float32
    """
    f0 = f0.copy()
    f0[confidence < value] = 0.0
    return f0


def hysteresis(
    f0: np.ndarray,
    confidence: np.ndarray,
    lower_bound: float = 0.19,
    upper_bound: float = 0.31,
    width: float = 2.0,
    stds: float = 1.0,
) -> np.ndarray:
    """Hysteresis thresholding (torchcrepe style).

    Uses statistical properties of pitch and confidence to suppress
    short spurious voiced segments.

    Args:
        f0: (T,) float32
        confidence: (T,) float32
        lower_bound: absolute lower bound
        upper_bound: absolute upper bound for hysteresis
        width: parabolic threshold width
        stds: standard deviations for normalization

    Returns:
        f0_masked: (T,) float32
    """
    f0 = f0.copy()
    T = len(f0)

    # Step 1: lower bound
    pitch_mask = confidence >= lower_bound

    # Step 2: whiten pitch
    voiced_f0 = f0[pitch_mask]
    if len(voiced_f0) > 1:
        mu = np.mean(voiced_f0)
        sigma = np.std(voiced_f0)
        if sigma > 0:
            whitened = np.abs(f0 - mu) / sigma
        else:
            whitened = np.zeros_like(f0)
    else:
        whitened = np.zeros_like(f0)

    # Step 3: build parabolic threshold
    adaptive_thresh = np.maximum(lower_bound, width * whitened * whitened - width * stds * stds)

    # Step 4: apply hysteresis to suppress spurious segments
    final_mask = confidence >= adaptive_thresh
    if np.any(final_mask):
        # Extend voiced segments through hysteresis
        pending = False
        pending_start = 0
        final = np.zeros_like(final_mask)

        for i in range(T):
            if final_mask[i]:
                if pending:
                    # Check if any frame in pending region exceeds upper_bound
                    if np.any(confidence[pending_start:i] >= upper_bound):
                        final[pending_start:i] = True
                    pending = False
                final[i] = True
            elif confidence[i] >= lower_bound:
                # Start pending
                if not pending:
                    pending_start = i
                    pending = True
            else:
                if pending:
                    if np.any(confidence[pending_start:i] >= upper_bound):
                        final[pending_start:i] = True
                    pending = False

        if pending:
            if np.any(confidence[pending_start:] >= upper_bound):
                final[pending_start:] = True

        f0[~final] = 0.0

    return f0


def silence_mask(
    f0: np.ndarray,
    confidence: np.ndarray,
    loudness: np.ndarray,
    threshold_db: float = -60.0,
) -> np.ndarray:
    """Mask silent frames based on loudness (torchcrepe style).

    Args:
        f0: (T,) float32
        confidence: (T,) float32
        loudness: (T,) float32 in dB
        threshold_db: silence threshold in dB

    Returns:
        f0_masked: (T,) float32
    """
    f0 = f0.copy()
    f0[loudness < threshold_db] = 0.0
    return f0
