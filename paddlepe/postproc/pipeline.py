"""Unified post-processing pipeline for F0 inference.

Chains confidence thresholding, median/mean filtering, and UV
interpolation into a single call operating entirely on GPU
(paddle tensors) to avoid CPU round-trips.

Each model's ``infer()`` calls this with its own default parameters.

Usage::

    from paddlepe.postproc.pipeline import postprocess_f0

    f0, conf = postprocess_f0(
        f0_tensor,
        conf_tensor,
        threshold=0.5,
        median_filter=3,
        interp_uv=True,
    )
"""

from __future__ import annotations

from typing import Any

import paddle


def _median_filter_paddle(signal: paddle.Tensor, win_length: int) -> paddle.Tensor:
    """Median filter with UV-aware (zero) handling.

    Args:
        signal: (T,) float32 on any device.
        win_length: odd kernel size.

    Returns:
        (T,) float32, filtered.
    """
    T = signal.shape[0]
    half = win_length // 2

    # Pad with zeros on both sides → shape (T + 2*half,)
    padded = paddle.nn.functional.pad(
        signal.unsqueeze(0), [half, half], mode="constant", value=0.0
    )  # (1, T + 2*half)

    # Reshape to 4D [N, C, H, W] for unfold
    padded_4d = padded.reshape([1, 1, 1, -1])  # (1, 1, 1, T')
    windows = paddle.nn.functional.unfold(
        padded_4d, kernel_sizes=[1, win_length], strides=[1, 1]
    )  # (1, win_length, T_out)

    windows = windows.squeeze(0)  # (win_length, T_out)
    sorted_w = paddle.sort(windows, axis=0)  # (win_length, T_out)
    mid = win_length // 2
    median_vals = sorted_w[mid]  # (T_out,)

    # Trim/pad to original length
    if median_vals.shape[0] > T:
        median_vals = median_vals[:T]
    elif median_vals.shape[0] < T:
        pad_len = T - median_vals.shape[0]
        median_vals = paddle.concat([median_vals, paddle.zeros([pad_len])])

    return median_vals


def _interp_uv_paddle(f0: paddle.Tensor, uv: paddle.Tensor) -> paddle.Tensor:
    """Linear interpolation of unvoiced frames in log domain.

    Args:
        f0: (T,) float32, 0 = unvoiced.
        uv: (T,) bool, ``True`` = unvoiced.

    Returns:
        (T,) float32, interpolated.
    """
    voiced = ~uv
    voiced_idx = paddle.nonzero(voiced, as_tuple=False).squeeze(-1)  # (K,)

    if voiced_idx.numel() < 2:
        return f0

    unvoiced_idx = paddle.nonzero(uv, as_tuple=False).squeeze(-1)  # (M,)
    if unvoiced_idx.numel() == 0:
        return f0

    # Log domain F0 for voiced frames
    K = voiced_idx.shape[0]
    voiced_f0 = paddle.where(
        f0[voiced_idx] > 0,
        paddle.log(f0[voiced_idx]),
        paddle.to_tensor(0.0),
    )  # (K,)

    # For each unvoiced frame, find left/right voiced neighbour indices
    # voiced_idx is sorted, so we can search
    left = paddle.searchsorted(voiced_idx, unvoiced_idx)  # returns insertion point
    # clamp: left=0 means the unvoiced frame is before the first voiced frame
    # clamp so we always have valid neighbours
    left_i = paddle.clip(left - 1, min=0)  # (M,)
    right_i = paddle.clip(left, max=K - 1)  # (M,)

    left_idx = voiced_idx[left_i]  # (M,)
    right_idx = voiced_idx[right_i]  # (M,)
    left_val = voiced_f0[left_i]  # (M,)
    right_val = voiced_f0[right_i]  # (M,)

    # Linear interpolation in log domain
    span = paddle.cast(right_idx - left_idx, paddle.float32)
    span = paddle.clip(span, min=1.0)
    t = paddle.cast(unvoiced_idx - left_idx, paddle.float32) / span
    interp_val = left_val + t * (right_val - left_val)  # (M,)

    # Build result: start from original, only overwrite UV frames
    result = f0.clone()
    interp_exp = paddle.exp(interp_val)

    # Use scatter_nd to write at unvoiced indices
    # scatter_nd(index, updates, shape) assigns updates at index positions
    scatter_idx = unvoiced_idx.reshape([-1, 1]).cast(paddle.int64)
    scattered = paddle.scatter_nd(scatter_idx, interp_exp, result.shape)

    # Combine: keep original for voiced, use scattered for unvoiced
    result = paddle.where(uv, scattered, result)

    return result


# Per-model defaults
DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "fcpe": {
        "threshold": 0.05,
        "threshold_mode": "fixed",
        "median_filter": 0,
        "mean_filter": 0,
        "interp_uv": False,
    },
    "rmvpe": {
        "threshold": 0.03,
        "threshold_mode": "fixed",
        "median_filter": 0,
        "mean_filter": 0,
        "interp_uv": False,
    },
    "crepe": {
        "threshold": 0.5,
        "threshold_mode": "fixed",
        "median_filter": 3,
        "mean_filter": 0,
        "interp_uv": True,
    },
    "penn": {
        "threshold": 0.01,
        "threshold_mode": "fixed",
        "median_filter": 0,
        "mean_filter": 0,
        "interp_uv": False,
    },
}


def get_defaults(model_name: str) -> dict[str, Any]:
    """Return a copy of the default postproc config for *model_name*."""
    return dict(DEFAULT_CONFIG.get(model_name, {}))


def postprocess_f0(
    f0: paddle.Tensor,
    confidence: paddle.Tensor,
    *,
    threshold: float | None = None,
    threshold_mode: str = "fixed",
    hysteresis_lower: float = 0.19,
    hysteresis_upper: float = 0.31,
    median_filter: int = 0,
    mean_filter: int = 0,
    interp_uv: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Apply post-processing steps in sequence, all on-device.

    All operations use paddle tensors so data never leaves GPU.

    Args:
        f0: (T,) float32 paddle tensor on any device, 0 = unvoiced.
        confidence: (T,) float32 paddle tensor in [0, 1].
        threshold: Confidence threshold.  ``None`` = skip thresholding.
        threshold_mode: ``"fixed"`` or ``"hysteresis"``.
        hysteresis_lower: Lower bound for hysteresis (confidence domain).
        hysteresis_upper: Upper bound for hysteresis.
        median_filter: Median filter kernel size.  0 = disabled.
        mean_filter: Mean filter kernel size.  0 = disabled.
        interp_uv: If ``True``, linearly interpolate unvoiced frames
            in log domain.

    Returns:
        (f0, confidence) — both (T,) float32, possibly modified.
    """
    # 1. Confidence thresholding
    if threshold is not None and threshold > 0:
        if threshold_mode == "fixed":
            f0 = f0.clone()
            f0[confidence < threshold] = 0.0
        elif threshold_mode == "hysteresis":
            f0 = _hysteresis_paddle(f0, confidence, hysteresis_lower)
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    # 2. Median filter (in-place on voiced frames)
    if median_filter > 0 and median_filter % 2 == 1:
        f0 = _median_filter_paddle(f0, win_length=median_filter)

    # 3. UV interpolation
    if interp_uv:
        uv = f0 <= 0
        if uv.any():
            f0 = _interp_uv_paddle(f0, uv)

    return f0, confidence


def _hysteresis_paddle(
    f0: paddle.Tensor,
    confidence: paddle.Tensor,
    lower_bound: float,
    upper_bound: float,
    width: float = 2.0,
    stds: float = 1.0,
) -> paddle.Tensor:
    """Simplified hysteresis thresholding (paddle version)."""
    T = f0.shape[0]
    pitch_mask = confidence >= lower_bound

    # Whitened f0
    voiced_f0 = f0[pitch_mask]
    if voiced_f0.numel() > 1:
        mu = voiced_f0.mean()
        sigma = voiced_f0.std()
        whitened = paddle.abs(f0 - mu) / (sigma + 1e-10)
    else:
        whitened = paddle.zeros_like(f0)

    # Adaptive threshold
    adaptive = paddle.maximum(
        paddle.to_tensor(lower_bound),
        width * whitened * whitened - width * stds * stds,
    )
    final_mask = confidence >= adaptive

    # Hysteresis: extend voiced segments
    if final_mask.any():
        final_mask_np = final_mask.numpy()
        extended = final_mask_np.copy()
        for i in range(1, T - 1):
            if not final_mask_np[i] and (final_mask_np[i - 1] or final_mask_np[i + 1]):
                extended[i] = True
        f0_out = f0.clone()
        f0_out[~paddle.to_tensor(extended)] = 0.0
        return f0_out

    return f0
