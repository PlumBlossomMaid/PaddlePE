"""CREPE pitch extraction inference.

Inference pipeline:
  1. preprocess -- resample to 16 kHz, frame with 1024-sample window
     and 160-sample (10 ms) hop
  2. forward   -- CrepeBackbone produces (B, 360) pitch probabilities
  3. decode    -- argmax / weighted_argmax with optional threshold
  4. return    -- (f0_hz, confidence)
"""

from __future__ import annotations

from pathlib import Path

import paddle
from paddle import nn

from paddlepe.models.base import BasePE
from paddlepe.models.crepe.backbone import CrepeBackbone
from paddlepe.registry import registry

# ---------------------------------------------------------------------------
# Constants (mirrors torchcrepe)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
WINDOW_SIZE = 1024
HOP_LENGTH = 160  # 10 ms at 16 kHz
PITCH_BINS = 360
CENTS_PER_BIN = 20
MAX_FMAX = 2006.0


# ---------------------------------------------------------------------------
# Conversion helpers  (ported from torchcrepe/convert.py)
# ---------------------------------------------------------------------------
def _bins_to_cents(bins: paddle.Tensor) -> paddle.Tensor:
    """Convert pitch bins (0--359) to musical cents."""
    return CENTS_PER_BIN * bins + 1997.3794084376191


def _cents_to_frequency(cents: paddle.Tensor) -> paddle.Tensor:
    """Convert cents to frequency in Hz."""
    return 10.0 * 2.0 ** (cents / 1200.0)


def _bins_to_frequency(bins: paddle.Tensor) -> paddle.Tensor:
    """Convert pitch bins to frequency in Hz."""
    return _cents_to_frequency(_bins_to_cents(bins))


def _frequency_to_bins(
    frequency: paddle.Tensor, quantize_fn=paddle.floor
) -> paddle.Tensor:
    """Convert frequency in Hz to pitch bins."""
    cents = 1200.0 * paddle.log2(frequency / 10.0)
    bins = (cents - 1997.3794084376191) / CENTS_PER_BIN
    return quantize_fn(bins).astype(paddle.int64)


# ---------------------------------------------------------------------------
# Decoders  (ported from torchcrepe/decode.py)
# ---------------------------------------------------------------------------
def _decode_argmax(
    logits: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Decode by taking the argmax bin. Returns (bin_indices, f0_hz)."""
    bins = logits.argmax(axis=1)  # (B,)
    return bins, _bins_to_frequency(bins)


def _decode_weighted_argmax(
    logits: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Decode by weighted sum near argmax (sub-bin resolution).

    Mirrors torchcrepe.decode.weighted_argmax:
      1. Find argmax bin
      2. Mask out bins outside [argmax-4, argmax+5]
      3. Weighted sum over remaining bins using cents as weights
    """
    bins = logits.argmax(axis=1)  # (T,)

    # Build weight vector lazily
    if not hasattr(_decode_weighted_argmax, "_weights"):
        w = _bins_to_cents(paddle.arange(PITCH_BINS, dtype=paddle.float32))
        _decode_weighted_argmax._weights = w  # (360,)

    weights = _decode_weighted_argmax._weights  # (360,)

    # Build window mask: zero out bins outside [argmax-4, argmax+5]
    arange = paddle.arange(PITCH_BINS, dtype=paddle.int64)  # (360,)
    bins_exp = bins.unsqueeze(-1)  # (T, 1)
    mask = (arange >= (bins_exp - 4)) & (arange <= (bins_exp + 5))  # (T, 360)

    # Convert logits to probabilities and apply mask
    probs = nn.functional.sigmoid(logits)  # (T, 360)
    probs = probs * mask.astype(probs.dtype)

    # Weighted sum over the 360 bins
    weighted = (weights * probs).sum(axis=1)  # (T,)
    normalizer = probs.sum(axis=1) + 1e-12  # (T,)
    cents = weighted / normalizer

    return bins, _cents_to_frequency(cents)


# ---------------------------------------------------------------------------
# Preprocess  (ported from torchcrepe/core.py::preprocess)
# ---------------------------------------------------------------------------
def _preprocess(
    audio: paddle.Tensor,
    sample_rate: int,
    hop_length: int | None = None,
) -> paddle.Tensor:
    """Convert audio waveform to framed input for the model.

    Args:
        audio: (S,) mono waveform
        sample_rate: source sample rate
        hop_length: frame step in samples (defaults to sr/100 = 10 ms)

    Returns:
        frames: (T, 1024) mean-centered & variance-scaled audio frames
    """
    audio = audio.squeeze()
    hop_len = hop_length if hop_length is not None else sample_rate // 100

    # Resample to 16 kHz
    if sample_rate != SAMPLE_RATE:
        scale = SAMPLE_RATE / sample_rate
        new_len = int(audio.shape[0] * scale)
        audio = nn.functional.interpolate(
            audio.unsqueeze(0).unsqueeze(0),
            size=[new_len],
            mode="linear",
        ).squeeze()
        hop_len = int(hop_len * SAMPLE_RATE / sample_rate)

    # Pad for framing
    audio = nn.functional.pad(
        audio.unsqueeze(0),
        (WINDOW_SIZE // 2, WINDOW_SIZE // 2),
        mode="constant",
        value=0,
    ).squeeze(0)

    # Unfold into frames
    frames = nn.functional.unfold(
        audio.unsqueeze(0).unsqueeze(0).unsqueeze(0),  # (1, 1, 1, S)
        kernel_sizes=[1, WINDOW_SIZE],
        strides=[1, hop_len],
    )  # (1, 1024, T)
    frames = frames.squeeze(0).t()  # (T, 1024)

    # Mean-center and scale
    mean = frames.mean(axis=1, keepdim=True)
    std = frames.std(axis=1, keepdim=True)
    std = paddle.clip(std, min=1e-10)
    frames = (frames - mean) / std

    return frames  # (T, 1024)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
@registry.register("crepe")
class CrepePE(BasePE):
    """CREPE pitch extraction model.

    Ported from torchcrepe with a Conv1d backbone.
    """

    trainable = True
    support_onnx = True

    def __init__(self, capacity: str = "full"):
        super().__init__()
        self.capacity = capacity
        self.backbone = CrepeBackbone(capacity=capacity)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Training forward pass.

        Args:
            x: (B, 1024) or (B, 1, 1024) audio frames

        Returns:
            (B, 360) pitch probabilities
        """
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # (B, 1024)
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(
        self,
        wav: paddle.Tensor,
        sr: int,
        decoder: str = "weighted_argmax",
        threshold: float | None = None,
        hop_length: int | None = None,
        interp_uv: bool | None = None,
        median_filter: int | None = None,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Infer F0 from audio waveform.

        Pipeline: preprocess -> forward -> decode -> postprocess

        Args:
            wav: (S,) or (1, S) float32 audio
            sr: sample rate in Hz
            decoder: ``"argmax"`` or ``"weighted_argmax"``
            threshold: confidence threshold.  ``None`` = model default (0.5).
            hop_length: frame step in samples (default 10 ms)
            interp_uv: interpolate unvoiced frames.  ``None`` = model default.
            median_filter: median filter kernel size.  ``None`` = model default.
            **kwargs: overrides for :func:`postprocess_f0`.

        Returns:
            f0_hz: (T,) float32, 0 = unvoiced
            confidence: (T,) float32
        """
        from paddlepe.postproc.pipeline import get_defaults, postprocess_f0

        wav = self._to_tensor(wav)
        wav = wav.squeeze()
        if wav.ndim == 0:
            wav = wav.unsqueeze(0)

        # Preprocess: waveform -> frames (T, 1024)
        frames = _preprocess(wav, sr, hop_length)  # (T, 1024)

        if frames.shape[0] == 0:
            return paddle.to_tensor([], dtype=paddle.float32), None

        # Forward through backbone
        # backbone expects (B, 1024)
        logits = self.backbone(frames)  # (T, 360)

        # Decode
        if decoder == "argmax":
            _, f0_hz = _decode_argmax(logits)
        elif decoder == "weighted_argmax":
            _, f0_hz = _decode_weighted_argmax(logits)
        else:
            raise ValueError(
                f"Unknown decoder '{decoder}'. Choose from: argmax, weighted_argmax"
            )

        # Confidence = max probability
        confidence = logits.max(axis=1)  # (T,)

        # Apply threshold (crepe internal: sets low-conf frames to 0)
        _th = threshold if threshold is not None else 0.5
        if _th > 0.0:
            mask = confidence < _th
            f0_hz = paddle.where(mask, paddle.zeros_like(f0_hz), f0_hz)

        # Apply post-processing pipeline
        cfg = get_defaults("crepe")
        cfg["interp_uv"] = interp_uv if interp_uv is not None else cfg["interp_uv"]
        cfg["median_filter"] = (
            median_filter if median_filter is not None else cfg["median_filter"]
        )
        cfg["threshold"] = None  # we already applied threshold above
        cfg.update(kwargs)

        f0_hz, confidence = postprocess_f0(f0_hz, confidence, **cfg)

        return f0_hz, confidence

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    @classmethod
    def default_ckpt(cls) -> str:
        """Path to default checkpoint."""
        return str(
            Path(__file__).parent.parent.parent.parent / "ckpts" / "crepe.pdparams"
        )
