"""PENN (FCNF0++) pitch extraction inference.

Inference pipeline:
  1. preprocess -- resample to 8 kHz, frame with 1024-sample window
     and 80-sample (10 ms) hop
  2. forward   -- PennBackbone produces (B, 1440, 1) logits
  3. decode    -- argmax with frequency conversion + confidence
  4. return    -- (f0_hz, confidence)
"""

from __future__ import annotations

from pathlib import Path

import paddle
from paddle import nn

from paddlepe.models.base import BasePE
from paddlepe.models.penn.backbone import PennBackbone
from paddlepe.registry import registry

# ---------------------------------------------------------------------------
# Constants (mirrors penn.config.defaults)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 8000
WINDOW_SIZE = 1024
HOP_LENGTH = 80  # 10 ms at 8 kHz
PITCH_BINS = 1440
CENTS_PER_BIN = 5.0
FMIN = 31.0
OCTAVE = 1200  # cents per octave

# FMAX = FMIN * 2^(PITCH_BINS * CENTS_PER_BIN / OCTAVE)
#      = 31.0 * 2^6 = 1984.0 Hz
FMAX = FMIN * (2.0 ** (PITCH_BINS * CENTS_PER_BIN / OCTAVE))


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def _bins_to_cents(bins: paddle.Tensor) -> paddle.Tensor:
    """Convert pitch bins to cents."""
    return CENTS_PER_BIN * bins.astype(paddle.float32)


def _cents_to_frequency(cents: paddle.Tensor) -> paddle.Tensor:
    """Convert cents to frequency in Hz."""
    return FMIN * (2.0 ** (cents / OCTAVE))


def _bins_to_frequency(bins: paddle.Tensor) -> paddle.Tensor:
    """Convert pitch bins directly to frequency in Hz."""
    return _cents_to_frequency(_bins_to_cents(bins))


# ---------------------------------------------------------------------------
# Preprocess: waveform -> framed input
# ---------------------------------------------------------------------------
def _preprocess(
    audio: paddle.Tensor,
    sample_rate: int,
) -> paddle.Tensor:
    """Convert audio waveform to framed input for the PENN model.

    Args:
        audio: (S,) mono waveform
        sample_rate: source sample rate

    Returns:
        frames: (T, 1, 1024) audio frames
    """
    audio = audio.squeeze()

    # Resample to 8 kHz
    if sample_rate != SAMPLE_RATE:
        scale = SAMPLE_RATE / sample_rate
        new_len = max(1, int(audio.shape[0] * scale))
        audio = (
            nn.functional.interpolate(
                audio.unsqueeze(0).unsqueeze(0),  # (1, 1, S)
                size=[new_len],
                mode="linear",
            )
            .squeeze(0)
            .squeeze(0)
        )  # (new_len,) — remove batch+channel dims

    # Pad half-window on both sides
    audio = nn.functional.pad(
        audio.unsqueeze(0),  # (1, S)
        (WINDOW_SIZE // 2, WINDOW_SIZE // 2),
        mode="constant",
        value=0,
    ).squeeze(0)  # (S + WINDOW_SIZE,)

    # Unfold into frames using unfold
    # audio: (S,) -> (1, 1, 1, S_padded) -> unfold -> (1, WINDOW_SIZE, T)
    audio_4d = audio.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, S)
    frames = nn.functional.unfold(
        audio_4d,
        kernel_sizes=[1, WINDOW_SIZE],
        strides=[1, HOP_LENGTH],
    )  # (1, WINDOW_SIZE, T)
    frames = frames.squeeze(0).t()  # (T, WINDOW_SIZE)

    # Add channel dim: (T, 1024) -> (T, 1, 1024)
    frames = frames.unsqueeze(1)

    return frames  # (T, 1, 1024)


# ---------------------------------------------------------------------------
# Decode: logits -> f0_hz, confidence
# ---------------------------------------------------------------------------
def _decode_argmax(
    logits: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Decode pitch by argmax over pitch bins.

    Args:
        logits: (T, 1440) or (T, 1440, 1) or (B, 1440, T)

    Returns:
        f0_hz: (T,) float32, 0 = unvoiced
        confidence: (T,) float32, [0, 1]
    """
    # Squeeze spatial dim if present
    if logits.ndim == 3:
        logits = logits.squeeze(-1)  # (T, 1440) or (B, 1440)
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits.squeeze(0)  # (1440,) -> would be wrong
    # Handle (1, 1440, 1) -> (1440,)
    # General case: (B, 1440) or (T, 1440)

    if logits.ndim == 2:
        # Batched: (B, 1440) -> process each batch item
        # For inference we should have (T, 1440) shape
        pass

    # Convert to probabilities via softmax
    probs = nn.functional.softmax(logits, axis=-1)  # (..., 1440)

    # Argmax bins
    bins = probs.argmax(axis=-1)  # (...)

    # Convert to frequency
    f0_hz = _bins_to_frequency(bins)  # (...)

    # Confidence = max probability
    confidence = probs.max(axis=-1)  # (...)

    return f0_hz, confidence


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
@registry.register("penn")
class PennPE(BasePE):
    """PENN (FCNF0++) pitch extraction model.

    Ported from the original FCNF0++ PyTorch implementation.
    Processes audio at 8 kHz, outputs f0 at 10 ms frame rate.
    """

    trainable = True
    support_onnx = True

    def __init__(self):
        super().__init__()
        self.backbone = PennBackbone()

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Training forward pass.

        Args:
            x: (B, 1, 1024) audio frames

        Returns:
            (B, 1440, 1) pitch logits
        """
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(
        self,
        wav: paddle.Tensor,
        sr: int,
        decoder: str = "argmax",
        threshold: float | None = None,
        interp_uv: bool | None = None,
        median_filter: int | None = None,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Infer F0 from audio waveform.

        Pipeline: preprocess -> forward -> decode -> postprocess

        Args:
            wav: (S,) or (1, S) float32 audio
            sr: sample rate in Hz
            decoder: only ``"argmax"`` is supported
            threshold: confidence threshold.  ``None`` = model default.
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

        # Preprocess: waveform -> frames (T, 1, 1024)
        frames = _preprocess(wav, sr)  # (T, 1, 1024)

        if frames.shape[0] == 0:
            return paddle.to_tensor([], dtype=paddle.float32), None

        # Forward through backbone
        logits = self.backbone(frames)  # (T, 1440, 1)

        # Decode
        f0_hz, confidence = _decode_argmax(logits)

        # Apply threshold
        _th = threshold if threshold is not None else 0.01
        if _th > 0.0:
            mask = confidence < _th
            f0_hz = paddle.where(mask, paddle.zeros_like(f0_hz), f0_hz)

        # Apply post-processing pipeline
        cfg = get_defaults("penn")
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
    # Checkpoint helpers
    # ------------------------------------------------------------------
    @classmethod
    def default_ckpt(cls) -> str:
        """Path to default checkpoint."""
        return str(
            Path(__file__).parent.parent.parent.parent / "ckpts" / "penn.pdparams"
        )
