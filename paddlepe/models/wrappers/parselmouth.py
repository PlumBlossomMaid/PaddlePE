"""ParselmouthPE: pitch extraction via Praat (Parselmouth wrapper).

Wraps parselmouth (Python bindings for Praat) to extract F0 contours.
Uses Praat's built-in autocorrelation pitch tracking.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import paddle

from paddlepe.models.base import BasePE
from paddlepe.registry import registry


@registry.register("parselmouth")
class ParselmouthPE(BasePE):
    """Pitch extraction via Parselmouth (Praat wrapper).

    This is NOT a neural network model. It wraps parselmouth library calls
    and returns pitch via Praat's autocorrelation algorithm. No training
    or ONNX export is supported.
    """

    trainable: ClassVar[bool] = False
    support_onnx: ClassVar[bool] = False

    def __init__(
        self,
        time_step: float = 0.01,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0,
    ):
        """Initialize ParselmouthPE.

        Args:
            time_step: time step between frames in seconds (default: 0.01)
            pitch_floor: minimum pitch in Hz (default: 75.0)
            pitch_ceiling: maximum pitch in Hz (default: 600.0)
        """
        super().__init__()
        self.time_step = time_step
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling

        self._check_import()

    @staticmethod
    def _check_import():
        """Verify parselmouth is installed, raise ImportError otherwise."""
        try:
            import parselmouth  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "parselmouth is not installed. "
                "Install it with: pip install praat-parselmouth"
            ) from e

    def forward(self, x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        """Not supported for wrapper models."""
        raise NotImplementedError(
            "ParselmouthPE does not support forward() for training. "
            "It is a wrapper around the Praat library for inference only."
        )

    def infer(
        self,
        wav: paddle.Tensor,
        sr: int,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Extract F0 using Praat's autocorrelation pitch tracking.

        Args:
            wav: audio waveform, (S,) or (1, S) float32
            sr: sample rate
            **kwargs: override init-time params (time_step, pitch_floor, pitch_ceiling)

        Returns:
            (f0_hz, None)
            f0_hz: (T,) float32, 0 = unvoiced
            confidence: None (Parselmouth does not provide confidence)
        """
        import parselmouth

        # Convert paddle tensor to numpy float64 (Parselmouth requires float64)
        if isinstance(wav, paddle.Tensor):
            wav_np = wav.numpy().astype(np.float64)
        else:
            wav_np = np.asarray(wav, dtype=np.float64)

        # Ensure 1D
        wav_np = wav_np.squeeze()
        if wav_np.ndim != 1:
            raise ValueError(f"Expected 1D waveform, got shape {wav_np.shape}")

        # Override params from kwargs if provided
        time_step = kwargs.get("time_step", self.time_step)
        pitch_floor = kwargs.get("pitch_floor", self.pitch_floor)
        pitch_ceiling = kwargs.get("pitch_ceiling", self.pitch_ceiling)

        # Create Parselmouth Sound and extract pitch
        sound = parselmouth.Sound(wav_np, sampling_frequency=sr)
        pitch = sound.to_pitch(
            time_step=time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )

        # Extract F0 values; unvoiced frames have value 0
        f0_np = pitch.selected_array["frequency"]

        # Convert to paddle tensor
        f0 = paddle.to_tensor(f0_np, dtype=paddle.float32)

        # No confidence available from Parselmouth
        return f0, None
