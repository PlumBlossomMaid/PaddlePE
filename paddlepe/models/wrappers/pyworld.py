"""WorldPE: pitch extraction via the WORLD vocoder (pyworld wrapper).

Wraps pyworld to extract F0 contours using Harvest or DIO algorithms.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import paddle

from paddlepe.models.base import BasePE
from paddlepe.registry import registry


@registry.register("world")
class WorldPE(BasePE):
    """Pitch extraction via pyworld (WORLD vocoder).

    This is NOT a neural network model. It wraps pyworld library calls
    to extract F0 using Harvest or DIO algorithms. No training or ONNX
    export is supported.
    """

    trainable: ClassVar[bool] = False
    support_onnx: ClassVar[bool] = False

    def __init__(self, hop_length: int = 160, sample_rate: int = 16000):
        """Initialize WorldPE.

        Args:
            hop_length: hop length in samples, used to derive
                frame period (default: 160)
            sample_rate: target sample rate in Hz (default: 16000)
        """
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self._check_import()

    @staticmethod
    def _check_import():
        """Verify pyworld is installed, raise ImportError otherwise."""
        try:
            import pyworld  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pyworld is not installed. Install it with: pip install pyworld"
            ) from e

    def forward(self, x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        """Not supported for wrapper models."""
        raise NotImplementedError(
            "WorldPE does not support forward() for training. "
            "It is a wrapper around the pyworld library for inference only."
        )

    def infer(
        self,
        wav: paddle.Tensor,
        sr: int,
        method: str = "harvest",
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Extract F0 using WORLD vocoder algorithms.

        Args:
            wav: audio waveform, (S,) or (1, S) float32
            sr: sample rate in Hz
            method: "harvest" (default, more accurate) or "dio" (faster)
            **kwargs: passed through to the underlying pyworld function

        Returns:
            (f0_hz, confidence)
            f0_hz: (T,) float32, 0 = unvoiced
            confidence: (T,) float32, inverted unvoiced flag
                (1.0 = voiced, 0.0 = unvoiced)
        """
        import pyworld

        # Convert paddle tensor to numpy float64 (pyworld requires float64)
        if isinstance(wav, paddle.Tensor):
            wav_np = wav.numpy().astype(np.float64)
        else:
            wav_np = np.asarray(wav, dtype=np.float64)

        # Ensure 1D
        wav_np = wav_np.squeeze()
        if wav_np.ndim != 1:
            raise ValueError(f"Expected 1D waveform, got shape {wav_np.shape}")

        # Compute frame period in milliseconds
        frame_period = self.hop_length / sr * 1000.0

        # Select and run the pitch extraction algorithm
        method = method.lower()
        if method == "harvest":
            f0_np, t_np = pyworld.harvest(
                wav_np, sr, frame_period=frame_period, **kwargs
            )
        elif method == "dio":
            f0_np, t_np = pyworld.dio(wav_np, sr, frame_period=frame_period, **kwargs)
            # Apply Stonemask refinement for DIO (standard practice)
            f0_np = pyworld.stonemask(wav_np, f0_np, t_np, sr)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'harvest', 'dio'")

        # Build confidence from unvoiced flag
        # pyworld marks unvoiced frames as 0.0
        uv = f0_np == 0.0
        confidence_np = (~uv).astype(np.float32)

        # Convert to paddle tensors
        f0 = paddle.to_tensor(f0_np, dtype=paddle.float32)
        confidence = paddle.to_tensor(confidence_np, dtype=paddle.float32)

        return f0, confidence
