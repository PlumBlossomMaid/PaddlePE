"""BasePE: abstract base class for all pitch extraction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import paddle
from paddle import nn


class BasePE(nn.Layer, ABC):
    """Abstract base class for all pitch extraction models.

    Subclasses must implement:
      - forward(x) for training
      - infer(wav, sr, **kwargs) for inference

    Class variables:
      - trainable: whether the model supports fine-tuning
      - support_onnx: whether the model can be exported to ONNX
    """

    trainable: ClassVar[bool] = False
    support_onnx: ClassVar[bool] = False

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        """Model forward pass for training.

        Args:
            x: input tensor, shape varies by model
            args, kwargs: additional model-specific inputs (e.g. labels)

        Returns:
            model output (logits, embeddings, or loss)
        """
        ...

    @abstractmethod
    def infer(
        self,
        wav: paddle.Tensor,
        sr: int,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Full inference pipeline: preprocess → forward → decode.

        Args:
            wav: audio waveform, (S,) or (1, S) float32
            sr: sample rate in Hz
            kwargs: model-specific inference parameters

        Returns:
            (f0_hz, confidence)
            f0_hz: (T,) float32, 0 = unvoiced
            confidence: (T,) float32 or None, [0, 1]
        """
        ...

    def get_pitch(
        self,
        wav: paddle.Tensor,
        sr: int,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Alias for infer()."""
        return self.infer(wav, sr, **kwargs)

    @property
    def device(self) -> paddle.CPUPlace | paddle.CUDAPlace:
        """Get the device this layer is on."""
        try:
            return next(iter(self.parameters())).place
        except StopIteration:
            return paddle.CPUPlace()
