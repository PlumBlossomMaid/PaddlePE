"""PENN (FCNF0++) pitch estimation backbone.

PaddlePaddle port of the original PyTorch FCNF0++ model.

Architecture (matches the original):
  6x Block(Conv1D -> ReLU -> MaxPool1D -> LayerNorm)
  followed by Conv1D(512, 1440, 4)

Each Block is:
  Conv1d(in_channels, out_channels, kernel_size=32)
  -> LeakyReLU (original uses ReLU)
  -> MaxPool1d (optional, with kernel_size=stride=2)
  -> LayerNorm((out_channels, length))

Checkpoint key naming (torch Sequential indices -> Paddle named layers):
  torch 0.0.weight -> blocks.0.conv.weight  (Block 0, Conv1d)
  torch 0.3.weight -> blocks.0.norm.weight  (Block 0, LayerNorm)
  torch 6.weight   -> final_conv.weight     (Output Conv1d)
"""

from __future__ import annotations

import paddle
from paddle import nn

PITCH_BINS = 1440


class PennBlock(nn.Layer):
    """A single block of the PENN backbone.

    Composed of Conv1d -> ReLU -> (optional MaxPool1d) -> LayerNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        pooling: tuple[int, int] | None = None,
        kernel_size: int = 32,
    ):
        super().__init__()
        self.conv = nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
        )
        self.pool: nn.MaxPool1D | None = None
        if pooling is not None:
            self.pool = nn.MaxPool1D(pooling[0], stride=pooling[1])
        self.norm = nn.LayerNorm((out_channels, length))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = nn.functional.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        x = self.norm(x)
        return x


class PennBackbone(nn.Layer):
    """PENN (FCNF0++) pitch estimation backbone.

    Takes (B, 1, 1024) audio frames at 8 kHz and outputs
    (B, 1440, 1) logits over 1440 pitch bins (31 Hz -- 1984 Hz).

    The input is expected to have 16 samples trimmed from left
    and 15 from right of the time dimension (done in forward).
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.LayerList(
            [
                PennBlock(1, 256, 481, pooling=(2, 2), kernel_size=32),
                PennBlock(256, 32, 225, pooling=(2, 2), kernel_size=32),
                PennBlock(32, 32, 97, pooling=(2, 2), kernel_size=32),
                PennBlock(32, 128, 66, pooling=None, kernel_size=32),
                PennBlock(128, 256, 35, pooling=None, kernel_size=32),
                PennBlock(256, 512, 4, pooling=None, kernel_size=32),
            ]
        )
        self.final_conv = nn.Conv1D(512, PITCH_BINS, 4, padding=0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            x: (B, 1, 1024) audio frames

        Returns:
            (B, 1440, 1) pitch logits
        """
        # Trim 16 samples from left, 15 from right (as in original FCNF0++)
        x = x[:, :, 16:-15]  # (B, 1, 993)

        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)  # (B, 1440, 1)
        return x
