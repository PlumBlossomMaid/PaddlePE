"""CREPE pitch estimation backbone.

PaddlePaddle port of the original torchcrepe model architecture.

Architecture (matches the original Conv2d-based CREPE):
  6x (Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D)
  followed by Flatten -> Linear(2048, 360) -> Sigmoid

Layer naming convention (used for weight compatibility):
  convs.0.weight / convs.0.bias       -- Conv2D layers
  bns.0.weight / bns.0.bias           -- BatchNorm gamma / beta
  bns.0._mean / bns.0._variance       -- BatchNorm running stats
  fc2.weight / fc2.bias                -- Output Linear (called "classifier" in torch)

NOTE: Unlike the torch original which uses named attributes (conv1, conv1_BN, ...),
      this port uses ModuleList for clean iteration while preserving the same
      computational graph.
"""

from __future__ import annotations

from typing import ClassVar

import paddle
from paddle import nn


class CrepeBackbone(nn.Layer):
    """CREPE pitch estimation backbone.

    Takes (B, 1024) audio frames at 16 kHz and outputs (B, 360) pitch
    probabilities over 360 bins (32.7 Hz -- 2006 Hz).

    The input is expected to be mean-centered and variance-scaled
    single-channel audio frames.

    Args:
        capacity: 'full' or 'tiny'

    Architecture (full):
        conv1:  Conv2D(1, 1024, (512,1), stride=(4,1))  + pad (0,0,254,254)
        pool1:  MaxPool2D((2,1), stride=(2,1))
        conv2:  Conv2D(1024, 128, (64,1), stride=(1,1)) + pad (0,0,31,32)
        pool2:  MaxPool2D((2,1), stride=(2,1))
        conv3:  Conv2D(128, 128, (64,1), stride=(1,1))  + pad (0,0,31,32)
        pool3:  MaxPool2D((2,1), stride=(2,1))
        conv4:  Conv2D(128, 128, (64,1), stride=(1,1))  + pad (0,0,31,32)
        pool4:  MaxPool2D((2,1), stride=(2,1))
        conv5:  Conv2D(128, 256, (64,1), stride=(1,1))  + pad (0,0,31,32)
        pool5:  MaxPool2D((2,1), stride=(2,1))
        conv6:  Conv2D(256, 512, (64,1), stride=(1,1))  + pad (0,0,31,32)
        pool6:  MaxPool2D((2,1), stride=(2,1))
        fc2:    Linear(2048, 360) + Sigmoid
    """

    CHANNEL_CONFIGS: ClassVar[dict[str, list[list[int]]]] = {
        "full": {
            "in_channels": [1, 1024, 128, 128, 128, 256],
            "out_channels": [1024, 128, 128, 128, 256, 512],
            "kernel_sizes": [
                (512, 1),
                (64, 1),
                (64, 1),
                (64, 1),
                (64, 1),
                (64, 1),
            ],
            "strides": [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            "paddings": [
                (0, 0, 254, 254),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
            ],
            "in_features": 2048,
        },
        "tiny": {
            "in_channels": [1, 128, 16, 16, 16, 32],
            "out_channels": [128, 16, 16, 16, 32, 64],
            "kernel_sizes": [
                (512, 1),
                (64, 1),
                (64, 1),
                (64, 1),
                (64, 1),
                (64, 1),
            ],
            "strides": [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            "paddings": [
                (0, 0, 254, 254),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
                (0, 0, 31, 32),
            ],
            "in_features": 256,
        },
    }

    def __init__(self, capacity: str = "full"):
        super().__init__()

        if capacity not in self.CHANNEL_CONFIGS:
            raise ValueError(
                f"Unknown capacity '{capacity}'. "
                f"Choose from {list(self.CHANNEL_CONFIGS.keys())}."
            )

        cfg = self.CHANNEL_CONFIGS[capacity]
        self.capacity = capacity
        self.in_features = cfg["in_features"]

        convs = []
        bns = []
        for i in range(6):
            convs.append(
                nn.Conv2D(
                    in_channels=cfg["in_channels"][i],
                    out_channels=cfg["out_channels"][i],
                    kernel_size=cfg["kernel_sizes"][i],
                    stride=cfg["strides"][i],
                    padding=0,  # We apply padding manually (asymmetric)
                )
            )
            bns.append(
                nn.BatchNorm2D(
                    num_features=cfg["out_channels"][i],
                    epsilon=0.0010000000474974513,
                    momentum=0.0,
                )
            )

        self.convs = nn.LayerList(convs)
        self.bns = nn.LayerList(bns)

        # Padding configurations for each layer (asymmetric padding)
        # PaddlePaddle's pad format for 4D: (left, right, top, bottom)
        self._paddings = cfg["paddings"]

        # Output classifier: Linear(2048, 360) in original
        self.fc2 = nn.Linear(cfg["in_features"], 360)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            x: (B, 1024) audio frames (mean-centered & variance-scaled)

        Returns:
            (B, 360) pitch probabilities (sigmoid output)
        """
        # Add channel and spatial dims: (B, 1024) -> (B, 1, 1024, 1)
        x = x.unsqueeze(1).unsqueeze(-1)  # (B, 1, 1024, 1)

        # 6 Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D layers
        for i in range(6):
            # Apply asymmetric padding manually
            pad = self._paddings[i]  # (left, right, top, bottom)
            x = nn.functional.pad(x, pad=pad, mode="constant", value=0)

            x = self.convs[i](x)
            x = nn.functional.relu(x)
            x = self.bns[i](x)
            x = nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))

        # Flatten: (B, C, H, 1) -> (B, H*C)
        # Must match torch's permute(0, 2, 1, 3).reshape(-1, in_features)
        # so elements are ordered (batch, height, channel), not (batch, channel, height)
        x = x.transpose([0, 2, 1, 3])  # (B, H, C, 1)
        x = x.reshape((x.shape[0], -1))  # (B, H*C)

        # Classifier
        x = self.fc2(x)  # (B, 360)
        return nn.functional.sigmoid(x)

    def embed(self, x: paddle.Tensor) -> paddle.Tensor:
        """Extract pitch embedding (output after 5th pooling layer).

        This matches torchcrepe's ``embed`` functionality used for
        representation learning.

        Args:
            x: (B, 1024) audio frames

        Returns:
            (B, 256, 8, 1) embedding tensor (full) / (B, 32, 8, 1) (tiny)
        """
        x = x.unsqueeze(1).unsqueeze(-1)  # (B, 1, 1024, 1)

        for i in range(5):  # Only first 5 layers
            pad = self._paddings[i]
            x = nn.functional.pad(x, pad=pad, mode="constant", value=0)
            x = self.convs[i](x)
            x = nn.functional.relu(x)
            x = self.bns[i](x)
            x = nn.functional.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))

        return x  # (B, C, H, 1)
