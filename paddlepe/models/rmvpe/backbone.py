"""RMVPE backbone: exact port of E2E0 from RMVPE_paddle.

Matches reference architecture:
  - ConvBlockRes with shortcut connections + ReLU (not LeakyReLU)
  - ResEncoderBlock with n_blocks=4 conv blocks per layer
  - Encoder/Decoder/Intermediate structure from reference
  - DeepUnet0 (no TimbreFilter)
"""

from __future__ import annotations

import paddle
import paddle.nn.functional as F
from paddle import nn

N_MELS = 128
N_CLASS = 360


class ConvBlockRes(nn.Layer):
    """Residual convolutional block matching reference."""

    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 3, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, 3, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2D(in_channels, out_channels, 1)
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Layer):
    """Encoder block with n_blocks residual conv blocks + pooling."""

    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.LayerList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2D(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Layer):
    """Decoder block: transposed conv + concat skip + residual conv blocks."""

    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.Conv2DTranspose(
                in_channels, out_channels, (3, 3),
                stride=stride, padding=(1, 1), output_padding=out_padding,
                bias_attr=False,
            ),
            nn.BatchNorm2D(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.LayerList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = paddle.concat([x, concat_tensor], axis=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Layer):
    """Encoder: BN → N x ResEncoderBlock."""

    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks,
                 out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2D(in_channels, momentum=momentum)
        self.layers = nn.LayerList()
        self.latent_channels = []
        for i in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate(nn.Layer):
    """Intermediate: N x ResEncoderBlock (no pooling)."""

    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.LayerList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Layer):
    """Decoder: N x ResDecoderBlock."""

    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.LayerList()
        self.n_decoders = n_decoders
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-(i + 1)])
        return x


class DeepUnet(nn.Layer):
    """DeepUNet with TimbreFilter (for E2E)."""

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                 in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size,
                               n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel,
            inter_layers, n_blocks,
        )
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        concat_tensors = self.tf(concat_tensors)
        x = self.decoder(x, concat_tensors)
        return x


class DeepUnet0(nn.Layer):
    """DeepUNet0 without TimbreFilter (for E2E0 / RMVPEUNet)."""

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                 in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size,
                               n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel,
            inter_layers, n_blocks,
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class TimbreFilter(nn.Layer):
    """TimbreFilter for DeepUnet (E2E)."""

    def __init__(self, latent_rep_channels):
        super().__init__()
        self.layers = nn.LayerList()
        for ch, _ in latent_rep_channels:
            self.layers.append(ConvBlockRes(ch, ch))

    def forward(self, x_tensors):
        return [layer(t) for layer, t in zip(self.layers, x_tensors)]


class BiGRU(nn.Layer):
    """Bidirectional GRU wrapper."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, direction="bidirectional")

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class RMVPEBase(nn.Layer):
    """RMVPE base model (E2E with DeepUnet + TimbreFilter)."""

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2),
                 en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers,
                             in_channels, en_out_channels)
        self.cnn = nn.Conv2D(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel):
        mel = mel.transpose([0, 2, 1]).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose([0, 2, 1]).flatten(-2)
        x = self.fc(x)
        return x


class RMVPEUNet(nn.Layer):
    """RMVPE UNet model (E2E0 with DeepUnet0, no TimbreFilter).

    This is the main backbone used by RMVPEPE.
    Matches reference E2E0 architecture exactly.
    """

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2),
                 en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers,
                              in_channels, en_out_channels)
        self.cnn = nn.Conv2D(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel):
        """Forward pass.

        Args:
            mel: (B, N_MELS, T)

        Returns:
            logits: (B, T, N_CLASS)
        """
        mel = mel.transpose([0, 2, 1]).unsqueeze(1)  # (B, 1, T, N_MELS)
        x = self.cnn(self.unet(mel))  # (B, 3, T, N_MELS)
        x = x.transpose([0, 2, 1, 3]).flatten(-2)  # (B, T, 3 * N_MELS)
        x = self.fc(x)  # (B, T, N_CLASS)
        return x
