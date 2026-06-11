"""RMVPE backbone: RMVPEUNet - UNet + BiGRU pitch estimator.

Port of E2E0 with renamed classes.
"""

from __future__ import annotations

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ConvBlock(nn.Layer):
    """Convolutional block for DeepUNet."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2D(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2D(out_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm2D(out_ch)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = self.norm(self.conv2(x))
        x = F.leaky_relu(x, 0.01)
        return x


class EncoderBlock(nn.Layer):
    """Encoder block: ConvBlock + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2D(2)

    def forward(self, x: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
        skip = self.conv(x)
        out = self.pool(skip)
        return out, skip


class DecoderBlock(nn.Layer):
    """Decoder block: upsampling + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Conv2DTranspose(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x: paddle.Tensor, skip: paddle.Tensor) -> paddle.Tensor:
        x = self.up(x)
        # Handle size mismatch
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = paddle.concat([x, skip], axis=1)
        return self.conv(x)


class DeepUNet(nn.Layer):
    """DeepUNet for pitch estimation."""

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.n_blocks = n_blocks
        self.en_de_layers = en_de_layers

        # Encoder
        self.encoder = nn.LayerList()
        ch = in_channels
        for i in range(en_de_layers):
            out_ch = en_out_channels * (2 ** min(i, 4))
            self.encoder.append(EncoderBlock(ch, out_ch))
            ch = out_ch

        # Intermediate
        self.inter = nn.LayerList([
            ConvBlock(ch, ch) for _ in range(inter_layers)
        ])

        # Decoder
        self.decoder = nn.LayerList()
        for i in range(en_de_layers - 1, -1, -1):
            out_ch = en_out_channels * (2 ** min(i, 4))
            self.decoder.append(DecoderBlock(ch, out_ch))
            ch = out_ch

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        skips = []
        for enc in self.encoder:
            x, skip = enc(x)
            skips.append(skip)

        for block in self.inter:
            x = block(x)

        for i, dec in enumerate(self.decoder):
            x = dec(x, skips[-(i + 1)])

        return x


class DeepUNet0(nn.Layer):
    """DeepUNet variant 0 for RMVPEUNet."""

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.n_blocks = n_blocks
        self.en_de_layers = en_de_layers

        self.encoder = nn.LayerList()
        ch = in_channels
        for i in range(en_de_layers):
            out_ch = en_out_channels * (2 ** min(i, 4))
            self.encoder.append(EncoderBlock(ch, out_ch))
            ch = out_ch

        self.inter = nn.LayerList([
            ConvBlock(ch, ch) for _ in range(inter_layers)
        ])

        self.decoder = nn.LayerList()
        for i in range(en_de_layers - 1, -1, -1):
            out_ch = en_out_channels * (2 ** min(i, 4))
            self.decoder.append(DecoderBlock(ch, out_ch))
            ch = out_ch

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        skips = []
        for enc in self.encoder:
            x, skip = enc(x)
            skips.append(skip)

        for block in self.inter:
            x = block(x)

        for i, dec in enumerate(self.decoder):
            x = dec(x, skips[-(i + 1)])

        return x


class BiGRU(nn.Layer):
    """Bidirectional GRU wrapper."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, direction="bidirectional")

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = x.unsqueeze(0) if x.ndim == 2 else x
        out, _ = self.gru(x)
        return out.squeeze(0) if out.shape[0] == 1 else out


N_MELS = 128
N_CLASS = 360


class RMVPEBase(nn.Layer):
    """RMVPE base model (original E2E)."""

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        super().__init__()
        self.unet = DeepUNet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
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

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        mel = mel.transpose([0, 2, 1]).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose([0, 2, 1]).flatten(-2)
        x = self.fc(x)
        return x


class RMVPEUNet(nn.Layer):
    """RMVPE UNet model (original E2E0) - main backbone for RMVPE."""

    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        super().__init__()
        self.unet = DeepUNet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
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

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            mel: (B, N_MELS, T)

        Returns:
            logits: (B, T, N_CLASS)
        """
        mel = mel.transpose([0, 2, 1]).unsqueeze(1)  # (B, 1, T, N_MELS)
        x = self.cnn(self.unet(mel))  # (B, 3, T, N_MELS)
        x = x.transpose([0, 2, 1, 3]).flatten(-2)  # (B, T, 3*N_MELS)
        x = self.fc(x)  # (B, T, N_CLASS)
        return x
