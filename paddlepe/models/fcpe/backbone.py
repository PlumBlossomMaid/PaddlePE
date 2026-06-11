"""MelConformerF0: FCPE backbone - Conformer-based pitch estimator
from Mel spectrogram.

Port of CFNaiveMelPE with renamed classes.
"""

from __future__ import annotations

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.utils import weight_norm


class Transpose(nn.Layer):
    """Wrapper for paddle.transpose."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        ndim = x.ndim
        perm = list(range(ndim))
        perm[self.dim0], perm[self.dim1] = self.dim1, self.dim0
        return x.transpose(perm)


class GLU(nn.Layer):
    """Gated Linear Unit."""

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        dim = x.shape[1] // 2
        return x[:, :dim] * F.sigmoid(x[:, dim:])


class Swish(nn.Layer):
    """Swish activation."""

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x * F.sigmoid(x)


class DepthWiseConv1d(nn.Layer):
    """Depth-wise 1D convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1D(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=channels,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.conv(x)


class ConformerConvModule(nn.Layer):
    """Conformer convolution module."""

    def __init__(
        self,
        dim: int,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(1, 2),
            nn.Conv1D(dim, inner_dim * 2, 1),
            GLU(),
            DepthWiseConv1d(inner_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, inner_dim),
            Swish(),
            nn.Conv1D(inner_dim, dim, 1),
            nn.Dropout(dropout),
            Transpose(1, 2),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.net(x)


class FastAttention(nn.Layer):
    """Fast attention with optional LayerNorm."""

    def __init__(self, dim: int, heads: int, use_norm: bool = True):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias_attr=False)
        self.to_out = nn.Linear(dim, dim)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.use_norm:
            x = self.norm(x)
        q, k, v = paddle.chunk(self.to_qkv(x), 3, axis=-1)
        B, T, D = q.shape
        H = self.heads
        q = q.reshape([B, T, H, -1]).transpose([0, 2, 1, 3])
        k = k.reshape([B, T, H, -1]).transpose([0, 2, 1, 3])
        v = v.reshape([B, T, H, -1]).transpose([0, 2, 1, 3])

        scale = q.shape[-1] ** -0.5
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale
        attn = F.softmax(attn, axis=-1)
        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3]).reshape([B, T, D])
        return self.to_out(out)


class SelfAttention(nn.Layer):
    """Self-attention with residual and optional LayerNorm."""

    def __init__(self, dim: int, heads: int, use_norm: bool = True):
        super().__init__()
        self.attn = FastAttention(dim, heads, use_norm)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x + self.attn(x)


class ConformerEncoderLayer(nn.Layer):
    """Single Conformer encoder layer (FFN + Attention + Conv + FFN)."""

    def __init__(
        self,
        dim: int,
        heads: int,
        use_norm: bool = True,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.0,
    ):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Dropout(atten_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(atten_dropout),
        )
        if not conv_only:
            self.attn = SelfAttention(dim, heads, use_norm)
        else:
            self.attn = nn.Identity()
        self.conv = ConformerConvModule(dim, dropout=conv_dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Dropout(atten_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(atten_dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = x + self.ff1(x) * 0.5
        x = self.attn(x)
        x = x + self.conv(x)
        x = x + self.ff2(x) * 0.5
        return self.norm(x)


class ConformerEncoder(nn.Layer):
    """Stack of Conformer encoder layers."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        use_norm: bool = True,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.LayerList(
            [
                ConformerEncoderLayer(
                    dim,
                    num_heads,
                    use_norm,
                    conv_only,
                    conv_dropout,
                    atten_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MelConformerF0(nn.Layer):
    """Conformer-based pitch estimator from Mel spectrogram.

    Input:  (B, T, mel_bins) Mel spectrogram
    Output: (B, T, out_dims) sigmoid pitch distribution

    Args:
        mel_bins: number of Mel bins
        out_dims: number of output pitch bins
        hidden_dims: Conformer hidden dimension
        n_layers: number of Conformer layers
        n_heads: number of attention heads
        f0_min: minimum F0 in Hz
        f0_max: maximum F0 in Hz
        conv_only: use only convolution (no attention)
    """

    def __init__(
        self,
        mel_bins: int = 128,
        out_dims: int = 360,
        hidden_dims: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        f0_min: float = 32.70,
        f0_max: float = 1975.5,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.0,
    ):
        super().__init__()
        self.mel_bins = mel_bins
        self.out_dims = out_dims
        self.hidden_dims = hidden_dims
        self.f0_min = f0_min
        self.f0_max = f0_max

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1D(mel_bins, hidden_dims, 3, 1, 1),
            nn.GroupNorm(4, hidden_dims),
            nn.LeakyReLU(),
            nn.Conv1D(hidden_dims, hidden_dims, 3, 1, 1),
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(
            dim=hidden_dims,
            num_layers=n_layers,
            num_heads=n_heads,
            conv_only=conv_only,
            conv_dropout=conv_dropout,
            atten_dropout=atten_dropout,
        )

        # Output projection
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(
            nn.Linear(hidden_dims, out_dims), "weight", 1
        )

        # Cent table for decoding
        cent_min = self._f0_to_cent(paddle.to_tensor([f0_min])).item()
        cent_max = self._f0_to_cent(paddle.to_tensor([f0_max])).item()
        cent_table = paddle.linspace(cent_min, cent_max, out_dims)
        self.register_buffer("cent_table", cent_table, persistable=True)

    def _f0_to_cent(self, f0: paddle.Tensor) -> paddle.Tensor:
        return 1200.0 * paddle.log2(f0 / 10.0 + 1e-10)

    def _cent_to_f0(self, cent: paddle.Tensor) -> paddle.Tensor:
        return 10.0 * (2.0 ** (cent / 1200.0))

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Forward pass for training.

        Args:
            mel: (B, T, mel_bins)

        Returns:
            latent: (B, T, out_dims) sigmoid outputs
        """
        x = mel.transpose([0, 2, 1])  # (B, mel_bins, T)
        # Input projection with GroupNorm workaround
        x = self.input_proj[0](x)
        x = self.input_proj[1](x.unsqueeze(-2)).squeeze(-2)
        x = self.input_proj[2](x)
        x = self.input_proj[3](x)
        x = x.transpose([0, 2, 1])  # (B, T, hidden_dims)

        x = self.encoder(x)
        x = self.norm(x)
        x = self.output_proj(x)
        return F.sigmoid(x)

    def infer(
        self,
        mel: paddle.Tensor,
        decoder: str = "local_argmax",
        threshold: float = 0.05,
    ) -> paddle.Tensor:
        """Inference from Mel spectrogram.

        Args:
            mel: (B, T, mel_bins) or (1, T, mel_bins)
            decoder: "argmax" or "local_argmax"
            threshold: UV confidence threshold

        Returns:
            f0: (B, T, 1) Hz
        """
        latent = self.forward(mel)  # (B, T, out_dims)
        B, N, D = latent.shape
        cent_table = self.cent_table[None, None, :].expand([B, N, -1])

        if decoder == "argmax":
            rtn = paddle.sum(cent_table * latent, axis=-1, keepdim=True) / (
                paddle.sum(latent, axis=-1, keepdim=True) + 1e-10
            )
            confident = paddle.max(latent, axis=-1, keepdim=True)
            confident_mask = paddle.where(
                confident <= threshold,
                paddle.to_tensor(float("-inf"), dtype=rtn.dtype),
                paddle.ones_like(rtn),
            )
            rtn = rtn * confident_mask
        elif decoder == "local_argmax":
            confident, max_idx = (
                paddle.max(latent, axis=-1, keepdim=True),
                paddle.argmax(latent, axis=-1, keepdim=True),
            )
            local_idx = paddle.arange(0, 9) + (max_idx - 4)
            local_idx = paddle.clip(local_idx, 0, D - 1)
            ci_local = paddle.take_along_axis(
                cent_table, axis=-1, indices=local_idx
            )
            y_local = paddle.take_along_axis(latent, axis=-1, indices=local_idx)
            rtn = paddle.sum(ci_local * y_local, axis=-1, keepdim=True) / (
                paddle.sum(y_local, axis=-1, keepdim=True) + 1e-10
            )
            confident_mask = paddle.where(
                confident <= threshold,
                paddle.to_tensor(float("-inf"), dtype=rtn.dtype),
                paddle.ones_like(rtn),
            )
            rtn = rtn * confident_mask
        else:
            raise ValueError(f"Unknown decoder: {decoder}")

        return self._cent_to_f0(rtn)  # (B, T, 1)

    def train_and_loss(
        self, mel: paddle.Tensor, gt_f0: paddle.Tensor, loss_scale: float = 10.0
    ) -> paddle.Tensor:
        """Training step with loss computation.

        Args:
            mel: (B, T, mel_bins)
            gt_f0: (B, T, 1) ground truth F0 in Hz
            loss_scale: BCE loss scale factor

        Returns:
            loss
        """
        # Align lengths
        _len = min(mel.shape[1], gt_f0.shape[1])
        mel = mel[:, :_len, :]
        gt_f0 = gt_f0[:, :_len, :]

        # Ground truth → Gaussian-blurred cent target
        gt_cent = self._f0_to_cent(gt_f0)
        cent_table = self.cent_table[None, None, :].expand(
            [mel.shape[0], _len, -1]
        )
        x_gt = paddle.exp(-((cent_table - gt_cent) ** 2) / 1250.0)
        x_gt = x_gt * (gt_cent > 0.1).astype(x_gt.dtype)

        # Prediction
        x_pred = self.forward(mel)
        loss = F.binary_cross_entropy(x_pred, x_gt) * loss_scale
        return loss
