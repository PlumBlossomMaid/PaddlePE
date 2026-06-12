"""MelConformerF0: FCPE backbone matching original FCPE architecture exactly.

Port of CFNaiveMelPE with renamed classes. Uses the original
ConformerConvModule from the FCPE Paddle port.
"""

from __future__ import annotations

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.utils import weight_norm

from .conformer_encoder import ConformerEncoder


class MelConformerF0(nn.Layer):
    """Conformer-based pitch estimator — architecture matches original FCPE.

    Input:  (B, mel_bins, T)
    Output: (B, T, out_dims) sigmoid
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
        use_fa_norm: bool = False,
        conv_only: bool = False,
        conv_dropout: float = 0.0,
        atten_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_stack = nn.Sequential(
            nn.Conv1D(mel_bins, hidden_dims, 3, 1, 1),
            nn.GroupNorm(4, hidden_dims),
            nn.LeakyReLU(),
            nn.Conv1D(hidden_dims, hidden_dims, 3, 1, 1),
        )
        self.net = ConformerEncoder(
            num_layers=n_layers,
            num_heads=n_heads,
            dim_model=hidden_dims,
            use_norm=use_fa_norm,
            conv_only=conv_only,
            conv_dropout=conv_dropout,
            atten_dropout=atten_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dims)
        self.output_proj = weight_norm(
            nn.Linear(hidden_dims, out_dims), "weight", 1
        )
        cent_min = float(1200.0 * np.log2(f0_min / 10.0))
        cent_max = float(1200.0 * np.log2(f0_max / 10.0))
        self.register_buffer(
            "cent_table",
            paddle.linspace(cent_min, cent_max, out_dims),
            persistable=True,
        )

    def _cent_to_f0(self, cent: paddle.Tensor) -> paddle.Tensor:
        return 10.0 * (2.0 ** (cent / 1200.0))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Forward pass.

        Args:
            x: (B, T, mel_bins) — note: (batch, time, channels) format
        Returns:
            out: (B, T, out_dims) sigmoid
        """
        # Input stack: (B, T, C) → transpose → (B, C, T) → conv1d → transpose → (B, T, C)
        x = self.input_stack[0](x.transpose([0, 2, 1]))  # Conv1D: (B, C, T)
        x = self.input_stack[1](x.unsqueeze(-2)).squeeze(
            -2
        )  # GroupNorm: need 4D
        x = self.input_stack[2](x)  # LeakyReLU
        x = self.input_stack[3](x)  # Conv1D
        x = x.transpose([0, 2, 1])  # (B, T, C) for Conformer

        x = self.net(x)  # (B, T, C)
        x = self.norm(x)  # (B, T, C)
        x = self.output_proj(x)  # (B, T, D)
        return F.sigmoid(x)

    def infer(
        self,
        mel: paddle.Tensor,
        decoder: str = "local_argmax",
        threshold: float = 0.05,
    ) -> paddle.Tensor:
        latent = self.forward(mel)
        B, N, D = latent.shape
        ct = self.cent_table[None, None, :].expand([B, N, -1])
        confident, max_idx = (
            paddle.max(latent, axis=-1, keepdim=True),
            paddle.argmax(latent, axis=-1, keepdim=True),
        )
        local_idx = paddle.arange(0, 9) + (max_idx - 4)
        local_idx = paddle.clip(local_idx, 0, D - 1)
        ci_local = paddle.take_along_axis(ct, axis=-1, indices=local_idx)
        y_local = paddle.take_along_axis(latent, axis=-1, indices=local_idx)
        rtn = paddle.sum(ci_local * y_local, axis=-1, keepdim=True) / (
            paddle.sum(y_local, axis=-1, keepdim=True) + 1e-10
        )
        rtn = rtn * paddle.where(
            confident <= threshold,
            paddle.to_tensor(float("-inf"), dtype=rtn.dtype),
            paddle.ones_like(rtn),
        )
        return self._cent_to_f0(rtn)
