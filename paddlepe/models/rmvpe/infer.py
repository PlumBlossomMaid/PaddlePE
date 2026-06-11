"""RMVPE inference: wav → Mel → model → F0 pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlepe.models.base import BasePE
from paddlepe.models.rmvpe.backbone import N_CLASS, N_MELS, RMVPEUNet
from paddlepe.registry import registry


class MelSpectrogram(nn.Layer):
    """Mel spectrogram extraction for RMVPE."""

    def __init__(
        self,
        n_mels=128,
        sr=16000,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        f_min=0.0,
        f_max=8000.0,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max

        # Build mel filterbank
        mel_filters = self._create_mel_filterbank(n_fft // 2 + 1, sr)
        self.register_buffer("mel_filters", mel_filters, persistable=True)

        # Precompute Hann window
        n = paddle.arange(win_length, dtype=paddle.float32)
        hann = 0.5 * (1.0 - paddle.cos(2.0 * paddle.pi * n / (win_length - 1)))
        self.register_buffer("_hann_window", hann, persistable=True)

    def _create_mel_filterbank(self, n_freq: int, sr: int) -> paddle.Tensor:
        low_mel = 2595.0 * np.log10(1.0 + self.f_min / 700.0)
        high_mel = 2595.0 * np.log10(1.0 + self.f_max / 700.0)
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        freq_bins = np.linspace(0, sr / 2, n_freq)

        filters = np.zeros((self.n_mels, n_freq), dtype=np.float32)
        for i in range(1, self.n_mels + 1):
            left = hz_points[i - 1]
            center = hz_points[i]
            right = hz_points[i + 1]
            idx_left = (freq_bins >= left) & (freq_bins < center)
            filters[i - 1, idx_left] = (freq_bins[idx_left] - left) / (
                center - left
            )
            idx_right = (freq_bins >= center) & (freq_bins <= right)
            filters[i - 1, idx_right] = (right - freq_bins[idx_right]) / (
                right - center
            )
        return paddle.to_tensor(filters)

    def forward(self, wav: paddle.Tensor, center: bool = True) -> paddle.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)

        pad = self.win_length // 2 if center else 0
        if pad > 0:
            wav = F.pad(wav.unsqueeze(-1), [pad, pad]).squeeze(-1)

        stft = paddle.signal.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann_window,
            center=False,
        )
        mag = paddle.abs(stft)
        mel = paddle.matmul(self.mel_filters, mag)
        mel = paddle.log(mel + 1e-5)
        return mel


@registry.register("rmvpe")
class RMVPEPE(BasePE):
    """RMVPE pitch extraction model.

    Uses UNet + BiGRU backbone.
    """

    trainable = True
    support_onnx = True

    def __init__(
        self,
        n_blocks: int = 4,
        n_gru: int = 1,
        sample_rate: int = 16000,
        hop_length: int = 160,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = N_MELS
        self.n_class = N_CLASS

        self.backbone = RMVPEUNet(n_blocks=n_blocks, n_gru=n_gru)
        self.mel_extractor = MelSpectrogram(
            n_mels=N_MELS, sr=sample_rate, hop_length=hop_length
        )

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Training forward pass.

        Args:
            mel: (B, N_MELS, T)

        Returns:
            logits: (B, T, N_CLASS)
        """
        return self.backbone(mel)

    @paddle.no_grad()
    def infer(
        self,
        wav: paddle.Tensor,
        sr: int,
        threshold: float = 0.05,
        use_viterbi: bool = False,
        interp_uv: bool = False,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Infer F0 from audio.

        Args:
            wav: (S,) or (1, S) float32 audio
            sr: sample rate
            threshold: UV confidence threshold
            use_viterbi: use Viterbi decoding
            interp_uv: interpolate unvoiced frames

        Returns:
            (f0_hz, confidence)
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)

        # Resample if needed
        if sr != self.sample_rate:
            scale = self.sample_rate / sr
            new_len = int(wav.shape[-1] * scale)
            wav = (
                F.interpolate(
                    wav.unsqueeze(-1).transpose([0, 2, 1]),
                    size=new_len,
                    mode="linear",
                )
                .transpose([0, 2, 1])
                .squeeze(-1)
            )

        # Extract Mel
        mel = self.mel_extractor(wav, center=True)  # (B, N_MELS, T)
        n_frames = mel.shape[-1]

        # Pad to multiple of 32
        pad_len = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad_len > 0:
            mel = F.pad(mel, [0, pad_len])

        # Forward
        logits = self.backbone(mel)  # (B, T, N_CLASS)
        logits = logits[:, :n_frames, :]  # (B, n_frames, N_CLASS)

        # Decode to F0
        # Calculate cent table
        cent_min = 1200.0 * np.log2(32.70 / 10.0)
        cent_max = 1200.0 * np.log2(2100.0 / 10.0)
        cent_table = paddle.linspace(cent_min, cent_max, N_CLASS)

        if use_viterbi:
            from paddlepe.postproc.decode import viterbi

            npy_logits = logits.squeeze(0).numpy()
            npy_cent = cent_table.numpy()
            _, f0_npy = viterbi(npy_logits, 10.0 * (2.0 ** (npy_cent / 1200.0)))
            f0 = paddle.to_tensor(f0_npy)
        else:
            # Local weighted average
            B, T, D = logits.shape
            latent = F.sigmoid(logits)
            ct = cent_table[None, None, :].expand([B, T, -1])
            f0_cent = paddle.sum(ct * latent, axis=-1) / (
                paddle.sum(latent, axis=-1) + 1e-10
            )
            confident = paddle.max(latent, axis=-1)
            f0 = 10.0 * (2.0 ** (f0_cent / 1200.0))
            f0[confident <= threshold] = 0.0

        f0 = f0.squeeze(0)
        confidence = paddle.max(F.sigmoid(logits).squeeze(0), axis=-1)

        if interp_uv:
            f0_np = f0.numpy()
            uv = f0_np <= 0
            if uv.any():
                from paddlepe.postproc.filter import interpolate_uv as iuv

                f0_np = iuv(f0_np, uv)
                f0 = paddle.to_tensor(f0_np)

        return f0, confidence

    @classmethod
    def default_ckpt(cls) -> str:
        return str(
            Path(__file__).parent.parent.parent.parent
            / "ckpts"
            / "rmvpe.pdparams"
        )
