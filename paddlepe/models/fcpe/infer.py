"""FCPE inference: wav → Mel → model → F0 pipeline.

Uses librosa.filters.mel for mel filterbank (matching training preprocessing).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import paddle

from paddlepe.models.base import BasePE
from paddlepe.models.fcpe.backbone import MelConformerF0
from paddlepe.registry import registry

logger = logging.getLogger(__name__)

try:
    from librosa.filters import mel as librosa_mel_fn
except ImportError:
    logger.warning("librosa not found; FCPE Mel accuracy may be degraded")
    librosa_mel_fn = None


@registry.register("fcpe")
class FCPEPE(BasePE):
    """FCPE pitch extraction model.

    Uses Conformer-based MelConformerF0 backbone.
    Mel preprocessing matches the original FCPE training pipeline.
    """

    trainable = True
    support_onnx = True

    def __init__(
        self,
        mel_bins: int = 128,
        hidden_dims: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        f0_min: float = 32.70,
        f0_max: float = 1975.5,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        conv_only: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.mel_bins = mel_bins
        self.f0_min = f0_min
        self.f0_max = f0_max

        # Hann window matching original FCPE training code
        self._hann_window = paddle.audio.functional.get_window(
            "hann", win_length, fftbins=True, dtype="float32"
        )

        self.backbone = MelConformerF0(
            mel_bins=mel_bins,
            hidden_dims=hidden_dims,
            n_layers=n_layers,
            n_heads=n_heads,
            f0_min=f0_min,
            f0_max=f0_max,
            conv_only=conv_only,
        )

        # Librosa-based mel filterbank (matching training config)
        if librosa_mel_fn is not None:
            mel_basis = librosa_mel_fn(
                sr=sample_rate, n_fft=n_fft, n_mels=mel_bins,
                fmin=f_min, fmax=f_max,
            )
        else:
            mel_basis = self._create_mel_filterbank(int(n_fft // 2 + 1), sr=sample_rate)
        self.register_buffer(
            "_mel_basis", paddle.to_tensor(mel_basis, dtype=paddle.float32),
            persistable=True,
        )

    def _wav_to_mel(self, wav: paddle.Tensor, sr: int) -> paddle.Tensor:
        """Convert waveform to Mel spectrogram, matching original FCPE pipeline.

        Args:
            wav: (S,) or (1, S) float32
            sr: sample rate

        Returns:
            mel: (1, T, mel_bins) float32
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim > 2:
            wav = wav.squeeze(-1) if wav.shape[-1] == 1 else wav[:, 0]

        # Resample if needed
        if sr != self.sample_rate:
            scale = self.sample_rate / sr
            new_len = int(wav.shape[-1] * scale)
            wav = paddle.nn.functional.interpolate(
                wav.unsqueeze(0), size=[new_len], mode="linear"
            ).squeeze(0)

        # Pad matching original MelExtractor reflect padding
        pad_left = (self.win_length - self.hop_length) // 2
        pad_right = max(
            (self.win_length - self.hop_length + 1) // 2,
            self.win_length - wav.shape[-1] - pad_left,
        )
        if pad_right < wav.shape[-1]:
            wav = paddle.nn.functional.pad(
                wav.unsqueeze(1),
                [pad_left, pad_right], mode="reflect", data_format="NCL",
            ).squeeze(1)
        else:
            wav = paddle.nn.functional.pad(
                wav.unsqueeze(1),
                [pad_left, pad_right], mode="constant", data_format="NCL",
            ).squeeze(1)

        # STFT
        stft = paddle.signal.stft(
            wav, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            window=self._hann_window,
            center=False, pad_mode="reflect",
            normalized=False, onesided=True,
        )
        mag = paddle.sqrt(stft.real().pow(2) + stft.imag().pow(2) + 1e-9)

        # Mel projection via precomputed librosa filterbank
        mel = paddle.matmul(self._mel_basis, mag)
        mel = paddle.log(paddle.clip(mel, min=1e-5))
        return mel.transpose([0, 2, 1])  # (B, T, mel_bins)

    def _create_mel_filterbank(self, n_freq: int, sr: int) -> paddle.Tensor:
        """Fallback: triangular mel filterbank when librosa unavailable."""
        low_mel = 2595.0 * np.log10(1.0 + self.f_min / 700.0)
        high_mel = 2595.0 * np.log10(1.0 + self.f_max / 700.0)
        mel_points = np.linspace(low_mel, high_mel, self.mel_bins + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        freq_bins = np.linspace(0, sr / 2, n_freq)
        filters = np.zeros((self.mel_bins, n_freq), dtype=np.float32)
        for i in range(1, self.mel_bins + 1):
            left = hz_points[i - 1]
            center = hz_points[i]
            right = hz_points[i + 1]
            idx_left = (freq_bins >= left) & (freq_bins < center)
            filters[i - 1, idx_left] = (freq_bins[idx_left] - left) / (center - left)
            idx_right = (freq_bins >= center) & (freq_bins <= right)
            filters[i - 1, idx_right] = (right - freq_bins[idx_right]) / (right - center)
        return paddle.to_tensor(filters)

    def forward(self, mel: paddle.Tensor) -> paddle.Tensor:
        """Training forward pass.

        Args:
            mel: (B, T, mel_bins)

        Returns:
            latent: (B, T, out_dims)
        """
        return self.backbone(mel)

    def infer(
        self,
        wav: paddle.Tensor,
        sr: int,
        decoder: str = "local_argmax",
        threshold: float = 0.05,
        interp_uv: bool = False,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Infer F0 from audio.

        Args:
            wav: (S,) or (1, S) float32 audio
            sr: sample rate
            decoder: "argmax" or "local_argmax"
            threshold: UV confidence threshold
            interp_uv: interpolate unvoiced frames

        Returns:
            (f0_hz, confidence)
        """
        wav = self._to_tensor(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() > 2:
            wav = wav.squeeze(-1) if wav.shape[-1] == 1 else wav[:, 0]

        mel = self._wav_to_mel(wav, sr)  # (B, T, mel_bins)

        f0 = self.backbone.infer(mel, decoder=decoder, threshold=threshold)
        # Extract confidence
        with paddle.no_grad():
            latent = self.backbone(mel)
            confidence = paddle.max(latent, axis=-1)  # (B, T)

        f0 = f0.squeeze(-1).squeeze(0)  # (T,)
        confidence = confidence.squeeze(0)  # (T,)

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
        """Path to default checkpoint."""
        return str(
            Path(__file__).parent.parent.parent.parent
            / "ckpts"
            / "fcpe.pdparams"
        )
