"""RMVPE inference: wav → Mel → model → F0 pipeline.

Matches reference RMVPE_paddle implementation:
  - librosa.filters.mel with htk=True, fmin=30
  - STFT with center=True
  - Local weighted average F0 decoding with CONST=1997.3794
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlepe.models.base import BasePE
from paddlepe.models.rmvpe.backbone import N_CLASS, N_MELS, RMVPEUNet
from paddlepe.registry import registry

try:
    from librosa.filters import mel as librosa_mel_fn
except ImportError:
    librosa_mel_fn = None

# Constants matching reference
CONST = 1997.3794084376191
SAMPLE_RATE = 16000
WINDOW_LENGTH = 1024
MEL_FMIN = 30
MEL_FMAX = 8000


class MelSpectrogram(nn.Layer):
    """Mel spectrogram extraction matching reference RMVPE_paddle.

    Uses librosa.filters.mel(htk=True) and STFT with center=True.
    """

    def __init__(
        self,
        n_mels=N_MELS,
        sr=SAMPLE_RATE,
        win_length=WINDOW_LENGTH,
        hop_length=160,
        n_fft=None,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sr
        self.n_mels = n_mels
        self.clamp = clamp

        # Build mel filterbank using librosa with htk=True (matching reference)
        if librosa_mel_fn is not None:
            mel_basis = librosa_mel_fn(
                sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax, htk=True
            )
        else:
            # Fallback: triangular mel
            mel_basis = self._create_mel_filterbank(n_fft // 2 + 1, sr, n_mels, mel_fmin, mel_fmax)
        self.register_buffer("mel_basis", paddle.to_tensor(mel_basis, dtype="float32"), persistable=True)

        # Hann window
        self.hann_window = paddle.audio.functional.get_window(
            "hann", win_length, fftbins=True, dtype="float32"
        )

    @staticmethod
    def _create_mel_filterbank(n_freq, sr, n_mels, fmin, fmax):
        low_mel = 2595.0 * np.log10(1.0 + fmin / 700.0)
        high_mel = 2595.0 * np.log10(1.0 + fmax / 700.0)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        freq_bins = np.linspace(0, sr / 2, n_freq)
        filters = np.zeros((n_mels, n_freq), dtype=np.float32)
        for i in range(1, n_mels + 1):
            left = hz_points[i - 1]
            center = hz_points[i]
            right = hz_points[i + 1]
            idx_left = (freq_bins >= left) & (freq_bins < center)
            filters[i - 1, idx_left] = (freq_bins[idx_left] - left) / (center - left)
            idx_right = (freq_bins >= center) & (freq_bins <= right)
            filters[i - 1, idx_right] = (right - freq_bins[idx_right]) / (right - center)
        return filters

    def forward(self, audio, center=True):
        """Extract log-Mel spectrogram.

        Args:
            audio: (B, T_audio) float32
            center: pad audio with win_length//2 on both sides

        Returns:
            mel: (B, n_mels, T_frames)
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        stft = paddle.signal.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=center,
        )
        magnitude = stft.abs()
        mel_output = paddle.matmul(self.mel_basis, magnitude)
        log_mel = paddle.log(paddle.clip(mel_output, min=self.clamp))
        return log_mel


@registry.register("rmvpe")
class RMVPEPE(BasePE):
    """RMVPE pitch extraction model.

    Uses UNet + BiGRU backbone (E2E0).
    Mel preprocessing and F0 decoding match reference RMVPE_paddle.
    """

    trainable = True
    support_onnx = True

    def __init__(
        self,
        n_blocks: int = 4,
        n_gru: int = 1,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = 160,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = N_MELS
        self.n_class = N_CLASS
        self.const = CONST

        self.backbone = RMVPEUNet(n_blocks=n_blocks, n_gru=n_gru)
        self.mel_extractor = MelSpectrogram(
            n_mels=N_MELS, sr=sample_rate, hop_length=hop_length,
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
        threshold: float = 0.03,
        use_viterbi: bool = False,
        interp_uv: bool = False,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """Infer F0 from audio.

        Args:
            wav: (S,) or (1, S) float32 audio
            sr: sample rate
            threshold: UV confidence threshold (reference default 0.03)
            use_viterbi: use Viterbi decoding
            interp_uv: interpolate unvoiced frames

        Returns:
            (f0_hz, confidence)
        """
        wav = self._to_tensor(wav)
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

        # Extract Mel — matching reference: center=True
        mel = self.mel_extractor(wav, center=True)  # (B, N_MELS, T)
        n_frames = mel.shape[-1]

        # Pad to multiple of 32 (U-Net downsamples 5 times: 2^5 = 32)
        pad_len = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad_len > 0:
            mel = F.pad(mel, [0, pad_len], mode="constant", data_format="NCL")

        # Forward
        logits = self.backbone(mel)  # (B, T, N_CLASS)
        logits = logits[:, :n_frames, :]  # (B, n_frames, N_CLASS)

        # Decode — matching reference to_local_average_f0
        f0 = self._decode_local_average(logits, threshold)

        if use_viterbi:
            f0 = self._decode_viterbi(logits, f0, threshold)

        f0 = f0.squeeze(0)  # (T,)
        confidence = paddle.max(logits.squeeze(0), axis=-1)  # (T,)

        if interp_uv:
            f0_np = f0.numpy()
            uv = f0_np <= 0
            if uv.any():
                from paddlepe.postproc.filter import interpolate_uv as iuv

                f0_np = iuv(f0_np, uv)
                f0 = paddle.to_tensor(f0_np)

        return f0, confidence

    def _decode_local_average(self, hidden, thred=0.03):
        """Local weighted average F0 decoding matching reference.

        cents = idx * 20 + CONST
        f0 = 10 * 2^(cents / 1200)
        """
        B, T, N = hidden.shape
        idx_f = paddle.arange(N, dtype=paddle.float32)[None, None, :]  # (1, 1, N)
        idx_cents = idx_f * 20 + self.const  # (1, 1, N)

        center_i = paddle.argmax(hidden, axis=2, keepdim=True)  # (B, T, 1) int64
        center_f = center_i.astype(paddle.float32)
        start = paddle.clip(center_f - 4, min=0.0)
        end = paddle.clip(center_f + 5, max=float(N))
        idx_mask = (idx_f >= start) & (idx_f < end)  # (B, T, N)
        weights = hidden * idx_mask.astype(paddle.float32)
        product_sum = paddle.sum(weights * idx_cents, axis=2)  # (B, T)
        weight_sum = paddle.sum(weights, axis=2)  # (B, T)
        cents = product_sum / (weight_sum + (weight_sum == 0).astype(paddle.float32))
        f0 = 10.0 * (2.0 ** (cents / 1200.0))
        uv = paddle.max(hidden, axis=2) < thred
        f0 = f0 * (~uv).astype(paddle.float32)
        return f0

    def _decode_viterbi(self, hidden, f0, thred=0.03):
        """Viterbi decoding matching reference."""
        import librosa

        hidden_np = hidden.squeeze(0).cpu().numpy()
        prob = hidden_np.T
        prob = prob / prob.sum(axis=0)

        N = N_CLASS
        xx, yy = np.meshgrid(range(N), range(N))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)

        path = librosa.sequence.viterbi(prob, transition).astype(np.int64)
        center = paddle.to_tensor(path).unsqueeze(0).unsqueeze(-1)
        return self._decode_local_average(hidden, center=center, thred=thred)

    @classmethod
    def default_ckpt(cls) -> str:
        return str(
            Path(__file__).parent.parent.parent.parent
            / "ckpts"
            / "rmvpe.pdparams"
        )
