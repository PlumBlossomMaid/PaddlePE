"""RMVPE collator: waveform → Mel → Gaussian pitch label."""

from __future__ import annotations

from typing import Any

import numpy as np
import paddle

from paddlepe.training.collators.base import BaseCollator

# Constants from RMVPE reference training
N_CLASS = 360
CONST = 1997.3794084376191


class RMVPECollator(BaseCollator):
    """Convert HDF5 samples to RMVPE training batches.

    Pipeline (per sample):
      1. Resample waveform to 16 kHz
      2. Compute log-Mel (128 bins, htk=True, fmin=30, hop_length=20)
      3. Convert F0 to Gaussian-smoothed pitch labels (360 bins)
      4. Return (mel, pitch_label)

    Note: RMVPE trains at a higher temporal resolution (hop_length=20,
    ~1.25ms) than the F0 annotation (hop_length=160, ~10ms). The
    Gaussian label is computed at annotation resolution and repeated
    to match the Mel frame rate.

    Args:
        sr: target sample rate (16000)
        n_mels: Mel bins (128)
        n_fft: FFT size (1024)
        hop_length: Mel frame hop during training (20)
        win_length: STFT window (1024)
        mel_fmin: min Mel frequency (30)
        mel_fmax: max Mel frequency (8000)
        sigma: Gaussian width for pitch label smoothing (1.25)
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 20,
        win_length: int = 1024,
        mel_fmin: float = 30.0,
        mel_fmax: float = 8000.0,
        sigma: float = 1.25,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sigma = sigma

        # Precompute Hann window
        self._hann = {}

        # Precompute Mel filterbank (librosa, htk=True)
        self._mel_basis = self._build_mel_basis()

    def _build_mel_basis(self) -> paddle.Tensor:
        from librosa.filters import mel as librosa_mel_fn

        mel_basis = librosa_mel_fn(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            htk=True,
        )
        return paddle.to_tensor(mel_basis, dtype=paddle.float32)

    def _hann_window(self, n_fft: int) -> paddle.Tensor:
        key = str(n_fft)
        if key not in self._hann:
            self._hann[key] = paddle.audio.functional.get_window(
                "hann", n_fft, fftbins=True, dtype="float32"
            )
        return self._hann[key]

    def _wav_to_mel(
        self, wav: paddle.Tensor, sample_rate: int
    ) -> paddle.Tensor:
        """Convert (S,) waveform → (n_mels, T) log-Mel at training resolution."""
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Resample to target SR
        if sample_rate != self.sr:
            scale = self.sr / sample_rate
            new_len = int(wav.shape[-1] * scale)
            wav = paddle.nn.functional.interpolate(
                wav.unsqueeze(0), size=[new_len], mode="linear"
            ).squeeze(0)

        # STFT with center=True matching RMVPE reference
        stft = paddle.signal.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann_window(self.win_length),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )
        magnitude = stft.abs()

        # Mel projection
        mel = paddle.matmul(self._mel_basis, magnitude)
        return paddle.log(paddle.clip(mel, min=1e-5))  # (n_mels, T)

    @staticmethod
    def _hz_to_gaussian_label(
        f0: np.ndarray,
        n_frames: int,
    ) -> np.ndarray:
        """Convert F0 (Hz) → Gaussian-smoothed label (T_mel, 360).

        For each frame: cent = 1200 * log2(f0/10)
        bin_index = (cent - CONST) / 20
        label = exp(-(arange - index)^2 / 2 / sigma^2)
        """
        nz = f0 > 0
        hz = f0.copy()
        hz[~nz] = 1.0  # avoid log(0)
        cent = 1200.0 * np.log2(hz / 10.0)
        index = (cent - CONST) / 20.0  # float bin index

        label = np.zeros((n_frames, N_CLASS), dtype=np.float32)
        arange = np.arange(N_CLASS, dtype=np.float32)

        for t in range(n_frames):
            if nz[t]:
                label[t] = np.exp(-((arange - index[t]) ** 2) / 2.0 / (1.25**2))

        return label

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Collate a batch of HDF5 samples.

        Returns:
            (mel, pitch_label)
            mel: (B, n_mels, T)
            pitch_label: (B, T, 360)
        """
        mel_list, label_list = [], []

        for sample in batch:
            wav = paddle.to_tensor(sample["waveform"], dtype=paddle.float32)
            sr = sample["sr"]
            f0_np = sample["f0"]
            hop = sample["hop"]

            mel = self._wav_to_mel(wav, sr).squeeze(0)  # (n_mels, T_mel)

            # Pad to multiple of 32 (RMVPE UNet downsamples 5×: 2^5=32)
            pad_len = 32 * ((mel.shape[-1] - 1) // 32 + 1) - mel.shape[-1]
            if pad_len > 0:
                mel = paddle.nn.functional.pad(
                    mel.unsqueeze(0),
                    [0, pad_len],
                    mode="constant",
                    data_format="NCL",
                ).squeeze(0)
            T_mel = mel.shape[-1]

            # Align F0 to the FULL (padded) Mel length
            if hop != self.hop_length:
                f0_aligned = np.interp(
                    np.arange(T_mel) * self.hop_length / hop,
                    np.arange(len(f0_np)),
                    f0_np,
                ).astype(np.float32)
                nearest = np.clip(
                    np.round(np.arange(T_mel) * self.hop_length / hop).astype(
                        int
                    ),
                    0,
                    len(f0_np) - 1,
                )
                f0_aligned[f0_np[nearest] == 0] = 0.0
            else:
                f0_aligned = np.pad(f0_np, (0, max(0, T_mel - len(f0_np))))[
                    :T_mel
                ]

            label = self._hz_to_gaussian_label(f0_aligned, T_mel)

            mel_list.append(mel)
            label_list.append(paddle.to_tensor(label, dtype=paddle.float32))

        # Pad time dimension to match longest in batch
        # mel: (n_mels, T), label: (T, 360)
        # Paddle pad format: [d0_left, d0_right, d1_left, d1_right, ...]
        max_T = max(m.shape[-1] for m in mel_list)
        mel_padded, label_padded = [], []
        for mel, label in zip(mel_list, label_list):
            T = mel.shape[-1]
            if T < max_T:
                pad = max_T - T
                mel = paddle.nn.functional.pad(
                    mel, [0, 0, 0, pad], mode="constant", value=0.0
                )
                label = paddle.nn.functional.pad(
                    label, [0, pad, 0, 0], mode="constant", value=0.0
                )
            mel_padded.append(mel)
            label_padded.append(label)

        return paddle.stack(mel_padded), paddle.stack(label_padded)
