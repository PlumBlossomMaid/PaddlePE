"""FCPE collator: waveform → Mel spectrogram → Gaussian pitch label."""

from __future__ import annotations

from typing import Any

import numpy as np
import paddle

from paddlepe.training.collators.base import BaseCollator

# Constants matching FCPE inference
CONST = 1997.3794084376191
N_CLASS = 360


class FCPECollator(BaseCollator):
    """Convert HDF5 samples to FCPE training batches.

    Pipeline (per sample):
      1. Resample waveform to 16 kHz
      2. Compute log-Mel spectrogram (128 bins, 1024 FFT, 160 hop)
      3. Convert F0 to Gaussian-smoothed pitch labels (360 bins)
      4. Return (mel, pitch_label)

    Args:
        sr: target sample rate (16000)
        n_mels: Mel bins (128)
        n_fft: FFT size (1024)
        hop_length: Mel frame hop (160 samples)
        win_length: STFT window (1024)
        f_min: min Mel frequency (0)
        f_max: max Mel frequency (8000)
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max

        # Precompute Hann window
        self._hann = paddle.audio.functional.get_window(
            "hann", win_length, fftbins=True, dtype="float32"
        )

        # Precompute Mel filterbank (librosa compatible)
        self._mel_basis = self._build_mel_basis()

    def _build_mel_basis(self) -> paddle.Tensor:
        from librosa.filters import mel as librosa_mel_fn

        mel_basis = librosa_mel_fn(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        return paddle.to_tensor(mel_basis, dtype=paddle.float32)

    def _wav_to_mel(
        self, wav: paddle.Tensor, sample_rate: int
    ) -> paddle.Tensor:
        """Convert (S,) waveform → (n_mels, T) log-Mel."""
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        if sample_rate != self.sr:
            scale = self.sr / sample_rate
            new_len = int(wav.shape[-1] * scale)
            wav = paddle.nn.functional.interpolate(
                wav.unsqueeze(0), size=[new_len], mode="linear"
            ).squeeze(0)

        pad = (self.win_length - self.hop_length) // 2
        wav = paddle.nn.functional.pad(
            wav.unsqueeze(1), [pad, pad], mode="reflect", data_format="NCL"
        ).squeeze(1)

        stft = paddle.signal.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )
        mag = paddle.sqrt(stft.real().pow(2) + stft.imag().pow(2) + 1e-9)
        mel = paddle.matmul(self._mel_basis, mag)
        return paddle.log(paddle.clip(mel, min=1e-5)).transpose(
            [0, 2, 1]
        )  # (1, T, n_mels)

    @staticmethod
    def _hz_to_gaussian_label(f0: np.ndarray, n_frames: int) -> np.ndarray:
        """Convert F0 (Hz) → Gaussian label (T, 360)."""
        nz = f0 > 0
        hz = f0.copy()
        hz[~nz] = 1.0
        cent = 1200.0 * np.log2(hz / 10.0)
        index = (cent - CONST) / 20.0

        label = np.zeros((n_frames, N_CLASS), dtype=np.float32)
        arange = np.arange(N_CLASS, dtype=np.float32)
        for t in range(n_frames):
            if nz[t]:
                label[t] = np.exp(-((arange - index[t]) ** 2) / 2.0 / (1.25**2))
        return label

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Collate a batch.

        Returns:
            (mel, pitch_label): (B, n_mels, T), (B, T, 360)
        """
        mel_list, label_list = [], []

        for sample in batch:
            wav = paddle.to_tensor(sample["waveform"], dtype=paddle.float32)
            sr = sample["sr"]
            f0_np = sample["f0"]
            hop = sample["hop"]

            mel = self._wav_to_mel(wav, sr).squeeze(0)  # (T_mel, n_mels)
            T_mel = mel.shape[0]

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
        # mel: (T_mel, n_mels), label: (T_mel, 360)
        # Paddle pad format: [d0_left, d0_right, d1_left, d1_right, ...]
        max_T = max(m.shape[0] for m in mel_list)
        for i in range(len(mel_list)):
            T = mel_list[i].shape[0]
            if T < max_T:
                pad = max_T - T
                mel_list[i] = paddle.nn.functional.pad(
                    mel_list[i], [0, pad, 0, 0],
                    mode="constant",
                )
                label_list[i] = paddle.nn.functional.pad(
                    label_list[i], [0, pad, 0, 0],
                    mode="constant",
                    value=0.0,
                )

        return paddle.stack(mel_list), paddle.stack(label_list)
