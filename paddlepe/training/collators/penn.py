"""PENN collator: waveform → 8kHz frames → model input."""

from __future__ import annotations

from typing import Any

import numpy as np
import paddle

from paddlepe.training.collators.base import BaseCollator

# Constants from PENN inference
SAMPLE_RATE = 8000
WINDOW_SIZE = 1024
HOP_LENGTH = 80


class PENNCollator(BaseCollator):
    """Convert HDF5 samples to PENN training batches.

    Pipeline:
      1. Resample to 8 kHz
      2. Frame waveform (1024 window, 80 hop)
      3. Softmax cross-entropy loss over 1440 pitch bins
      4. Return (frames, f0_bin_target) for loss

    Args:
        sr: target sample rate (8000)
    """

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        frame_list, f0_list = [], []

        for sample in batch:
            wav = paddle.to_tensor(sample["waveform"], dtype=paddle.float32)
            sr = sample["sr"]
            f0_np = sample["f0"]
            hop = sample["hop"]

            # Resample to 8 kHz
            if sr != self.sr:
                scale = self.sr / sr
                new_len = int(wav.shape[-1] * scale)
                wav = paddle.nn.functional.interpolate(
                    wav.reshape([1, 1, -1]), size=[new_len], mode="linear"
                ).reshape([-1])
                hop = int(hop * self.sr / sr)

            # Pad and frame
            wav_pad = paddle.nn.functional.pad(
                wav.unsqueeze(0),
                (WINDOW_SIZE // 2, WINDOW_SIZE // 2),
                mode="constant",
                value=0,
            ).squeeze(0)

            frames = (
                paddle.nn.functional.unfold(
                    wav_pad.reshape([1, 1, 1, -1]),
                    kernel_sizes=[1, WINDOW_SIZE],
                    strides=[1, HOP_LENGTH],
                )
                .squeeze(0)
                .t()
            )  # (T, 1024)

            T = frames.shape[0]

            # Align F0 to frame rate
            if hop != HOP_LENGTH:
                f0_aligned = np.interp(
                    np.arange(T) * HOP_LENGTH / hop,
                    np.arange(len(f0_np)),
                    f0_np,
                ).astype(np.float32)
                nearest = np.clip(
                    np.round(np.arange(T) * HOP_LENGTH / hop).astype(int),
                    0,
                    len(f0_np) - 1,
                )
                f0_aligned[f0_np[nearest] == 0] = 0.0
            else:
                f0_aligned = f0_np[:T]

            frame_list.append(frames)
            f0_list.append(
                paddle.to_tensor(f0_aligned[:T], dtype=paddle.float32)
            )

        return paddle.stack(frame_list), paddle.stack(f0_list)
