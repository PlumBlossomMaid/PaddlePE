"""Format writers for .f0 and .csv files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from .formats import encode_header


def write_f0(
    path: str | Path,
    f0: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    hop_length: int = 160,
    f0_min: float = 32.0,
    f0_max: float = 2100.0,
):
    """Write .f0 binary file.

    Args:
        f0: (T,) float32, 0=unvoiced
        confidence: (T,) float32, optional
    """
    f0 = np.asarray(f0, dtype=np.float32)
    has_conf = confidence is not None
    header = encode_header(sample_rate, hop_length, len(f0), f0_min, f0_max, has_confidence=has_conf)

    with open(path, "wb") as f:
        f.write(header)
        f.write(f0.tobytes())
        if has_conf:
            f.write(np.asarray(confidence, dtype=np.float32).tobytes())


def write_csv(
    path: str | Path,
    f0: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    hop_length: int = 160,
):
    """Write CSV file with F0 data.

    Columns: time,f0_hz[,confidence]
    """
    f0 = np.asarray(f0, dtype=np.float32)
    has_conf = confidence is not None
    frame_period = hop_length / sample_rate

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if has_conf:
            writer.writerow(["time", "f0_hz", "confidence"])
            for i in range(len(f0)):
                writer.writerow([i * frame_period, f"{f0[i]:.4f}", f"{confidence[i]:.6f}"])
        else:
            writer.writerow(["time", "f0_hz"])
            for i in range(len(f0)):
                writer.writerow([i * frame_period, f"{f0[i]:.4f}"])


def write(
    path: str | Path,
    f0: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    hop_length: int = 160,
):
    """Auto-detect format by suffix and write."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".f0":
        write_f0(path, f0, confidence, sample_rate, hop_length)
    elif suffix == ".csv":
        write_csv(path, f0, confidence, sample_rate, hop_length)
    else:
        raise ValueError(f"Unknown format: {suffix}")
