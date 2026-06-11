"""Format readers for .f0, .csv, .pv, .tsv files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .formats import (
    HEADER_SIZE,
    decode_header,
)


def read_f0(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    """Read .f0 binary file.

    Returns:
        (f0_hz, confidence, sample_rate, hop_length)
        f0_hz: (T,) float32, 0=unvoiced
        confidence: (T,) float32 or None
    """
    data = Path(path).read_bytes()
    header = decode_header(data[:HEADER_SIZE])
    offset = header.header_size
    dtype = np.dtype("<f4")

    # Read f0
    f0 = np.frombuffer(
        data, dtype=dtype, count=header.num_frames, offset=offset
    ).copy()
    offset += header.num_frames * 4

    # Read confidence if present
    confidence: np.ndarray | None = None
    if header.flags & 0x01:
        confidence = np.frombuffer(
            data, dtype=dtype, count=header.num_frames, offset=offset
        ).copy()

    return f0, confidence, header.sample_rate, header.hop_length


def read_csv(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    """Read CSV file with F0 data.

    Expected columns (header row required):
        time,f0_hz[,confidence]

    Returns:
        (f0_hz, confidence, sample_rate, hop_length)
    """
    import csv

    times: list[float] = []
    f0s: list[float] = []
    confs: list[float] = []
    has_conf = False

    with open(path) as f:
        reader = csv.reader(f)
        try:
            header_row = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV file: {path}")
        has_conf = len(header_row) >= 3

        for row in reader:
            if not row:
                continue
            times.append(float(row[0]))
            f0s.append(float(row[1]))
            if has_conf and len(row) >= 3:
                confs.append(float(row[2]))

    f0 = np.array(f0s, dtype=np.float32)
    confidence = (
        np.array(confs, dtype=np.float32) if has_conf and confs else None
    )
    # Infer sample_rate and hop_length from timestamps
    if len(times) >= 2:
        hop_sec = times[1] - times[0]
        sample_rate = int(round(1.0 / hop_sec)) if hop_sec > 0 else 16000
        hop_length = int(round(sample_rate * hop_sec))
    else:
        sample_rate = 16000
        hop_length = 160
    return f0, confidence, sample_rate, hop_length


def read_pv(path: str | Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read .pv format (RMVPE style: one value per line,
    0=unvoiced, else semitones).

    Returns:
        (f0_hz, uv, sample_rate, hop_length)
        uv: (T,) bool, True=unvoiced
    """
    lines = Path(path).read_text().strip().splitlines()
    values = np.array(
        [float(line.strip()) for line in lines if line.strip()],
        dtype=np.float32,
    )
    uv = values == 0
    # Convert semitones to Hz: f = 440 * 2^((st - 69) / 12)
    f0 = np.where(uv, 0.0, 440.0 * (2.0 ** ((values - 69.0) / 12.0)))
    return f0, uv, 16000, 160


def read_tsv(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read .tsv format (MIR-ST500 style: onset/offset/midi-note per line).

    Returns:
        (f0_hz, uv, sample_rate, hop_length)
        Frame-level with hop_length=160 (10ms @ 16kHz).
    """

    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    sr = 16000
    hop = 160
    # Determine total frames from max offset
    if data.ndim == 1:
        data = data.reshape(1, -1)
    max_time = float(np.max(data[:, 1])) if len(data) > 0 else 1.0
    num_frames = int(np.ceil(max_time * sr / hop)) + 1

    f0 = np.zeros(num_frames, dtype=np.float32)
    uv = np.ones(num_frames, dtype=bool)
    for onset, offset, note in data:
        left = int(round(onset * sr / hop))
        right = int(round(offset * sr / hop)) + 1
        freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
        f0[left:right] = freq
        uv[left:right] = False
    return f0, uv, sr, hop


def read(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    """Auto-detect format by suffix and read.

    Returns:
        (f0_hz, confidence/uv, sample_rate, hop_length)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".f0":
        return read_f0(path)
    elif suffix == ".csv":
        return read_csv(path)
    elif suffix == ".pv":
        f0, uv, sr, hop = read_pv(path)
        return f0, uv.astype(np.float32), sr, hop
    elif suffix == ".tsv":
        f0, uv, sr, hop = read_tsv(path)
        return f0, uv.astype(np.float32), sr, hop
    else:
        raise ValueError(f"Unknown format: {suffix}")
