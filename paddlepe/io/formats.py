"""PADDLEF0 binary format constants and structures.

Cross-language binary format for F0 data.

Header layout (37 bytes, packed):
  Offset  Size  Field
  0       8     magic: "PADDLEF0"
  8       1     version: 0x01
  9       4     header_size: 37
  13      4     sample_rate: Hz
  17      4     hop_length: samples
  21      4     num_frames
  25      4     f0_min: Hz
  29      4     f0_max: Hz
  33      1     flags: bit0=has_confidence
  34      3     reserved
  --- data ---
  37            f0_hz[N]: float32 LE
  37+4N         confidence[N]: float32 LE (if flags & 0x01)
"""

import struct
from typing import NamedTuple

MAGIC = b"PADDLEF0"
VERSION = 0x01
HEADER_SIZE = 37

FLAG_CONFIDENCE = 0x01


class PitchHeader(NamedTuple):
    """Decoded .f0 file header."""

    version: int
    header_size: int
    sample_rate: int
    hop_length: int
    num_frames: int
    f0_min: float
    f0_max: float
    flags: int


HEADER_FORMAT = "<8sBIIIIffB3x"
HEADER_FORMAT_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_FORMAT_SIZE == HEADER_SIZE, (
    f"Header size mismatch: {HEADER_FORMAT_SIZE} != {HEADER_SIZE}"
)


def encode_header(
    sample_rate: int,
    hop_length: int,
    num_frames: int,
    f0_min: float = 32.0,
    f0_max: float = 2100.0,
    has_confidence: bool = False,
) -> bytes:
    """Encode header to bytes."""
    flags = FLAG_CONFIDENCE if has_confidence else 0
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        HEADER_SIZE,
        sample_rate,
        hop_length,
        num_frames,
        f0_min,
        f0_max,
        flags,
    )


def decode_header(data: bytes) -> PitchHeader:
    """Decode header from bytes. Returns PitchHeader."""
    magic, ver, hdr_size, sr, hop, nf, fmin, fmax, flags = struct.unpack(
        HEADER_FORMAT, data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: expected {MAGIC!r}, got {magic!r}")
    return PitchHeader(
        version=ver,
        header_size=hdr_size,
        sample_rate=sr,
        hop_length=hop,
        num_frames=nf,
        f0_min=fmin,
        f0_max=fmax,
        flags=flags,
    )
