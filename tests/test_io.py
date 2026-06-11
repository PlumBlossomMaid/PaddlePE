"""Tests for .f0 binary format I/O."""

from __future__ import annotations

import numpy as np
import pytest

from paddlepe.io import (
    read_csv,
    read_f0,
    read_pv,
    read_tsv,
    write_csv,
    write_f0,
)
from paddlepe.io.formats import (
    HEADER_SIZE,
    decode_header,
    encode_header,
)


class TestFormats:
    """Test binary format encoding/decoding."""

    def test_encode_decode_header(self):
        """Header round-trips correctly."""
        data = encode_header(
            16000, 160, 100, f0_min=32.0, f0_max=2100.0, has_confidence=True
        )
        assert len(data) == HEADER_SIZE
        header = decode_header(data)
        assert header.sample_rate == 16000
        assert header.hop_length == 160
        assert header.num_frames == 100
        assert header.f0_min == 32.0
        assert header.f0_max == 2100.0
        assert header.flags & 0x01  # has_confidence

    def test_invalid_magic_raises(self):
        """Invalid magic raises ValueError."""
        bad_data = b"BADMAGIC" + b"\x00" * (HEADER_SIZE - 8)
        with pytest.raises(ValueError, match="Invalid magic"):
            decode_header(bad_data)


class TestReadWriteF0:
    """Test .f0 binary I/O."""

    def test_write_read_roundtrip(self, tmp_dir, sample_f0, sample_confidence):
        """Write and read .f0 file."""
        path = tmp_dir / "test.f0"
        write_f0(
            path,
            sample_f0,
            sample_confidence,
            sample_rate=16000,
            hop_length=160,
        )
        f0, conf, sr, hop = read_f0(path)
        assert np.allclose(f0, sample_f0)
        assert conf is not None
        assert np.allclose(conf, sample_confidence)
        assert sr == 16000
        assert hop == 160

    def test_write_read_without_confidence(self, tmp_dir, sample_f0):
        """Write and read .f0 file without confidence."""
        path = tmp_dir / "test_no_conf.f0"
        write_f0(
            path, sample_f0, confidence=None, sample_rate=16000, hop_length=160
        )
        f0, conf, sr, hop = read_f0(path)
        assert np.allclose(f0, sample_f0)
        assert conf is None

    def test_read_nonexistent_file_raises(self):
        """Reading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_f0("/nonexistent/path.f0")


class TestReadWriteCSV:
    """Test CSV I/O."""

    def test_write_read_roundtrip(self, tmp_dir, sample_f0, sample_confidence):
        """Write and read CSV file."""
        path = tmp_dir / "test.csv"
        write_csv(
            path,
            sample_f0,
            sample_confidence,
            sample_rate=16000,
            hop_length=160,
        )
        f0, conf, sr, hop = read_csv(path)
        assert np.allclose(f0, sample_f0)
        assert conf is not None
        assert np.allclose(conf, sample_confidence, atol=1e-4)

    def test_write_read_without_confidence(self, tmp_dir, sample_f0):
        """Write and read CSV file without confidence."""
        path = tmp_dir / "test_no_conf.csv"
        write_csv(path, sample_f0, confidence=None)
        f0, conf, sr, hop = read_csv(path)
        assert np.allclose(f0, sample_f0)
        assert conf is None


class TestReadPV:
    """Test .pv format reader."""

    def test_read_pv(self, tmp_dir):
        """Read .pv file."""
        pv_path = tmp_dir / "test.pv"
        # Write .pv format: semitones per line, 0 for unvoiced
        values = [
            69.0,
            70.0,
            0.0,
            71.0,
            72.0,
        ]  # A4=440Hz, B4≈493.88, silence, C5≈523.25, D5≈587.33
        pv_path.write_text("\n".join(str(v) for v in values))

        f0, uv, sr, hop = read_pv(pv_path)
        assert len(f0) == 5
        assert f0[0] == pytest.approx(440.0, rel=1e-2)
        assert f0[2] == 0.0
        assert uv[2] == True  # noqa: E712
        assert uv[0] == False  # noqa: E712


class TestReadTSV:
    """Test .tsv format reader."""

    def test_read_tsv(self, tmp_dir):
        """Read .tsv file."""
        tsv_path = tmp_dir / "test.tsv"
        # MIR-ST500 style: onset(sec)  offset(sec)  midi_note
        tsv_path.write_text("onset\toffset\tnote\n0.0\t0.5\t69\n0.5\t1.0\t71\n")

        f0, uv, sr, hop = read_tsv(tsv_path)
        assert len(f0) > 0
        assert f0[0] == pytest.approx(440.0, rel=1e-2)  # MIDI 69 = A4 = 440Hz
        assert sr == 16000
        assert hop == 160
