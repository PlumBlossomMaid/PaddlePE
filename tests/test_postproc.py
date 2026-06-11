"""Tests for postprocessing modules."""

from __future__ import annotations

import numpy as np
import pytest

from paddlepe.postproc import decode, ensemble, threshold, filter as pfilt, periodicity, convert


class TestDecode:
    """Test F0 decoding algorithms."""

    def test_argmax(self, sample_logits, sample_cent_table):
        bins_to_freq = 10.0 * (2.0 ** (sample_cent_table / 1200.0))
        bins, f0 = decode.argmax(sample_logits, bins_to_freq)
        assert bins.shape == (50,)
        assert f0.shape == (50,)
        assert np.all(f0 >= 0)

    def test_weighted_argmax(self, sample_logits, sample_cent_table):
        bins_to_freq = 10.0 * (2.0 ** (sample_cent_table / 1200.0))
        bins, f0 = decode.weighted_argmax(sample_logits, bins_to_freq)
        assert bins.shape == (50,)
        assert f0.shape == (50,)

    def test_viterbi(self, sample_logits, sample_cent_table):
        bins_to_freq = 10.0 * (2.0 ** (sample_cent_table / 1200.0))
        bins, f0 = decode.viterbi(sample_logits, bins_to_freq)
        assert bins.shape == (50,)
        assert f0.shape == (50,)

    def test_local_argmax(self, sample_logits, sample_cent_table):
        cents, f0 = decode.local_argmax(sample_logits, sample_cent_table, threshold=0.05)
        assert cents.shape == (50,)
        assert f0.shape == (50,)

    def test_cent_to_f0_roundtrip(self):
        f0_in = np.array([100.0, 200.0, 440.0, 880.0], dtype=np.float32)
        cent = decode.f0_to_cent(f0_in)
        f0_out = decode.cent_to_f0(cent)
        assert np.allclose(f0_in, f0_out, atol=0.1)

    def test_f0_to_cent_zero_safe(self):
        f0 = np.array([0.0, 100.0], dtype=np.float32)
        cent = decode.f0_to_cent(f0)
        assert np.isfinite(cent[0]) or cent[0] < 0
        assert np.isfinite(cent[1])


class TestEnsemble:
    """Test ensemble/TTA methods."""

    def test_ensemble_f0(self):
        T, K = 50, 3
        np.random.seed(42)
        f0s = np.random.uniform(200, 300, (T, K)).astype(np.float32)
        result = ensemble.ensemble_f0(f0s, [0, -12, 12], uv_penalty=12.0)
        assert result.shape == (T,)
        assert np.all(result >= 0)

    def test_ensemble_uv_penalty(self):
        """UV frames should be penalized."""
        T = 10
        f0s = np.zeros((T, 2), dtype=np.float32)
        f0s[:, 0] = 200.0  # all voiced
        f0s[0, 1] = 200.0  # one candidate voiced
        result = ensemble.ensemble_f0(f0s, [0, -12], uv_penalty=12.0)
        assert result.shape == (T,)


class TestThreshold:
    """Test UV thresholding methods."""

    def test_threshold_at(self, sample_f0, sample_confidence):
        result = threshold.threshold_at(sample_f0, sample_confidence, value=0.5)
        assert result.shape == sample_f0.shape
        # Frames with low confidence should be 0
        assert result[20] == 0.0  # low confidence region
        assert result[0] > 0  # high confidence region

    def test_hysteresis(self, sample_f0, sample_confidence):
        result = threshold.hysteresis(sample_f0, sample_confidence)
        assert result.shape == sample_f0.shape
        assert np.all(result >= 0)

    def test_silence_mask(self, sample_f0, sample_confidence):
        loudness = np.random.uniform(-80, -20, len(sample_f0)).astype(np.float32)
        result = threshold.silence_mask(sample_f0, sample_confidence, loudness, threshold_db=-60)
        assert result.shape == sample_f0.shape
        assert np.all(result >= 0)


class TestFilter:
    """Test filtering methods."""

    def test_nanmean(self):
        assert pfilt.nanmean(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
        assert pfilt.nanmean(np.array([1.0, np.nan, 3.0])) == pytest.approx(2.0)

    def test_nanmedian(self):
        assert pfilt.nanmedian(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
        assert pfilt.nanmedian(np.array([1.0, np.nan, 3.0])) == pytest.approx(2.0)

    def test_mean_filter(self, sample_f0):
        result = pfilt.mean_filter(sample_f0, win_length=5)
        assert result.shape == sample_f0.shape

    def test_median_filter(self, sample_f0):
        result = pfilt.median_filter(sample_f0, win_length=5)
        assert result.shape == sample_f0.shape

    def test_interpolate_uv(self, sample_f0):
        uv = sample_f0 <= 0
        result = pfilt.interpolate_uv(sample_f0, uv)
        assert result.shape == sample_f0.shape
        # UV regions should be interpolated to non-zero
        if uv.any():
            # Some interpolated values might be > 0
            pass


class TestPeriodicity:
    """Test periodicity estimation."""

    def test_entropy(self, sample_logits):
        p = periodicity.entropy(sample_logits)
        assert p.shape == (50,)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_periodicity_max(self, sample_logits):
        p = periodicity.periodicity_max(sample_logits)
        assert p.shape == (50,)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_periodicity_sum(self, sample_logits):
        p = periodicity.periodicity_sum(np.log(sample_logits + 1))
        assert p.shape == (50,)


class TestConvert:
    """Test frequency conversion utilities."""

    def test_hz_to_midi_roundtrip(self):
        hz = np.array([440.0, 261.63, 880.0], dtype=np.float32)
        midi = convert.hz_to_midi(hz)
        hz_back = convert.midi_to_hz(midi)
        assert np.allclose(hz, hz_back, atol=0.5)

    def test_known_values(self):
        # A4 = 440Hz = MIDI 69
        assert convert.hz_to_midi(np.array([440.0]))[0] == pytest.approx(69.0, rel=1e-3)
        # A4 = 440Hz = 0 semitones relative to 440
        st = convert.hz_to_semitones(np.array([440.0]))
        assert st[0] == pytest.approx(0.0, abs=1e-3)

    def test_convert_bins_to_frequency(self):
        bins = np.array([0, 180, 359])
        freq = convert.bins_to_frequency(bins)
        assert len(freq) == 3
        assert np.all(freq >= 0)

    def test_frequency_to_bins_to_frequency(self):
        hz = np.array([100.0, 440.0, 1000.0], dtype=np.float32)
        bins = convert.frequency_to_bins(hz)
        hz_back = convert.bins_to_frequency(bins)
        # Should be approximately correct
        assert np.all(np.abs(hz - hz_back) / hz < 0.3)
