"""Frequency conversion utilities: bin ↔ cent ↔ Hz ↔ MIDI."""

from __future__ import annotations

import numpy as np


def bins_to_cents(
    bins: np.ndarray,
    cents_per_bin: float = 20.0,
    base_cents: float = 1997.3794084376191,
) -> np.ndarray:
    """Convert pitch bin index to cents."""
    return bins.astype(np.float64) * cents_per_bin + base_cents


def bins_to_frequency(
    bins: np.ndarray,
    cents_per_bin: float = 20.0,
    base_cents: float = 1997.3794084376191,
) -> np.ndarray:
    """Convert pitch bin index to frequency in Hz."""
    cents = bins_to_cents(bins, cents_per_bin, base_cents)
    return cents_to_frequency(cents)


def cents_to_bins(
    cents: np.ndarray,
    cents_per_bin: float = 20.0,
    base_cents: float = 1997.3794084376191,
) -> np.ndarray:
    """Convert cents to pitch bin index."""
    return ((cents - base_cents) / cents_per_bin).astype(np.int64)


def cents_to_frequency(cents: np.ndarray) -> np.ndarray:
    """Convert cents to Hz. f0 = 10 * 2^(cent / 1200)"""
    return 10.0 * (2.0 ** (cents / 1200.0))


def frequency_to_cents(frequency: np.ndarray) -> np.ndarray:
    """Convert Hz to cents. cent = 1200 * log2(freq / 10)"""
    return 1200.0 * np.log2(np.maximum(frequency, 1e-10) / 10.0)


def frequency_to_bins(
    frequency: np.ndarray,
    cents_per_bin: float = 20.0,
    base_cents: float = 1997.3794084376191,
) -> np.ndarray:
    """Convert frequency in Hz to pitch bin index."""
    cents = frequency_to_cents(frequency)
    return cents_to_bins(cents, cents_per_bin, base_cents)


def hz_to_midi(frequency: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI note number. 69 = A4 = 440Hz."""
    return 12.0 * np.log2(np.maximum(frequency, 1e-10) / 440.0) + 69.0


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """Convert MIDI note number to Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_semitones(frequency: np.ndarray, ref: float = 440.0) -> np.ndarray:
    """Convert Hz to semitones relative to reference."""
    return 12.0 * np.log2(np.maximum(frequency, 1e-10) / ref)


def semitones_to_hz(semitones: np.ndarray, ref: float = 440.0) -> np.ndarray:
    """Convert semitones to Hz."""
    return ref * (2.0 ** (semitones / 12.0))
