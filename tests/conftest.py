"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_f0() -> np.ndarray:
    """Synthetic F0 contour."""
    np.random.seed(42)
    T = 100
    f0 = np.sin(np.linspace(0, 4 * np.pi, T)) * 50 + 200  # 150-250 Hz
    f0[20:30] = 0  # unvoiced segment
    f0[60:70] = 0  # another unvoiced segment
    return f0.astype(np.float32)


@pytest.fixture
def sample_confidence() -> np.ndarray:
    """Synthetic confidence values."""
    np.random.seed(42)
    T = 100
    conf = np.random.uniform(0.5, 1.0, T).astype(np.float32)
    conf[20:30] = 0.1  # low confidence = unvoiced
    conf[60:70] = 0.05
    return conf


@pytest.fixture
def sample_logits() -> np.ndarray:
    """Synthetic pitch logits (T, 360)."""
    np.random.seed(42)
    T, D = 50, 360
    logits = np.random.randn(T, D).astype(np.float32)
    # Make one bin dominant per frame
    for t in range(T):
        dominant = (150 + t * 2) % 360
        logits[t, dominant] += 10
    return logits


@pytest.fixture
def sample_cent_table() -> np.ndarray:
    """Cent table mapping bins to cent values."""
    cent_min = 1200.0 * np.log2(32.70 / 10.0)
    cent_max = 1200.0 * np.log2(2100.0 / 10.0)
    return np.linspace(cent_min, cent_max, 360).astype(np.float32)
