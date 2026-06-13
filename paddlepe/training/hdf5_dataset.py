"""HDF5Dataset — unified reader for preprocessed HDF5 datasets.

Each HDF5 file stores preprocessed audio and F0 annotations in a
standard format, independent of the original dataset layout.
All models share this reader; model-specific transforms go in
the collator.

HDF5 format (per sample group)::

    /sample_0000/
    ├── waveform     (S,) float32   — mono audio @ model's target SR
    ├── f0           (T,) float32   — ground truth F0 in Hz, 0=unvoiced
    ├── sr           scalar int     — sample rate
    ├── hop          scalar int     — F0 frame hop in samples
    └── name         scalar str     — utterance identifier
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from paddle.io import Dataset

logger = logging.getLogger(__name__)


class HDF5Dataset(Dataset):
    """Read a single preprocessed HDF5 file.

    Each group in the HDF5 is one sample with ``waveform``, ``f0``,
    ``sr``, ``hop``, and ``name`` datasets.

    Args:
        h5_path: path to the .h5 file
        min_f0_hz: frames with F0 below this are treated as unvoiced (0)
    """

    def __init__(
        self,
        h5_path: str | Path,
        min_f0_hz: float = 40.0,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.min_f0_hz = min_f0_hz

        # Index all sample groups in the HDF5
        with h5py.File(self.h5_path, "r") as f:
            self.sample_keys = sorted(
                k
                for k in f.keys()
                if isinstance(f[k], h5py.Group) and "waveform" in f[k]
            )

        if not self.sample_keys:
            raise RuntimeError(f"No sample groups found in {self.h5_path}")

        logger.info(
            "HDF5Dataset: %d samples from %s",
            len(self.sample_keys),
            self.h5_path,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a unified sample dict.

        Returns:
            dict with keys: waveform, f0, sr, hop, name
        """
        key = self.sample_keys[index]
        with h5py.File(self.h5_path, "r") as f:
            grp = f[key]
            waveform = grp["waveform"][()].astype(np.float32)
            f0_raw = grp["f0"][()].astype(np.float32)
            sr = int(grp["sr"][()])
            hop = int(grp["hop"][()])
            name = str(grp.attrs.get("name", f"sample_{index:05d}"))

        # Apply min_f0 threshold
        f0_raw[f0_raw < self.min_f0_hz] = 0.0

        return {
            "waveform": waveform,
            "f0": f0_raw,
            "sr": sr,
            "hop": hop,
            "name": name,
        }

    def __len__(self) -> int:
        return len(self.sample_keys)

    @property
    def metadata(self) -> dict:
        """Return dataset-level metadata stored as HDF5 attributes."""
        with h5py.File(self.h5_path, "r") as f:
            return dict(f.attrs)
