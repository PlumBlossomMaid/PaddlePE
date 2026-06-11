"""Dataset base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import paddle
from paddle.io import Dataset


class PEDataset(Dataset, ABC):
    """Base class for pitch estimation datasets.

    Subclasses should define:
    - url: download URL
    - sample_rate: target sample rate
    - hop_length: frame hop in samples
    """

    url: ClassVar[Optional[str]] = None
    sample_rate: ClassVar[int] = 16000
    hop_length: ClassVar[int] = 160

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return a dict with at least 'wav' and 'f0' keys."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
