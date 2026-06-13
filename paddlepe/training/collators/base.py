"""Base collator: interface for all model-specific collators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCollator(ABC):
    """Convert a batch of unified HDF5 dicts → model input tensors.

    Subclasses must implement ``__call__`` which receives a list of
    ``{waveform, f0, sr, hop, name}`` dicts and returns whatever the
    model's ``training_step`` expects (typically ``(inputs, labels)``).
    """

    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> tuple[Any, Any]: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
