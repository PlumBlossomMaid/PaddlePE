"""PE: unified facade for all pitch extraction models.

Usage:
    from paddlepe import PE

    # List available models
    PE.list_models()

    # Create a pitch extractor
    pe = PE.create("fcpe")
    f0, conf = pe.infer(wav, sr=16000)
"""

from __future__ import annotations

from typing import Optional

from paddlepe.models.base import BasePE
from paddlepe.registry import registry


class PE:
    """Unified facade for pitch extraction."""

    @staticmethod
    def create(name: str, ckpt: Optional[str] = None, **kwargs) -> BasePE:
        """Create a pitch extraction model.

        Args:
            name: model name (e.g. 'fcpe', 'rmvpe')
            ckpt: path to .pdparams checkpoint (optional, uses default if available)
            kwargs: additional model arguments

        Returns:
            BasePE instance
        """
        return registry.create(name, ckpt, **kwargs)

    @staticmethod
    def list_models() -> list[str]:
        """List all registered pitch extraction models."""
        return registry.list_models()

    @staticmethod
    def register(name: str):
        """Decorator to register a model class."""
        return registry.register(name)
