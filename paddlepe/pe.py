"""PE: unified facade for all pitch extraction models.

Usage:
    from paddlepe import PE

    # List available models
    PE.list_models()

    # Create a pitch extractor (auto-detects CUDA conflicts)
    pe = PE.create("fcpe")
    f0, conf = pe.infer(wav, sr=16000)

    # Force remote (subprocess) mode for PyTorch users
    pe = PE.create("fcpe", force_remote=True)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from paddlepe.registry import registry

if TYPE_CHECKING:
    from paddlepe.models.base import BasePE

logger = logging.getLogger(__name__)


class PE:
    """Unified facade for pitch extraction."""

    @staticmethod
    def create(
        name: str,
        ckpt: str | None = None,
        force_remote: bool = False,
        **kwargs,
    ) -> BasePE:
        """Create a pitch extraction model.

        Auto-detects CUDA context conflicts (e.g. when PyTorch already
        owns CUDA) and transparently falls back to a subprocess server.

        Args:
            name: model name (e.g. 'fcpe', 'rmvpe')
            ckpt: path to .pdparams checkpoint
            force_remote: skip direct mode, always use subprocess server
            kwargs: additional model arguments (passed to both modes)

        Returns:
            BasePE instance (direct or remote)
        """
        if force_remote:
            return _create_remote(name, ckpt)

        # Try direct mode first
        try:
            return registry.create(name, ckpt, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            # Check for CUDA-related failures
            if any(
                kw in err_str
                for kw in ["cuda", "gpu", "out of memory", "cublas", "driver"]
            ):
                # rank-zero-only: avoid spam from forked DataLoader workers
                if os.environ.get("LOCAL_RANK", "0") == "0":
                    logger.warning("CUDA unavailable, entering server mode")
                return _create_remote(name, ckpt)
            # Re-raise non-CUDA errors
            raise

    @staticmethod
    def list_models() -> list[str]:
        """List all registered pitch extraction models."""
        return registry.list_models()

    @staticmethod
    def register(name: str):
        """Decorator to register a model class."""
        return registry.register(name)


def _create_remote(name: str, ckpt: str | None = None) -> BasePE:
    """Create a RemotePE instance (subprocess server)."""
    from paddlepe.remote import RemotePE

    return RemotePE(model=name, ckpt=ckpt)  # type: ignore
