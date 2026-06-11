"""paddlePE: Unified Pitch Extraction Toolkit for PaddlePaddle.

Provides a unified API for multiple pitch estimation models
with shared postprocessing, format I/O, and training infra.
"""

from paddlepe.pe import PE
from paddlepe.registry import registry

__all__ = ["PE"]
