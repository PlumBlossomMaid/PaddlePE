"""Model-specific collators: convert unified HDF5 dict → model input."""

from paddlepe.training.collators.base import BaseCollator
from paddlepe.training.collators.crepe import CREPECollator
from paddlepe.training.collators.fcpe import FCPECollator
from paddlepe.training.collators.penn import PENNCollator
from paddlepe.training.collators.rmvpe import RMVPECollator

__all__ = [
    "BaseCollator",
    "FCPECollator",
    "RMVPECollator",
    "CREPECollator",
    "PENNCollator",
]
