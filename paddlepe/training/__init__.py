"""Training components for paddlePE — DataModule, datasets, collators.

Uses ocean for model checkpointing and logging primitives.
Only available when PaddlePaddle can be loaded directly.
"""

from __future__ import annotations

from paddlepe._compat import require_paddle

require_paddle("training")

from paddle import nn  # noqa: E402

from paddlepe.training.hdf5_dataset import HDF5Dataset  # noqa: E402
from paddlepe.training.pe_datamodule import PEDataModule  # noqa: E402

__all__ = [
    "HDF5Dataset",
    "PEDataModule",
    "PETask",
]


class PETask(nn.Layer):
    """Base training task for pitch estimation models.

    Provides common training loop utilities.
    Each model subpackage implements its own PETask subclass
    with model-specific forward/loss logic.
    """

    def __init__(self, backbone: nn.Layer):
        super().__init__()
        self.backbone = backbone

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        """Override in subclass."""
        raise NotImplementedError

    def validation_step(self, batch, *args, **kwargs):
        """Override in subclass."""
        raise NotImplementedError
