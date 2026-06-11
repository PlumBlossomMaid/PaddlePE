"""Training task base for paddlePE models.

Uses ocean for model checkpointing and logging primitives.
"""

from __future__ import annotations

from paddle import nn


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
