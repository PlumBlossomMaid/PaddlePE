"""CREPE pitch estimation module."""

from paddlepe.models.crepe.backbone import CrepeBackbone
from paddlepe.models.crepe.infer import CrepePE

__all__ = [
    "CrepeBackbone",
    "CrepePE",
]
