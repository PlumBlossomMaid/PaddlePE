"""Wrapper models for third-party pitch extraction libraries.

These models wrap non-neural pitch extractors (Parselmouth/Praat, pyworld/WORLD)
and expose them through the BasePE interface. They do not support training or
ONNX export.
"""

# Import wrappers to trigger registry registration
import paddlepe.models.wrappers.parselmouth
import paddlepe.models.wrappers.pyworld  # noqa: F401
