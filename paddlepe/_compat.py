"""Compatibility utilities for PaddlePaddle availability checks.

Provides input type conversion helpers used by both direct and
client-only inference modes.
"""

from __future__ import annotations

from typing import Any


def _paddle_available() -> bool:
    """Check if PaddlePaddle is importable."""
    try:
        import paddle  # noqa: F401

        return True
    except (OSError, ImportError):
        return False


def require_paddle(module: str = "") -> None:
    """Raise a clear error if PaddlePaddle is not importable.

    Args:
        module: The optional submodule name that requires Paddle.

    Raises:
        RuntimeError: If PaddlePaddle cannot be imported.
    """
    if not _paddle_available():
        label = f" ({module})" if module else ""
        raise RuntimeError(
            f"PaddlePaddle is not available in this process{label}. "
            "paddlePE is running in client-only mode (inference via "
            "subprocess server). Training and export require a process "
            "where PaddlePaddle can be imported directly.\n\n"
            "Use a separate Python process or run without PyTorch "
            "to access training and export functionality."
        )


def to_paddle_tensor(x: Any):
    """Convert various array-like objects to paddle.Tensor.

    Accepts: paddle.Tensor, numpy.ndarray, or any object with
    a ``.numpy()`` method (e.g. torch.Tensor).

    In direct (Paddle) mode, this is called at the start of each
    model's ``infer()`` to allow flexible input types.
    """
    import numpy as np
    import paddle

    if isinstance(x, paddle.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return paddle.to_tensor(x)
    if hasattr(x, "numpy"):
        return paddle.to_tensor(x.numpy())
    raise TypeError(
        f"Unsupported input type: {type(x).__name__}. "
        "Expected paddle.Tensor, numpy.ndarray, or an object "
        "with a .numpy() method."
    )


def to_numpy(x: Any):
    """Convert various array-like objects to numpy.ndarray.

    Accepts: numpy.ndarray, or any object with a ``.numpy()``
    method (e.g. torch.Tensor, paddle.Tensor).

    Does NOT import paddle — safe to call in client-only mode.

    In client (remote) mode, this is called at the start of
    ``RemotePE.infer()`` to normalize input before serialization.
    """
    import numpy as np

    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy(), dtype=np.float32)
    raise TypeError(
        f"Unsupported input type: {type(x).__name__}. "
        "Expected numpy.ndarray or an object with a .numpy() method."
    )
