"""paddlePE: Unified Pitch Extraction Toolkit for PaddlePaddle.

When imported, tries to load PaddlePaddle.  If the DLL load fails
(e.g. PyTorch already owns CUDA on Windows), falls back to
client-only mode with a subprocess server.

Use the :func:`paddlepe.logger.get_logger` function to obtain a
project-aware logger instead of ``print()``.
"""

from paddlepe.logger import get_logger, warn_once

logger = get_logger(__name__)

try:
    import paddle  # noqa: F401

    _PADDLE_AVAILABLE = True
except (OSError, ImportError) as e:
    _PADDLE_AVAILABLE = False
    warn_once(
        f"PaddlePaddle import failed ({e}). "
        "paddlePE runs in client-only mode (subprocess server).",
        key="paddle_unavailable",
    )

if _PADDLE_AVAILABLE:
    import paddlepe.models  # noqa: F401 — trigger model registration
    from paddlepe.pe import PE
    from paddlepe.registry import registry

    __all__ = ["PE", "registry"]
else:
    from paddlepe.client import ClientPE as PE  # type: ignore[assignment]

    __all__ = ["PE"]
