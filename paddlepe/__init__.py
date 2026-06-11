"""paddlePE: Unified Pitch Extraction Toolkit for PaddlePaddle.

When imported, tries to load PaddlePaddle. If the DLL load fails
(e.g. PyTorch already owns CUDA on Windows), falls back to
client-only mode with a subprocess server.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import paddle  # noqa: F401

    _PADDLE_AVAILABLE = True
except (OSError, ImportError) as e:
    _PADDLE_AVAILABLE = False
    logger.warning(
        "PaddlePaddle import failed (%s). "
        "paddlePE runs in client-only mode (subprocess server).",
        e,
    )

if _PADDLE_AVAILABLE:
    import paddlepe.models  # noqa: F401 — trigger model registration
    from paddlepe.pe import PE
    from paddlepe.registry import registry

    __all__ = ["PE", "registry"]
else:
    from paddlepe.remote import RemotePE as PE  # type: ignore[assignment]

    __all__ = ["PE"]
