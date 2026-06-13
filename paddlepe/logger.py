"""Logging utilities for paddlePE.

Provides a project-wide logger and helpers for one-time warnings.
Prevents raw print() statements scattered across the codebase.
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

# ── Project logger ──────────────────────────────────────────────
_NAME = "paddlepe"
_logger: logging.Logger | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Get the project logger or a named child logger.

    Examples:
        >>> from paddlepe.logger import get_logger
        >>> logger = get_logger(__name__)  # child of "paddlepe"
        >>> logger.info("model loaded")
    """
    return logging.getLogger(f"{_NAME}.{name}" if name else _NAME)


def configure_logging(
    level: int = logging.INFO,
    format_str: str | None = None,
    stream: Any = sys.stderr,
) -> None:
    """Configure the project logger.

    Call once at application entry point (CLI / server).  By default
    writes structured log lines to stderr so stdout stays clean for
    data output (e.g. CSV, JSON).
    """
    fmt = format_str or "[%(levelname)s] %(name)s: %(message)s"
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt))
    logger = get_logger()
    logger.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    _logger = logger


# ── One-time warnings ───────────────────────────────────────────
_sent_warnings: set[str] = set()


def warn_once(message: str, category: type = UserWarning, key: str | None = None) -> None:
    """Emit a warning only once per process (by message or explicit key).

    Use for import-time or config-time warnings that should not spam
    users on repeated calls (e.g. "Paddle unavailable → client mode").
    """
    k = key or message
    if k not in _sent_warnings:
        _sent_warnings.add(k)
        warnings.warn(message, category, stacklevel=2)


# ── Backward compatibility helpers ──────────────────────────────
# Modules that still use logger.warning/logger.info can migrate lazily
# by adding ``from paddlepe.logger import get_logger`` at the top.
