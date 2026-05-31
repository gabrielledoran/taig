"""
logging.py
----------
Structured logging for TAIG.

All modules should obtain a logger via get_logger(__name__) rather than
calling logging.getLogger directly, so that format and level stay consistent.

Usage
-----
    from taig.utils.logging import get_logger

    log = get_logger(__name__)
    log.info("Processing %d messages", n)
    log.warning("Low confidence entity", extra={"entity": name, "score": score})
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

_INITIALIZED = False

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_FORMAT_PLAIN = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FORMAT_JSON = (
    '{"time": "%(asctime)s", "level": "%(levelname)s", '
    '"logger": "%(name)s", "message": "%(message)s"}'
)


def _configure_root(level: str, use_json: bool) -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. Jupyter re-runs the cell)
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    fmt = _FORMAT_JSON if use_json else _FORMAT_PLAIN
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)

    # Silence noisy third-party loggers at WARNING unless we are at DEBUG
    if level != "DEBUG":
        for noisy in ("transformers", "sentence_transformers", "bertopic", "umap"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_logging(
    level: LogLevel | None = None,
    json: bool | None = None,
) -> None:
    """Configure root logging. Call once at application entry point.

    Parameters
    ----------
    level:
        Log level string. Falls back to TAIG_LOG_LEVEL env var, then "INFO".
    json:
        Emit JSON lines if True. Falls back to TAIG_LOG_JSON env var ("1"/"true").
    """
    global _INITIALIZED
    resolved_level = (
        level
        or os.environ.get("TAIG_LOG_LEVEL", "INFO").upper()
    )
    resolved_json = json if json is not None else os.environ.get("TAIG_LOG_JSON", "0") in ("1", "true")
    _configure_root(resolved_level, resolved_json)
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Initializes root logging on first call."""
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
