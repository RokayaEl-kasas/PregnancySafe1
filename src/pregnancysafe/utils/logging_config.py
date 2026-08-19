"""Central logging setup so every module (ingestion, retrieval, agent) logs
in a consistent format instead of ad-hoc print() calls scattered around."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly (e.g. once per module)."""
    global _CONFIGURED
    if not _CONFIGURED:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
